#![feature(float_extras)]

extern crate cgmath;
use cgmath::{Vector, Vector2, Vector3, Vector4, zero, vec2, vec3, Point3, Sphere, Ray3, Matrix4};

extern crate sdl2;
use sdl2::pixels::Color;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;

extern crate rand;

mod core;
use std::mem;

fn present(res : Vector2<usize>, rx : std::sync::mpsc::Receiver<(u32, Texture<RGBSpectrum>)>) {
	let mut sdl_context = sdl2::init().video().unwrap();

	let window = sdl_context.window("nray", res.x as u32, res.y as u32)
		.position_centered()
		.opengl()
		.build()
		.unwrap();

	let mut renderer = window.renderer().build().unwrap();
	renderer.set_draw_color(Color::RGB(0, 0, 0));


	let mut running = true;

	while running {
		for event in sdl_context.event_pump().poll_iter() {
			use sdl2::event::Event;

			match event {
				Event::Quit {..} | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
					running = false
				},
				_ => {}
			}
		}

		let data = rx.try_recv();
		match data {
			Ok(d) => {
				let (times, tex) = d;
				let mut texture = renderer.create_texture_streaming(PixelFormatEnum::RGB24, (tex.width as u32, tex.height as u32)).unwrap();

				let divisor = times as f32;
				texture.with_lock(None, |buffer: &mut [u8], pitch: usize| {
					for y in (0..tex.height) {
						for x in (0..tex.width) {
							let offset = y*pitch + x*3;
							let v : RGBSpectrum = *tex.get(x,y) / divisor;
							let (r,g,b) = v.to_sRGB();
							buffer[offset + 0] = r;
							buffer[offset + 1] = g;
							buffer[offset + 2] = b;
						}
					}
				}).unwrap();
				renderer.clear();
				renderer.copy(&texture, None, Some(Rect::new_unwrap(0, 0, tex.width as u32, tex.height as u32)));
				renderer.present();
			},
			Err(_) => ()
		}
		std::thread::sleep_ms(120);
	}
}

use core::renderer::{render, Sampler, GenericSampler, Camera, PinholeCamera, resolution_to_ndc, Texture, Material};
use core::intersectable::{Intersectable, BruteForceContainer, ShadedIntersectable, Face};
use core::spectrum::RGBSpectrum;
use std::sync::{mpsc};
use std::io::{self, BufRead, StdinLock};


//util functions for read_input_data
fn read_value<I: std::iter::Iterator<Item=String>, T : std::str::FromStr + cgmath::BaseNum>(rdr: &mut I) -> T {
	rdr.next().unwrap().parse::<T>().ok().unwrap()
}
fn read_vector2<I: std::iter::Iterator<Item=String>, T : std::str::FromStr + cgmath::BaseNum>(rdr: &mut I) -> Vector2<T> {
	Vector2::new(rdr.next().unwrap().parse::<T>().ok().unwrap(),
				rdr.next().unwrap().parse::<T>().ok().unwrap())
}
//returns samples, resolution, camera transform, fov, materials, geometry(intersectable/scene)
fn read_input_data(source: StdinLock) -> (u32, Vector2<usize>, Matrix4<f32>, f32, Vec<Material>, Box<Intersectable>) {
	let mut rdr = source.lines().flat_map(|e| e.unwrap().split(' ').map(|s| s.to_string()).collect::<Vec<String>>());
	//read magic
	let mgc : String = rdr.next().unwrap();
	if mgc != "NRAY-INTERNAL" {
		panic!("Unrecognized input format.");
	}
	//read samples
	let samples : u32 = read_value(&mut rdr);

	let resolution : Vector2<usize> = read_vector2(&mut rdr);

	let fov : f32 = read_value(&mut rdr);

	let materials : Vec<Material> = vec![];
	println!("{}", samples);
	let geo : BruteForceContainer<&Intersectable> = BruteForceContainer { items: vec![] };
	(samples, resolution, Matrix4::<f32>::identity(), fov, materials, Box::new(geo))
}

fn main() {
	let sin = io::stdin();
	let (samples, resolution, cam_transform, fov, materials, scene) = read_input_data(sin.lock());
	let headless = false;

	let s = GenericSampler;
	//let c = PinholeCamera::new(&Point3::new(0.0f32, 0.0f32, 0.0f32), &Point3::new(0.0f32, 0.0f32, 1.0f32), 60.0f32, resolution.x as f32 / resolution.y as f32);
	let c = PinholeCamera::new(&cam_transform, fov, resolution.x as f32 / resolution.y as f32);


	//let materials : Vec<Material> = vec![Material {albedo: RGBSpectrum::white(), metalness: 0.3f32, roughness: 0.0001f32, emissiveness: 0.0f32},
	//									Material {albedo: RGBSpectrum::white(), metalness: 0.0f32, roughness: 0.0f32, emissiveness: 1.0f32},
	//									Material {albedo: RGBSpectrum::from_sRGB(255, 64, 64), metalness: 0.0f32, roughness: 0.0f32, emissiveness: 0.0f32}];


	//render it
	let mut working_tex = Texture::<RGBSpectrum>::new(resolution.x, resolution.y, &RGBSpectrum::black());
	//let samples_per_pixel = 024;
	let (tx, rx) = mpsc::channel();

	let th = if !headless {
		std::thread::spawn(move || {
			present(resolution, rx);
		})
	}else{
		std::thread::spawn(|| {})
	};
	for i in 0..samples {
		render(&s, &c, &*scene, &materials, &mut working_tex);
		if i % 4 == 0 {
			tx.send((i+1, working_tex.clone()));
		}
	}
	th.join();
}
