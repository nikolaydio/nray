#![feature(float_extras)]

extern crate cgmath;
use cgmath::{Vector, Vector2, Vector3, Vector4, zero, vec2, vec3, Point3, Sphere, Ray3};

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
							let v = tex.get(x,y);
							buffer[offset + 0] = (v.chans[0] / divisor * 255.0f32) as u8;
							buffer[offset + 1] = (v.chans[1] / divisor * 255.0f32) as u8;
							buffer[offset + 2] = (v.chans[2] / divisor * 255.0f32) as u8;
						}
					}
				}).unwrap();
				renderer.clear();
				renderer.copy(&texture, None, Some(Rect::new_unwrap(0, 0, tex.width as u32, tex.height as u32)));
				renderer.present();
			},
			Err(_) => ()
		}
		std::thread::sleep_ms(240);
	}
}

use core::renderer::{render, Sampler, GenericSampler, Camera, PinholeCamera, resolution_to_ndc, Texture, Material};
use core::scene::{Intersectable, Scene};
use core::spectrum::RGBSpectrum;
use std::sync::{Arc, Mutex, RwLock, mpsc};
fn main() {
	let resolution = Vector2::new(800, 600);
	let s = GenericSampler;
	let c = PinholeCamera::new(&Point3::new(0.0f32, 0.0f32, 0.0f32), &Point3::new(0.0f32, 0.0f32, 1.0f32), 60.0f32, resolution.x as f32 / resolution.y as f32);

	let scene = Scene {
		objects: vec![Sphere { center: Point3::new(0.0f32, 0.0f32, 5.0f32), radius: 1.0f32 },
						Sphere { center: Point3::new(3.0f32, 0.0f32, 5.0f32), radius: 0.5f32 }]
		};

	let materials : Vec<Material> = vec![Material {albedo: RGBSpectrum::white(), metalness: 0.0f32, roughness: 0.0f32, emissiveness: 0.6f32}];


	//render it
	let mut working_tex = Texture::<RGBSpectrum>::new(resolution.x, resolution.y, &RGBSpectrum::black());
	let samples_per_pixel = 64;
	let (tx, rx) = mpsc::channel();

	std::thread::spawn(move || {
		for i in 0..samples_per_pixel {

			render(&s, &c, &scene, &materials, &mut working_tex);
			tx.send((i+1, working_tex.clone()));
		}
	});
	present(resolution, rx);
}
