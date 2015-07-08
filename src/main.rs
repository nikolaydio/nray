#![feature(float_extras)]

extern crate cgmath;
use cgmath::{Vector, Vector2, Vector3, Vector4, Point, zero, vec2, vec3, Point3, Sphere, Ray3, Matrix4, Matrix};

extern crate sdl2;
use sdl2::pixels::Color;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;

extern crate rand;

mod core;
use std::mem;

fn headless_present(res : Vector2<usize>, rx : std::sync::mpsc::Receiver<(u32, Texture<RGBSpectrum>)>) {
	loop {
		let data = rx.recv();
		match data {
			Ok((times, tex)) => {
				let divisor = times as f32;
				for y in (0..tex.height) {
					for x in (0..tex.width) {
						let v : RGBSpectrum = *tex.get(tex.width-x-1,tex.height-y-1) / divisor;
						let (r,g,b) = v.to_sRGB();
						println!("{} {} {}", r, g, b);
					}
				}
			},
			Err(E) => break
		}
	}
}
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
							let v : RGBSpectrum = *tex.get(tex.width-x-1, y) / divisor;
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
use core::intersectable::{Intersectable, BruteForceContainer, ShadedIntersectable, Face, ObjTransform};
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
fn read_vector3<I: std::iter::Iterator<Item=String>, T : std::str::FromStr + cgmath::BaseNum>(rdr: &mut I) -> Vector3<T> {
	Vector3::new(rdr.next().unwrap().parse::<T>().ok().unwrap(),
				rdr.next().unwrap().parse::<T>().ok().unwrap(),
				rdr.next().unwrap().parse::<T>().ok().unwrap())
}
fn read_vector4<I: std::iter::Iterator<Item=String>, T : std::str::FromStr + cgmath::BaseNum>(rdr: &mut I) -> Vector4<T> {
	Vector4::new(rdr.next().unwrap().parse::<T>().ok().unwrap(),
				rdr.next().unwrap().parse::<T>().ok().unwrap(),
				rdr.next().unwrap().parse::<T>().ok().unwrap(),
				rdr.next().unwrap().parse::<T>().ok().unwrap())
}
fn read_matrix4<I: std::iter::Iterator<Item=String>, T : std::str::FromStr + cgmath::BaseFloat>(rdr: &mut I) -> Matrix4<T> {
	Matrix4::from_cols(read_vector4(rdr),
				read_vector4(rdr),
				read_vector4(rdr),
				read_vector4(rdr)).transpose()
}
fn read_a_material<I: std::iter::Iterator<Item=String>>(rdr: &mut I) -> Material {
	let c : Vector3<f32> = read_vector3(rdr);
	let metalness : f32 = read_value(rdr);
	let roughness : f32 = read_value(rdr);
	let emissiveness : f32 = read_value(rdr);
	Material { albedo: RGBSpectrum::from_sRGB((c.x * 255.0f32) as u8, (c.y * 255.0f32) as u8, (c.z * 255.0f32) as u8),
	 		   metalness: metalness,
				roughness: roughness,
				emissiveness: emissiveness}
}
fn read_materials<I: std::iter::Iterator<Item=String>>(rdr: &mut I) -> Vec<Material> {
	let mut materials = Vec::<Material>::new();
	let count : u32 = read_value(rdr);
	for i in 0..count {
		let mat : Material = read_a_material(rdr);
		materials.push(mat);
	}
	materials
}
fn read_a_mesh<I: std::iter::Iterator<Item=String>>(rdr: &mut I) -> BruteForceContainer<Face> {
	let count : u32 = read_value(rdr);
	let mut faces = Vec::<Face>::new();
	for i in 0..count {
		let p0 : Vector3<f32> = read_vector3(rdr);
		let p1 : Vector3<f32> = read_vector3(rdr);
		let p2 : Vector3<f32> = read_vector3(rdr);
		faces.push(Face { points: [Point3::<f32>::from_vec(&p0), Point3::<f32>::from_vec(&p1), Point3::<f32>::from_vec(&p2)] });
	}
	BruteForceContainer {
		items: faces
	}
}
fn read_meshes<I: std::iter::Iterator<Item=String>>(rdr: &mut I) -> Vec<BruteForceContainer<Face>> {
	let mut meshes = Vec::<BruteForceContainer<Face>>::new();
	let count : u32 = read_value(rdr);
	for i in 0..count {
		meshes.push(read_a_mesh(rdr));
	}
	meshes
}
fn read_scene<'a, I: std::iter::Iterator<Item=String>>(rdr: &mut I, meshes: & Vec<BruteForceContainer<Face>>)
		-> BruteForceContainer<ShadedIntersectable<ObjTransform<BruteForceContainer<Face>>>> {
	let count : u32 = read_value(rdr);
	let mut objs = Vec::<ShadedIntersectable<ObjTransform<BruteForceContainer<Face>>>>::new();
	//println!("{}", count);
	for i in 0..count {
		let mat_id : i32 = read_value(rdr);
		let mesh_id : u32 = read_value(rdr);

		let transform : Matrix4<f32> = read_matrix4(rdr).invert().unwrap();
		objs.push(ShadedIntersectable { material_index: mat_id,
			 	  intersectable:  ObjTransform {
					tform: transform,
					next: meshes[mesh_id as usize].clone() }
				});
	}
	BruteForceContainer{ items: objs }
}

//returns samples, resolution, camera transform, fov, materials, geometry(intersectable/scene)
fn read_input_data(source: StdinLock)
	-> (u32, Vector2<usize>, Matrix4<f32>, f32, Vec<Material>, BruteForceContainer<ShadedIntersectable<ObjTransform<BruteForceContainer<Face>>>>) {
	let mut rdr = source.lines().flat_map(|e| e.unwrap().split_terminator(' ').map(|s| s.to_string()).collect::<Vec<String>>());
	//read magic
	let mgc : String = rdr.next().unwrap();
	if mgc != "NRAY-INTERNAL" {
		panic!("Unrecognized input format.");
	}
	//read samples
	let samples : u32 = read_value(&mut rdr);

	let resolution : Vector2<usize> = read_vector2(&mut rdr);

	let cam_transform : Matrix4<f32> = read_matrix4(&mut rdr);

	let fov : f32 = read_value(&mut rdr);

	let materials : Vec<Material> = read_materials(&mut rdr);

	let meshes : Vec<BruteForceContainer<Face>> = read_meshes(&mut rdr);


	let geo = read_scene(&mut rdr, &meshes);
	(samples, resolution, cam_transform.invert().unwrap(), fov, materials, geo)
}

use std::env;

fn main() {
	let sin = io::stdin();
	let (samples, resolution, cam_transform, fov, materials, scene) = read_input_data(sin.lock());
	//println!("Scene - loaded.");
	let headless = match env::args().nth(1) {
		Some(txt) => txt == "headless",
		None => false
	};

	let s = GenericSampler;
	//let c = PinholeCamera::new_hf(&Point3::new(0.0f32, 2.0f32, -4.0f32), &Point3::new(0.0f32, 0.0f32, 1.0f32), 60.0f32, resolution.x as f32 / resolution.y as f32);
	let c = PinholeCamera::new(&cam_transform, fov, resolution.x as f32 / resolution.y as f32);


	//let materials : Vec<Material> = vec![Material {albedo: RGBSpectrum::white(), metalness: 0.3f32, roughness: 0.0001f32, emissiveness: 0.0f32},
	//									Material {albedo: RGBSpectrum::white(), metalness: 0.0f32, roughness: 0.0f32, emissiveness: 1.0f32},
	//									Material {albedo: RGBSpectrum::from_sRGB(255, 64, 64), metalness: 0.0f32, roughness: 0.0f32, emissiveness: 0.0f32}];


	//render it
	let mut working_tex = Texture::<RGBSpectrum>::new(resolution.x, resolution.y, &RGBSpectrum::black());
	//let samples_per_pixel = 024;
	let (tx, rx) = mpsc::channel();


	let th = std::thread::spawn(move || {
		if !headless {
			present(resolution, rx);
		}else {
			headless_present(resolution, rx);
		}
	});
	for i in 0..samples {
		render(&s, &c, &scene, &materials, &mut working_tex);
		if headless {
			if i % 4 == 0 {
				tx.send((i+1, working_tex.clone()));
			}
		}else{
			if i % 4 == 0 {
				tx.send((i+1, working_tex.clone()));
			}
		}
	}
	th.join();
}
