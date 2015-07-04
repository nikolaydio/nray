#![feature(float_extras)]

extern crate cgmath;
use cgmath::{Vector, Vector2, Vector3, Vector4, zero, vec2, vec3, Point3, Sphere, Ray3};

extern crate sdl2;
use sdl2::pixels::Color;
use sdl2::keyboard::Keycode;

extern crate rand;

mod core;
use std::mem;

fn present() {
	let mut sdl_context = sdl2::init().video().unwrap();

	let window = sdl_context.window("nray", 640, 480)
		.position_centered()
		.opengl()
		.build()
		.unwrap();

	let mut renderer = window.renderer().build().unwrap();

	renderer.set_draw_color(Color::RGB(255, 255, 255));
	renderer.clear();
	renderer.present();

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
		// The rest of the game loop goes here...
	}
}

use core::renderer::{Sampler, GenericSampler, Camera, PinholeCamera, resolution_to_ndc};
use core::scene::{Intersectable, Scene};
fn main() {
	let s = GenericSampler;
	let sams = s.create_samples(Vector2::new(10, 10));
	let ndcs = resolution_to_ndc(&sams, 10.0f32, 10.0f32);
	let c = PinholeCamera::new(&Point3::new(0.0f32,0.0f32,0.0f32), &Point3::new(0.0f32, 0.0f32, 1.0f32), 60.0f32, 4.0f32 / 4.0f32);
	let rays = c.create_rays(&ndcs);
	let scene = Scene { objects: vec![Sphere { center: Point3::new(0.0f32, 1.5f32, 2.0f32), radius: 1.0f32 }; 1] };
	let gd = scene.intersect(Ray3::<f32>::new(Point3::new(0.0f32, 0.0f32, 0.0f32), Vector3::new(0.0f32, 0.0f32, 1.0f32)));
	match gd {
			Some(gd) => println!("{}", gd.d),
			None => println!("nope")
	}
	for r in rays {
		//println!("{} {} {}", gd.);
	}
}
