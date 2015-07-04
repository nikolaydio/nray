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

fn present(tex: &Texture<RGBSpectrum>, times: f32) {
	let mut sdl_context = sdl2::init().video().unwrap();

	let window = sdl_context.window("nray", 640, 480)
		.position_centered()
		.opengl()
		.build()
		.unwrap();

	let mut renderer = window.renderer().build().unwrap();

	let mut texture = renderer.create_texture_streaming(PixelFormatEnum::RGB24, (tex.width as u32, tex.height as u32)).unwrap();
    // Create a red-green gradient
    texture.with_lock(None, |buffer: &mut [u8], pitch: usize| {
        for y in (0..tex.height) {
            for x in (0..tex.width) {
                let offset = y*pitch + x*3;
				let v = tex.get(x,y);
                buffer[offset + 0] = (v.chans[0] / times * 255.0f32) as u8;
                buffer[offset + 1] = (v.chans[1] / times * 255.0f32) as u8;
                buffer[offset + 2] = (v.chans[2] / times * 255.0f32) as u8;
            }
        }
    }).unwrap();

	renderer.set_draw_color(Color::RGB(0, 0, 0));
	renderer.clear();
	renderer.copy(&texture, None, Some(Rect::new_unwrap(0, 0, tex.width as u32, tex.height as u32)));
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
		std::thread::sleep_ms(200);
	}
}

use core::renderer::{render, Sampler, GenericSampler, Camera, PinholeCamera, resolution_to_ndc, Texture};
use core::scene::{Intersectable, Scene};
use core::spectrum::RGBSpectrum;
fn main() {
	let s = GenericSampler;
	let sams = s.create_samples(Vector2::new(10, 10));
	let ndcs = resolution_to_ndc(&sams, 10.0f32, 10.0f32);
	let c = PinholeCamera::new(&Point3::new(0.0f32, 0.0f32, 0.0f32), &Point3::new(0.0f32, 0.0f32, 1.0f32), 60.0f32, 640.0f32 / 480.0f32);
	let rays = c.create_rays(&ndcs);
	let scene = Scene {
		objects: vec![Sphere { center: Point3::new(0.0f32, 0.0f32, 2.0f32), radius: 1.0f32 },
						Sphere { center: Point3::new(3.0f32, 0.0f32, 2.0f32), radius: 0.5f32 }]
		};




	//render it
	let mut tex = Texture::<RGBSpectrum>::new(640, 480, &RGBSpectrum::black());
	let samples_per_pixel = 64;
	for i in 0..samples_per_pixel {
		render(&s, &c, &scene, &mut tex);
	}
	present(&tex, samples_per_pixel as f32);
}
