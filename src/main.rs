#![feature(float_extras)]

extern crate cgmath;
use cgmath::{Vector, Vector2, Vector3, Vector4, zero, vec2, vec3};

extern crate sdl2;
use sdl2::pixels::Color;
use sdl2::keyboard::Keycode;

extern crate rand;

mod core;
use std::mem;

fn main() {
	println!("Sizeof array {}", std::mem::size_of::<i32>());

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
