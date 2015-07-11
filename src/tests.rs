use rand;
use rand::*;
use cgmath::{Vector, Vector2, Vector3, Vector4, Point, zero, vec2, vec3, Point3, Sphere, Ray3, Matrix4, Matrix, Aabb, Aabb3, EuclideanVector};
use core::bvh::{BVH};
use core::renderer::{render, Sampler, GenericSampler, Camera, PinholeCamera, resolution_to_ndc, Texture, Material};
use core::intersectable::{Intersectable, BruteForceContainer, ShadedIntersectable, Face, ObjTransform, GeomDiff};
use core::spectrum::RGBSpectrum;
use std::sync::{mpsc};
use std::io::{self, BufRead, StdinLock};
use time;

#[test]
fn test_containers() {
	let mut primitives = Vec::<Face>::new();
	let mut rng = rand::thread_rng();
	for i in 0..10000 {
		primitives.push(Face { points: [Point3::new(rng.gen::<f32>(),rng.gen::<f32>(),rng.gen::<f32>()),
											Point3::new(rng.gen::<f32>(),rng.gen::<f32>(),rng.gen::<f32>()),
											Point3::new(rng.gen::<f32>(),rng.gen::<f32>(),rng.gen::<f32>())] });
	}
	let bvh = BVH::new(64, &mut primitives);
	let bf = BruteForceContainer { items: primitives };

	let rays : Vec<Ray3<f32>> = (0..20000).map(|i| {
		Ray3::<f32>::new(
			Point3::new(rng.gen::<f32>(),rng.gen::<f32>(),rng.gen::<f32>()).mul_s(5.0f32),
			Vector3::new(rng.gen::<f32>(),rng.gen::<f32>(),rng.gen::<f32>()).normalize())
	}).collect();

	let start = time::precise_time_s();
	let bf_results : Vec<Option<GeomDiff>> = rays.iter().map(|r| {
		bf.intersect(*r)
	}).collect();
	let end = time::precise_time_s();
	println!("BF {}", end - start);

	let start = time::precise_time_s();
	let bvh_results : Vec<Option<GeomDiff>> = rays.iter().map(|r| {
		bvh.intersect(*r)
	}).collect();
	let end = time::precise_time_s();
	println!("BVH {}", end - start);


	for (bf_r, bvh_r) in bf_results.iter().zip(bvh_results.iter()) {
		let isect1 : &Option<GeomDiff> = bf_r;
		let isect2 : &Option<GeomDiff> = bvh_r;
		match (isect1, isect2) {
			(&Some(ref geom1), &Some(ref geom2)) => assert!(geom1.d == geom2.d),
			(&None, &None) => assert!(true),
			(&Some(ref geom1), &None) => assert!(false),
			(&None, &Some(ref geom2)) => assert!(false)
		}
	}
}
