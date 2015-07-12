use rand;
use rand::*;
use cgmath::{Vector, Vector2, Vector3, Vector4, Point, zero, vec2, vec3, Point3, Sphere, Ray3, Matrix4, Matrix, Aabb, Aabb3, EuclideanVector};
use core::bvh::{BVH};
use core::renderer::{render, Sampler, GenericSampler, Camera, PinholeCamera, resolution_to_ndc, Texture, Material};
use core::intersectable::{Intersectable, BruteForceContainer, ShadedIntersectable, Face, ObjTransform, GeomDiff, intersect_ray_aabb};
use core::spectrum::RGBSpectrum;
use std::sync::{mpsc};
use std::io::{self, BufRead, StdinLock};
use time;

#[test]
fn test_ray_aabb() {
    let aabb = Aabb3::new(Point3::new(-1.0f32,-1.0f32,-1.0f32),
                        Point3::new(1.0f32,1.0f32,1.0f32));
    let ray1 = Ray3::new(Point3::new(0.0f32, 0.0f32, -5.0f32), Vector3::new(0.0f32, 0.0f32, 1.0f32));
    let ray2 = Ray3::new(Point3::new(0.0f32, 0.0f32, -5.0f32), Vector3::new(0.0f32, 0.0f32, -1.0f32));
    let ray3 = Ray3::new(Point3::new(-5.0f32, 0.0f32, -5.0f32), Vector3::new(0.7f32, 0.0f32, 0.7f32));
    let ray4 = Ray3::new(Point3::new(-5.0f32, 0.0f32, -5.0f32), Vector3::new(-0.7f32, 0.0f32, 0.7f32));
    let ray5 = Ray3::new(Point3::new(-5.0f32, 0.0f32, -5.0f32), Vector3::new(0.0f32, 0.0f32, 1.0f32));
    assert!(intersect_ray_aabb(&ray1, &aabb) == true);
    assert!(intersect_ray_aabb(&ray2, &aabb) == false);
    assert!(intersect_ray_aabb(&ray3, &aabb) == true);
    assert!(intersect_ray_aabb(&ray4, &aabb) == false);
    assert!(intersect_ray_aabb(&ray5, &aabb) == false);
}

#[test]
fn test_aabb() {
    let a =     Aabb3 { min: Point3::from_vec(&Vector3::from_value(::std::f32::INFINITY)),
                max: Point3::from_vec(&Vector3::from_value(::std::f32::NEG_INFINITY))};
    let b = a.grow(&Point3::new(0.5f32, 0.5f32, 0.5f32));
    assert!(a.min.x == ::std::f32::INFINITY);
    assert!(b.min.x == 0.5f32);
    assert!(b.min.y == 0.5f32);
    assert!(b.min.z == 0.5f32);
    assert!(b.max.x == 0.5f32);
    assert!(b.max.y == 0.5f32);
    assert!(b.max.z == 0.5f32);
}

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
			(&Some(ref geom1), &Some(ref geom2)) => {
                assert!(geom1.d == geom2.d);
                assert!(geom1.pos == geom2.pos);
                assert!(geom1.n == geom2.n);
            },
			(&None, &None) => assert!(true),
			(&Some(ref geom1), &None) => assert!(false),
			(&None, &Some(ref geom2)) => assert!(false)
		}
	}
}
