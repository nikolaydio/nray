
//define operations
//1. Generate samples
// (samples_per_pixel, area) + sampler_function -> plane_coordinates
//2. Transform coords
// (plane_coordinates, resolution) + generic_trnfm_fn -> plane_coordinates_ndc
//3. Calculate world space rays
// (plane_coordinates_ndc, view_matrix, hfov, vfov) + cam_function -> primary_rays
//4. Intersection(4. rays = primary_rays)
// (scene_geometry, rays) + traversal + intersection -> diff_geoms
//5. Shading sub-process
// (diff_geoms, materials, throughput) + path_integrator -> (secondary_rays, throughput)
//6. loop 2 times + round robin or until emissive object
// Intersection(rays = secondary_rays)
// Shading sub-process
//7. Calculate spectrums
// (throughput, emission) + multiplication -> SPDs
//8. Average over pixels
// (SPDs) + avg -> SPD_pixel
//9. Map to XYZ
// (SPD_pixel, XYZ graphs) -> XYZ_pixel
//10. XYZ to sRGB
// (XYZ_pixel, sRGB graphs) -> sRGB_pixel


use cgmath::{Ray3,Vector,  Vector2, Vector3, Vector4, Point3, Matrix4, Matrix, EuclideanVector};
use core::spectrum::RGBSpectrum;
use rand::Rng;
use rand;
use core::scene::{Intersectable, GeomDiff};
use std::f32::consts::{PI, FRAC_1_PI};

use std::ops::{Add, Sub, Mul};
use std::clone::Clone;

#[derive(Clone)]
pub struct Texture<T> {
    pub entries: Vec<T>,
    pub width: usize,
    pub height: usize
}



impl<T: Clone> Texture<T> {
    pub fn get(&self, x: usize, y: usize) -> &T {
        &self.entries[y * self.width + x]
    }
    pub fn get_mut(&mut self, x: usize, y:usize) -> &mut T {
        &mut self.entries[y * self.width + x]
    }
    pub fn set(&mut self, x: usize, y:usize, t: T) {
        self.entries[y * self.width + x] = t;
    }
    pub fn new(width: usize, height: usize, default: &T) -> Texture<T> {
        Texture::<T> { width: width, height: height, entries: vec![default.clone(); width * height] }
    }
}
fn add_to_texture(out : &mut Texture<RGBSpectrum>, idx: usize, c : RGBSpectrum) {
    let v = out.entries[idx] + c;
    out.entries[idx] = v;
}


pub trait Camera {
    fn create_rays(&self, NDC: &[Vector2<f32>]) -> Vec<Ray3<f32>>;
}
pub struct PinholeCamera {
    cam_to_world: Matrix4<f32>,
    field_of_view: Vector2<f32>
}
fn vector3_to_point(v : &Vector3<f32>) -> Point3<f32> {
    Point3::new(v.x, v.y, v.z)
}
impl Camera for PinholeCamera {
    fn create_rays(&self, NDC: &[Vector2<f32>]) -> Vec<Ray3<f32>> {
        NDC.iter().map(|&ndc| {
            let origin = Vector4::<f32>::new(0.0f32, 0.0f32, 0.0f32, 1.0f32);
            let translated = ndc - Vector2::<f32>::new(0.5f32, 0.5f32);
            let translated_scaled = translated * self.field_of_view;

            let mut direction = translated_scaled.extend(1.0f32).normalize();
            direction.z = -direction.z;

            Ray3::new(vector3_to_point(&self.cam_to_world.mul_v(&origin).truncate()),
                      (self.cam_to_world.mul_v(&direction.extend(0.0f32))).truncate())
        }).collect()
    }
}
impl PinholeCamera {
    pub fn new(eye: &Point3<f32>, lookat: &Point3<f32>, fov: f32, aspect_ratio: f32) -> PinholeCamera {
        let mut m = Matrix4::<f32>::look_at(eye, lookat, &Vector3::new(0.0f32, 1.0f32, 0.0f32));
        m.invert_self();

        let fov = Vector2::new(fov, fov / aspect_ratio);
        let half_fov = fov.div_s(2.0f32);

        let coeff_x = 1.0f32 / (90.0f32 - half_fov.x).to_radians().sin();
        let coeff_y = 1.0f32 / (90.0f32 - half_fov.y).to_radians().sin();

        let _fov = Vector2::new(coeff_x * half_fov.x.to_radians().sin(), coeff_y * half_fov.y.to_radians().sin());

        PinholeCamera { cam_to_world: m, field_of_view: _fov.mul_s(2.0f32) }
    }
}

pub trait Sampler {
    fn create_samples(&self, resolution: Vector2<i32>) -> Vec<Vector2<f32>>;
}
pub struct GenericSampler;
impl Sampler for GenericSampler {
    fn create_samples(&self, resolution: Vector2<i32>) ->  Vec<Vector2<f32>>{
        let count = (resolution.x * resolution.y);
        let mut samples = Vec::<Vector2<f32>>::new();
        let mut rng = rand::thread_rng();
        for i in 0..count {
            let pixel = Vector2::new((i % resolution.x) as f32, (i / resolution.x) as f32);
            let rand_range = Vector2::new(rng.gen::<f32>(), rng.gen::<f32>());

            samples.push(pixel + rand_range);
        }
        samples
    }
}
pub struct Material {
    pub albedo : RGBSpectrum,
    pub metalness : f32,
    pub roughness : f32,
    pub emissiveness : f32
}
fn uniform_sample_hemisphere(seed: Vector2<f32>) -> (Vector3<f32>, f32) {
    let z = seed.x;
    let r = 0f32.max(1.0f32 - z * z).sqrt();
    let phi = 2.0f32 * PI * seed.y;
    let x = r * phi.cos();
    let y = r * phi.sin();
    (Vector3::new(x,y,z), FRAC_1_PI / 2.0f32)
}

impl Material {
    fn eval_brdf(&self, wo: Vector3<f32>, wi: Vector3<f32>) -> RGBSpectrum {
        self.albedo * FRAC_1_PI
    }
    //return spectrum and pdf
    fn sample_brdf(&self, wo: Vector3<f32>, seed: Vector2<f32>) -> (RGBSpectrum, Vector3<f32>, f32) {
        let (new_dir, pdf) = uniform_sample_hemisphere(seed);
        (self.eval_brdf(wo, new_dir), new_dir, pdf)
    }
    fn bounce_ray(&self, r: Ray3<f32>, gd: &GeomDiff, seed: Vector2<f32>, tp: &mut RGBSpectrum) -> Ray3<f32> {
        //convert everything to local shading space
        let temp_u = if gd.n.x.abs() > 0.1f32 { Vector3::new(0.0f32, 1.0f32, 0.0f32) } else { Vector3::new(1.0f32, 0.0f32, 0.0f32) };
        let u = temp_u.cross(&gd.n).normalize();
        let v = gd.n.cross(&u);
        let wo = -Vector3::new(r.direction.dot(&u), r.direction.dot(&v), r.direction.dot(&gd.n));

        //shading calcs
        let (spec, wi, pdf) = self.sample_brdf(wo, seed);
        *tp = (*tp) * (spec * wi.z.abs() / pdf);

        //convert back to world space
        let dir = Vector3::new(u.x * wi.x + v.x * wi.y + gd.n.x * wi.z,
                               u.y * wi.x + v.y * wi.y + gd.n.y * wi.z,
                               u.z * wi.x + v.z * wi.y + gd.n.z * wi.z);
        Ray3::new(gd.pos, dir)
    }
}
trait Integrator {
    fn radiance(&self, geom_diffs : &[GeomDiff], materials : &[Material], throughput: &[RGBSpectrum]);
}


pub fn resolution_to_ndc(buffer: &[Vector2<f32>], width: f32, height: f32) -> Vec<Vector2<f32>> {
    let resolution = Vector2::new(width, height);
    buffer.iter().map(|elem| { *elem / resolution }).collect()
}


//dynamic dispatch, no point in having static one here. Thus this function is expected to be quite big
//all loops should be internal in the different components
pub fn render(sampler: &Sampler, camera: &Camera, scene: &Intersectable, materials: &Vec<Material>, out : &mut Texture<RGBSpectrum>) {
    let mut rng = rand::thread_rng();
    let elems = out.width * out.height;

    let samples = sampler.create_samples(Vector2::new(out.width as i32, out.height as i32));
    println!("Generated {} samples", elems);

    //translate resolution to NDC
    let ndc = resolution_to_ndc(&samples[..], out.width as f32, out.height as f32);
    println!("Translated them to NDC");

    //idx, ray tuples for processing
    let mut ray_pool : Vec<(usize, Ray3<f32>)> = camera.create_rays(&ndc[..]).iter().map(|&e|e).
    enumerate().collect();
    println!("Created {} primary rays", ray_pool.len());

    let mut throughputs = vec![RGBSpectrum::white(); elems];

    for i in 0..3 {
        let intersections : Vec<(usize, GeomDiff, Ray3<f32>)> = ray_pool.iter()
        .filter_map(|&(idx, ray)| {
            match scene.intersect(ray) {
                Some(g) => Some((idx, g, ray)),
                None => None
            }
        }).collect();
        println!("From which {} intersected with geometry", intersections.len());

        if i == 2 {
            break;
        }
        //choose secondary rays and light sampling rays
        ray_pool = intersections.iter().map(|&(idx, ref geo, ray)| {
            let ref material : Material = materials[geo.mat_id as usize];
            //add the emissive part
            add_to_texture(out, idx, throughputs[idx] * (material.albedo * material.emissiveness));

            //calculate the secondary ray and return it, update throughput
            (idx, material.bounce_ray(ray, geo, Vector2::new(rng.gen::<f32>(), rng.gen::<f32>()), &mut throughputs[idx]))
        }).collect();
        println!("Generated {} secondary rays for bounce {}", ray_pool.len(), i);
    }

}
