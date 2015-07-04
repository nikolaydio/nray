
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

struct GeomDiff {
    pos: Vector3<f32>,
    normal: Vector3<f32>,
    ts: Vector3<f32>,
    ss: Vector3<f32>,
    mat_id: i32, //used to identify, who did the ray hit.
}

trait Intersectable {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff>;
}

struct Texture<T> {
    pub entries: Vec<T>,
    pub width: usize,
    pub height: usize
}
use std::ops::{Add, Sub, Mul};
impl<T> Texture<T> {
    pub fn get(&self, x: usize, y: usize) -> &T {
        &self.entries[y * self.width + x]
    }
    pub fn get_mut(&mut self, x: usize, y:usize) -> &mut T {
        &mut self.entries[y * self.width + x]
    }
    pub fn set(&mut self, x: usize, y:usize, t: T) {
        self.entries[y * self.width + x] = t;
    }
}
fn add_to_texture(out : &mut Texture<RGBSpectrum>, idx: usize, c : RGBSpectrum) {
    let v = out.entries[idx] + c;
    out.entries[idx] = v;
}


trait Camera {
    fn create_rays(&self, NDC: &[Vector2<f32>]) -> Vec<Ray3<f32>>;
}
struct PinholeCamera {
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

            let direction = translated_scaled.extend(1.0f32).normalize();

            Ray3::new(vector3_to_point(&self.cam_to_world.mul_v(&origin).truncate()),
                      (self.cam_to_world.mul_v(&direction.extend(0.0f32))).truncate())
        }).collect()
    }
}
impl PinholeCamera {
    fn new(eye: &Point3<f32>, lookat: &Point3<f32>, fov: f32, aspect_ratio: f32) -> PinholeCamera {
        let mut m = Matrix4::<f32>::look_at(eye, lookat, &Vector3::new(0.0f32, 1.0f32, 0.0f32));
        m.invert_self();

        let fov = Vector2::new(fov, fov / aspect_ratio);
        let half_fov = fov.div_s(2.0f32);

        let coeff_x = 1.0f32 / (90.0f32 - half_fov.x).to_radians().sin();
        let coeff_y = 1.0f32 / (90.0f32 - half_fov.y).to_radians().sin();

        let _fov = Vector2::new(coeff_x * half_fov.x.to_radians().sin(), coeff_y * half_fov.y.to_radians().sin());

        PinholeCamera { cam_to_world: m, field_of_view: _fov }
    }
}

trait Sampler {
    fn create_samples(&self, resolution: Vector2<i32>) -> Vec<Vector2<f32>>;
}
struct GenericSampler;
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
struct Material {
    albedo : RGBSpectrum,
    metalness : f32,
    roughness : f32,
    emissiveness : f32
}
impl Material {
    fn bounce_ray(&self, r: Ray3<f32>, gd: &GeomDiff, tp: &mut RGBSpectrum) -> Ray3<f32> {
        Ray3::new(Point3::new(0f32,0f32,0f32), Vector3::new(1f32,1f32,1f32))
    }
}
trait Integrator {
    fn radiance(&self, geom_diffs : &[GeomDiff], materials : &[Material], throughput: &[RGBSpectrum]);
}


fn resolution_to_ndc(buffer: &[Vector2<f32>], width: f32, height: f32) -> Vec<Vector2<f32>> {
    let resolution = Vector2::new(width, height);
    buffer.iter().map(|elem| { *elem / resolution }).collect()
}


//dynamic dispatch, no point in having static one here. Thus this function is expected to be quite big
//all loops should be internal in the different components
fn render(sampler: &Sampler, camera: &Camera, scene: &Intersectable, shader: &Integrator, out : &mut Texture<RGBSpectrum>) {
    let elems = out.width * out.height;

    let samples = sampler.create_samples(Vector2::new(out.width as i32, out.height as i32));

    //translate resolution to NDC
    let ndc = resolution_to_ndc(&samples[..], out.width as f32, out.height as f32);

    //idx, ray tuples for processing
    let mut ray_pool : Vec<(usize, Ray3<f32>)> = camera.create_rays(&ndc[..]).iter().map(|&e|e).
    enumerate().collect();

    let mut throughputs = vec![RGBSpectrum::white(); elems];
    let materials : Vec<Material> = Vec::new();
    for i in 0..3 {
        let intersections : Vec<(usize, GeomDiff, Ray3<f32>)> = ray_pool.iter()
        .filter_map(|&(idx, ray)| {
            match scene.intersect(ray) {
                Some(g) => Some((idx, g, ray)),
                None => None
            }
        }).collect();

        //choose secondary rays and light sampling rays
        ray_pool = intersections.iter().map(|&(idx, ref geo, ray)| {
            let ref material : Material = materials[geo.mat_id as usize];
            //add the emissive part
            add_to_texture(out, idx, throughputs[idx] * (material.albedo * material.emissiveness));

            //calculate the secondary ray and return it, update throughput
            (idx, material.bounce_ray(ray, geo, &mut throughputs[idx]))
        }).collect();
    }

}
