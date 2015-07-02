
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

use cgmath::{Ray3, Vector2, Vector3};
use core::spectrum::RGBSpectrum;

struct GeomDiff {
    pos: Vector3<f32>,
    normal: Vector3<f32>,
    ts: Vector3<f32>,
    ss: Vector3<f32>,
    elem_id: i32 //used to identify, who did the ray hit.
}

trait Intersectable {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff>;
}

struct Texture<T> {
    entries: Vec<T>,
    width: i32,
    height: i32
}

trait Camera {
    fn create_rays(&self, NDC: &[Vector2<f32>], rays: &[Ray3<f32>]);
}

trait Sampler {
    fn create_samples(&self, buffer:&mut[Vector2<f32>]);
}
struct Material {
    albedo : RGBSpectrum,
    metalness : f32,
    roughness : f32,
    emissiveness : f32
}
trait Integrator {
    fn radiance(geom_diffs : &[GeomDiff], materials : &[Material], throughput: &[RGBSpectrum]);
}

//dynamic dispatch, no point in having static one here. Thus this function is expected to be quite big
//all loops should be internal in the different components
fn render(sampler: &Sampler, camera: &Camera, scene: &Intersectable, shader: &Integrator, out : &mut Texture<RGBSpectrum>) {

    //sampler.create_samples();
}
