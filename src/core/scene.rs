

/*
trait Scene {
    //returns point on a light source
    fn sample_light_sources(&self, seed: &Vector3<f32>) -> Vector3<f32>;
}

struct GenericScene<'a, T:Intersectable, E: Intersectable> {
    lights: Vec<&'a E>,
    geometry: 'a T,
}

impl<T:Intersectable, E: Intersectable> Scene for GenericScene {
    fn sample_light_sources(&self, seed: &Vector3<f32>) -> Vector3<f32> {
        let l = (seed.z * self.lights.len() as f32) as usize;
        let light = self.lights[l];
        light.sample(seed.truncate())
    }
}
*/
