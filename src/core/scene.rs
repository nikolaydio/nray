use cgmath::{Ray3,Vector,  Vector2, Vector3, Vector4, Point, Point3, Matrix4, Matrix, EuclideanVector, Sphere};

pub trait Intersectable {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff>;
}
pub struct GeomDiff {
    pos: Point3<f32>,
    normal: Vector3<f32>,
    pub d: f32,
    pub mat_id: i32, //used to identify, who did the ray hit.
}
impl GeomDiff {
    fn gmad_id(&self) -> i32 {
        self.mat_id
    }
}


impl Intersectable for Sphere<f32> {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff> {
        let L = self.center.sub_p(&ray.origin);
        let tca = L.dot(&ray.direction);
        if tca < 0.0f32 {
            return None;
        }
        let d2 = L.dot(&L) - tca * tca;
        if d2 > self.radius {
            return None;
        }

        let thc = (self.radius - d2).sqrt();
        let t0 = tca - thc;
        let t1 = tca + thc;

        let lesser = t0.min(t1);
        if lesser < 0.0f32 {
            return None;
        }

        let poss = ray.origin.add_v(&ray.direction.mul_s(lesser));
        Some(GeomDiff { pos: poss,
                        normal: poss.sub_p(&self.center).normalize(),
                        d: lesser,
                        mat_id: 0i32})
    }
}

pub struct Scene {
    pub objects: Vec<Sphere<f32>>
}
impl Intersectable for Scene {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff> {
        let mut best_candidate : Option<GeomDiff> = None;
        for i in self.objects.iter() {
            best_candidate = match best_candidate {
                Some(geom) => {
                    match i.intersect(ray) {
                        Some(new_g) => if new_g.d < geom.d { Some(new_g) }
                                       else { Some(geom) },
                        None => Some(geom)
                    }
                },
                None => i.intersect(ray)
            }
        }

        best_candidate
    }
}
