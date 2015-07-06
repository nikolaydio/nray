use cgmath::{Ray3,Vector,  Vector2, Vector3, Vector4, Point, Point3, Matrix4, Matrix, EuclideanVector, Sphere};


pub trait Intersectable {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff>;
}

pub struct GeomDiff {
    pub pos: Point3<f32>,
    pub n: Vector3<f32>,
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
                        n: poss.sub_p(&self.center).normalize(),
                        d: lesser,
                        mat_id: 0i32})
    }
}


pub struct ShadedIntersectable<T: Intersectable> {
    pub material_index: i32,
    pub intersectable: T
}
impl<T:Intersectable> Intersectable for ShadedIntersectable<T> {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff> {
        match self.intersectable.intersect(ray) {
            Some(geo) => Some( GeomDiff{mat_id: self.material_index, .. geo}),
            None => None
        }
    }
}

pub struct BruteForceContainer<T: Intersectable> {
    pub items: Vec<T>
}

impl<T:Intersectable> Intersectable for BruteForceContainer<T> {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff> {
        let mut best_candidate : Option<GeomDiff> = None;
        for i in self.items.iter() {
            let isect = i.intersect(ray);

            best_candidate = match best_candidate {
                Some(geom) => {
                    match isect {
                        Some(new_g) => if new_g.d < geom.d { Some(new_g) }
                                       else { Some(geom) },
                        None => Some(geom)
                    }
                },
                None => isect
            }
        }

        best_candidate
    }
}




pub struct Face {
    pub points: [Point3<f32>; 3]
}

impl Intersectable for Face {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff> {
        let (p0, p1, p2) = (self.points[0], self.points[1], self.points[2]);
        let e1 = p1.sub_p(&p0);
        let e2 = p2.sub_p(&p0);
        let s1 = ray.direction.cross(&e2);
        let divisor = s1.dot(&e1);

        if divisor.abs() < 0.000001 {
            return None;
        }
        let invDivisor = 1.0f32 / divisor;
        //first barycentric coord
        let d = ray.origin.sub_p(&p0);
        let b1 = d.dot(&s1) * invDivisor;
        if b1 < 0.0f32 || b1 > 1.0f32 {
            return None;
        }
        let s2 = d.cross(&e1);
        let b2 = ray.direction.dot(&s2) * invDivisor;
        if b2 < 0.0f32 || b2 > 1.0f32 {
            return None;
        }
        let t = e2.dot(&s2) * invDivisor;

        if t < 0.0001 {
            return None;
        }

        Some(GeomDiff { pos: ray.origin.add_v(&ray.direction.mul_s(t)),
                   d: t,
                   n: e1.cross(&e2).normalize() ,
                   mat_id: 0})
    }
}
