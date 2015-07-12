use cgmath::{Ray3,Vector,  Vector2, Vector3, Vector4, Point, Point3, Matrix4, Matrix, EuclideanVector, Sphere, Transform,
Aabb, Aabb3};
use std::f32;
use std::f32::INFINITY;
use std::f32::NEG_INFINITY;
use core::bvh::overbox;

pub fn intersect_ray_aabb(ray: &Ray3<f32>, bbox: &Aabb3<f32>) -> bool {
    let dirfrac = Vector3::from_value(1.0f32).div_v(&ray.direction);

    let t1 = (bbox.min().x - ray.origin.x)*dirfrac.x;
    let t2 = (bbox.max().x - ray.origin.x)*dirfrac.x;
    let t3 = (bbox.min().y - ray.origin.y)*dirfrac.y;
    let t4 = (bbox.max().y - ray.origin.y)*dirfrac.y;
    let t5 = (bbox.min().z - ray.origin.z)*dirfrac.z;
    let t6 = (bbox.max().z - ray.origin.z)*dirfrac.z;

    let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
    let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

    if (tmax < 0.0f32) {
        return false;
    }
    if (tmin > tmax){
        return false;
    }
    return true;
}
pub trait Intersectable {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff>;
    fn bounding_box(&self) -> Aabb3<f32>;
}
impl<'a> Intersectable for &'a Intersectable {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff> {
        (*self).intersect(ray)
    }
    fn bounding_box(&self) -> Aabb3<f32> {
        (*self).bounding_box()
    }
}

//unsafe impl<'a> Sync for Intersectable { }


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
unsafe impl Sync for GeomDiff {}

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
    fn bounding_box(&self) -> Aabb3<f32> {
        let diag = Vector3::new(self.radius, self.radius, self.radius);
        Aabb3::new(self.center.add_v(&(-diag)), self.center.add_v(&diag))
    }
}

#[derive(Clone)]
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
    fn bounding_box(&self) -> Aabb3<f32> {
        self.intersectable.bounding_box()
    }
}

#[derive(Clone)]
pub struct BruteForceContainer<T: Intersectable> {
    pub items: Vec<T>
}

impl<T: Intersectable> Intersectable for BruteForceContainer<T> {
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
    fn bounding_box(&self) -> Aabb3<f32> {
        self.items.iter().fold(overbox(), |acc, item| {
            item.bounding_box().grow(acc.min()).grow(acc.max())
        })
    }
}




#[derive(Clone)]
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

        if divisor.abs() < 0.00001f32 {
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
        if b2 < 0.0f32 || b1 + b2 > 1.0f32 {
            return None;
        }
        let t = e2.dot(&s2) * invDivisor;

        if t < 0.00001f32 {
            return None;
        }

        Some(GeomDiff { pos: ray.origin.add_v(&ray.direction.mul_s(t)),
                   d: t,
                   n: e1.cross(&e2).normalize() ,
                   mat_id: 0})
    }
    fn bounding_box(&self) -> Aabb3<f32> {
        Aabb3::new(self.points[0], self.points[1]).grow(&self.points[2])
    }
}

#[derive(Clone)]
pub struct ObjTransform<T : Intersectable> {
    pub tform: Matrix4<f32>,
    pub next: T
}
impl<T : Intersectable> Intersectable for ObjTransform<T> {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff> {
        //transform ray into obj space
        let inner_ray = Ray3::new(Point3::from_vec(&self.tform.mul_v(&(Point3::to_vec(&ray.origin).extend(1.0f32))).truncate()),
                                self.tform.mul_v(&ray.direction.extend(0.0f32)).truncate());
        let isect = self.next.intersect(inner_ray);

        match isect {
            Some(geo) =>  {
                let tm = self.tform.invert().unwrap();
                let outer_pos = Point3::from_vec(&tm.mul_v(&Point3::to_vec(&geo.pos).extend(1.0f32)).truncate());
                Some(GeomDiff {
                    pos: outer_pos,
                    n: tm.mul_v(&geo.n.extend(0.0f32)).truncate().normalize(),
                    d: outer_pos.sub_p(&ray.origin).length(),
                    mat_id: geo.mat_id
                    })
            },
            None => None
        }
    }
    fn bounding_box(&self) -> Aabb3<f32> {
        let bbox = self.next.bounding_box();
        let tm = self.tform.invert().unwrap();
        let corners = bbox.to_corners();


        corners.iter().fold(overbox(), |acc, &item| {
            let p = Point3::to_vec(&item).extend(1.0f32);
            let world_space_corner = Point3::from_vec(&tm.mul_v(&p).truncate());
            acc.grow(&world_space_corner)
        })
    }
}
