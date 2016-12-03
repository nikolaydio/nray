
use std::f32;
use std::f32::INFINITY;
use std::f32::NEG_INFINITY;
use core::bvh::overbox;
use cgmath::prelude::*;
use cgmath::{Vector3, Point3, Matrix4, Vector2};
pub use collision::Aabb3;
pub use collision::{Ray, Ray2, Ray3};
use collision::Aabb;
use std::ops::*;
/* pub struct Ray3<S> {
    pub origin: Point3<S>,
    pub direction: Vector3<S>
} */

pub fn vec_length(p: Vector3<f32>) -> f32
{
    return (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
}

pub fn intersect_ray_aabb(ray: &Ray3<f32>, bbox: &Aabb3<f32>) -> bool {
    let dirfrac = Vector3::from_value(1.0f32).div_element_wise(ray.direction);

    let t1 = (bbox.min.x - ray.origin.x)*dirfrac.x;
    let t2 = (bbox.max.x - ray.origin.x)*dirfrac.x;
    let t3 = (bbox.min.y - ray.origin.y)*dirfrac.y;
    let t4 = (bbox.max.y - ray.origin.y)*dirfrac.y;
    let t5 = (bbox.min.z - ray.origin.z)*dirfrac.z;
    let t6 = (bbox.max.z - ray.origin.z)*dirfrac.z;

    let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
    let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

    if tmax < 0.0f32 {
        return false;
    }
    if tmin > tmax {
        return false;
    }
    return true;
}
pub trait Intersectable {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff>;
    fn bounding_box(&self) -> Aabb3<f32>;
    //fn sample(&self, seed: Vector2<f32>) -> Vector3<f32>;
}
impl<'a> Intersectable for &'a Intersectable {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff> {
        (*self).intersect(ray)
    }
    fn bounding_box(&self) -> Aabb3<f32> {
        (*self).bounding_box()
    }

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
unsafe impl Sync for GeomDiff {}



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
            item.bounding_box().grow(acc.min).grow(acc.max)
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
        let e1 = p1.sub(p0);
        let e2 = p2.sub(p0);
        let s1 = ray.direction.cross(e2);
        let divisor = s1.dot(e1);

        if divisor.abs() < 0.00001f32 {
            return None;
        }
        let invDivisor = 1.0f32 / divisor;
        //first barycentric coord
        let d = ray.origin.sub(p0);
        let b1 = d.dot(s1) * invDivisor;
        if b1 < 0.0f32 || b1 > 1.0f32 {
            return None;
        }
        let s2 = d.cross(e1);
        let b2 = ray.direction.dot(s2) * invDivisor;
        if b2 < 0.0f32 || b1 + b2 > 1.0f32 {
            return None;
        }
        let t = e2.dot(s2) * invDivisor;

        if t < 0.00001f32 {
            return None;
        }

        Some(GeomDiff { pos: ray.origin.add(ray.direction.mul(t)),
                   d: t,
                   n: e1.cross(e2).normalize() ,
                   mat_id: 0})
    }
    fn bounding_box(&self) -> Aabb3<f32> {
        Aabb3::new(self.points[0], self.points[1]).grow(self.points[2])
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
        let inner_ray = Ray3::new(Point3::from_vec(self.tform.mul(&(Point3::to_vec(ray.origin).extend(1.0f32))).truncate()),
                                self.tform.mul(ray.direction.extend(0.0f32)).truncate());
        let isect = self.next.intersect(inner_ray);

        match isect {
            Some(geo) =>  {
                let tm = self.tform.invert().unwrap();
                let outer_pos = Point3::from_vec(tm.mul(Point3::to_vec(geo.pos).extend(1.0f32)).truncate());
                Some(GeomDiff {
                    pos: outer_pos,
                    n: tm.mul(geo.n.extend(0.0f32)).truncate().normalize(),
                    d: vec_length(outer_pos.sub(ray.origin)),
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
            let p = Point3::to_vec(item).extend(1.0f32);
            let world_space_corner = Point3::from_vec(tm.mul(p).truncate());
            acc.grow(world_space_corner)
        })
    }
}
