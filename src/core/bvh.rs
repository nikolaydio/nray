use core::intersectable::{Intersectable, intersect_ray_aabb, GeomDiff};
use cgmath::{Vector3, Vector, Ray3, Aabb, Point, Point3};
use cgmath::Aabb3;
use std::clone::Clone;

pub struct PInfo {
    idx : usize,
    bbox: Aabb3<f32>,
}

pub enum Node {
    Interior(Aabb3<f32>, Box<Node>, Box<Node>),
    Leaf(Aabb3<f32>, usize, usize)
}
fn longest_extend(v: Vector3<f32>) -> u32 {
    if v.x > v.y {
        if v.x > v.z {
            0
        }else{
            2
        }
    }else {
        if v.y > v.z {
            1
        }else {
            2
        }
    }
}

fn overbox() -> Aabb3<f32> {
    Aabb3 { min: Point3::from_vec(&Vector3::from_value(::std::f32::INFINITY)),
            max: Point3::from_vec(&Vector3::from_value(::std::f32::NEG_INFINITY))}
}

fn ray_bbox(ray: Ray3<f32>, bbox: Aabb3<f32>) -> bool {
    true
}


pub struct BVH<T: Intersectable+Clone> {
    pub root: Box<Node>,
    pub primitives : Vec<T>,
}
impl<T:Intersectable+Clone> Intersectable for BVH<T> {
    fn intersect(&self, ray: Ray3<f32>) -> Option<GeomDiff> {
        BVH::<T>::hit(&self.primitives[..], &*self.root, ray)
    }
    fn bounding_box(&self) -> Aabb3<f32> {
        match *self.root {
            Node::Interior(aabb, _, _) => aabb,
            Node::Leaf(aabb, _, _) => aabb
        }
    }
}
impl<T: Intersectable+Clone> BVH<T> {
    pub fn new(max_prims_node: u32, primitives : &mut Vec<T>) -> BVH<T> {
        let mut working_set : Vec<PInfo> = primitives.iter().enumerate().map(|(i, p)| {
            PInfo { idx: i, bbox: p.bounding_box() }
        }).collect();
        let root = BVH::<T>::build(&mut working_set[..], 0, max_prims_node);
        let mut new_prims = Vec::<T>::with_capacity(primitives.len());

        for i in 0..primitives.len() {
            new_prims.push(primitives[working_set[i].idx].clone());
        }
        BVH { root: root, primitives: new_prims}
    }
    pub fn build(elems: &mut [PInfo], offset: usize, max_prims_node: u32) -> Box<Node> {
        let bbox = elems.iter().fold(overbox(), |acc, item| {
            acc.grow(&item.bbox.min()).grow(&item.bbox.max())
        });
        if elems.len() == 1 {
            Box::new(Node::Leaf(bbox, offset, elems.len()))
        }else {
            //find bounding centroid
            let bcent = elems.iter().fold(overbox(), |acc, item| {
                acc.grow(&item.bbox.center())
            });
            let dimensions = bcent.max().sub_p(bcent.min());
            let axis = longest_extend(dimensions) as usize;

            let mid = elems.len() / 2;
            //return if no volume
            if bcent.max()[axis] == bcent.min()[axis] {
                if elems.len() < max_prims_node as usize {
                    return Box::new(Node::Leaf(bbox, offset, elems.len()))
                }else{
                    return Box::new(Node::Interior(bbox,
                        BVH::<T>::build(&mut elems[0..mid], offset, max_prims_node),
                        BVH::<T>::build(&mut elems[mid..], offset + mid, max_prims_node)))
                }
            }

            //split em, but first - sort
            elems.sort_by(|a, b| { a.bbox.center()[axis].partial_cmp(&b.bbox.center()[axis]).unwrap_or(::std::cmp::Ordering::Equal) });
            Box::new(Node::Interior(bbox,
                BVH::<T>::build(&mut elems[0..mid], offset, max_prims_node),
                BVH::<T>::build(&mut elems[mid..], offset + mid, max_prims_node)))
        }
    }
    pub fn hit(primitives: &[T], node: &Node, ray: Ray3<f32>) -> Option<GeomDiff> {
        match node {
            &Node::Interior(bbox, ref left, ref right) => {
                if !intersect_ray_aabb(ray, bbox) {
                    return None;
                }
                let lh = BVH::<T>::hit(primitives, &*left, ray);
                let rh = BVH::<T>::hit(primitives, &*right, ray);
                match lh {
                    Some(lgeo) => {
                        match rh {
                            Some(rgeo) => {
                                if lgeo.d < rgeo.d {
                                    Some(lgeo)
                                }else {
                                    Some(rgeo)
                                }
                            },
                            None => Some(lgeo)
                        }
                    },
                    None => {
                        match rh {
                            Some(rgeo) => Some(rgeo),
                            None => None
                        }
                    }
                }
            },
            &Node::Leaf(bbox, start, count) => {
                if !intersect_ray_aabb(ray, bbox) {
                    return None;
                }
                //brute force all in the list
                primitives[start..start+count].iter().fold(None, |acc, item| {
                    let isect = item.intersect(ray);
                    match acc {
                        Some(geom) => {
                            match isect {
                                Some(ngeom) => if geom.d < ngeom.d { Some(geom) } else { Some(ngeom) },
                                None => Some(geom)
                            }
                        },
                        None => isect
                    }
                })
            }
        }
    }
}
