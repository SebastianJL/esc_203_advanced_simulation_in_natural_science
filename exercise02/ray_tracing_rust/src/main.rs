use std::time::Instant;

use image::ImageBuffer;
use itertools_num::linspace;
use nalgebra::{vector, Vector3};

use f64 as Real;

#[derive(Debug)]
struct Ray {
    origin: Vector3<Real>,
    direction: Vector3<Real>,
}

impl Ray {
    fn new(origin: Vector3<Real>, direction: Vector3<Real>) -> Self {
        Self { origin, direction }
    }
}

trait SceneObject {
    /// Return t where closest_intersection = t*ray_dir + ray_origin or None if no intersection.
    ///
    /// This function is supposed to only return if both roots are positive, since
    /// otherwise this would mean that the ray origin is inside the object.
    fn smallest_positive_intersect(&self, ray: &Ray) -> Option<Real>;

    fn color(&self) -> Vector3<u8>;
    fn normal(&self, coord: Vector3<Real>) -> Vector3<Real>;
}

struct Sphere {
    color: Vector3<u8>,
    center: Vector3<Real>,
    radius: Real,
}

impl Sphere {
    fn new(color: Vector3<u8>, center: Vector3<Real>, radius: Real) -> Self {
        Sphere { color, center, radius }
    }
}

impl SceneObject for Sphere {
    fn smallest_positive_intersect(&self, ray: &Ray) -> Option<Real> {
        let dist_vec = ray.origin - self.center;
        let b = 2. * dist_vec.dot(&ray.direction);
        let c = dist_vec.dot(&dist_vec) - self.radius.powi(2);
        let discriminant = b.powi(2) - 4. * c;
        if discriminant >= 0. {
            let q = -0.5 * (b + b.signum() * discriminant.sqrt());
            let t1 = q;
            let t2 = c / q;

            if t1 > 0. && t2 > 0. {
                return Some(t1.min(t2));
            }
        }
        None
    }

    fn color(&self) -> Vector3<u8> {
        self.color
    }

    fn normal(&self, intersection: Vector3<Real>) -> Vector3<Real> {
        (intersection - self.center).normalize()
    }
}

fn find_closest_intersecting_object<'a>(
    objects: &'a Vec<Box<dyn SceneObject>>, ray: &Ray)
    -> (Option<&'a Box<dyn SceneObject>>, Real) {
    let mut t_min = Real::INFINITY;
    let mut closest_object = None;

    for object in objects {
        let t_opt = object.smallest_positive_intersect(ray);
        if let Some(t) = t_opt {
            if t < t_min {
                t_min = t;
                closest_object = Some(object);
            }
        }
    }
    (closest_object, t_min)
}

fn render() {
    let width = 2400;
    let height = 1600;
    let ratio = width as Real / height as Real;

    let camera_pos = vector![0_f64, 0., -1.];
    let light_pos = vector![4., 4., -3.];
    let scene_objects: Vec<Box<dyn SceneObject>> = vec![
        Box::new(Sphere::new(vector![255, 0, 0], vector![0.0, 0.0, 10.0], 5.)),
        Box::new(Sphere::new(vector![0, 255, 0], vector![0.5, 0.4, 3.5], 0.4)),
        Box::new(Sphere::new(vector![0, 255, 170], vector![-0.5, 0.4, 4.5], 0.4)),
        Box::new(Sphere::new(vector![0, 255, 255], vector![0.7, 0.7, 2.5], 0.1)),
    ];

    let mut pixels: ImageBuffer<image::Rgb<u8>, _> = ImageBuffer::new(width, height);

    for (i, x) in linspace::<Real>(-1., 1., width as usize).enumerate() {
        for (j, y) in linspace::<Real>(-1. / ratio, 1. / ratio, height as usize).enumerate() {
            let pixel_pos = vector![x, y, 0.];
            let direction = (pixel_pos - camera_pos).normalize();
            let primary = Ray::new(camera_pos, direction);

            let (nearest_object, t_min) = find_closest_intersecting_object(&scene_objects, &primary);
            if nearest_object.is_none() {
                continue;
            }
            let nearest_object = nearest_object.unwrap();

            let intersection = primary.direction * t_min + camera_pos;
            let surface_normal = nearest_object.normal(intersection);
            let shadow_origin = intersection + 1e-5 * surface_normal;
            let shadow = Ray::new(
                shadow_origin,
                (light_pos - shadow_origin).normalize(),
            );

            let (_shadowing_object, t_min) = find_closest_intersecting_object(
                &scene_objects,
                &shadow,
            );
            let is_shadowed = t_min < (light_pos - intersection).norm();
            if is_shadowed {
                continue;
            }

            pixels.put_pixel(
                i as u32, height - 1 - j as u32,
                image::Rgb(nearest_object.color().into()
                ));
        }
    }

    pixels.save("rt_image.png").unwrap();
}

fn main() {
    let start = Instant::now();
    render();
    let duration = start.elapsed();
    println!("{:?}", duration);
}
