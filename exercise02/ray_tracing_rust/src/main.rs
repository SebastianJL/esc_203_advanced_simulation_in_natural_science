use std::cmp;
use std::time::Instant;

use image::ImageBuffer;
use itertools_num::linspace;
use nalgebra::{DMatrix, point, Point3, UnitVector3, vector, Vector3};

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

    fn color(&self) -> [u8; 3];
}

struct Sphere {
    color: [u8; 3],
    center: Vector3<Real>,
    radius: Real,
}

impl Sphere {
    fn new(color: [u8; 3], center: Vector3<Real>, radius: Real) -> Self {
        Sphere { color, center, radius }
    }
}

impl SceneObject for Sphere {
    fn smallest_positive_intersect(&self, ray: &Ray) -> Option<Real> {
        let dist_vec = &ray.origin - &self.center;
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

    fn color(&self) -> [u8; 3] {
        self.color
    }
}

fn find_closest_intersecting_object<'a>(objects: &'a [Sphere], ray: &Ray) -> (Option<&'a Sphere>, Real) {
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
    let scene_objects = vec![
        Sphere::new([255, 0, 0], vector![0.0, 0.0, 10.0], 5.),
        Sphere::new([0, 255, 0], vector![0.5, 0.4, 3.5], 0.4),
        Sphere::new([0, 255, 170], vector![-0.5, 0.4, 4.5], 0.4),
        // Sphere(color = np.array([0, 1, 0]), center = np.array([0.5, 0.4, 3.5]), radius =.4),
        // Sphere(color = np.array([0, 1, 0.7]), center = np.array([-0.5, 0.4, 4.5]), radius =.4),
    ];

    let mut pixels: ImageBuffer<image::Rgb<u8>, _> = ImageBuffer::new(width, height);

    'outer:
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
            // Todo: Return surface normal from find_closest_intersecting_object().
            let surface_normal = (intersection - &nearest_object.center).normalize();
            let shadow_origin = &intersection + 1e-5 * &surface_normal;
            let shadow = Ray::new(
                shadow_origin,
                (&light_pos - &shadow_origin).normalize(),
            );

            let (shadowing_object, t_min) = find_closest_intersecting_object(
                &scene_objects,
                &shadow,
            );
            let is_shadowed = t_min < (light_pos - intersection).norm();
            if is_shadowed {
                continue;
            }

            pixels.put_pixel(i as u32, height - 1 - j as u32, image::Rgb(nearest_object.color));
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
