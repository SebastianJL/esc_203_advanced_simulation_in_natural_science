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

    fn color(&self) -> Vector3<Real>;
    fn normal(&self, coord: Vector3<Real>) -> Vector3<Real>;
}

struct Sphere {
    color: Vector3<Real>,
    center: Vector3<Real>,
    radius: Real,
}

impl Sphere {
    fn new(color: Vector3<Real>, center: Vector3<Real>, radius: Real) -> Self {
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

    fn color(&self) -> Vector3<Real> {
        self.color
    }

    fn normal(&self, intersection: Vector3<Real>) -> Vector3<Real> {
        (intersection - self.center).normalize()
    }
}

struct Light {
    position: Vector3<Real>,
    color: Vector3<Real>,
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
    let width = 600;
    let height = 400;
    let ratio = width as Real / height as Real;

    let camera_pos = vector![0_f64, 0., -1.];
    let lights = vec![
        Light { position: vector![4., 4., -3.], color: vector![1., 1., 1.] }
    ];
    let scene_objects: Vec<Box<dyn SceneObject>> = vec![
        Box::new(Sphere::new(vector![1., 0., 0.], vector![0.0, 0.0, 10.0], 5.)),
        Box::new(Sphere::new(vector![0., 1., 0.], vector![0.5, 0.4, 3.5], 0.4)),
        Box::new(Sphere::new(vector![0., 1., 0.7], vector![-0.5, 0.4, 4.5], 0.4)),
        Box::new(Sphere::new(vector![0., 1., 1.], vector![0.7, 0.7, 2.5], 0.1)),
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

            let mut light_rays: Vec<Ray> = vec![];
            for light in lights.iter() {
                let shadow_origin = intersection + 1e-5 * surface_normal;
                let shadow = Ray::new(
                    shadow_origin,
                    (light.position - shadow_origin).normalize(),
                );

                let (_shadowing_object, t_min) = find_closest_intersecting_object(
                    &scene_objects,
                    &shadow,
                );
                let is_shadowed = t_min < (light.position - intersection).norm();
                if !is_shadowed {
                    light_rays.push(shadow);
                }
            }

            if light_rays.is_empty() {
                continue;
            }

            let rgb_value: [u8; 3]= (nearest_object.color() * 255.)
                .iter()
                .cloned()
                .map(|x| x as u8) // Saturating cast.
                .collect::<Vec<u8>>()
                .try_into().unwrap();
            pixels.put_pixel(
                i as u32,
                height - 1 - j as u32,
                image::Rgb(rgb_value),
            );
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


#[cfg(test)]
mod tests {
    use image::io::Reader as ImageReader;

    #[test]
    fn compare_image() {
        crate::render();
        let img1 = ImageReader::open("rt_image.png").unwrap().decode().unwrap().into_rgb8();
        let img2 = ImageReader::open("rt_image_reference.png").unwrap().decode().unwrap().into_rgb8();
        let width = img1.width();
        let height = img1.height();
        assert_eq!(width, img2.width());
        assert_eq!(height, img2.height());

        for (x, y, p1) in img1.enumerate_pixels() {
            let p2 = img2.get_pixel(x, y);
            assert_eq!(p1, p2);
        }
    }
}
