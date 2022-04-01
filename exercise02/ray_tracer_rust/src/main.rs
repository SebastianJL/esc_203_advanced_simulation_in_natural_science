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

#[derive(Debug, Clone)]
struct SurfaceParameters {
    // Ambient
    a: Real,
    // Diffuse
    d: Real,
    // Specular
    s: Real,
    // Specular phong exponent
    sp: Real,
    // Specular metalness
    sm: Real,
}

trait SceneObject {
    /// Return t where closest_intersection = t*ray_dir + ray_origin or None if no intersection.
    ///
    /// This function is supposed to only return if both roots are positive, since
    /// otherwise this would mean that the ray origin is inside the object.
    fn smallest_positive_intersect(&self, ray: &Ray) -> Option<Real>;

    fn color(&self) -> Vector3<Real>;
    fn normal(&self, coord: Vector3<Real>) -> Vector3<Real>;
    fn surface_parameters(&self) -> SurfaceParameters;
}

struct Sphere {
    color: Vector3<Real>,
    surface_parameters: SurfaceParameters,
    center: Vector3<Real>,
    radius: Real,
}

impl Sphere {
    fn new(color: Vector3<Real>, center: Vector3<Real>, radius: Real) -> Self {
        Sphere {
            color,
            surface_parameters: SurfaceParameters {
                a: 1.,
                d: 1.,
                s: 1.,
                sp: 40.,
                sm: 0.2,
            },
            center,
            radius,
        }
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

    fn surface_parameters(&self) -> SurfaceParameters {
        self.surface_parameters.clone()
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

fn phong_shading(light_rays: &Vec<Ray>, lights: Vec<&Light>, normal: Vector3<Real>, view_direction:
Vector3<Real>, object: &Box<dyn SceneObject>) -> Vector3<Real> {
    #![allow(non_snake_case)]
    // Source for mathematical model are the lecture notes for phong shading.
    let V = view_direction;
    let N = normal;
    let SurfaceParameters { a: k_a, d: k_d, s: k_s, sp: k_sp, sm: k_sm } = object
        .surface_parameters();
    let mut sum: Vector3<Real> = Vector3::zeros();
    for (light_ray, light) in light_rays.into_iter().zip(lights) {
        let L = light_ray.direction;
        let R = 2. * L.dot(&N) * N - L;

        // Diffuse reflection.
        let diffuse_color = k_d * light.color.component_mul(&object.color()) * Real::max(L.dot(&N), 0.);
        sum += diffuse_color;

        // Specular reflection.
        let specular_highlight_color: Vector3<Real> = k_sm * object.color() + (1. - k_sm) * vector![1., 1.,1.];
        let specular_color = k_s * specular_highlight_color.component_mul(&light.color) *
            Real::max(R.dot(&V), 0.).powf(k_sp);
        sum += specular_color;
    }

    // Ambient light.
    let ambient_light = vector![0.3,0.3,0.3];  // Todo: Remove hard coding of ambient lighting.
    let ambient_color = k_a * object.color().component_mul(&ambient_light);
    sum += ambient_color;

    sum
}

fn trace(lights: &Vec<Light>, scene_objects: &Vec<Box<dyn SceneObject>>, ray: &Ray, max_recursion: u8) -> Option<Vector3<Real>> {
    let (nearest_object, t_min) = find_closest_intersecting_object(&scene_objects, &ray);
    if nearest_object.is_none() {
        return None;
    }
    let nearest_object = nearest_object.unwrap();

    let intersection = ray.direction * t_min + ray.origin;
    let surface_normal = nearest_object.normal(intersection);

    // Find light sources which are not shadowed by an object.
    let mut light_rays: Vec<Ray> = vec![];
    let mut active_lights: Vec<&Light> = vec![];
    for light in lights.iter() {
        // Move out intersection slightly to avoid self intersection problem.
        let light_ray_origin = intersection + 1e-5 * surface_normal;
        let light_ray = Ray::new(
            light_ray_origin,
            (light.position - light_ray_origin).normalize(),
        );

        let (_shadowing_object, t_min) = find_closest_intersecting_object(
            &scene_objects,
            &light_ray,
        );
        let is_shadowed = t_min < (light.position - intersection).norm();
        if !is_shadowed {
            light_rays.push(light_ray);
            active_lights.push(light);
        }
    }

    let phong_color: Vector3<Real> = phong_shading(
        &light_rays,
        active_lights,
        surface_normal,
        -ray.direction,
        nearest_object);

    // Todo: Trace reflected ray.

    // Todo: Trace refracted ray.

    // Todo: Mix colors
    let mixed_color = phong_color;

    Some(mixed_color)
}

fn render() {
    let multiplier = 1;
    let width = multiplier * 600;
    let height = multiplier * 400;
    let ratio = width as Real / height as Real;
    let screen_size = 2.;
    let max_recursion = 1;
    let bg_color = [170; 3];
    let camera_pos = vector![0. as Real, 0., -4.];
    let lights = vec![
        Light { position: vector![4., 4., -3.], color: vector![1., 1., 1.] }
    ];
    let scene_objects: Vec<Box<dyn SceneObject>> = vec![
        Box::new(Sphere::new(vector![1., 0., 0.], vector![0.0, 0.0, 10.0], 5.)),
        Box::new(Sphere::new(vector![0., 1., 0.], vector![0.5, 1.1, 3.5], 0.4)),
        Box::new(Sphere::new(vector![0., 1., 0.7], vector![-0.5, 0.4, 4.5], 0.4)),
        Box::new(Sphere::new(vector![0., 1., 1.], vector![1.2, 1.8, 2.0], 0.1)),
        Box::new(Sphere::new(vector![0.2, 0.2, 0.2], vector![2.5, 0.2, 2.5], 0.2)),
    ];

    let mut pixels: ImageBuffer<image::Rgb<u8>, _> = ImageBuffer::from_fn(
        width, height, |_, _| image::Rgb(bg_color),
    );

    for (i, x) in linspace::<Real>(-screen_size, screen_size, width as usize).enumerate() {
        for (j, y) in linspace::<Real>(-screen_size / ratio, screen_size / ratio, height as
            usize).enumerate() {
            let pixel_pos = vector![x, y, 0.];
            let direction = (pixel_pos - camera_pos).normalize();
            let primary_ray = Ray::new(camera_pos, direction);

            let color = match trace(&lights, &scene_objects, &primary_ray, max_recursion) {
                Some(color) => color,
                None => continue,
            };

            let rgb_value: [u8; 3] = (color * 255.)
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
