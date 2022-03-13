use itertools_num::linspace;
use nalgebra::{DMatrix, point, Point3, UnitVector3, vector, Vector3};

use f64 as Real;

#[derive(Debug)]
struct Ray {
    origin: Point3<Real>,
    direction: UnitVector3<Real>,
}

impl Ray {
    fn new(origin: Point3<Real>, direction: UnitVector3<Real>) -> Self {
        Self { origin, direction }
    }
}

trait SceneObject {
    /// Return t where closest_intersection = t*ray_dir + ray_origin or None if no intersection.
    ///
    /// This function is supposed to only return if both roots are positive, since
    /// otherwise this would mean that the ray origin is inside the object.
    fn smallest_positive_intersect(&self, ray: Ray) -> Option<Real>;

    fn color(&self) -> &Vector3<Real>;
}

struct Sphere {
    color: Vector3<Real>,
    center: Point3<Real>,
    radius: Real,
}

impl Sphere {
    fn new(color: Vector3<Real>, center: Point3<Real>, radius: Real) -> Self {
        Sphere { color, center, radius }
    }
}

impl SceneObject for Sphere {
    fn smallest_positive_intersect(&self, ray: Ray) -> Option<Real> {
        todo!()
    }

    fn color(&self) -> &Vector3<Real> {
        &self.color
    }
}

fn main() {
    let width = 600;
    let height = 400;
    let ratio = width as Real / height as Real;

    let camera_pos = point![0_f64, 0., -1.];
    let light_pos = point![4., 4., -3.];
    let scene_objects = vec![
        Sphere::new(vector![1., 0., 0.], point![0.0, 0.0, 10.0], 5.),
        // Sphere(color = np.array([0, 1, 0]), center = np.array([0.5, 0.4, 3.5]), radius =.4),
        // Sphere(color = np.array([0, 1, 0.7]), center = np.array([-0.5, 0.4, 4.5]), radius =.4),
    ];

    let pixels = DMatrix::<f32>::zeros(height, width);

    'outer:
    for (i, x) in linspace::<Real>(-1., 1., width).enumerate() {
        for (j, y) in linspace::<Real>(-1. / ratio, 1. / ratio, height).enumerate() {
            let pixel_pos = point![x, y, 0.];
            let direction = UnitVector3::new_normalize(pixel_pos - camera_pos);
            let primary = Ray::new(camera_pos, direction);
            println!("{:?}", primary);

            break 'outer;
        }
    }
}
