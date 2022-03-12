from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class SceneObject(ABC):
    color: np.ndarray

    @abstractmethod
    def smallest_positive_intersect(self, ray_dir: np.ndarray, ray_origin: np.ndarray) -> Optional[float]:
        """Return t where closest_intersection = t*ray_dir + ray_origin or None if no intersection.

        This function is supposed to only return if both roots are positive, since otherwise this would mean that
        the ray origin is inside the object.
        """
        ...


@dataclass
class Sphere(SceneObject):
    center: np.ndarray
    radius: float

    def smallest_positive_intersect(self, ray_dir, ray_origin) -> object:
        assert (np.isclose(np.linalg.norm(ray_dir), 1))
        dist_vec = ray_origin - self.center
        b = 2*dist_vec.dot(ray_dir)
        c = dist_vec.dot(dist_vec) - self.radius**2
        discriminant = b**2 - 4*c
        if discriminant >= 0:
            q = -0.5*(b + np.sign(b)*np.sqrt(discriminant))
            t1 = q
            t2 = c/q

            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None


def normalize(x, axis=None):
    return x/np.linalg.norm(x, axis=axis)


def find_closest_intersecting_object(objects: list[SceneObject], ray_dir: np.ndarray, ray_origin: np.ndarray) -> (
        SceneObject, float):
    """Find closest intersecting object and distance parameter t."""
    assert (np.isclose(np.linalg.norm(ray_dir), 1))

    t_min = np.inf
    closest_object = None
    for obj in objects:
        t = obj.smallest_positive_intersect(ray_dir, ray_origin)
        if t is not None and t < t_min:
            t_min = t
            closest_object = obj

    return closest_object, t_min


if __name__ == '__main__':

    width = 600
    height = 400
    ratio = width/height
    camera_pos = np.array([0, 0, -1])
    light_pos = np.array([4, 4, -4])
    scene_objects = [
        Sphere(color=np.array([1, 0, 0]), center=np.array([0.0, 0.0, 10.0]), radius=5),
        Sphere(color=np.array([0, 1, 0]), center=np.array([0.5, 0.4, 4]), radius=.4),
        # Sphere(color=np.array([0, 0, 1]), center=np.array([0.0, 0.0, 0.5]), radius=.1),
    ]

    pixels = np.zeros((width, height, 3))

    # Cartesian coordinates for screen in the 3d scene.
    x_coords = np.linspace(-1, 1, width)
    y_coords = np.linspace(-1/ratio, 1/ratio, height)

    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            pixel_pos = np.array([x, y, 0])
            primary_direction = normalize(pixel_pos - camera_pos)
            primary_origin = camera_pos

            nearest_object, t_min = find_closest_intersecting_object(scene_objects, primary_direction, primary_origin)
            if nearest_object is None:
                continue

            intersection = t_min*primary_direction + camera_pos
            # Todo: Return surface normal from find_closest_intersecting_object() instead of calculating it here.
            surface_normal = normalize(intersection - nearest_object.center)
            shadow_origin = intersection + 1e-5*surface_normal  # This places the origin slightly outside the object.
            shadow_direction = normalize(light_pos - shadow_origin)

            shadowing_object, t_min = find_closest_intersecting_object(scene_objects, shadow_direction, shadow_origin)
            is_shadowed = t_min < np.linalg.norm(light_pos - intersection)

            if is_shadowed:
                continue

            pixels[i, j] = nearest_object.color

    # Transform to screen coordinates.
    pixels = pixels.swapaxes(0, 1)
    pixels = pixels[::-1, :]

    # Show
    plt.imshow(pixels)
    plt.show()
