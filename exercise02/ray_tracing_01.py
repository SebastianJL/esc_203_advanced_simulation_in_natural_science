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

    width = 300
    height = 200
    ratio = width/height
    camera = np.array([0, 0, -1])
    pixels = np.zeros((height, width, 3))

    x_coords = np.linspace(-1, 1, width)
    y_coords = np.linspace(-1/ratio, 1/ratio, height)

    objects = [
        Sphere(color=np.array([0.2, 0.3, 0.9]), center=np.array([0., -1., 5.]), radius=2.)
    ]

    light = np.array([1, 4, 4])

    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            pixel = np.array([x, y, 0])
            primary_direction = normalize(pixel - camera)
            primary_origin = camera

            nearest_object, t_min = find_closest_intersecting_object(objects, primary_direction, primary_origin)
            if nearest_object is None:
                continue

            intersection = t_min*primary_direction + camera
            shadow_direction = normalize(light - intersection)
            shadow_origin = intersection

            shadowing_object, t_min = find_closest_intersecting_object(objects, shadow_direction, shadow_origin)
            is_shadowed = t_min < np.linalg.norm(light - intersection)

            if is_shadowed:
                continue

            pixels[j, i] = nearest_object.color

    pixels = pixels[::-1, :]
    plt.imshow(pixels)
    plt.show()
