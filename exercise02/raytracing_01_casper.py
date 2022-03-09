# Simple ray tracing
# based on https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9

# Imports
import numpy as np
from matplotlib import pyplot as plt

# Functions

def normalize(vector):
    """Normalizes a vector"""
    return vector / np.linalg.norm(vector)

def sphere_intersect(center, radius, ray_origin, ray_direction):
    """Checks if a given ray intersects a sphere
    ray direction normalized -> a = 1"""
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta >= 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)  # Only return nearest intersection
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    """Finds the closest object that intersects with a ray"""
    distances = [sphere_intersect(obj["center"], obj["radius"], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

# Setup

width = 600
height = 400
ratio = float(width) / height

camera = np.array([0, 0, 1])
light = { "position": np.array([5, 5, 5]) }
# screen edges (left, top, right, bottom)
screen = (-1, 1/ratio, 1, -1/ratio)     # screen on x/y plane

image = np.zeros((height, width, 3))

# Add objects to scene
objects = [
    { "center": np.array([-0.2, 0, -1]), "radius": 0.7 , "color": np.array([0.5, 0.5, 0.5])},
    { "center": np.array([0.1, -0.3, 0]), "radius": 0.1 , "color": np.array([0, 1, 0])},
    { "center": np.array([-0.3, 0, 0]), "radius": 0.15 , "color": np.array([1, 0, 0])}
]

# Loop over all pixels
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        # Cast ray
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        # Check for intersections
        nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
        if nearest_object is None:
            continue

        # Compute intersection point between ray and nearest object
        intersection = origin + min_distance * direction
        normal_to_surface = normalize(intersection - nearest_object["center"])
        shifted_point = intersection + 1e-5 * normal_to_surface # so that we don't cast a ray from "inside" the object
        intersection_to_light = normalize(light["position"] - shifted_point)

        _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
        intersection_to_light_distance = np.linalg.norm(light["position"] - intersection)
        is_shadowed = min_distance < intersection_to_light_distance

        # Get color
        if not is_shadowed:
            pixel_color = nearest_object["color"]
            # Set image color
            image[i, j] = np.clip(pixel_color, 0, 1)


        # image[i, j] = ...
        print("%d/%d" % (i + 1, height))

# Saving image
plt.imsave("rt_image.png", image)
