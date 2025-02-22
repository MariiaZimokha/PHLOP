import random
import numpy as np
import json


def set_position_and_velocity(obj):
    collision_radius = random.uniform(1, 2)

    if obj["mode"] == "collision":
        angle = random.uniform(0, 2 * np.pi)
        x = collision_radius * np.cos(angle)
        y = collision_radius * np.sin(angle)
        speed = random.uniform(2, 5)
        obj["velocity"] = [
            -speed * np.cos(angle),
            -speed * np.sin(angle),
            random.uniform(-0.2, 0.2),
        ]

    if obj["mode"] == "sliding":
        r = collision_radius * np.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 2 * np.pi)
        x, y = r * np.cos(theta), r * np.sin(theta)
        speed = random.uniform(1, 3)
        phi = random.uniform(0, 2 * np.pi)
        obj["velocity"] = [
            speed * np.cos(phi),
            speed * np.sin(phi),
            random.uniform(-0.1, 0.1),
        ]

    if obj["mode"] == "stationary":
        r = collision_radius * np.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 2 * np.pi)
        x, y = r * np.cos(theta), r * np.sin(theta)
        obj["velocity"] = [0.0, 0.0, 0.0]

    if obj["mode"] == "offset":
        r = random.uniform(1.2, 1.5) * collision_radius
        angle = random.uniform(0, 2 * np.pi)
        x, y = r * np.cos(angle), r * np.sin(angle)
        speed = random.uniform(1, 2)
        obj["velocity"] = [
            -speed * np.cos(angle),
            -speed * np.sin(angle),
            random.uniform(-0.1, 0.1),
        ]
    obj["init_possition_x"] = x
    obj["init_possition_y"] = y
    obj["collision_radius"] = collision_radius
    return obj


def set_physics_properties(obj):
    # Compute shape volume to get a mass = density * volume
    shape = obj["shape"]
    dims = obj["dimensions"]
    if shape == "ball":
        # volume of a sphere = 4/3*pi*r^3
        radius = dims["radius"]
        volume = (4.0 / 3.0) * np.pi * (radius**3)
        base_z = radius
        geom_type = "sphere"
        size_str = f"{radius:.4f}"
    elif shape == "cylinder":
        # volume of a cylinder = pi*r^2*h
        radius = dims["radius"]
        height = dims["height"]
        volume = np.pi * (radius**2) * height
        base_z = height / 2
        geom_type = "cylinder"
        size_str = f"{radius:.4f} {height / 2:.4f}"
    elif shape in ["cube", "block"]:
        if shape == "cube":
            side = dims["side"]
            volume = side**3
            base_z = side / 2
            # half-extents
            size_str = f"{side/2:.4f} {side/2:.4f} {side/2:.4f}"
        else:
            length = dims["length"]
            width = dims["width"]
            height = dims["height"]
            volume = length * width * height
            base_z = height / 2
            size_str = f"{length/2:.4f} {width/2:.4f} {height/2:.4f}"
        geom_type = "box"

    obj["volume"] = volume
    obj["base_z"] = base_z
    obj["geom_type"] = geom_type
    obj["size_str"] = size_str
    return obj


def save_file(path, data):
    def convert(obj):
        if isinstance(obj, set):
            return list(obj)  # Convert set to list
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)

    print(f"JSON file created: {path}")
