import random
import numpy as np
from dataset.world.constants import (
    SHAPES,
    Shapes,
    MATERIAL_MIXTURES,
    MATERIAL_VISUALS,
    MATERIAL_SHININESS,
    COLORS,
    DENSITY_SCALING_FACTOR,
)


class Object:
    def __init__(self):
        self.material_mixtures = MATERIAL_MIXTURES
        self.material_visuals = MATERIAL_VISUALS
        self.material_shininess = MATERIAL_SHININESS
        self.colors = COLORS
        self.density_scaling_factor = DENSITY_SCALING_FACTOR

    def __sample_from_mixture(self, mixture_list):
        """
        (mean, std, weight)
        """
        # Randomly pick a distribution
        weights = [comp[-1] for comp in mixture_list]
        dist_id = np.random.choice(len(mixture_list), size=1, p=weights)[0]
        mu, std, weight = mixture_list[dist_id]
        return np.random.normal(mu, std)

    def __get_visual(self, material):
        random_color_str = random.choice(list(self.colors.values()))
        r, g, b, _ = list(map(float, random_color_str.split()))
        alpha = self.material_visuals[material]["alpha"]
        specular = self.material_visuals[material]["specular"]
        final_rgba_str = f"{r:.3f} {g:.3f} {b:.3f} {alpha:.3f}"
        return {"rgba": final_rgba_str, "specular": specular}

    def get_object(self, shape=None, material=None):
        if shape is None or shape not in SHAPES:
            shape = random.choice(SHAPES)

        material_keys = list(self.material_mixtures.keys())
        if material is None or material not in material_keys:
            material = random.choice(material_keys)

        mixture_data = self.material_mixtures[material]

        elasticity_val = self.__sample_from_mixture(mixture_data["elasticity_dist"])
        density_val = self.__sample_from_mixture(mixture_data["density_dist"])
        mu = self.__sample_from_mixture(mixture_data["friction_dist_lateral"])
        friction_str = f"{mu:.2f} {mu:.2f} {mu:.2f}"
        # friction_static = self.__sample_from_mixture(mixture_data["friction_dist_lateral"])
        # friction_dynamic = self.__sample_from_mixture(mixture_data["friction_dist_lateral"])
        # friction_rolling = self.__sample_from_mixture(mixture_data["friction_dist_lateral"])
        # friction_str = f"{friction_static:.2f} {friction_dynamic:.2f} {friction_rolling:.2f}"

        visual = self.__get_visual(material)

        if shape == Shapes.BALL:
            radius = random.uniform(0.05, 0.1)
            volume = (4 / 3) * np.pi * (radius**3)
            dimensions = {"radius": radius}
        elif shape == Shapes.CYLINDER:
            radius = random.uniform(0.05, 0.1)
            height = random.uniform(0.01, 0.2)
            volume = np.pi * (radius**2) * height
            dimensions = {"radius": radius, "height": height}
        elif shape == Shapes.CUBE:
            side = random.uniform(0.05, 0.1)
            volume = side**3
            dimensions = {"side": side}
        else:  # Shapes.BLOCK
            length = random.uniform(0.01, 0.1)
            width = random.uniform(0.05, 0.1)
            height = random.uniform(0.05, 0.1)
            volume = length * width * height
            dimensions = {"length": length, "width": width, "height": height}

        density_val *= self.density_scaling_factor
        raw_mass = density_val * volume
        mass_val = max(raw_mass, 1e-6)

        linear_velocity = np.random.uniform(-1, 1, size=3)
        angular_velocity = np.random.uniform(-1, 1, size=3)

        return {
            "shape": shape,
            "material": material,
            "material_shininess": self.material_shininess[material],
            "dimensions": dimensions,
            "mass": mass_val,
            "density": density_val,
            "elasticity": min(float(f"{elasticity_val:.3f}"), 1),  # round to 3 decimals
            "friction": friction_str,
            "visual": visual,
            "velocity": linear_velocity.tolist(),
            "angular_velocity": angular_velocity.tolist(),
        }
