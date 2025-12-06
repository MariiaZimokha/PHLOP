import random
import numpy as np
from phlop.world.constants import (
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

    def _sample_from_mixture(self, mixture_list):
        """
        (mean, std, weight)
        """
        # Randomly pick a distribution
        weights = [comp[-1] for comp in mixture_list]
        dist_id = np.random.choice(len(mixture_list), size=1, p=weights)[0]
        mu, std, weight = mixture_list[dist_id]
        return np.random.normal(mu, std)

    def _sample_from_component_subset(self, mixture_list, component_indices):
        if not component_indices:
            return self._sample_from_mixture(mixture_list)

        valid = [i for i in component_indices if 0 <= i < len(mixture_list)]
        if not valid:
            return self._sample_from_mixture(mixture_list)

        subset = [mixture_list[i] for i in valid]
        weights = np.array([c[-1] for c in subset], dtype=float)
        if np.any(np.isnan(weights)) or weights.sum() <= 0:
            weights = np.ones(len(subset))
        weights /= weights.sum()

        idx = int(np.random.choice(len(subset), p=weights))
        mu, std, _ = subset[idx]
        return float(np.random.normal(mu, std))

    def _get_visual(self, material):
        random_color_str = random.choice(list(self.colors.values()))
        r, g, b, _ = list(map(float, random_color_str.split()))
        alpha = self.material_visuals[material]["alpha"]
        specular = self.material_visuals[material]["specular"]
        final_rgba_str = f"{r:.3f} {g:.3f} {b:.3f} {alpha:.3f}"
        return {"rgba": final_rgba_str, "specular": specular}

    def get_object(
        self,
        shape=None,
        material=None,
        density_idx=None,
        friction_idx=None,
        elasticity_idx=None,
    ):
        if shape is None or shape not in SHAPES:
            shape = random.choice(SHAPES)

        material_keys = list(self.material_mixtures.keys())

        if material is None or material not in material_keys:
            material = random.choice(material_keys)

        mixture = self.material_mixtures[material]

        density_val = self._sample_from_component_subset(
            mixture["density_dist"], density_idx
        )
        elasticity = self._sample_from_component_subset(
            mixture["elasticity_dist"], elasticity_idx
        )
        base_fric = self._sample_from_component_subset(
            mixture["friction_dist_lateral"], friction_idx
        )

        fric_static = max(0.0, base_fric + np.random.normal(0.0, 0.02))
        fric_dynamic = max(0.0, base_fric * 0.95 + np.random.normal(0.0, 0.02))
        fric_rolling = max(0.0, base_fric * 0.60 + np.random.normal(0.0, 0.01))
        friction_str = f"{fric_static:.3f} {fric_dynamic:.3f} {fric_rolling:.3f}"

        visual = self._get_visual(material)

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

        # linear_velocity = np.random.uniform(-1, 1, size=3)
        # angular_velocity = np.random.uniform(-1, 1, size=3)
        # Slow dynamic behavior
        linear_velocity = (0.20 * np.random.uniform(-1, 1, 3)).tolist()
        angular_velocity = (0.15 * np.random.uniform(-1, 1, 3)).tolist()

        return {
            "shape": shape,
            "material": material,
            "material_shininess": float(self.material_shininess.get(material, 5)),
            "dimensions": dimensions,
            "mass": float(mass_val),
            "density": float(density_val),
            "elasticity": float(round(elasticity, 3)),  # round to 3 decimals
            "friction": friction_str,
            "visual": visual,
            "velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
