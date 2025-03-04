import random
import numpy as np


class Object:
    def __init__(self):
        self.material_mixtures = {
            "metal": {
                "elasticity_dist": [(0.85, 0.02, 0.4), (0.92, 0.01, 0.6)],
                "density_dist": [
                    (7850, 50, 0.30),  # Steel
                    (7950, 40, 0.20),  # Stainless steel (average)
                    (2700, 30, 0.15),  # Aluminum
                    (8940, 60, 0.10),  # Copper
                    (7140, 70, 0.08),  # Zinc
                    (8500, 100, 0.07),  # Brass (average)
                    (8900, 80, 0.05),  # Bronze (average)
                    (7300, 200, 0.03),  # Cast iron
                    (19320, 100, 0.02),  # Gold
                ],
                "friction_dist_lateral": [(0.28, 0.02, 0.5), (0.32, 0.02, 0.5)],
            },
            "wood": {
                "elasticity_dist": [(0.35, 0.05, 0.6), (0.45, 0.05, 0.4)],
                "density_dist": [
                    (500, 50, 0.4),  # Softwoods
                    (600, 50, 0.3),  # Hardwoods
                    (700, 50, 0.3),  # Dense hardwoods
                ],
                "friction_dist_lateral": [(0.45, 0.05, 0.6), (0.55, 0.05, 0.4)],
            },

            "rubber": {
                "elasticity_dist": [(0.90, 0.05, 0.7), (0.98, 0.05, 0.3)],
                "density_dist": [
                    (900, 50, 0.4),  # Natural rubber
                    (1100, 50, 0.3),  # Synthetic rubber
                    (1200, 50, 0.3),  # High-density rubber
                ],
                "friction_dist_lateral": [(0.9, 0.05, 0.7), (1.1, 0.05, 0.3)],
            },
            "glass": {
                "elasticity_dist": [(0.55, 0.05, 0.6), (0.65, 0.05, 0.4)],
                "density_dist": [
                    (2400, 50, 0.5),  # Soda-lime glass
                    (2500, 50, 0.3),  # Borosilicate glass
                    (2600, 50, 0.2),  # Lead glass
                ],
                "friction_dist_lateral": [(0.18, 0.02, 0.6), (0.22, 0.02, 0.4)],
            },

            "plastic": {
                "elasticity_dist": [(0.65, 0.05, 0.5), (0.75, 0.05, 0.5)],
                "density_dist": [
                    (1000, 50, 0.3),  # Low-density polyethylene
                    (1200, 50, 0.3),  # Polypropylene
                    (1400, 50, 0.2),  # High-density polyethylene
                    (1800, 50, 0.2),  # Polycarbonate
                ],
                "friction_dist_lateral": [(0.35, 0.05, 0.5), (0.45, 0.05, 0.5)],
            },

            # "wood": {
            #     "elasticity": 0.4,
            #     "density": 600,
            #     "friction": "0.5",
            # },
            # "rubber": {
            #     "elasticity": 0.95,
            #     "density": 1100,
            #     "friction": "1.0",
            # },
            # "glass": {
            #     "elasticity": 0.6,
            #     "density": 2500,
            #     "friction": "0.2",
            # },
            # "plastic": {
            #     "elasticity": 0.7,
            #     "density": 1200,
            #     "friction": "0.4",
            # },
        }

        self.material_visuals = {
            "glass": {"alpha": 0.5, "specular": 0.6},
            "metal": {"alpha": 1.0, "specular": 1.0},
            "wood": {"alpha": 1.0, "specular": 0.2},
            "rubber": {"alpha": 1.0, "specular": 0.0},
            "plastic": {"alpha": 1.0, "specular": 0.3},
        }

        self.material_shininess = {
            "glass": 50,
            "metal": 100,
            "wood": 10,
            "rubber": 0,
            "plastic": 5,
        }

        self.colors = {
            "gray": "0.5 0.5 0.5 1.0",
            "blue": "0.0 0.2 0.7 1.0",
            "brown": "0.5 0.3 0.1 1.0",
            "cyan": "0.0 0.6 0.6 1.0",
            "green": "0.0 0.6 0.0 1.0",
            "purple": "0.5 0.1 0.5 1.0",
            "red": "0.7 0.1 0.1 1.0",
            "yellow": "0.7 0.7 0.1 1.0",
            "vibrant_pink": "1.0 0.2 0.8 1.0",
            "teal": "0.3 0.8 1.0 1.0",
            "strong_yellow": "1.0 0.8 0.1 1.0",
            "bright_red": "1.0 0.3 0.3 1.0",
            "bright_green": "0.6 1.0 0.2 1.0",
        }

        self.density_scaling_factor = 1

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
        shapes = ["ball", "cylinder", "cube", "block"]
        if shape is None or shape not in shapes:
            shape = random.choice(shapes)

        material_keys = list(self.material_mixtures.keys())
        if material is None or material not in material_keys:
            material = random.choice(material_keys)

        mixture_data = self.material_mixtures[material]

        elasticity_val = self.__sample_from_mixture(mixture_data["elasticity_dist"])
        density_val = self.__sample_from_mixture(mixture_data["density_dist"])
            friction_lateral = self.__sample_from_mixture(
                mixture_data["friction_dist_lateral"]
            )
        friction_str = f"{friction_lateral:.2f}"

        visual = self.__get_visual(material)

        # Dimension logic remains the same
        if shape == "ball":
            radius = random.uniform(0.05, 0.1)
            volume = (4 / 3) * np.pi * (radius**3)
            dimensions = {"radius": radius}
        elif shape == "cylinder":
            radius = random.uniform(0.05, 0.1)
            height = random.uniform(0.01, 0.2)
            volume = np.pi * (radius**2) * height
            dimensions = {"radius": radius, "height": height}
        elif shape == "cube":
            side = random.uniform(0.05, 0.1)
            volume = side**3
            dimensions = {"side": side}
        else:  # "block"
            length = random.uniform(0.01, 0.1)
            width = random.uniform(0.05, 0.1)
            height = random.uniform(0.05, 0.1)
            volume = length * width * height
            dimensions = {"length": length, "width": width, "height": height}

        density_val *= self.density_scaling_factor
        raw_mass = density_val * volume
        mass_val = max(raw_mass, 1e-6)

        # Random velocities
        linear_velocity = np.random.uniform(-5, 5, size=3)
        angular_velocity = np.random.uniform(-5, 5, size=3)

        return {
            "shape": shape,
            "material": material,
            "material_shininess": self.material_shininess[material],
            "dimensions": dimensions,
            "mass": mass_val,
            "density": density_val,
            "elasticity": float(f"{elasticity_val:.3f}"),  # round to 3 decimals
            "friction": friction_str,
            "visual": visual,
            "velocity": linear_velocity.tolist(),
            "angular_velocity": angular_velocity.tolist(),
        }
