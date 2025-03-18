import random
import numpy as np


class Light:
    def __init__(self, min_distance=1):
        self.min_distance = min_distance

    def get_settings(self, num_lights=1, cutoff_range=[30, 180]):
        lights = []
        positions = []

        for _ in range(num_lights):
            while True:
                pos = self._generate_random_position()
                if all(self._is_far_enough(pos, existing_pos) for existing_pos in positions):
                    positions.append(pos)
                    break

            light = {
                "pos": pos,
                "diffuse": self._generate_diffuse(),
                # reflection of light from shiny surfaces
                "specular": self._generate_specular(),
                "cutoff": random.uniform(cutoff_range[0], cutoff_range[1]),
                "directional": False
            }
            lights.append(light)

        return lights

    def _generate_random_position(self):
        return [
            random.uniform(-1, 1),  # x
            random.uniform(-1, 1),  # y
            random.uniform(0.1, 0.7),  # z
        ]

    def _generate_diffuse(self):
        return f"{random.uniform(0.6, 1)} {random.uniform(0.8, 1)} {random.uniform(0.8, 1)}"

    def _generate_specular(self):
        # reflection of light from shiny surfaces
        return f"{random.uniform(0.5, 0.8)} {random.uniform(0.5, 1)} {random.uniform(0.5, 1)}"

    def _is_far_enough(self, pos1, pos2):
        distance = np.array(pos1) - np.array(pos2)
        return np.linalg.norm(distance) >= self.min_distance
