import random
from dataset.world.constants import FLOOR_TEXTURE


class Floor:
    def __init__(self, min_fric=0.01, max_fric=1) -> None:
        self.min_fric = min_fric
        self.max_fric = max_fric

    def get_settings(self):
        friction = self.get_friction()

        return {
            "friction": " ".join(map(str, friction)),
            "texture": random.choice(FLOOR_TEXTURE),
            **self.get_visual_properties(friction)
        }

    def get_friction(self):
        friction_static = random.uniform(self.min_fric, self.max_fric)
        friction_dynamic = random.uniform(self.min_fric, friction_static)
        friction_spin = random.uniform(self.min_fric, friction_dynamic)
        return (friction_static, friction_dynamic, friction_spin)

    def get_visual_properties(self, friction=[]):
        if not friction:
            return {}

        friction_static = friction[0]
        normalized = (friction_static - self.min_fric) / (self.max_fric - self.min_fric)

        # high val - shiny, low val - matte
        specular = 0.8 - 0.6 * normalized
        shininess = 0.8 - 0.6 * normalized

        # lower friction produces brighter floor
        rgba = f"{1 - 0.2 * normalized:.2f} {1 - 0.2 * normalized:.2f} {1 - 0.2 * normalized:.2f} 1"

        return {
            "rgba": rgba,
            "specular": f"{specular:.6f}",
            "shininess": f"{shininess:.6f}"
        }
