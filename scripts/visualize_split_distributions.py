import matplotlib.pyplot as plt
import numpy as np
from phlop.world.object import Object
from phlop.split_config import SPLIT_CONFIG


def sample_objects(split, n=300):
    cfg = SPLIT_CONFIG[split]
    obj_gen = Object()
    out = {"density": [], "elasticity": [], "friction": []}

    for _ in range(n):
        mat = np.random.choice(cfg["materials"])
        comps = cfg["material_components"][mat]
        shape = np.random.choice(cfg["shapes"])

        o = obj_gen.get_object(
            shape=shape,
            material=mat,
            density_idx=comps["density_idx"],
            friction_idx=comps["friction_idx"],
            elasticity_idx=comps["elasticity_idx"],
        )

        out["density"].append(o["density"])
        out["elasticity"].append(o["elasticity"])
        out["friction"].append(float(o["friction"].split()[0]))  # static friction
    return out


def plot_split(split_data, split_name):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(split_data["density"], bins=30)
    axs[0].set_title(f"{split_name} Density")

    axs[1].hist(split_data["elasticity"], bins=30)
    axs[1].set_title(f"{split_name} Elasticity")

    axs[2].hist(split_data["friction"], bins=30)
    axs[2].set_title(f"{split_name} Friction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        print("Sampling", split)
        data = sample_objects(split)
        plot_split(data, split)
