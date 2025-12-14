# material_components are based in constant.py
SPLIT_CONFIG = {
    "train": {
        "shapes": ["ball", "cube", "cylinder"],
        "materials": ["metal", "wood", "plastic"],
        "object_count": (2, 6),
        "camera_profile": "broad",
        "collision_modes": ["collision", "sliding", "stationary"],
        "floor_textures": ["checkerboard", "flat"],
        # component indices for each property per material
        "material_components": {
            "metal": {
                "density_idx": [0, 1, 2],  # steel, stainless, aluminum
                "friction_idx": [0],  # metal friction lower component
                "elasticity_idx": [0],  # lower elasticity component
            },
            "wood": {
                "density_idx": [0, 1],
                "friction_idx": [0],
                "elasticity_idx": [0],
            },
            "plastic": {
                "density_idx": [0, 1],
                "friction_idx": [0],
                "elasticity_idx": [0],
            },
        },
    },
    "val": {
        "shapes": ["ball", "cube"],
        "materials": ["metal", "wood", "glass"],
        "object_count": (2, 5),
        "camera_profile": "narrow",
        "collision_modes": ["collision", "sliding"],
        "floor_textures": ["flat", "gradient"],
        "material_components": {
            "metal": {
                "density_idx": [3, 4, 5],  # copper, zinc, brass
                "friction_idx": [0, 1],
                "elasticity_idx": [0, 1],
            },
            "wood": {
                "density_idx": [1, 2],
                "friction_idx": [0],
                "elasticity_idx": [1],
            },
            "glass": {
                "density_idx": [0, 1],
                "friction_idx": [0],
                "elasticity_idx": [0],
            },
        },
    },
    "test": {
        "shapes": ["block", "cylinder", "cube"],
        "materials": ["glass", "rubber"],
        "object_count": (3, 7),
        "camera_profile": "extreme",
        "collision_modes": ["collision", "offset", "sliding"],
        "floor_textures": ["checkerboard", "gradient"],
        "material_components": {
            "metal": {
                "density_idx": [6, 7, 8],  # bronze, cast iron, gold
                "friction_idx": [1],
                "elasticity_idx": [1],
            },
            "rubber": {
                "density_idx": [0, 1, 2],
                "friction_idx": [0, 1],
                "elasticity_idx": [0, 1],
            },
            "glass": {
                "density_idx": [2],
                "friction_idx": [0],
                "elasticity_idx": [1],
            },
        },
    },
}
