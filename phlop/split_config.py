"""
Split Configuration for Train/Val/Test Sets

This module contains all split-specific configuration including:
- Object shapes and materials
- Object count ranges
- Camera settings (azimuth, elevation, distance, lookat)
- Collision modes
- Floor textures
- Material component indices

All split-specific settings should be defined here, not hardcoded in other modules.
"""
# material_components are based in constant.py
SPLIT_CONFIG = {
    "train": {
        "shapes": ["ball", "cube", "cylinder"],
        "materials": ["metal", "wood", "plastic"],
        "object_count": (2, 6),
        "camera_profile": "broad",
        "collision_modes": ["collision", "sliding", "stationary"],
        "floor_textures": ["checkerboard", "flat"],
        # Camera settings
        "camera": {
            "azimuth_range": (-180, 180),  # Broad azimuth coverage
            "elevation_range": (-45, -15),  # Moderate elevation
            "distance_range": (1.0, 2.5),  # Adaptive distance range
            "lookat_z_range": (0.3, 0.7),  # Lookat height range
            "limits": {
                "az_range": (-180, 180),
                "el_range": (-89, 0),
                "dist_range": (0.8, 10.0),
            },
        },
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
        # Camera settings
        "camera": {
            "azimuth_range": (-90, 90),  # Narrower azimuth - test angle robustness
            "elevation_range": (-60, -35),  # Steeper elevation - more top-down views
            "distance_range": (1.0, 2.5),  # Adaptive distance range (pulled back slightly)
            "lookat_z_range": (0.4, 0.6),  # Lookat height range
            "limits": {
                "az_range": (-180, 180),
                "el_range": (-89, 0),
                "dist_range": (0.8, 10.0),
            },
        },
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
        # Camera settings
        "camera": {
            "azimuth_range": (-180, 180),  # Full azimuth - test all rotations
            "elevation_range": (-80, -10),  # Extreme elevation - test extreme angles
            "distance_range": (1.0, 2.5),  # Adaptive distance range (wide range)
            "lookat_z_range": (0.2, 0.9),  # Lookat height range
            "limits": {
                "az_range": (-180, 180),
                "el_range": (-89, 0),
                "dist_range": (0.8, 10.0),
            },
        },
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
