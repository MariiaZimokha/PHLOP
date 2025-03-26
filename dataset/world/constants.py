
FLOOR_TEXTURE = ["checkerboard", "flat", "gradient"]
MODES = ["collision", "sliding", "stationary", "offset"]


class MaterialTypes:
    METAL = "metal"
    WOOD = "wood"
    RUBBER = "rubber"
    GLASS = "glass"
    PLASTIC = "plastic"


class Shapes:
    BALL = "ball"
    CYLINDER = "cylinder"
    CUBE = "cube"
    BLOCK = "block"


SHAPES = [Shapes.BALL, Shapes.CYLINDER, Shapes.CUBE, Shapes.BLOCK]

MATERIAL_MIXTURES = {
    MaterialTypes.METAL: {
        #  (mean, std, weight)
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
        #  (mean, std, weight)
        "friction_dist_lateral": [(0.28, 0.02, 0.5), (0.32, 0.02, 0.5)],
    },
    MaterialTypes.WOOD: {
        "elasticity_dist": [(0.35, 0.05, 0.6), (0.45, 0.05, 0.4)],
        "density_dist": [
            (500, 50, 0.4),  # Softwoods
            (600, 50, 0.3),  # Hardwoods
            (700, 50, 0.3),  # Dense hardwoods
        ],
        #  (mean, std, weight)
        "friction_dist_lateral": [(0.45, 0.05, 0.6), (0.55, 0.05, 0.4)],
    },
    MaterialTypes.RUBBER: {
        "elasticity_dist": [(0.90, 0.05, 0.7), (0.98, 0.05, 0.3)],
        "density_dist": [
            (900, 50, 0.4),   # Natural rubber
            (1100, 50, 0.3),  # Synthetic rubber
            (1200, 50, 0.3),  # High-density rubber
        ],
        "friction_dist_lateral": [(0.9, 0.05, 0.7), (1.1, 0.05, 0.3)],
    },
    MaterialTypes.GLASS: {
        "elasticity_dist": [(0.55, 0.05, 0.6), (0.65, 0.05, 0.4)],
        "density_dist": [
            (2400, 50, 0.5),  # Soda-lime glass
            (2500, 50, 0.3),  # Borosilicate glass
            (2600, 50, 0.2),  # Lead glass
        ],
        "friction_dist_lateral": [(0.18, 0.02, 0.6), (0.22, 0.02, 0.4)],
    },
    MaterialTypes.PLASTIC: {
        "elasticity_dist": [(0.65, 0.05, 0.5), (0.75, 0.05, 0.5)],
        "density_dist": [
            (1000, 50, 0.3),  # Low-density polyethylene
            (1200, 50, 0.3),  # Polypropylene
            (1400, 50, 0.2),  # High-density polyethylene
            (1800, 50, 0.2),  # Polycarbonate
        ],
        "friction_dist_lateral": [(0.35, 0.05, 0.5), (0.45, 0.05, 0.5)],
    },
}

MATERIAL_VISUALS = {
    MaterialTypes.GLASS: {"alpha": 0.5, "specular": 0.6},
    MaterialTypes.METAL: {"alpha": 1.0, "specular": 1.0},
    MaterialTypes.WOOD: {"alpha": 1.0, "specular": 0.2},
    MaterialTypes.RUBBER: {"alpha": 1.0, "specular": 0.0},
    MaterialTypes.PLASTIC: {"alpha": 1.0, "specular": 0.3},
}

MATERIAL_SHININESS = {
    MaterialTypes.GLASS: 50,
    MaterialTypes.METAL: 100,
    MaterialTypes.WOOD: 10,
    MaterialTypes.RUBBER: 0,
    MaterialTypes.PLASTIC: 5,
}

COLORS = {
    "GRAY": "0.5 0.5 0.5 1.0",
    "BLUE": "0.0 0.2 0.7 1.0",
    "BROWN": "0.5 0.3 0.1 1.0",
    "CYAN": "0.0 0.6 0.6 1.0",
    "GREEN": "0.0 0.6 0.0 1.0",
    "PURPLE": "0.5 0.1 0.5 1.0",
    "RED": "0.7 0.1 0.1 1.0",
    "YELLOW": "0.7 0.7 0.1 1.0",
    "VIBRANT_PINK": "1.0 0.2 0.8 1.0",
    "TEAL": "0.3 0.8 1.0 1.0",
    "STRONG_YELLOW": "1.0 0.8 0.1 1.0",
    "BRIGHT_RED": "1.0 0.3 0.3 1.0",
    "BRIGHT_GREEN": "0.6 1.0 0.2 1.0",
}

DENSITY_SCALING_FACTOR = 1
