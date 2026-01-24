import random
import numpy as np
import json
import mujoco
import matplotlib.colors as mcolors


def is_cylinder_upright(objects, model, data, object_id):
    """
    Returns:
        True if cylinder is upright, False if on its side or if check fails
    """
    if model is None or data is None or object_id is None:
        return False

    # find the object index from object_id
    obj_index = None
    for i, obj in enumerate(objects):
        if obj.get("id") == object_id:
            obj_index = i
            break

    if obj_index is None:
        return False

    try:
        geom_name = f"geom_obj{obj_index}"
        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )

        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            return None

        geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
        cylinder_axis = geom_xmat[:, 2]

        world_up = np.array([0, 0, 1])
        alignment = abs(np.dot(cylinder_axis, world_up))

        radius = model.geom_size[geom_id][0]
        half_height = model.geom_size[geom_id][1]
        height = 2.0 * half_height

        aspect_ratio = height / (2.0 * radius)

        # short cylinders needs stricter alignment
        threshold = 0.95 if aspect_ratio < 0.5 else 0.8
        return alignment >= threshold
    except:
        return None


def set_position_and_velocity(obj):
    collision_radius = random.uniform(1, 2)

    if obj["mode"] == "collision":
        angle = random.uniform(0, 2 * np.pi)
        x = collision_radius * np.cos(angle)
        y = collision_radius * np.sin(angle)
        speed = random.uniform(2, 3.5)
        obj["velocity"] = [
            -speed * np.cos(angle),
            -speed * np.sin(angle),
            random.uniform(-0.2, 0.2),
        ]

    if obj["mode"] == "sliding":
        r = collision_radius * np.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 2 * np.pi)
        x, y = r * np.cos(theta), r * np.sin(theta)
        speed = random.uniform(1, 2.5)
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
        speed = random.uniform(0.5, 1.5)
        obj["velocity"] = [
            -speed * np.cos(angle),
            -speed * np.sin(angle),
            random.uniform(-0.1, 0.1),
        ]
    obj["init_position_x"] = x
    obj["init_position_y"] = y
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
            size_str = f"{side / 2:.4f} {side / 2:.4f} {side / 2:.4f}"
        else:
            length = dims["length"]
            width = dims["width"]
            height = dims["height"]
            volume = length * width * height
            base_z = height / 2
            size_str = f"{length / 2:.4f} {width / 2:.4f} {height / 2:.4f}"
        geom_type = "box"

    obj["volume"] = volume
    obj["base_z"] = base_z
    obj["geom_type"] = geom_type
    obj["size_str"] = size_str
    return obj


def save_file(path, data):
    def convert(obj):
        """Custom converter for JSON serialization of numpy types."""
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)

    # print(f"JSON file created: {path}")


def rgba_to_name(rgba):
    """Convert RGBA tuple/list to closest CSS color name."""
    if not rgba or len(rgba) < 3:
        return "unknown color"
    rgb = tuple(rgba[:3])
    min_dist = float("inf")
    best_name = "unknown color"
    for name, hex_val in mcolors.CSS4_COLORS.items():
        named_rgb = mcolors.to_rgb(hex_val)
        dist = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, named_rgb))
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name.replace("grey", "gray")


def describe_object_basic(obj, rgba_to_name_func=None):
    """Get basic color+shape description without ID."""
    if not obj:
        return "unknown object"

    shape = obj.get("geom_type", "object")

    # Try to get color from visual properties
    rgba_str = obj.get("visual", {}).get("rgba", "")
    if rgba_str:
        try:
            rgba = [float(x) for x in rgba_str.split()]
            if rgba_to_name_func:
                color = rgba_to_name_func(rgba)
            else:
                color = rgba_to_name(rgba)
        except (ValueError, AttributeError):
            color = "unknown color"
    else:
        color = "unknown color"

    return f"{color} {shape}"


def load_json(path: str) -> dict:
    """Load JSON file and return as dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def get_appeared_object_ids(frames: list) -> set:
    """Extract set of object IDs that appear in frames (have non-zero bounding boxes)."""
    appeared = set()
    for frame in frames:
        for obj_id, obj_state in frame.get("objects", {}).items():
            bbox = obj_state.get("bbox", [[0, 0], [0, 0]])
            if bbox != [[0, 0], [0, 0]]:
                appeared.add(obj_id)
    return appeared


def describe_object_unique(
    target_id: str,
    objects: list,
    frames: list,
    appeared_obj_ids: set,
    rgba_to_name_func=None,
) -> str:
    """
    Unique description string like "the red cube" or "the red cube on the left"
    """
    # Find target object
    target_obj = next((o for o in objects if o.get("id") == target_id), None)
    if not target_obj:
        return target_id

    # Get basic description (e.g., "red cube")
    target_desc = describe_object_basic(target_obj, rgba_to_name_func)

    # Check for ambiguity among appeared objects
    confusors = []
    for obj_id in appeared_obj_ids:
        if obj_id == target_id:
            continue
        other_obj = next((o for o in objects if o.get("id") == obj_id), None)
        if other_obj:
            other_desc = describe_object_basic(other_obj, rgba_to_name_func)
            if other_desc == target_desc:
                confusors.append(obj_id)

    if confusors:
        # Ambiguity detected! Add spatial context or unique identifier.
        spatial_context = None

        if frames:
            # Get positions from first frame (initial positions are most stable for disambiguation)
            first_frame = frames[0]
            target_obj_state = first_frame.get("objects", {}).get(target_id)
            target_pos = None
            if target_obj_state:
                target_pos = target_obj_state.get("position", [0, 0, 0])

            confusor_positions = []
            for conf_id in confusors:
                conf_obj_state = first_frame.get("objects", {}).get(conf_id)
                if conf_obj_state:
                    conf_pos = conf_obj_state.get("position", [0, 0, 0])
                    confusor_positions.append(conf_pos)

            if target_pos and confusor_positions and len(target_pos) >= 2:
                target_x = target_pos[0]
                target_y = target_pos[1] if len(target_pos) > 1 else 0

                # Extract x and y coordinates from confusors
                confusor_x_positions = [
                    pos[0] for pos in confusor_positions if len(pos) > 0
                ]
                confusor_y_positions = [
                    pos[1] for pos in confusor_positions if len(pos) > 1
                ]

                # Determine spatial context using quadrant-based approach
                if confusor_x_positions:
                    min_confusor_x = min(confusor_x_positions)
                    max_confusor_x = max(confusor_x_positions)
                    avg_confusor_x = sum(confusor_x_positions) / len(
                        confusor_x_positions
                    )

                    # Use a threshold to avoid noise (0.15 units)
                    threshold = 0.15

                    # Check if target is clearly to the left or right
                    if target_x < min_confusor_x - threshold:
                        spatial_context = "on the left"
                    elif target_x > max_confusor_x + threshold:
                        spatial_context = "on the right"
                    elif target_x < avg_confusor_x - threshold:
                        spatial_context = "on the left"
                    elif target_x > avg_confusor_x + threshold:
                        spatial_context = "on the right"

                    # If x-position is ambiguous, try y-position (front/back)
                    if not spatial_context and confusor_y_positions:
                        min_confusor_y = min(confusor_y_positions)
                        max_confusor_y = max(confusor_y_positions)
                        avg_confusor_y = sum(confusor_y_positions) / len(
                            confusor_y_positions
                        )

                        if target_y < min_confusor_y - threshold:
                            spatial_context = "in the front"
                        elif target_y > max_confusor_y + threshold:
                            spatial_context = "in the back"
                        elif target_y < avg_confusor_y - threshold:
                            spatial_context = "in the front"
                        elif target_y > avg_confusor_y + threshold:
                            spatial_context = "in the back"

                    # If still ambiguous, use quadrant description
                    if not spatial_context:
                        # Determine quadrant relative to confusors
                        if target_x < avg_confusor_x and target_y < avg_confusor_y:
                            spatial_context = "in the front-left"
                        elif target_x < avg_confusor_x and target_y >= avg_confusor_y:
                            spatial_context = "in the back-left"
                        elif target_x >= avg_confusor_x and target_y < avg_confusor_y:
                            spatial_context = "in the front-right"
                        else:
                            spatial_context = "in the back-right"

        if spatial_context:
            return f"the {target_desc} {spatial_context}"
        else:
            # Fallback: Extract numeric ID suffix for a more natural description
            # e.g., "geom_obj0" -> "0", "obj_1" -> "1"
            obj_id_suffix = target_id
            if "_" in target_id:
                parts = target_id.split("_")
                # Try to find a numeric suffix
                for part in reversed(parts):
                    if part.replace("obj", "").isdigit():
                        obj_id_suffix = part.replace("obj", "")
                        break
                else:
                    obj_id_suffix = parts[-1]
            elif target_id[-1].isdigit():
                # Extract trailing digits
                obj_id_suffix = ""
                for char in reversed(target_id):
                    if char.isdigit():
                        obj_id_suffix = char + obj_id_suffix
                    else:
                        break

            return f"the {target_desc} (Object {obj_id_suffix})"

    return f"the {target_desc}"
