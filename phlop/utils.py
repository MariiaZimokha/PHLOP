import random
import numpy as np
import json
import mujoco
import matplotlib.colors as mcolors

def is_cylinder_upright(objects, model, data, object_id, alignment_threshold=0.7):
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
        
        # Get quaternion from MuJoCo data
        joint_name = f"obj{obj_index}_free"
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            adr = model.jnt_dofadr[joint_id]
            quat = data.qpos[adr + 3 : adr + 7]
            
            # Convert quaternion to rotation matrix
            rot_mat = np.zeros((3, 3))
            mujoco.mju_quat2Mat(rot_mat.ravel(), quat)
            
            # For a cylinder, check if its z-axis (cylinder axis) is aligned with world vertical (0, 0, 1)
            # If the cylinder is on its end edge, the z-axis of the rotation matrix should point up
            cylinder_axis = rot_mat[:, 2]  # z-axis of the cylinder in world coordinates
            world_up = np.array([0, 0, 1])
            
            # Check alignment: dot product close to 1 means aligned (upright)
            alignment = np.dot(cylinder_axis, world_up)
            
            # Return True if cylinder is upright (alignment >= threshold)
            return alignment >= alignment_threshold
        except:
            # If we can't get orientation, return False for safety
            return False


def set_position_and_velocity(obj):
    collision_radius = random.uniform(1, 2)

    if obj["mode"] == "collision":
        angle = random.uniform(0, 2 * np.pi)
        x = collision_radius * np.cos(angle)
        y = collision_radius * np.sin(angle)
        speed = random.uniform(2, 3.5)  # Reduced from (2, 5) to (0.5, 1.5) for slower movement
        obj["velocity"] = [
            -speed * np.cos(angle),
            -speed * np.sin(angle),
            random.uniform(-0.2, 0.2),  # Reduced from (-0.2, 0.2) to (-0.1, 0.1)
        ]

    if obj["mode"] == "sliding":
        r = collision_radius * np.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 2 * np.pi)
        x, y = r * np.cos(theta), r * np.sin(theta)
        speed = random.uniform(1, 2.5)  # Reduced from (1, 3) to (0.3, 1.0) for slower movement
        phi = random.uniform(0, 2 * np.pi)
        obj["velocity"] = [
            speed * np.cos(phi),
            speed * np.sin(phi),
            random.uniform(-0.1, 0.1),  # Reduced from (-0.1, 0.1) to (-0.05, 0.05)
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
        speed = random.uniform(0.5, 1.5)  # Reduced from (1, 2) to (0.3, 0.8) for slower movement
        obj["velocity"] = [
            -speed * np.cos(angle),
            -speed * np.sin(angle),
            random.uniform(-0.1, 0.1),  # Reduced from (-0.1, 0.1) to (-0.05, 0.05)
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


def describe_object_unique(
    target_id: str,
    objects: list,
    frames: list,
    appeared_obj_ids: set,
    rgba_to_name_func=None
) -> str:
    """
    Generates a unique description. If color/shape is not unique among visible objects,
    adds spatial context or ID.
    
    Args:
        target_id: The object ID to describe
        objects: List of object dictionaries
        frames: List of frame dictionaries
        appeared_obj_ids: Set of object IDs that appeared in frames
        rgba_to_name_func: Optional function to convert RGBA to color name (for compatibility)
    
    Returns:
        Unique description string like "the red cube" or "the red cube on the left"
    """
    # Find target object
    target_obj = next((o for o in objects if o.get("id") == target_id), None)
    if not target_obj:
        return target_id
    
    # Get basic description
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
        # Ambiguity detected! Add spatial context or unique ID.
        # Try to use spatial context from first frame position
        spatial_context = None
        if frames:
            target_pos = None
            confusor_positions = []
            
            first_frame = frames[0]
            target_obj_state = first_frame.get("objects", {}).get(target_id)
            if target_obj_state:
                target_pos = target_obj_state.get("position", [0, 0, 0])
            
            for conf_id in confusors:
                conf_obj_state = first_frame.get("objects", {}).get(conf_id)
                if conf_obj_state:
                    conf_pos = conf_obj_state.get("position", [0, 0, 0])
                    confusor_positions.append((conf_id, conf_pos))
            
            # Use x-position to determine left/right
            if target_pos and confusor_positions:
                target_x = target_pos[0] if len(target_pos) > 0 else 0
                # Compare with confusors to determine relative position
                confusor_x_positions = [pos[0] for _, pos in confusor_positions if len(pos) > 0]
                if confusor_x_positions:
                    avg_confusor_x = sum(confusor_x_positions) / len(confusor_x_positions)
                    if target_x < avg_confusor_x - 0.1:  # Threshold to avoid noise
                        spatial_context = "on the left"
                    elif target_x > avg_confusor_x + 0.1:
                        spatial_context = "on the right"
        
        if spatial_context:
            return f"the {target_desc} {spatial_context}"
        else:
            # Fallback to ID suffix (e.g., "Object 0" from "geom_obj0")
            obj_id_suffix = target_id.split("_")[-1] if "_" in target_id else target_id[-1]
            return f"{target_desc} (Object {obj_id_suffix})"
    
    return f"the {target_desc}"
