import glob
import json
import numpy as np

def load_data(directory):
    """Load all JSON annotations containing object and camera data"""
    data = []
    files = glob.glob(f"{directory}/**/*.json", recursive=True)
    
    for filename in files:
        if "obj" in filename:  
            try:
                with open(filename, 'r') as file:
                    video_data = json.load(file)
                    data.append(video_data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading {filename}: {str(e)}")
    return data

def is_visible(camera, obj_position, obj_radius):
    """Check if object is within camera's view frustum"""
    # Extract camera parameters
    lookat = np.array(camera['lookat'])
    position = lookat - np.array([0, 0, camera['distance']])  # Simplified camera position
    view_dir = lookat - position
    view_dir /= np.linalg.norm(view_dir)
    
    # Simple visibility check (approximate)
    obj_pos = np.array(obj_position)
    to_obj = obj_pos - position
    distance = np.linalg.norm(to_obj)
    angle = np.arccos(np.dot(view_dir, to_obj/distance))
    
    # Basic frustum approximation (adjust based on your camera parameters)
    max_angle = np.radians(60)  # Assuming 60° FOV
    return angle <= max_angle + np.arctan(obj_radius/distance)

def calculate_optimal_duration(frames, world, max_duration=15.0, min_duration=3.0, frame_interval=0.0333):
    """Calculate video duration considering visibility"""
    last_active_frame = -1
    velocity_threshold = 0.001
    angular_threshold = 0.0005
    
    for frame_idx, frame in enumerate(frames):
        camera = world['camera']
        active_in_frame = False
        
        for obj_id, obj_data in frame['objects'].items():
            # Get position from current frame's object data
            if 'position' not in obj_data:
                continue  # Skip if no position

            # Get radius from the *initial* object definition (assuming it's constant)
            obj_radius = 0.05  # Default (example)
            
            if is_visible(camera, obj_data['position'], obj_radius):
                lin_vel = np.linalg.norm(obj_data['velocity'])
                ang_vel = np.linalg.norm(obj_data['angular_velocity'])
                
                if lin_vel > velocity_threshold or ang_vel > angular_threshold:
                    active_in_frame = True
                    break
        
        if active_in_frame:
            last_active_frame = frame_idx

    # Duration calculation
    if last_active_frame >= 0:
        motion_duration = (last_active_frame + 1) * frame_interval
        calculated_duration = max(motion_duration, min_duration)
    else:
        calculated_duration = min_duration
    
    return min(calculated_duration, max_duration)

def create_dataset(annotations):
    """Create dataset with visibility-aware duration"""
    dataset = []
    
    for video in annotations:
        try:
            # print( video['world'])
            # break
            # Access objects from the TOP LEVEL JSON (initial state)
            objects = video['objects']  # List of initial object data
            
            # Use FIRST frame's camera (ensure camera consistency)
            world = video["world"]
            first_frame_camera = video['world']['camera']  # Camera from world data
            
            initial_state = {
                'camera': first_frame_camera,
                'objects': []
            }
            
            # Collect objects from initial state
            for obj_data in objects:
                # Object radius (assuming constant) - CHECK YOUR DATA FOR THIS
                obj_radius = 0.05 # CHECK JSON
                
                # Extract relevant data (based on your JSON structure)
                obj = {
                    'position': [obj_data['init_possition_x'], obj_data['init_possition_y'], obj_data['base_z']],  # Initial
                    'velocity': obj_data['velocity'],
                    'angular_velocity': obj_data['angular_velocity'],
                    'mass': obj_data['mass'],
                    'elasticity': obj_data['elasticity'],
                    'friction': list(map(float, obj_data['friction'].split())),
                    'radius': obj_radius
                }
                initial_state['objects'].append(obj)
            
            duration = calculate_optimal_duration(video['frames'], world)
            
            # Feature vector
            feature_vector = []
            # Camera
            cam = initial_state['camera']
            feature_vector.extend(cam['lookat'])
            feature_vector.append(cam['distance'])
            feature_vector.append(cam['azimuth'])
            feature_vector.append(cam['elevation'])
            
            # Objects (visible in initial frame)
            for obj in initial_state['objects']:
                if is_visible(cam, obj['position'], obj['radius']):
                    feature_vector.extend(obj['position'])
                    feature_vector.extend(obj['velocity'])
                    feature_vector.extend(obj['angular_velocity'])
                    feature_vector.append(obj['mass'])
                    feature_vector.append(obj['elasticity'])
                    feature_vector.extend(obj['friction'])
                    feature_vector.append(obj['radius'])
            
            dataset.append({
                'features': np.array(feature_vector, dtype=np.float32),
                'duration': duration,
                'num_visible_objects': sum(
                    1 for obj in initial_state['objects'] 
                    if is_visible(cam, obj['position'], obj['radius'])
                )
            })
            
        except KeyError as e:
            print(f"Skipping video due to missing data: {str(e)}")
    
    return dataset


# Usage example
# directory = 'generated'
# annotations = load_data(directory)
# df = create_dataset(annotations)
