import numpy as np
from collections import deque

from dataset.physics_engine import PhysicsEngine

class PhysicsTaxonomy:
    def __init__(self, objects):
        self.objects = objects
        self.MOTION_WINDOW_SIZE = 5
        self.motion_history = {obj['id']: deque(maxlen=self.MOTION_WINDOW_SIZE) for obj in objects}
        # self.COLLISION_ELASTIC_FACTOR = 0.5  # Threshold for elastic collision classification
        self.FORCE_THRESHOLD = 10.0          # Example threshold for push/pull detection

        self.physics_engine = PhysicsEngine()

    def analyze_motion(self, object_id, vel_prev, vel_curr, dt):
        current_motion = self.physics_engine.detect_linear_motion(vel_prev, vel_curr, dt)
        return {
            "category": "Kinematic Events",
            "subcategory": "Linear motion",
            "labels": [current_motion],
        }

    def detect_collision(self, model, data, prev_frame, current_objects):
        """
        Detect collisions between objects and classify them as elastic or inelastic.
        Returns a dictionary of collision results for each interacting pair.
        """
        collision_results = {}
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            # Skip if either geom is the world (geom_id = 0) - floor
            if g1 == 0 or g2 == 0:
                continue
            pair = tuple(sorted((g1, g2)))

            # Compute normal direction from the contact frame
            normal = contact.frame[:3]
            normal = normal / (np.linalg.norm(normal) + 1e-6)

            # Get object IDs and their velocities before the collision
            vel1_id = next((obj["id"] for obj in self.objects if obj["geom_id"] == g1), None)
            vel2_id = next((obj["id"] for obj in self.objects if obj["geom_id"] == g2), None)
            vel1_pre = prev_frame.get(vel1_id, {}).get("velocity", [])
            vel2_pre = prev_frame.get(vel2_id, {}).get("velocity", [])
            vel1_post = current_objects.get(vel1_id, {}).get("velocity", [])
            vel2_post = current_objects.get(vel2_id, {}).get("velocity", [])
            if not vel1_pre or not vel2_pre or not vel1_post or not vel2_post:
                continue

            collision_type = self.physics_engine.detect_collision(vel1_pre, vel2_pre, vel1_post, vel2_post, normal)

            collision_results[pair] = {
                "category": "Interaction Events",
                "subcategory": "Collisions",
                "labels": [collision_type],
            }
        return collision_results

    def map_motion_state(self, motion):
        if motion == "Stationary":
            return "Stationary"
        elif motion in ("Constant Velocity", "Accelerating", "Decelerating"):
            return "Moving"
        else:
            return None

    def detect_state_transitions(self, cur_velocity, prev_velocity, object_id, dt):
        """
        if velocity magnitude approching 0 over time : Moving -> Stopping (due to friction)
        if prev_velocity is 0, and current >0 : Stationary -> Moving (due External Forces)
        """
        cur_motion = self.physics_engine.detect_linear_motion(prev_velocity, cur_velocity, dt)
        prev_motion = self.motion_history[object_id][-1] if self.motion_history[object_id] else None
        prev_motion_mapped = self.map_motion_state(prev_motion) if prev_motion else None

        state_transition = ""
        if prev_motion_mapped == "Moving" and cur_motion == "Decelerating":
            state_transition = "Moving to Stopping"
        elif prev_motion_mapped == "Stationary" and cur_motion in ("Accelerating", "Constant Velocity"):
            state_transition = "Stationary to Moving"

        if cur_motion:
            self.motion_history[object_id].append(cur_motion)
        return state_transition

    def get_state_transitions_labels(self, cur_velocity, prev_velocity, object_id, dt):
        subcategory = self.detect_state_transitions(cur_velocity, prev_velocity, object_id, dt)
        if subcategory:
            if subcategory == "Moving to Stopping":
                return {
                    "category": "State transitions",
                    "subcategory": subcategory,
                    "labels": ["Friction causes deceleration"]
                }
            if subcategory == "Stationary to Moving":
                return {
                    "category": "State transitions",
                    "subcategory": subcategory,
                    "labels": ["Force overcomes static friction"]
                }
        return None

    def get_rotational_motions(self, angular_velocity, linear_velocity, radius, shape):
        if shape == "ball":
            rotational_motion = self.physics_engine.detect_rotational_motion(angular_velocity, linear_velocity, radius)
            if rotational_motion:
                return {
                    "category": "Kinematic Events",
                    "subcategory": "Rotational Motion",
                    "labels": [rotational_motion],
                }
        return None

    def environmental_interactions(self, cur_velocity, prev_velocity, dt, friction_coefficient):
        cur_velocity = np.array(cur_velocity)
        prev_velocity = np.array(prev_velocity)
        acceleration = (cur_velocity - prev_velocity) / dt
        friction_event = self.physics_engine.detect_friction_event(cur_velocity, acceleration, friction_coefficient)
        if friction_event:
            return {
                "category": "Environmental Interactions",
                "subcategory": "Friction-Induced Events",
                "labels": [friction_event],
            }
        return None

    def get_friction_coefficients(self, friction):
        # Allow friction to be a single value or a list of coefficients.
        if isinstance(friction, list) and len(friction) == 3:
            return {
                "sliding": friction[0],  # Sliding friction
                "torsional": friction[1],  # Torsional friction
                "rolling": friction[2],  # Rolling friction
            }
        return {"sliding": friction}

    def get_taxonomy(self, model, data, dt, prev_frame, current_objects):
        results = {obj["id"]: [] for obj in self.objects}
        dt = max(dt, 1e-6)  # avoid division by zero
        collision_results = self.detect_collision(model, data, prev_frame, current_objects)

        for obj in self.objects:
            object_id = obj.get("id", '')
            friction = float(obj.get("friction", '0'))
            cur_velocity = current_objects.get(object_id, {}).get("velocity", [])
            cur_angular_velocity = current_objects.get(object_id, {}).get("angular_velocity", [])
            prev_velocity = prev_frame.get(object_id, {}).get("velocity", [])

            # Linear motion analysis
            linear_motion = self.analyze_motion(object_id, prev_velocity[:3], cur_velocity, dt)
            if linear_motion:
                results[object_id].append(linear_motion)

            # Detect state transitions
            state_transitions = self.get_state_transitions_labels(cur_velocity, prev_velocity[:3], object_id, dt)
            if state_transitions:
                results[object_id].append(state_transitions)

            # Detect rotational motion
            radius = obj.get("dimensions", {}).get("radius", 0)
            shape = obj.get("shape", "")
            rotational_motion = self.get_rotational_motions(cur_angular_velocity, cur_velocity, radius, shape)
            if rotational_motion:
                results[object_id].append(rotational_motion)

            # Detect enviroment interections
            friction_coefficients = self.get_friction_coefficients(friction)
            friction_event = self.environmental_interactions(
                cur_velocity, prev_velocity, dt, friction_coefficients["sliding"]
            )
            if friction_event:
                results[object_id].append(friction_event)

        # add collision results to the objects
        for (g1, g2), collision_info in collision_results.items():
            obj1_id = next(obj["id"] for obj in self.objects if obj["geom_id"] == g1)
            obj2_id = next(obj["id"] for obj in self.objects if obj["geom_id"] == g2)
            results[obj1_id].append(collision_info)
            results[obj2_id].append(collision_info)

        return results
