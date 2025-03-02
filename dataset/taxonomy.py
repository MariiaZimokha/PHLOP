import mujoco
import numpy as np
from collections import deque


class PhysicsTaxonomy:
    def __init__(self, objects):
        # Constants for physics calculations
        self.epsilon = 0.01  # Threshold for "acceleration ~ 0" in m/(s^2)
        self.COLLISION_ELASTIC_FACTOR = 0.5  # Threshold for elastic collision
        self.FORCE_THRESHOLD = 10.0  # Example threshold for push/pull detection
        self.G = 9.8  # Gravity in m/(s^2)
        self.precision = 5  # Decimal precision for rounding
        self.velocity_threshold = 1e-4
        self.acceleration_threshold = 1e-12
        self.MOTION_WINDOW_SIZE = 5

        self.objects = objects
        self.motion_history = {obj['id']: deque(maxlen=self.MOTION_WINDOW_SIZE) for obj in objects}

    def detect_linear_motion(self, vel_prev, vel_curr, dt):
        """
        Check if the acceleration is 0 over dt.
        a = (v_current - v_prev) / dt
        a = dv(t) / dt
        """
        vel_curr = np.array(vel_curr)
        vel_prev = np.array(vel_prev)
        vel_curr_magnitude = round(np.linalg.norm(vel_curr), self.precision)
        vel_prev_magnitude = round(np.linalg.norm(vel_prev), self.precision)

        accel = (vel_curr - vel_prev) / dt
        accel_magnitude = round(np.linalg.norm(accel), self.precision)

        if abs(vel_curr_magnitude - self.velocity_threshold) < self.velocity_threshold:
            return "Stationary"

        if abs(vel_curr_magnitude) <= self.velocity_threshold and abs(vel_prev_magnitude) <= self.velocity_threshold:
            return "Stationary"
        # Check if acceleration is negligible (constant velocity)
        elif accel_magnitude < self.acceleration_threshold:
            return "Constant Velocity"
        else:
            vel_accel_dot = np.dot(vel_curr, accel)

            # Determine if accelerating or decelerating
            if vel_accel_dot > 0:
                return "Accelerating"
            else:
                return "Decelerating"

    def analyze_motion(self, object_id, vel_prev, vel_curr, dt):
        current_motion = self.detect_linear_motion(vel_prev, vel_curr, dt)

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

            # Skip if either geom is the world (geom_id = 0)
            if g1 == 0 or g2 == 0:
                continue

            pair = tuple(sorted((g1, g2)))

            # Compute normal direction from the contact frame
            normal = contact.frame[:3]

            # Get object IDs and their velocities before the collision
            vel1_id = next((obj["id"] for obj in self.objects if obj["geom_id"] == g1), None)
            vel2_id = next((obj["id"] for obj in self.objects if obj["geom_id"] == g2), None)

            vel1_pre = prev_frame.get(vel1_id, {}).get("velocity", [])
            vel2_pre = prev_frame.get(vel2_id, {}).get("velocity", [])

            vel1_post = current_objects.get(vel1_id, {}).get("velocity", [])
            vel2_post = current_objects.get(vel2_id, {}).get("velocity", [])
            
            if not vel1_pre or not vel2_pre or not vel1_post or not vel2_post:
                continue

            normal = normal / (np.linalg.norm(normal) + 1e-6)

            # Calculate relative velocities before and after the collision
            rel_vel_pre = np.dot((np.array(vel1_pre) - np.array(vel2_pre)), normal)
            rel_vel_post = np.dot((np.array(vel1_post) - np.array(vel2_post)), normal)

            # Calculate elasticity ratio
            elasticity_ratio = (
                abs(rel_vel_post / rel_vel_pre) if rel_vel_pre != 0 else 0
            )

            # Classify collision type
            collision_type = (
                "Elastic Collision" if elasticity_ratio > self.COLLISION_ELASTIC_FACTOR
                else "Inelastic Collision"
            )

            collision_results[pair] = {
                "category": "Interaction Events",
                "subcategory": "Collision",
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

        cur_motion = self.detect_linear_motion(prev_velocity, cur_velocity, dt)
        prev_motion = self.motion_history[object_id][-1] if self.motion_history[object_id] else None

        prev_motion_mapped = self.map_motion_state(prev_motion) if prev_motion else None

        state_transitions = ""

        if prev_motion_mapped == "Moving" and cur_motion == "Decelerating":
            state_transitions = "Moving to Stopping"
        elif prev_motion_mapped == "Stationary" and cur_motion in ("Accelerating", "Constant Velocity"):
            state_transitions = "Stationary to Moving"

        if cur_motion:
            self.motion_history[object_id].append(cur_motion)

        return state_transitions

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

    def get_taxonomy(self, model, data, dt, prev_frame, current_objects):

        results = {obj["id"]: [] for obj in self.objects}
        dt = max(dt, 1e-6)  # division by zero
        collision_results = self.detect_collision(model, data, prev_frame, current_objects)

        for obj in self.objects:
            object_id = obj["id"]

            cur_velocity = current_objects.get(object_id, {}).get("velocity", [])
            prev_velocity = prev_frame.get(object_id, {}).get("velocity", [])
            prev_velocity = [float(x) for x in prev_velocity]
            # Detect linear motion
            linear_motion = self.analyze_motion(object_id, prev_velocity[:3], cur_velocity, dt)
            if linear_motion:
                results[object_id].append(linear_motion)

            # Detect state transitions
            state_transitions = self.get_state_transitions_labels(cur_velocity, prev_velocity[:3], object_id, dt)
            if state_transitions:
                results[object_id].append(state_transitions)

        # add collision results to the objects
        for (g1, g2), collision_info in collision_results.items():
            obj1_id = next(obj["id"] for obj in self.objects if obj["geom_id"] == g1)
            obj2_id = next(obj["id"] for obj in self.objects if obj["geom_id"] == g2)

            results[obj1_id].append(collision_info)
            results[obj2_id].append(collision_info)

        return results
