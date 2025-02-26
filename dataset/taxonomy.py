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
        # print(vel_curr_magnitude, "vel_curr_magnitude")
        if vel_curr_magnitude == self.velocity_threshold:
            return "Stationary"

        if vel_curr_magnitude <= self.velocity_threshold and vel_prev_magnitude <= self.velocity_threshold:
           # if np.linalg.norm(vel_curr) <  1e-6:
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

    def detect_collision(self, model, data, prev_frame):
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

            # Get joint addresses for post-collision velocities
            adr1 = model.jnt_dofadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{g1}_free")
            ]
            adr2 = model.jnt_dofadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{g2}_free")
            ]

            # Extract post-collision velocities
            vel1_post = data.qvel[adr1: adr1 + 3].tolist()
            vel2_post = data.qvel[adr2: adr2 + 3].tolist()

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

    def get_taxonomy(self, model, data, dt, prev_frame, current_objects):

        results = {obj["id"]: [] for obj in self.objects}
        dt = max(dt, 1e-6)  # division by zero
        collision_results = self.detect_collision(model, data, prev_frame)

        for obj in self.objects:
            object_id = obj["id"]

            # Get joint address for the object
            # joint_name = f"obj{obj['geom_id']}_free"
            # joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            # adr = model.jnt_dofadr[joint_id]

            # current velocity and position form mujoco model
            # velocity = data.qvel[adr : adr + 3].tolist()
            cur_velocity = current_objects.get(object_id, {}).get("velocity", [])
            prev_velocity = prev_frame.get(object_id, {}).get("velocity", [])
            prev_velocity = [float(x) for x in prev_velocity]
            # Detect linear motion
            linear_motion = self.analyze_motion(object_id, prev_velocity[:3], cur_velocity, dt)
            if linear_motion:
                results[object_id].append(linear_motion)

        # add collision results to the objects
        for (g1, g2), collision_info in collision_results.items():
            obj1_id = next(obj["id"] for obj in self.objects if obj["geom_id"] == g1)
            obj2_id = next(obj["id"] for obj in self.objects if obj["geom_id"] == g2)

            results[obj1_id].append(collision_info)
            results[obj2_id].append(collision_info)

        return results
