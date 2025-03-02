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
        self.air_density = 1.225
        self.precision = 5  # Decimal precision for rounding
        self.velocity_threshold = 1e-4
        self.acceleration_threshold = 1e-6
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

    def detect_rotational_motion(self, angular_velocity, linear_velocity, radius=1):
        """
        v - velocity, w - angular velocity
        - Pure Rotation: v = 0, w != 0
        - Rolling Motion: v = r * w
        - Rolling with sliding: v != r * w
        """
        angular_magnitude = np.linalg.norm(angular_velocity)
        linear_magnitude = np.linalg.norm(linear_velocity)

        # stationary
        if linear_magnitude <= self.velocity_threshold and angular_magnitude <= self.velocity_threshold:
            return

        if linear_magnitude <= self.velocity_threshold and angular_magnitude > self.velocity_threshold:
            return "Pure Rotation"

        rolling_motions = linear_magnitude - angular_magnitude * radius
        if abs(rolling_motions) < self.epsilon:
            return "Rolling Motion"

        if linear_magnitude > self.velocity_threshold and angular_magnitude > self.velocity_threshold:
            return "Rolling with Sliding"

        return None

    def get_rotational_motions(self, angular_velocity, linear_velocity, radius, shape):
        if shape == "ball":
            rotational_motion = self.detect_rotational_motion(angular_velocity, linear_velocity, radius)

            if rotational_motion:
                return {
                    "category": "Kinematic Events",
                    "subcategory": "Rotational Motion",
                    "labels": [rotational_motion],
                }
        return None

    # def detect_projectile_motion(self, velocity, acceleration, drag_coefficient, mass):
    #     """
    #     - Without air resistance: Acceleration is only due to gravity.
    #     - With air resistance: Acceleration includes drag force.
    #     """
    #     gravity_acceleration = np.array([0, -self.G, 0])  # Gravity acts downward

    #     #
    #     if np.allclose(acceleration, gravity_acceleration, atol=self.epsilon):
    #         return "Projectile Motion (No Air Resistance)"

    #     # if there is air resistance
    #     drag_force = -drag_coefficient * np.array(velocity)
    #     net_acceleration_with_drag = gravity_acceleration + (drag_force / mass))

    #     if np.allclose(net_acceleration, net_acceleration_with_drag, atol=self.epsilon):
    #         return "Projectile Motion (With Air Resistance)"

    #     return None

    # def get_projectile_motion(self, cur_velocity, prev_velocity, dt, mass, drag_coefficient=0.0):
    #     vel_curr = np.array(cur_velocity)
    #     vel_prev = np.array(prev_velocity)
    #     vel_curr_magnitude = round(np.linalg.norm(vel_curr), self.precision)
    #     vel_prev_magnitude = round(np.linalg.norm(vel_prev), self.precision)

    #     acceleration = (vel_curr - vel_prev) / dt

    #     projectile_motion = self.detect_projectile_motion(cur_velocity, acceleration, drag_coefficient, mass)
    #     return {
    #         "category": "Kinematic Events",
    #         "subcategory": "Projectile Motion",
    #         "labels": [projectile_motion],
    #     }

    def detect_friction_events(self, velocity, acceleration, friction_coefficient):
        velocity_magnitude = np.linalg.norm(velocity)
        acceleration_magnitude = np.linalg.norm(acceleration)
        # a_f = -μg
        expected_friction_accel = -friction_coefficient * self.G

        # Friction Stop
        if velocity_magnitude <= self.velocity_threshold:
            # if velocity_magnitude <= self.velocity_threshold and acceleration_magnitude < self.acceleration_threshold:
            return "Friction Stop"

        # Sliding with Friction: Deceleration matches friction force, and velocity is reducing
        if acceleration_magnitude > self.acceleration_threshold and np.dot(velocity, acceleration) < 0:
            if np.isclose(acceleration_magnitude, abs(expected_friction_accel), atol=0.1):
                return "Sliding with Friction"

        return None

    def environmental_interactions(self, cur_velocity, prev_velocity, dt, friction_coefficient):
        vel_curr = np.array(cur_velocity)
        vel_prev = np.array(prev_velocity)

        acceleration = (vel_curr - vel_prev) / dt
        friction_event = self.detect_friction_events(cur_velocity, acceleration, friction_coefficient)
        if friction_event:
            return {
                "category": "Environmental Interactions",
                "subcategory": "Friction-Induced Events",
                "labels": [friction_event],
            }
        return None
        pass

    def get_friction_coefficients(self, friction):
        if type(friction) is list and len(friction) == 3:
            return {
                "sliding": friction[0],  # Sliding friction
                "torsional": friction[1],  # Torsional friction
                "rolling": friction[2],  # Rolling friction
            }

        return {
            "sliding": friction
        }

    def get_taxonomy(self, model, data, dt, prev_frame, current_objects):

        results = {obj["id"]: [] for obj in self.objects}
        dt = max(dt, 1e-6)  # division by zero
        collision_results = self.detect_collision(model, data, prev_frame, current_objects)

        for obj in self.objects:
            object_id = obj.get("id", '')
            mass = obj.get("mass", 0.0)
            friction = float(obj.get("friction", '0'))

            cur_velocity = current_objects.get(object_id, {}).get("velocity", [])
            cur_angular_velocity = current_objects.get(object_id, {}).get("angular_velocity", [])

            prev_velocity = prev_frame.get(object_id, {}).get("velocity", [])

            # prev_velocity = [float(x) for x in prev_velocity]
            # Detect linear motion
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
                cur_velocity, prev_velocity, dt, friction_coefficients["sliding"])
            if friction_event:
                results[object_id].append(friction_event)

            # Detect projectile motions:
            # projectile_motion = self.get_projectile_motion(self, cur_velocity, prev_velocity, dt, mass, drag_coefficient)
            # if projectile_motion:
            #     results[object_id].append(projectile_motion)

        # add collision results to the objects
        for (g1, g2), collision_info in collision_results.items():
            obj1_id = next(obj["id"] for obj in self.objects if obj["geom_id"] == g1)
            obj2_id = next(obj["id"] for obj in self.objects if obj["geom_id"] == g2)

            results[obj1_id].append(collision_info)
            results[obj2_id].append(collision_info)

        return results
