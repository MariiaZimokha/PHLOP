import numpy as np
from collections import deque

from phlop.physics_engine import PhysicsEngine
from phlop.utils import is_cylinder_upright


class PhysicsTaxonomy:
    def __init__(self, objects):
        self.objects = objects
        self.MOTION_WINDOW_SIZE = 5
        self.motion_history = {
            obj["id"]: deque(maxlen=self.MOTION_WINDOW_SIZE) for obj in objects
        }
        self.physics_engine = PhysicsEngine()
        self.prev_collision_pairs = set()
        # Track rotational state per object to avoid redundant friction detection
        self.rotational_state = {obj["id"]: None for obj in objects}

    def analyze_motion(self, object_id, vel_prev, vel_curr, dt):
        """Analyze linear motion state."""
        current_motion = self.physics_engine.detect_linear_motion(
            vel_prev, vel_curr, dt
        )
        return {
            "category": "Kinematic Events",
            "subcategory": "Linear Motion",
            "labels": [current_motion],
        }

    def detect_collision(
        self, model, data, prev_frame, current_objects, geom_id_to_obj_id
    ):
        """
        Detect collisions and classify them.
        Only records collision ONCE when first detected.
        """
        collision_results = {}
        current_collision_pairs = set()

        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            # Skip floor collisions (geom_id = 0)
            if g1 == 0 or g2 == 0:
                continue

            pair = tuple(sorted((g1, g2)))
            current_collision_pairs.add(pair)

            # Skip if already detected in previous frame
            if pair in self.prev_collision_pairs:
                continue

            # Compute contact normal
            normal = contact.frame[:3]
            normal_mag = np.linalg.norm(normal)
            if normal_mag > 0:
                normal = normal / normal_mag

            # Get object IDs from geom_id mapping
            vel1_id = geom_id_to_obj_id.get(g1)
            vel2_id = geom_id_to_obj_id.get(g2)

            if vel1_id is None or vel2_id is None:
                continue

            # Get velocities before and after
            vel1_pre = prev_frame.get(vel1_id, {}).get("velocity", [])
            vel2_pre = prev_frame.get(vel2_id, {}).get("velocity", [])
            vel1_post = current_objects.get(vel1_id, {}).get("velocity", [])
            vel2_post = current_objects.get(vel2_id, {}).get("velocity", [])

            if not all([vel1_pre, vel2_pre, vel1_post, vel2_post]):
                continue

            # Retrieve masses BEFORE detecting collision
            # This ensures collision label aligns with energy_analysis
            mass1 = next((o["mass"] for o in self.objects if o["id"] == vel1_id), 1.0)
            mass2 = next((o["mass"] for o in self.objects if o["id"] == vel2_id), 1.0)

            # Pass masses (m1, m2) to detect_collision so it uses energy as ground truth
            # This returns: "Elastic Collision", "Partially Inelastic Collision", or "Highly Inelastic Collision"
            collision_type = self.physics_engine.detect_collision(
                vel1_pre, vel2_pre, vel1_post, vel2_post, normal, m1=mass1, m2=mass2
            )

            if collision_type:
                # Compute collision context
                context = self.physics_engine.compute_collision_context(
                    vel1_pre, vel2_pre, mass1, mass2, normal
                )

                # Compute energy transfer
                energy_analysis = self.physics_engine.detect_energy_transfer(
                    vel1_pre, vel2_pre, vel1_post, vel2_post, mass1, mass2
                )

                # Check momentum conservation
                momentum_check = self.physics_engine.calculate_momentum_conservation(
                    vel1_pre, vel2_pre, vel1_post, vel2_post, mass1, mass2
                )

                collision_results[pair] = {
                    "category": "Interaction Events",
                    "subcategory": "Collision",
                    "labels": [collision_type],
                    "context": context,
                    "energy_analysis": energy_analysis,
                    "momentum_check": momentum_check,
                }

        self.prev_collision_pairs = current_collision_pairs
        return collision_results

    def map_motion_state(self, motion):
        """Map motion to higher-level state."""
        if motion == "Stationary":
            return "Stationary"
        elif motion in ("Constant Velocity", "Accelerating", "Decelerating"):
            return "Moving"
        else:
            return None

    def detect_state_transitions(self, cur_velocity, prev_velocity, object_id, dt):
        """
        Detect state changes: Moving→Stopping (friction) or Stationary→Moving (force)
        """
        cur_motion = self.physics_engine.detect_linear_motion(
            prev_velocity, cur_velocity, dt
        )
        prev_motion = (
            self.motion_history[object_id][-1]
            if self.motion_history[object_id]
            else None
        )
        prev_motion_mapped = self.map_motion_state(prev_motion) if prev_motion else None

        state_transition = ""
        if prev_motion_mapped == "Moving" and cur_motion == "Decelerating":
            state_transition = "Moving to Stopping"
        elif prev_motion_mapped == "Stationary" and cur_motion in (
            "Accelerating",
            "Constant Velocity",
        ):
            state_transition = "Stationary to Moving"

        if cur_motion:
            self.motion_history[object_id].append(cur_motion)

        return state_transition

    def get_state_transitions_labels(self, cur_velocity, prev_velocity, object_id, dt):
        """Get state transition event."""
        label = self.detect_state_transitions(
            cur_velocity, prev_velocity, object_id, dt
        )
        if label:
            return {
                "category": "State Transitions",
                "subcategory": "Motion Change",
                "labels": [label],
            }
        return None

    def get_rotational_motions(
        self, angular_velocity, linear_velocity, radius, shape, model, data, object_id
    ):
        """
        Detect rotational motion for any shape.
        Returns: Pure Rotation, Rolling Motion, Rolling with Slipping, or None
        """
        if shape == "cylinder":
            upright = is_cylinder_upright(self.objects, model, data, object_id)
            if upright:
                return None

            rotational_motion = self.physics_engine.detect_rotational_motion(
                angular_velocity, linear_velocity, radius
            )
            if rotational_motion:
                return {
                    "category": "Kinematic Events",
                    "subcategory": "Rotational Motion",
                    "labels": [rotational_motion],
                }

        if shape in ["ball", "sphere"]:
            rotational_motion = self.physics_engine.detect_rotational_motion(
                angular_velocity, linear_velocity, radius
            )
            if rotational_motion:
                return {
                    "category": "Kinematic Events",
                    "subcategory": "Rotational Motion",
                    "labels": [rotational_motion],
                }
        return None

    def environmental_interactions(
        self,
        cur_velocity,
        prev_velocity,
        dt,
        friction_coefficient,
        shape=None,
        model=None,
        data=None,
        object_id=None,
    ):
        """
        Detect friction effects.
        Only detect sliding friction, NOT rolling friction.
        """
        # Skip friction detection for balls
        if shape == "ball":
            return None

        # cylinders, only detect friction if upright (on its end edge)
        if shape == "cylinder":
            upright = is_cylinder_upright(self.objects, model, data, object_id)
            if not upright:
                return None

        cur_velocity = np.array(cur_velocity)
        prev_velocity = np.array(prev_velocity)
        acceleration = (cur_velocity - prev_velocity) / max(dt, 1e-6)

        friction_event = self.physics_engine.detect_friction_event(
            cur_velocity, acceleration, friction_coefficient
        )

        if friction_event:
            return {
                "category": "Environmental Interactions",
                "subcategory": "Friction",
                "labels": [friction_event],
            }
        return None

    def get_friction_coefficients(self, friction):
        """Parse friction coefficient(s)."""
        if isinstance(friction, list) and len(friction) >= 1:
            return {"sliding": float(friction[0])}
        elif isinstance(friction, (int, float)):
            return {"sliding": float(friction)}
        elif isinstance(friction, str):
            try:
                parts = [float(f) for f in friction.split()]
                return {"sliding": float(parts[0])} if parts else {"sliding": 0.4}
            except (ValueError, IndexError):
                return {"sliding": 0.4}
        return {"sliding": 0.4}

    def get_taxonomy(
        self, model, data, dt, prev_frame, current_objects, geom_id_to_obj_id
    ):
        """
        Main taxonomy computation for all objects.

        Returns: Dictionary mapping object_id → list of taxonomy events
        """
        results = {obj["id"]: [] for obj in self.objects}
        dt = max(dt, 1e-6)

        # Detect collisions first
        collision_results = self.detect_collision(
            model, data, prev_frame, current_objects, geom_id_to_obj_id
        )

        # Analyze each object
        for obj in self.objects:
            object_id = obj.get("id", "")
            if not object_id:
                continue

            # Get current and previous states
            cur_velocity = current_objects.get(object_id, {}).get("velocity", [])
            prev_velocity = prev_frame.get(object_id, {}).get("velocity", [])
            cur_angular_velocity = current_objects.get(object_id, {}).get(
                "angular_velocity", []
            )

            if not cur_velocity or not prev_velocity:
                continue

            # 1. Linear motion
            linear_motion = self.analyze_motion(
                object_id, prev_velocity, cur_velocity, dt
            )
            if linear_motion:
                results[object_id].append(linear_motion)

            # 2. State transitions
            state_transitions = self.get_state_transitions_labels(
                cur_velocity, prev_velocity, object_id, dt
            )
            if state_transitions:
                results[object_id].append(state_transitions)

            # 3. Rotational motion
            radius = obj.get("dimensions", {}).get("radius", 0.1)
            rotational_motion = self.get_rotational_motions(
                cur_angular_velocity,
                cur_velocity,
                radius,
                obj.get("shape", ""),
                model,
                data,
                object_id,
            )
            if rotational_motion:
                results[object_id].append(rotational_motion)

            # 4. Friction effects
            friction_coeff = self.get_friction_coefficients(obj.get("friction", 0.4))[
                "sliding"
            ]
            friction_event = self.environmental_interactions(
                cur_velocity,
                prev_velocity,
                dt,
                friction_coeff,
                obj.get("shape", ""),
                model=model,
                data=data,
                object_id=object_id,
            )
            if friction_event:
                results[object_id].append(friction_event)

        # 5. Add collisions to objects involved
        for (g1, g2), collision_info in collision_results.items():
            obj1_id = geom_id_to_obj_id.get(g1)
            obj2_id = geom_id_to_obj_id.get(g2)

            if obj1_id and obj2_id:
                results[obj1_id].append(collision_info)
                results[obj2_id].append(collision_info)

        return results
