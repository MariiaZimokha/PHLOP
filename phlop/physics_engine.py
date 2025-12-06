import numpy as np


class PhysicsEngine:
    def __init__(
        self,
        velocity_threshold=1e-3,
        acceleration_threshold=1e-3,
        epsilon=0.01,
        gravity=9.81,
        collision_elastic_factor=0.5,
    ):
        """
        Physics Engine with comprehensive collision and motion analysis.

        Args:
            velocity_threshold: Below this (m/s), object is stationary (1 mm/s)
            acceleration_threshold: Below this (m/s²), motion is constant velocity
            epsilon: Tolerance for rolling condition check
            gravity: Gravitational acceleration (m/s²)
            collision_elastic_factor: Threshold for elastic vs inelastic
        """
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.epsilon = epsilon
        self.gravity = gravity
        self.collision_elastic_factor = collision_elastic_factor

    def detect_linear_motion(self, vel_prev, vel_curr, dt):
        """
        Detect linear motion state: Stationary, Constant Velocity, Accelerating, Decelerating

        Args:
            vel_prev: Previous velocity [vx, vy, vz]
            vel_curr: Current velocity [vx, vy, vz]
            dt: Time step

        Returns:
            Motion classification string
        """
        vel_prev = np.array(vel_prev)
        vel_curr = np.array(vel_curr)

        vel_curr_mag = np.linalg.norm(vel_curr)
        vel_prev_mag = np.linalg.norm(vel_prev)

        # Object is stationary if velocity is very small
        if vel_curr_mag < self.velocity_threshold:
            return "Stationary"

        #  acceleration
        accel = (vel_curr - vel_prev) / max(dt, 1e-6)
        accel_mag = np.linalg.norm(accel)

        # Constant velocity: no significant acceleration
        if accel_mag < self.acceleration_threshold:
            return "Constant Velocity"

        # Accelerating: velocity is increasing
        if vel_curr_mag > vel_prev_mag:
            return "Accelerating"

        # Decelerating: velocity is decreasing
        return "Decelerating"

    def detect_rotational_motion(self, angular_velocity, linear_velocity, radius=1):
        """
        Detect rotational motion type based on axis of rotation and velocity relationship.

        IMPORTANT: Rolling only occurs when rotation axis is perpendicular to motion direction.
        - Ball rolling down: rotation axis perpendicular to velocity → rolling
        - Coin spinning on table: rotation axis parallel to velocity → spinning (not rolling)
        - Cylinder sliding sideways with spin: depends on which axis is spinning

        Args:
            angular_velocity: Angular velocity [wx, wy, wz] (rad/s)
            linear_velocity: Linear velocity [vx, vy, vz] (m/s)
            radius: Object radius (m) - only matters for rolling checks

        Returns:
            Rotation classification or None
        """
        angular_velocity = np.array(angular_velocity)
        linear_velocity = np.array(linear_velocity)

        angular_mag = np.linalg.norm(angular_velocity)
        linear_mag = np.linalg.norm(linear_velocity)

        # Case 1: No significant motion at all
        if (
            linear_mag < self.velocity_threshold
            and angular_mag < self.velocity_threshold
        ):
            return None

        # Case 2: PURE SPINNING - rotating but not moving linearly
        # Example: coin spinning on table, object spinning in place
        min_angular_for_pure_rotation = 0.5  # ~30 deg/s
        if (
            linear_mag < self.velocity_threshold
            and angular_mag > min_angular_for_pure_rotation
        ):
            return "Pure Spinning"

        # Case 3: Only linear motion, no significant rotation
        if linear_mag > self.velocity_threshold and angular_mag < 0.1:
            return None

        # Case 4: Both linear AND angular motion - check rolling condition
        # Rolling occurs when rotation axis is perpendicular to velocity direction
        # AND v ≈ r * ω
        if linear_mag > self.velocity_threshold and angular_mag > 0.1 and radius > 0:
            expected_linear_vel = angular_mag * radius
            rolling_diff = abs(linear_mag - expected_linear_vel)

            # Check if rotation axis is perpendicular to velocity
            # (dot product should be close to 0 for perpendicular vectors)
            if angular_mag > 0:
                axis_direction = angular_velocity / angular_mag
                velocity_direction = linear_velocity / (linear_mag + 1e-6)
                axis_alignment = abs(np.dot(axis_direction, velocity_direction))
            else:
                axis_alignment = 1.0

            # True rolling: perpendicular axis AND v ≈ r*ω
            if (
                axis_alignment < 0.3 and rolling_diff < self.epsilon
            ):  # Perpendicular + matching velocities
                return "Rolling Motion"

            # Rolling with slipping: perpendicular axis BUT v ≠ r*ω
            if axis_alignment < 0.3 and rolling_diff >= self.epsilon:
                return "Rolling Motion with Slipping"

            # Spinning while sliding: axis parallel/mixed to velocity direction
            # Example: coin sliding while spinning on its face, or cylinder sliding with axial spin
            if axis_alignment > 0.5:  # Axis roughly parallel to velocity
                return "Spinning While Sliding"

        # No significant rotation detected
        return None

    def detect_friction_event(
        self, velocity, acceleration, friction_coefficient, drag_coefficient=None
    ):
        """
        Detect friction-related events: Friction Stop, Sliding with Friction, or None

        Args:
            velocity: Current velocity [vx, vy, vz]
            acceleration: Current acceleration [ax, ay, az]
            friction_coefficient: Coefficient of friction (μ)
            drag_coefficient: Not used (placeholder for future)

        Returns:
            Friction event classification or None
        """
        velocity = np.array(velocity)
        acceleration = np.array(acceleration)

        vel_mag = np.linalg.norm(velocity)
        accel_mag = np.linalg.norm(acceleration)

        # Case 1: Object at rest
        if (
            vel_mag < self.velocity_threshold
            and accel_mag < self.acceleration_threshold
        ):
            return "Friction Stop"

        # Case 2: Object moving with deceleration due to friction
        if vel_mag > self.velocity_threshold:
            dot_product = np.dot(velocity, acceleration)

            # Check if acceleration opposes velocity (deceleration)
            if dot_product < 0:
                expected_friction_accel = friction_coefficient * self.gravity

                # Check if deceleration magnitude matches expected friction
                # Allow 20% tolerance for numerical errors
                if (
                    abs(accel_mag - expected_friction_accel)
                    / (expected_friction_accel + 1e-6)
                    < 0.2
                ):
                    return "Sliding with Friction"

        return None

    def detect_collision(self, vel1_pre, vel2_pre, vel1_post, vel2_post, normal):
        """
        Detect and classify collisions as Elastic or Inelastic.

        Uses coefficient of restitution: e = -rel_vel_post / rel_vel_pre

        Args:
            vel1_pre: Object 1 velocity before collision
            vel2_pre: Object 2 velocity before collision
            vel1_post: Object 1 velocity after collision
            vel2_post: Object 2 velocity after collision
            normal: Contact normal direction (unit vector)

        Returns:
            "Elastic Collision", "Inelastic Collision", or None
        """
        vel1_pre = np.array(vel1_pre)
        vel2_pre = np.array(vel2_pre)
        vel1_post = np.array(vel1_post)
        vel2_post = np.array(vel2_post)
        normal = np.array(normal)

        # Normalize normal
        normal_mag = np.linalg.norm(normal)
        if normal_mag > 0:
            normal = normal / normal_mag

        # Relative velocities along contact normal
        rel_vel_pre = np.dot(vel1_pre - vel2_pre, normal)
        rel_vel_post = np.dot(vel1_post - vel2_post, normal)

        # No significant relative velocity → can't classify
        if abs(rel_vel_pre) < self.velocity_threshold:
            return None

        # Coefficient of restitution
        restitution = -rel_vel_post / rel_vel_pre

        # Elastic: high restitution (energy conserved)
        if restitution >= self.collision_elastic_factor - 0.1:
            return "Elastic Collision"

        # Inelastic: low restitution (energy lost)
        return "Inelastic Collision"

    def compute_collision_context(self, vel1_pre, vel2_pre, mass1, mass2, normal):
        """
        Compute detailed collision context for hard questions.

        Args:
            vel1_pre: Object 1 pre-collision velocity
            vel2_pre: Object 2 pre-collision velocity
            mass1: Object 1 mass
            mass2: Object 2 mass
            normal: Contact normal

        Returns:
            Dictionary with collision metrics
        """
        vel1_pre = np.array(vel1_pre)
        vel2_pre = np.array(vel2_pre)
        normal = np.array(normal)

        # Normalize normal
        normal_mag = np.linalg.norm(normal)
        if normal_mag > 0:
            normal = normal / normal_mag

        # Relative velocity
        rel_velocity = vel1_pre - vel2_pre
        rel_vel_mag = np.linalg.norm(rel_velocity)
        rel_vel_normal = np.dot(rel_velocity, normal)

        # Kinetic energy before collision
        v1_mag = np.linalg.norm(vel1_pre)
        v2_mag = np.linalg.norm(vel2_pre)
        ke1_pre = 0.5 * mass1 * v1_mag**2
        ke2_pre = 0.5 * mass2 * v2_mag**2
        total_ke_pre = ke1_pre + ke2_pre

        # Momentum
        momentum = mass1 * vel1_pre + mass2 * vel2_pre

        # Collision geometry: is it head-on?
        v1_normal_component = abs(np.dot(vel1_pre, normal)) / (v1_mag + 1e-6)
        v2_normal_component = abs(np.dot(vel2_pre, normal)) / (v2_mag + 1e-6)
        is_head_on = v1_normal_component > 0.8 and v2_normal_component > 0.8

        return {
            "relative_velocity_magnitude": float(rel_vel_mag),
            "relative_velocity_along_normal": float(rel_vel_normal),
            "total_kinetic_energy_before": float(total_ke_pre),
            "momentum": momentum.tolist(),
            "is_head_on": bool(is_head_on),
        }

    def detect_energy_transfer(self, vel1_pre, vel2_pre, vel1_post, vel2_post, m1, m2):
        """
        Analyze energy transfer during collision.

        Returns: Classification of energy conservation
        """
        vel1_pre = np.array(vel1_pre)
        vel2_pre = np.array(vel2_pre)
        vel1_post = np.array(vel1_post)
        vel2_post = np.array(vel2_post)

        ke_pre = (
            0.5 * m1 * np.linalg.norm(vel1_pre) ** 2
            + 0.5 * m2 * np.linalg.norm(vel2_pre) ** 2
        )
        ke_post = (
            0.5 * m1 * np.linalg.norm(vel1_post) ** 2
            + 0.5 * m2 * np.linalg.norm(vel2_post) ** 2
        )

        if ke_pre < 1e-6:
            return "Negligible Initial Energy"

        ke_ratio = ke_post / ke_pre

        if ke_ratio > 0.9:
            return "Elastic (Energy Conserved)"
        elif ke_ratio > 0.5:
            return "Partially Inelastic"
        else:
            return "Highly Inelastic"

    def calculate_momentum_conservation(
        self, vel1_pre, vel2_pre, vel1_post, vel2_post, m1, m2
    ):
        """
        Check momentum conservation during collision.

        Returns: Dictionary with conservation info
        """
        vel1_pre = np.array(vel1_pre)
        vel2_pre = np.array(vel2_pre)
        vel1_post = np.array(vel1_post)
        vel2_post = np.array(vel2_post)

        p_total_before = m1 * vel1_pre + m2 * vel2_pre
        p_total_after = m1 * vel1_post + m2 * vel2_post

        p_before_mag = np.linalg.norm(p_total_before)
        p_after_mag = np.linalg.norm(p_total_after)

        if p_before_mag < 1e-6:
            return {
                "conserved": True,
                "ratio": 1.0,
                "classification": "Zero Initial Momentum",
            }

        p_ratio = p_after_mag / p_before_mag

        is_conserved = abs(1.0 - p_ratio) < 0.1

        return {
            "conserved": is_conserved,
            "ratio": float(p_ratio),
            "classification": "Conserved" if is_conserved else "Not Conserved",
        }
