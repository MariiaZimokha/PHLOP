import numpy as np


class PhysicsEngine:
    def __init__(
        self,
        precision=5,
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
        self.precision = precision

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
        vel_curr_mag = round(np.linalg.norm(vel_curr), self.precision)
        vel_prev_mag = round(np.linalg.norm(vel_prev), self.precision)
        accel = (vel_curr - vel_prev) / dt
        accel_mag = round(np.linalg.norm(accel), self.precision)

        # If current velocity is near threshold or both previous and current velocities are near zero
        if abs(vel_curr_mag - self.velocity_threshold) < self.velocity_threshold:
            return "Stationary"
        if (
            abs(vel_curr_mag) <= self.velocity_threshold
            and abs(vel_prev_mag) <= self.velocity_threshold
        ):
            return "Stationary"
        # If acceleration is negligible then it's constant velocity
        if accel_mag < self.acceleration_threshold:
            return "Constant Velocity"
        else:
            # Determine if accelerating or decelerating based on the dot product between velocity and acceleration
            # if np.dot(vel_curr, accel) > 0:
            if vel_curr_mag > vel_prev_mag:
                return "Accelerating"
            else:
                return "Decelerating"

    def detect_rotational_motion(self, angular_velocity, linear_velocity, radius=1):
        """
        Returns:
        - None: No significant rotational motion
        - "Pure Rotation": Rotating in place (stationary)
        - "Rolling Motion": v ≈ r·ω (within epsilon tolerance)
        - "Rolling Motion with Slipping": v ≠ r·ω but both present
        """
        angular_velocity = np.array(angular_velocity)
        linear_velocity = np.array(linear_velocity)
        angular_mag = np.linalg.norm(angular_velocity)
        linear_mag = np.linalg.norm(linear_velocity)

        # If both linear and angular velocities are very low, treat as no significant rotation
        if (
            linear_mag <= self.velocity_threshold
            and angular_mag <= self.velocity_threshold
        ):
            return None
        if (
            linear_mag <= self.velocity_threshold
            and angular_mag > self.velocity_threshold
        ):
            return "Pure Rotation"

        rolling_diff = linear_mag - angular_mag * radius
        if abs(rolling_diff) < self.epsilon:
            return "Rolling Motion"
        if (
            linear_mag > self.velocity_threshold
            and angular_mag > self.velocity_threshold
        ):
            return "Rolling Motion with Slipping"
        return None

    def detect_friction_event(
        self, velocity, acceleration, friction_coefficient, drag_coefficient=None
    ):
        """
        Detect friction-related events: Friction Stop, Sliding with Friction, or None
        Returns:
        - "Sliding with Friction": During motion phase (v > threshold)
        - "Friction Stop": When object comes to stop due to friction
        - None: When stationary or not friction-related
        """
        velocity = np.array(velocity)
        acceleration = np.array(acceleration)
        vel_mag = np.linalg.norm(velocity)
        accel_mag = np.linalg.norm(acceleration)

        # expected frictional deceleration
        expected_friction_accel = -friction_coefficient * self.gravity

        # if vel_mag <= self.velocity_threshold:
        if (
            vel_mag <= self.velocity_threshold
            and accel_mag <= self.acceleration_threshold
        ):
            return "Friction Stop"

        # ensure the object is actually moving
        if vel_mag > self.velocity_threshold:
            # ensure acceleration is negative (deceleration) and opposes velocity
            if np.dot(velocity, acceleration) < 0:
                # Compute actual deceleration direction
                deceleration_vector = -velocity / vel_mag * accel_mag
                expected_deceleration_vector = (
                    -velocity / vel_mag * abs(expected_friction_accel)
                )

                # check if the actual deceleration is aligned with expected frictional force
                alignment = np.dot(deceleration_vector, expected_deceleration_vector)

                if alignment > 0.90:  # Threshold to ensure alignment
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
        momentum = mass1 * vel1_pre + mass2 * vel2_pre

        # Collision geometry
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
