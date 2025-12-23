import numpy as np


class PhysicsEngine:
    def __init__(
        self,
        precision=5,
        velocity_threshold=0.05,
        acceleration_threshold=0.1,
        epsilon=0.01,
        gravity=9.81,
        collision_elastic_factor=0.5,
    ):
        """
        Physics Engine with comprehensive collision and motion analysis.
        """
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.epsilon = epsilon
        self.gravity = gravity
        self.collision_elastic_factor = collision_elastic_factor
        self.precision = precision

    def detect_linear_motion(self, vel_prev, vel_curr, dt):
        vel_prev = np.array(vel_prev)
        vel_curr = np.array(vel_curr)
        vel_curr_mag = round(np.linalg.norm(vel_curr), self.precision)
        vel_prev_mag = round(np.linalg.norm(vel_prev), self.precision)
        dt = max(dt, 1e-6)

        accel = (vel_curr - vel_prev) / dt
        accel_mag = round(np.linalg.norm(accel), self.precision)

        if (
            abs(vel_curr_mag) <= self.velocity_threshold
            and abs(vel_prev_mag) <= self.velocity_threshold
        ):
            return "Stationary"

        if abs(vel_curr_mag) <= self.velocity_threshold:
            return "Stationary"

        vel_change = vel_curr_mag - vel_prev_mag
        vel_change_threshold = 1e-4

        if accel_mag >= self.acceleration_threshold:
            if vel_change > vel_change_threshold:
                return "Accelerating"
            elif vel_change < -vel_change_threshold:
                return "Decelerating"
            else:
                if np.dot(vel_curr, accel) > 0:
                    return "Accelerating"
                else:
                    return "Decelerating"
        else:
            return "Constant Velocity"

    def detect_rotational_motion(self, angular_velocity, linear_velocity, radius=1):
        angular_velocity = np.array(angular_velocity)
        linear_velocity = np.array(linear_velocity)
        angular_mag = np.linalg.norm(angular_velocity)
        linear_mag = np.linalg.norm(linear_velocity)

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
        velocity = np.array(velocity)
        acceleration = np.array(acceleration)
        vel_mag = round(np.linalg.norm(velocity), self.precision)
        accel_mag = round(np.linalg.norm(acceleration), self.precision)

        expected_friction_accel = -friction_coefficient * self.gravity
        accel_threshold = self.acceleration_threshold

        if vel_mag <= self.velocity_threshold:
            accel_threshold = self.acceleration_threshold * 1.5

        if vel_mag <= self.velocity_threshold and accel_mag <= accel_threshold:
            return "Friction Stop"

        if vel_mag > self.velocity_threshold:
            if np.dot(velocity, acceleration) < 0:
                deceleration_vector = -velocity / vel_mag * accel_mag
                expected_deceleration_vector = (
                    -velocity / vel_mag * abs(expected_friction_accel)
                )

                alignment = np.dot(deceleration_vector, expected_deceleration_vector)
                if alignment > 0.90:
                    return "Sliding with Friction"
        return None

    def detect_collision(
        self, vel1_pre, vel2_pre, vel1_post, vel2_post, normal, m1=None, m2=None
    ):
        """
        Detect and classify collisions using Energy Ratio as ground truth.

        Returns: "Elastic Collision", "Partially Inelastic Collision",
                 "Highly Inelastic Collision", or None

        PRIORITY LOGIC:
        If masses (m1, m2) are provided, Energy Conservation is used as the ground truth
        to prevent contradictions between the "Label" and "Energy Analysis".

        Args:
            vel1_pre, vel2_pre: Pre-collision velocities
            vel1_post, vel2_post: Post-collision velocities
            normal: Contact normal vector
            m1, m2: (Optional) Masses of the objects.
        """
        vel1_pre = np.array(vel1_pre)
        vel2_pre = np.array(vel2_pre)
        vel1_post = np.array(vel1_post)
        vel2_post = np.array(vel2_post)
        normal = np.array(normal)

        # 1. Energy Check (Ground Truth) - returns 3 types aligned with energy_analysis
        if m1 is not None and m2 is not None:
            ke_pre = (
                0.5 * m1 * np.linalg.norm(vel1_pre) ** 2
                + 0.5 * m2 * np.linalg.norm(vel2_pre) ** 2
            )
            ke_post = (
                0.5 * m1 * np.linalg.norm(vel1_post) ** 2
                + 0.5 * m2 * np.linalg.norm(vel2_post) ** 2
            )

            if ke_pre > 1e-9:
                ke_ratio = ke_post / ke_pre

                if ke_ratio > 0.9:
                    return "Elastic Collision"
                elif ke_ratio > 0.5:
                    return "Partially Inelastic Collision"
                else:
                    return "Highly Inelastic Collision"

        # 2. Kinematic Check (Coefficient of Restitution) - Fallback
        normal_mag = np.linalg.norm(normal)
        if normal_mag > 0:
            normal = normal / normal_mag

        rel_vel_pre = np.dot(vel1_pre - vel2_pre, normal)
        rel_vel_post = np.dot(vel1_post - vel2_post, normal)

        if abs(rel_vel_pre) < self.velocity_threshold:
            return None

        restitution = -rel_vel_post / rel_vel_pre

        if restitution >= self.collision_elastic_factor:
            return "Elastic Collision"

        return "Inelastic Collision"

    def compute_collision_context(self, vel1_pre, vel2_pre, mass1, mass2, normal):
        """Compute detailed collision context (metrics)."""
        vel1_pre = np.array(vel1_pre)
        vel2_pre = np.array(vel2_pre)
        normal = np.array(normal)

        normal_mag = np.linalg.norm(normal)
        if normal_mag > 0:
            normal = normal / normal_mag

        rel_velocity = vel1_pre - vel2_pre
        rel_vel_mag = np.linalg.norm(rel_velocity)
        rel_vel_normal = np.dot(rel_velocity, normal)

        v1_mag = np.linalg.norm(vel1_pre)
        v2_mag = np.linalg.norm(vel2_pre)
        ke1_pre = 0.5 * mass1 * v1_mag**2
        ke2_pre = 0.5 * mass2 * v2_mag**2
        total_ke_pre = ke1_pre + ke2_pre
        momentum = mass1 * vel1_pre + mass2 * vel2_pre

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
        """Analyze energy transfer (KE ratio)."""
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
        """Check momentum conservation."""
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
