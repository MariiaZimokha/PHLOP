import numpy as np


class PhysicsEngine:
    def __init__(
        self,
        precision=5,
        velocity_threshold=1e-6,
        acceleration_threshold=1e-6,
        epsilon=0.01,
        gravity=9.8,
        collision_elastic_factor=0.5,
    ):
        self.precision = precision
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.epsilon = epsilon
        self.gravity = gravity
        self.collision_elastic_factor = collision_elastic_factor

    def detect_linear_motion(self, vel_prev, vel_curr, dt):
        """
        Check if the acceleration is zero over dt.
        a = (v_current - v_prev) / dt
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
        Determines the rotational motion type:
          - Pure Rotation: When linear speed is near zero but angular speed is significant.
          - Rolling Motion: When v ≈ r * w - linear velocity is proportional to the angular velocity,
          - Rolling with Slipping: When linear speed deviates from r * w.
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
        Detect friction-related events.
          - "Friction Stop": When the object’s velocity is negligible.
          - "Sliding with Friction": When deceleration is consistent with friction.
          - "Sliding with Drag": When deceleration is consistent with drag force effects.

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
        vel1_pre = np.array(vel1_pre)
        vel2_pre = np.array(vel2_pre)
        vel1_post = np.array(vel1_post)
        vel2_post = np.array(vel2_post)

        rel_vel_pre = np.dot(vel1_pre - vel2_pre, normal)
        rel_vel_post = np.dot(vel1_post - vel2_post, normal)

        # If the pre-collision relative speed is too low, avoid division by zero.
        if abs(rel_vel_pre) < self.velocity_threshold:
            return

        # coefficient of restitution
        # (with the negative sign to account for direction reversal)
        restitution = -rel_vel_post / rel_vel_pre

        margin = 0.05

        if restitution >= self.collision_elastic_factor - margin:
            return "Elastic Collision"
        else:
            return "Inelastic Collision"
