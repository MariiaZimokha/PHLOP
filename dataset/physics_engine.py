import numpy as np


class PhysicsEngine:
    def __init__(self, precision=5, velocity_threshold=1e-4, acceleration_threshold=1e-6, epsilon=0.01, gravity=9.8, collision_elastic_factor=0.5):
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
        if abs(vel_curr_mag) <= self.velocity_threshold and abs(vel_prev_mag) <= self.velocity_threshold:
            return "Stationary"
        # If acceleration is negligible then it's constant velocity
        if accel_mag < self.acceleration_threshold:
            return "Constant Velocity"
        else:
            # Determine if accelerating or decelerating based on the dot product between velocity and acceleration
            if np.dot(vel_curr, accel) > 0:
                return "Accelerating"
            else:
                return "Decelerating"

    def detect_rotational_motion(self, angular_velocity, linear_velocity, radius=1):
        """
        Determines the rotational motion type:
          - Pure Rotation: When linear speed is near zero but angular speed is significant.
          - Rolling Motion: When v ≈ r * w.
          - Rolling with Slipping: When linear speed deviates from r * w.
        """
        angular_velocity = np.array(angular_velocity)
        linear_velocity = np.array(linear_velocity)
        angular_mag = np.linalg.norm(angular_velocity)
        linear_mag = np.linalg.norm(linear_velocity)

        # If both linear and angular velocities are very low, treat as no significant rotation
        if linear_mag <= self.velocity_threshold and angular_mag <= self.velocity_threshold:
            return None
        if linear_mag <= self.velocity_threshold and angular_mag > self.velocity_threshold:
            return "Pure Rotation"

        rolling_diff = linear_mag - angular_mag * radius
        if abs(rolling_diff) < self.epsilon:
            return "Rolling Motion"
        if linear_mag > self.velocity_threshold and angular_mag > self.velocity_threshold:
            return "Rolling Motion with Slipping"
        return None

    def detect_friction_event(self, velocity, acceleration, friction_coefficient, drag_coefficient=None):
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

        if vel_mag == 0 and accel_mag == 0:
            return "Friction Stop"

        if vel_mag <= self.velocity_threshold:
            return "Friction Stop"

        if accel_mag > self.acceleration_threshold and np.dot(velocity, acceleration) < 0:
            if np.isclose(accel_mag, abs(expected_friction_accel), atol=0.1):
                return "Sliding with Friction"
            # if drag-related deceleration.
            if drag_coefficient is not None:
                # air resistance at higher speeds
                # deceleration scales with the square of the velocity.
                expected_drag_accel = -drag_coefficient * (vel_mag ** 2)
                if np.isclose(accel_mag, abs(expected_drag_accel), atol=0.1):
                    return "Sliding with Drag"
        return None

    def detect_collision(self, vel1_pre, vel2_pre, vel1_post, vel2_post, normal):
        # relative velocities before and after the collision
        rel_vel_pre = np.dot((np.array(vel1_pre) - np.array(vel2_pre)), normal)
        rel_vel_post = np.dot((np.array(vel1_post) - np.array(vel2_post)), normal)
        elasticity_ratio = abs(rel_vel_post / rel_vel_pre) if rel_vel_pre != 0 else 0

        return (
            "Elastic Collision" if elasticity_ratio > self.collision_elastic_factor
            else "Inelastic Collision"
        )
