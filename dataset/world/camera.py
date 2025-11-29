import random
import numpy as np
import mujoco


class CameraSettings:
    def __init__(
        self, model=None, data=None, camera_name="camera", evaluation_mode=False
    ):
        self.camera_name = camera_name
        self.evaluation_mode = evaluation_mode
        if model and data:
            self.set_model(model, data)

    def set_model(self, model=None, data=None):
        self.model = model
        self.data = data
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)

        self.cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if self.cam_id == -1:
            raise ValueError(
                f"Camera with name '{self.camera_name}' not found in the model!"
            )

    def _init_smoothing_targets(self):
        self.target_lookat = self.camera.lookat.copy()
        self.target_distance = self.camera.distance

    def set_init_settings(self, data=None):
        if data is None:
            data = {}

        # Initial random or provided camera settings
        self.camera.lookat[:] = data.get(
            "lookat",
            [
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(0, 0.5),
            ],
        )
        self.camera.distance = data.get("distance", random.uniform(1.5, 3.5))
        self.camera.azimuth = data.get("azimuth", random.uniform(-90, -30))
        self.camera.elevation = data.get("elevation", random.uniform(-30, 0))

        # Store previous + initialize target
        self.prev_camera_lookat = self.camera.lookat.copy()
        self.prev_camera_distance = self.camera.distance
        self._init_smoothing_targets()

        # Smoothing factor
        self.alpha = 0.1

    def get_init_settings(self):
        return {
            "camera_name": self.camera_name,
            "lookat": self.camera.lookat.tolist(),
            "distance": self.camera.distance,
            "azimuth": self.camera.azimuth,
            "elevation": self.camera.elevation,
            "prev_lookat": self.prev_camera_lookat.tolist(),
            "prev_distance": self.prev_camera_distance,
            "alpha": self.alpha,
        }

    def compute_camera_position(self, num_objects):
        obj_positions = []
        for i in range(num_objects):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free"
            )
            if joint_id != -1:
                obj_positions.append(self.data.qpos[joint_id : joint_id + 3])

        if not obj_positions:
            return np.array([0, 0, 2]), 3.0

        obj_positions = np.array(obj_positions)

        # Center of mass
        lookat = np.mean(obj_positions, axis=0)
        lookat[2] = max(lookat[2], 0.5)

        # Spread-based zoom
        max_distance = np.max(np.linalg.norm(obj_positions - lookat, axis=1))
        distance = max(3.0, min(6.0, max_distance * 2))

        return lookat, distance

    def update_camera(self, num_objects, renderer, time_step=0.01, orbit_speed=0.4):
        """
        Cinematic orbit camera around all objects.

        time_step:   dt per frame (approx simulation step)
        orbit_speed: radians per second
        """
        obj_positions = []
        for i in range(num_objects):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free"
            )
            if joint_id != -1:
                obj_positions.append(self.data.qpos[joint_id : joint_id + 3])

        if obj_positions:
            obj_positions = np.array(obj_positions)
            center = np.mean(obj_positions, axis=0)
            center[2] = max(center[2], 0.4)
        else:
            center = np.array([0, 0, 0.5])

        # Smooth interpolation of lookat
        self.camera.lookat[:] = (
            self.alpha * center + (1 - self.alpha) * self.prev_camera_lookat
        )
        if not hasattr(self, "orbit_angle"):
            self.orbit_angle = 0.0

        self.orbit_angle += orbit_speed * time_step

        self.camera.azimuth = np.degrees(self.orbit_angle)
        self.camera.elevation = -15
        self.camera.distance = 1.5

        self.prev_camera_lookat = self.camera.lookat.copy()
        renderer.update_scene(self.data, camera=self.camera)

    def update_camera_1(self, num_objects, renderer, time_step=0.01, orbit_speed=0.4):
        # Step 1: Compute NEW desired targets
        new_lookat, new_distance = self.compute_camera_position(num_objects)

        # Step 2: Update *targets* smoothly (slow drift)
        beta = 0.2  # how quickly targets chase objects
        self.target_lookat = beta * new_lookat + (1 - beta) * self.target_lookat
        self.target_distance = beta * new_distance + (1 - beta) * self.target_distance

        # Step 3: Smooth camera towards the target
        self.camera.lookat[:] = (
            self.alpha * self.target_lookat + (1 - self.alpha) * self.prev_camera_lookat
        )
        self.camera.distance = (
            self.alpha * self.target_distance
            + (1 - self.alpha) * self.prev_camera_distance
        )

        # Step 4: Mild rotation for aesthetics
        self.camera.azimuth += 0.05
        self.camera.elevation = -15

        # Save for next frame
        self.prev_camera_lookat = self.camera.lookat.copy()
        self.prev_camera_distance = self.camera.distance

        renderer.update_scene(self.data, camera=self.camera)
