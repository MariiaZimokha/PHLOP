import random
import numpy as np
import mujoco


class CameraSettings:
    def __init__(self, model=None, data=None, camera_name="camera", evaluation_mode=False):
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

    def set_init_settings(self, data=None):
        if data is None:
            data = {}

        if self.evaluation_mode:
            self.camera.lookat[:] = data.get("lookat", [random.uniform(-0.5, 0.5),
                                             random.uniform(-0.5, 0.5), random.uniform(0, 0.5)])
            self.camera.distance = data.get("distance", random.uniform(1.5, 4))
            self.camera.azimuth = data.get("azimuth", random.uniform(-30, 0))
            self.camera.elevation = data.get("elevation", random.uniform(-30, 0))

        else:
            # specifies the point in 3D space that the camera is focused on
            self.camera.lookat[:] = data.get("lookat", [random.uniform(-0.5, 0.5),
                                             random.uniform(-0.5, 0.5), random.uniform(0, 0.5)])
            # how far the camera is from the lookat point
            self.camera.distance = data.get("distance", random.uniform(1.5, 3.5))
            # horizontal angle of the camera relative to the lookat point
            self.camera.azimuth = data.get("azimuth", random.uniform(-90, -30))
            #  vertical angle of the camera relative to the lookat poin
            self.camera.elevation = data.get("elevation", random.uniform(-30, 0))

        self.prev_camera_lookat = self.camera.lookat
        self.prev_camera_distance = self.camera.distance
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
                obj_positions.append(self.data.qpos[joint_id: joint_id + 3])

        if not obj_positions:
            return np.array([0, 0, 2]), 3.0  # Default lookat and distance

        obj_positions = np.array(obj_positions)

        # Compute center of mass of objects
        camera_lookat = np.mean(obj_positions, axis=0)
        camera_lookat[2] = max(camera_lookat[2], 0.5)  # above ground

        # Compute spread of objects to adjust distance
        max_distance = np.max(np.linalg.norm(obj_positions - camera_lookat, axis=1))
        camera_distance = max(3.0, min(6.0, max_distance * 2))

        return camera_lookat, camera_distance

    def update_camera(self, num_objects, renderer):
        new_camera_lookat, new_camera_distance = self.compute_camera_position(
            num_objects
        )
        self.camera.lookat = new_camera_lookat
        self.camera.distance = new_camera_distance

        # Apply smoothing transition to avoid jitter
        self.camera.lookat[:] = (
            self.alpha * new_camera_lookat + (1 - self.alpha) * self.prev_camera_lookat
        )
        self.camera.distance = (
            self.alpha * new_camera_distance
            + (1 - self.alpha) * self.prev_camera_distance
        )

        # Rotate slightly for better tracking
        self.camera.azimuth += 0.05
        self.camera.elevation = -15

        # Store previous values for smooth interpolation
        self.prev_camera_lookat = self.camera.lookat.copy()
        self.prev_camera_distance = self.camera.distance
        renderer.update_scene(self.data, camera=self.camera)
