import random
import numpy as np
import mujoco


class CameraSettings:
    """
    Robust camera controller for MuJoCo renderer.

    Features:
      - Accepts explicit init: {'lookat':[x,y,z],'azimuth':deg,'elevation':deg,'distance':m}
      - If init values missing -> random initialization
      - Limit enforcement: az_range, el_range, dist_range
      - Follow strategies: "largest", "fastest", "closest", integer index, "none"
      - "random" follow picks one of the above per scene
      - Smooth interpolation via alpha/beta
      - update_camera(...) performs mjv_updateCamera + renderer.update_scene(...)
    """

    def __init__(
        self,
        model=None,
        data=None,
        camera_name="camera",
        evaluation_mode=False,
        follow_object="none",
    ):
        self.camera_name = camera_name
        self.evaluation_mode = evaluation_mode
        self.follow_object = (
            follow_object  # "random", "largest", "fastest", "closest", int index,
        )
        self.model = None
        self.data = None

        # default limits (wide), can be overridden via set_limits()
        self.az_range = (-180.0, 180.0)
        self.el_range = (-89.0, 89.0)
        self.dist_range = (0.5, 10.0)

        # smoothing hyperparams (tweakable)
        self.alpha = 0.15  # how fast camera moves toward target each frame
        self.beta = 0.25  # how fast target chases measured object
        self.orbit_angle = 0.0

        if model is not None and data is not None:
            self.set_model(model, data)

    def set_model(self, model=None, data=None):
        """Attach mujoco model & data and initialize an mjvCamera"""
        self.model = model
        self.data = data
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)

        self.cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if self.cam_id == -1:
            raise ValueError(f"Camera '{self.camera_name}' not found in model.")

    def set_limits(self, az_range=None, el_range=None, dist_range=None):
        """Set allowed ranges for azimuth, elevation and distance."""
        if az_range is not None:
            self.az_range = az_range
        if el_range is not None:
            self.el_range = el_range
        if dist_range is not None:
            self.dist_range = dist_range

    def set_init_settings(self, init_cfg=None):
        """
        Initialize camera parameters. init_cfg is a dict with optional keys:
          - lookat: [x,y,z]
          - azimuth: degrees
          - elevation: degrees
          - distance: meters

        Missing values are randomized within sensible ranges.
        """
        if init_cfg is None:
            init_cfg = {}

        # lookat
        lookat = init_cfg.get(
            "lookat",
            [
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(0.0, 0.6),
            ],
        )
        self.camera.lookat[:] = lookat

        # angle & distance
        self.camera.azimuth = float(
            init_cfg.get("azimuth", random.uniform(-45.0, 45.0))
        )
        self.camera.elevation = float(
            init_cfg.get("elevation", random.uniform(-10.0, 25.0))
        )
        self.camera.distance = float(init_cfg.get("distance", random.uniform(3.5, 5.5)))

        # Clip initial values to limits (if limits have been set)
        self.camera.azimuth = float(np.clip(self.camera.azimuth, *self.az_range))
        self.camera.elevation = float(np.clip(self.camera.elevation, *self.el_range))
        self.camera.distance = float(np.clip(self.camera.distance, *self.dist_range))

        # smoothing state
        self.prev_camera_lookat = self.camera.lookat.copy()
        self.prev_camera_distance = self.camera.distance
        self.target_lookat = self.camera.lookat.copy()
        self.target_distance = self.camera.distance

    def _gather_object_info(self, num_objects):
        """
        Collects arrays of object positions and linear speeds for objects present.
        Returns:
          positions: list of np.array([x,y,z])
          speeds: list of floats (linear speed)
          indices: list of integer object indexes corresponding to these entries
        """
        positions = []
        speeds = []
        indices = []
        for i in range(num_objects):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free"
            )
            if joint_id != -1:
                pos = np.array(self.data.qpos[joint_id : joint_id + 3])
                vel = np.array(self.data.qvel[joint_id : joint_id + 3])
                speed = float(np.linalg.norm(vel))
                positions.append(pos)
                speeds.append(speed)
                indices.append(i)
        return positions, speeds, indices

    def _select_follow_target(self, num_objects):
        """
        Decide the target 3D point to look at depending on follow_object setting.
        Supports:
          - int index -> follow that object if exists
          - 'largest' -> fallback to COM (requires object sizes; if unavailable, COM)
          - 'fastest' -> follow object with max speed
          - 'closest' -> follow object closest to current camera lookat
          - 'random' -> randomly pick one strategy from ['fastest','closest','none']
          - 'none' or None -> follow COM
        """
        positions, speeds, indices = self._gather_object_info(num_objects)

        if len(positions) == 0:
            return np.array([0.0, 0.0, 0.5])

        # explicit index
        if isinstance(self.follow_object, int):
            idx = min(self.follow_object, len(positions) - 1)
            return positions[idx]

        # dynamic pick
        strategy = self.follow_object
        if strategy == "random":
            strategy = random.choice(["fastest", "closest", "none"])

        if strategy == "fastest":
            # pick index of highest speed
            pick_idx = int(np.argmax(speeds))
            return positions[pick_idx]

        if strategy == "closest":
            # find the object closest to current camera lookat
            cam_look = np.array(self.camera.lookat)
            dists = [np.linalg.norm(p - cam_look) for p in positions]
            pick_idx = int(np.argmin(dists))
            return positions[pick_idx]

        if strategy == "none" or strategy is None:
            # center-of-mass
            com = np.mean(np.stack(positions, axis=0), axis=0)
            com[2] = max(com[2], 0.2)
            return com

        # 'largest' requires object size info from metadata; fallback to COM
        if strategy == "largest":
            # try to use model geom sizes if available
            try:
                geom_sizes = []
                geom_names = []
                for i in indices:
                    geom_name = f"geom_obj{i}"
                    gid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                    )
                    if gid != -1:
                        s = np.array(self.model.geom_size[gid])  # may be (1,) or (3,)
                        geom_sizes.append(float(np.prod(s)))
                        geom_names.append(i)
                if geom_sizes:
                    pick_idx_local = int(np.argmax(geom_sizes))
                    pick_obj_index = geom_names[pick_idx_local]
                    # find position index of that object
                    for jdx, idx in enumerate(indices):
                        if idx == pick_obj_index:
                            return positions[jdx]
            except Exception:
                pass
            # fallback
            com = np.mean(np.stack(positions, axis=0), axis=0)
            com[2] = max(com[2], 0.2)
            return com

        # default fallback: COM
        com = np.mean(np.stack(positions, axis=0), axis=0)
        com[2] = max(com[2], 0.2)
        return com

    def update_camera(
        self, num_objects, renderer, dt=1.0 / 25.0, orbit_speed=0.35, dynamic=True
    ):
        """
        Orbit-only camera that stays close to the scene.
        """

        if not dynamic:
            renderer.update_scene(self.data, camera=self.camera)
            return

        # --- 1) Orbit angle ---
        self.orbit_angle += orbit_speed * dt
        self.camera.azimuth = float(np.degrees(self.orbit_angle))

        # Clamp azimuth
        self.camera.azimuth = float(np.clip(self.camera.azimuth, *self.az_range))

        # --- 2) Fixed elevation (downward tilt) ---
        self.camera.elevation = float(np.clip(self.camera.elevation, *self.el_range))

        # --- 3) Distance stays near predefined range ---
        self.camera.distance = float(np.clip(self.camera.distance, *self.dist_range))

        # --- 4) Center of world (simple orbit) ---
        self.camera.lookat[:] = [0.0, 0.0, 0.5]  # center slightly above floor

        # --- 5) Save state ---
        self.prev_camera_distance = self.camera.distance
        self.prev_camera_lookat = self.camera.lookat.copy()

        # --- 6) Render with the camera object ---
        renderer.update_scene(self.data, camera=self.camera)

    def get_init_settings(self):
        return {
            "camera_name": self.camera_name,
            "lookat": list(self.camera.lookat),
            "azimuth": float(self.camera.azimuth),
            "elevation": float(self.camera.elevation),
            "distance": float(self.camera.distance),
            "az_range": tuple(self.az_range),
            "el_range": tuple(self.el_range),
            "dist_range": tuple(self.dist_range),
            "follow_object": self.follow_object,
            "alpha": self.alpha,
            "beta": self.beta,
        }
