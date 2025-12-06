import random
from phlop.taxonomy import PhysicsTaxonomy
import numpy as np
import mujoco
import imageio

from phlop.utils import save_file, set_physics_properties, set_position_and_velocity
from phlop.world.camera import CameraSettings
from phlop.world.floor import Floor
from phlop.world.light import Light
from phlop.world.constants import MODES


class Simulation:
    def __init__(self, world_object, width=1920, height=1088, annotator=None):
        self.obj = world_object
        self.width, self.height = width, height
        self.modes = MODES
        self.annotator = annotator
        self.camera_settings = CameraSettings()
        self.floor = Floor()
        self.light = Light()
        self.seg_color_map = {}

        # Bidirectional mapping for object tracking
        self.geom_id_to_obj_id = {}  # geom_id (int) → obj["id"] (str)
        self.obj_id_to_geom_id = {}  # obj["id"] (str) → geom_id (int)

        # XML template
        self.header = f"""
<mujoco model="dynamic_objects">
    <size nconmax="200" njmax="200"/>
    <option timestep="0.002" gravity="0 0 -9.81" density="1.2" viscosity="0.0001"/>
    <visual>
        <global offwidth="{self.width}" offheight="{self.height}" />
        <quality shadowsize="2048" />
    </visual>"""

        self.world_body_start = """<worldbody>"""

        self.world_floor = """
            <geom name="floor" type="plane" 
              size="50 50 0.1" 
              pos="0 0 0" 
              rgba="{floor_rgba}"
              friction="{floor_friction}"
              group="0"
              material="floor_mat"/>
        """

        # Initial static camera definition
        self.world_body_end = """
        <camera name="camera" pos="-.1 -.1 0.1" xyaxes="0.78 -0.63 0 0.27 0.33 0.9"/>
    </worldbody>
</mujoco>"""

    def __get_mode(self, weights=(0.4, 0.2, 0.2, 0.2)):
        return random.choices(self.modes, weights=weights, k=1)[0]

    def __convert_segmentation_to_mask(self, seg_frame):
        """Convert segmentation IDs to a colored mask."""
        seg_ids = seg_frame[:, :, 0]  # Red channel contains segmentation IDs
        mask = np.zeros((seg_frame.shape[0], seg_frame.shape[1], 3), dtype=np.uint8)
        unique_ids = np.unique(seg_ids)

        for geom_id in unique_ids:
            if geom_id not in self.seg_color_map:
                self.seg_color_map[geom_id] = (
                    [0, 0, 0]
                    if geom_id == 0 # floor
                    else np.random.randint(0, 255, size=3).tolist()
                )
            mask[seg_ids == geom_id] = self.seg_color_map[geom_id]
        return mask

    def __get_world_objects(self, num_objects=3):
        objects = []
        for _ in range(num_objects):
            obj = self.obj.get_object()
            obj["mode"] = self.__get_mode()
            obj = set_position_and_velocity(obj)
            obj = set_physics_properties(obj)
            objects.append(obj)
        return objects

    def _build_objects_from_specs(self, specs):
        objects = []
        for spec in specs:
            obj = self.obj.get_object(
                shape=spec["shape"],
                material=spec["material"],
                density_idx=spec["density_idx"],
                friction_idx=spec["friction_idx"],
                elasticity_idx=spec["elasticity_idx"],
            )

            obj["mode"] = self.__get_mode()
            obj = set_position_and_velocity(obj)
            obj = set_physics_properties(obj)
            objects.append(obj)

        return objects

    def __build_assets_and_bodies(self, objects, floor, lights):
        asset_defs = []
        bodies_xml = []

        for i, obj in enumerate(objects):
            mat_name = f"mat_obj{i}"
            asset_defs.append(
                f'<material name="{mat_name}" specular="{obj["visual"]["specular"]}" '
                f'shininess="{obj["material_shininess"]}" rgba="{obj["visual"]["rgba"]}"/>'
            )
            bodies_xml.append(
                f"""
                <body name="obj{i}" pos="{obj["init_position_x"]:.4f} {obj["init_position_y"]:.4f} {obj["base_z"]:.4f}">
                    <freejoint name="obj{i}_free"/>
                    <geom name="geom_obj{i}" type="{obj["geom_type"]}" size="{obj["size_str"]}" mass="{obj["mass"]}"
                          friction="{obj["friction"]}" material="{mat_name}" group="1"/>
                </body>"""
            )

        asset_defs.append(
            f"""<material name="floor_mat" 
                    specular="{floor.get("specular", 0.2)}" 
                    shininess="{floor.get("shininess", 0.2)}"
                    rgba="{floor.get("rgba", "0.8 0.8 0.8 1.0")}"
                />"""
        )

        light_xml = ""
        for i, light in enumerate(lights):
            light_xml += f"""
            <light name="light{i}" pos="{light["pos"][0]} {light["pos"][1]} {light["pos"][2]}" 
                diffuse="{light["diffuse"]}" specular="{light["specular"]}" 
                cutoff="{light["cutoff"]}" directional="{str(light["directional"]).lower()}"/>
            """

        return "".join(asset_defs), "".join(bodies_xml), light_xml

    def __detect_collisions(self, model, data):
        colliding_pairs = set()
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 != 0 and g2 != 0:
                colliding_pairs.add(tuple(sorted((g1, g2))))
        return colliding_pairs

    def run_simulation(
        self,
        num_objects=3,
        objects=None,
        duration=5.0,
        framerate=25,
        camera=None,
        path="",
        floor=None,
        lights=None,
        object_specs=None,
    ):
        if object_specs is not None:
            objects = self._build_objects_from_specs(object_specs)
        elif objects is None:
            objects = self.__get_world_objects(num_objects)
        if camera is None:
            camera = {"mode": 0, "init": {}}
        if floor is None:
            floor = self.floor.get_settings()
        if lights is None:
            lights = self.light.get_settings(2)

        world_floor = self.world_floor.format(
            floor_rgba=floor["rgba"], floor_friction=floor["friction"]
        )

        asset_defs, bodies_xml, light_xml = self.__build_assets_and_bodies(
            objects, floor, lights
        )

        simulation_xml = (
            f"{self.header}<asset>{asset_defs}</asset>"
            f"{self.world_body_start}{light_xml}{world_floor}"
            f"{bodies_xml}{self.world_body_end}"
        )

        # Initialize MuJoCo
        model = mujoco.MjModel.from_xml_string(simulation_xml)
        data = mujoco.MjData(model)

        # --- CAMERA SETUP ---
        self.camera_settings.set_model(model, data)
        self.camera_settings.set_init_settings(camera.get("init", None))

        # Setup limits if they exist, otherwise sensible defaults
        init_cam = camera.get("init", {})
        az = init_cam.get("azimuth", self.camera_settings.camera.azimuth)
        el = init_cam.get("elevation", self.camera_settings.camera.elevation)
        dist = init_cam.get("distance", self.camera_settings.camera.distance)

        # Limits: keep camera from flipping or hitting floor
        # Elevation: -80 (top down) to -5 (near horizon)
        # Distance: 1.5m to 8.0m
        az_range = tuple(camera.get("limits", {}).get("az_range", (-180, 180)))
        el_range = tuple(camera.get("limits", {}).get("el_range", (-50, -20)))
        dist_range = tuple(camera.get("limits", {}).get("dist_range", (0.8, 1.6)))

        self.camera_settings.set_limits(
            az_range=az_range, el_range=el_range, dist_range=dist_range
        )

        if camera.get("mode", 0) == 1:
            self.camera_settings.follow_object = "orbit"
        else:
            self.camera_settings.follow_object = "none"

        # Initialize Velocities
        for i, obj in enumerate(objects):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free"
            )
            adr = model.jnt_dofadr[joint_id]
            data.qvel[adr : adr + 6] = [*obj["velocity"], *obj["angular_velocity"]]

            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_obj{i}")
            obj_id = obj.get("id", f"geom_obj{i}")

            # store bidirectional mapping
            self.geom_id_to_obj_id[geom_id] = obj_id
            self.obj_id_to_geom_id[obj_id] = geom_id

            objects[i]["geom_id"] = geom_id
            objects[i]["id"] = obj_id

        physics = PhysicsTaxonomy(objects)
        prev_frame_data = self._get_obj_data(model, data, objects)

        normal_frames = []
        segmentation_frames = []
        annotation_frames = []
        prev_time = data.time
        frame_count = 0  # Frame counter for timing

        # Prepare for rendering
        cam_mode = camera.get("mode", 0)
        is_moving_cam = cam_mode == 1

        with mujoco.Renderer(model, self.height, self.width) as renderer:
            renderer.enable_segmentation_rendering()

            while data.time < duration:
                mujoco.mj_step(model, data)
                if is_moving_cam:
                    self.camera_settings.update_camera(
                        len(objects),
                        renderer,
                        dt=model.opt.timestep,
                        orbit_speed=0.5,
                        dynamic=True,
                    )

                # --- Use frame counter instead of floating point comparison ---
                target_frame = int(data.time * framerate)
                if frame_count <= target_frame:
                    # 1. Decide which camera to use for rendering
                    render_cam = self.camera_settings.camera

                    # 2. Render RGB
                    renderer.disable_segmentation_rendering()
                    renderer.update_scene(data, camera=render_cam)
                    normal_frames.append(renderer.render())

                    # 3. Render Segmentation
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(data, camera=render_cam)
                    seg_frame = renderer.render()
                    segmentation_frames.append(
                        self.__convert_segmentation_to_mask(seg_frame)
                    )

                    # 4. Annotations & Physics
                    annotation = self.annotator.get_annotation(
                        seg_frame, objects, data, model
                    )
                    pairs = self.__detect_collisions(model, data)

                    current_time = data.time
                    dt_step = current_time - prev_time
                    prev_time = current_time

                    # Pass mapping to physics taxonomy
                    events = physics.get_taxonomy(
                        model,
                        data,
                        dt_step,
                        prev_frame_data,
                        annotation["objects"],
                        geom_id_to_obj_id=self.geom_id_to_obj_id,
                    )

                    annotation_all = {
                        obj_id: {
                            **annotation["objects"].get(obj_id, {}),
                            "taxonomy": events.get(obj_id, {}),
                            "prev_data": prev_frame_data.get(obj_id, {}),
                        }
                        for obj_id in set(annotation["objects"].keys()).union(
                            events.keys()
                        )
                    }

                    annotation_frames.append(
                        {
                            "time": current_time,
                            "frame_index": frame_count,
                            "objects": annotation_all,
                            "interactions": pairs,
                        }
                    )

                    # Update previous frame data
                    prev_frame_data = self._get_obj_data(model, data, objects)
                    frame_count += 1

        # Save outputs
        normal_video_filename = f"{path}simulation_objects.mp4"
        segmentation_video_filename = f"{path}simulation_objects_segmentation.mp4"
        imageio.mimsave(
            normal_video_filename, normal_frames, fps=framerate, codec="libx264"
        )
        imageio.mimsave(segmentation_video_filename, segmentation_frames, fps=framerate)

        data_export = {
            "world": {
                "floor": floor,
                "camera": self.camera_settings.get_init_settings(),
                "lights": lights,
            },
            "objects": objects,
            "frames": annotation_frames,
            "metadata": {
                "total_frames": frame_count,
                "duration": duration,
                "framerate": framerate,
                "geom_id_to_obj_id": {
                    str(k): v for k, v in self.geom_id_to_obj_id.items()
                },
            },
        }
        file_path = f"{path}meta.json"
        save_file(file_path, data_export)

        return {
            "video_file": normal_video_filename,
            "segmentation_video_filename": segmentation_video_filename,
            "file_path": file_path,
            "config": {
                "objects": objects,
                "floor": floor,
                "lights": lights,
                "camera_init": self.camera_settings.get_init_settings(),
            },
        }

    def _get_obj_data(self, model, data, objects):
        """Helper to extract object state."""
        out = {}
        for i, obj in enumerate(objects):
            obj_id = obj["id"]
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free"
            )
            adr = model.jnt_dofadr[joint_id]
            out[obj_id] = {
                "velocity": [float(x) for x in data.qvel[adr : adr + 3].tolist()],
                "angular_velocity": [
                    float(x) for x in data.qvel[adr + 3 : adr + 6].tolist()
                ],
                "position": data.qpos[adr : adr + 3].tolist(),
            }
        return out
