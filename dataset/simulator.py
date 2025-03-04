import random
from dataset.taxonomy import PhysicsTaxonomy
import numpy as np
import mujoco
import imageio
import json
import cv2

from dataset.utils import save_file, set_physics_properties, set_position_and_velocity
from dataset.camera import CameraSettings


class Simulation:
    def __init__(self, world_object, width=1920, height=1088, annotator=None):
        self.obj = world_object
        self.width, self.height = width, height
        self.modes = ["collision", "sliding", "stationary", "offset"]
        self.annotator = annotator
        self.camera_settings = CameraSettings()
        self.seg_color_map = {}

        # XML template for the MuJoCo simulation
        self.header = f"""
<mujoco model="dynamic_objects">
    <size nconmax="200" njmax="200"/>
    <option timestep="0.0005" gravity="0 0 -9.81"/>
    <visual>
        <global offwidth="{self.width}" offheight="{self.height}" />
    </visual>"""

        self.world_body_start = """
    <worldbody>
        <light name="light" pos="0 0 3"/>
        <geom name="floor" type="plane" 
              size="50 50 0.1" 
              pos="0 0 0" 
              rgba="1 1 1 1"
              friction="0.05 0.3 0.5" 
              group="0"
              material="floor_mat"/>
    """
        self.world_body_end = """
        <camera name="camera" pos="0 -2 1" xyaxes="0.8944 0 0 0 0.4472 0.8944"/>
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
                    [0, 0, 0] if geom_id == 0  # floor
                    else np.random.randint(0, 255, size=3).tolist()  # random color
                )
            mask[seg_ids == geom_id] = self.seg_color_map[geom_id]

        return mask

    def __get_world_objects(self, num_objects=3):
        """Generate a list of objects with random properties."""
        objects = []
        for _ in range(num_objects):
            obj = self.obj.get_object()
            obj["mode"] = self.__get_mode()
            obj = set_position_and_velocity(obj)
            obj = set_physics_properties(obj)
            objects.append(obj)
        return objects

    def __build_assets_and_bodies(self, objects):
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
                <body name="obj{i}" pos="{obj["init_possition_x"]:.4f} {obj["init_possition_y"]:.4f} {obj["base_z"]:.4f}">
                    <freejoint name="obj{i}_free"/>
                    <geom name="geom_obj{i}" type="{obj["geom_type"]}" size="{obj["size_str"]}" mass="{obj["mass"]}"
                          friction="{obj['friction']}" material="{mat_name}" group="1"/>
                </body>"""
            )

        asset_defs.append(
            """<material name="floor_mat" specular="0.0" shininess="0.0" rgba="0.8 0.8 0.8 1.0" />"""
        )
        return "".join(asset_defs), "".join(bodies_xml)

    def __detect_collisions(self, model, data):
        colliding_pairs = set()
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 != 0 and g2 != 0:  # skip floor collisions
                colliding_pairs.add(tuple(sorted((g1, g2))))
        return colliding_pairs

    def run_simulation(
        self, num_objects=3, objects=None, duration=5.0, framerate=25, camera=None, path=""
    ):
        """
        camera:
            mode:
                0 - static
                1 - dynamic
        """
        if objects is None:
            objects = self.__get_world_objects(num_objects)
        if camera is None:
            camera = {"mode": 0, "init": {}}

        num_objects = len(objects)
        asset_defs, bodies_xml = self.__build_assets_and_bodies(objects)

        simulation_xml = (
            f"{self.header}"
            f"<asset>{asset_defs}</asset>"
            f"{self.world_body_start}"
            f"{bodies_xml}"
            f"{self.world_body_end}"
        )

        # Initialize MuJoCo model and data
        model = mujoco.MjModel.from_xml_string(simulation_xml)
        data = mujoco.MjData(model)

        # Configure camera
        self.camera_settings.set_model(model, data)
        self.camera_settings.set_init_settings(camera.get("init", {}))
        camera_init_config = self.camera_settings.get_init_settings()

        # Initialize object velocities and IDs
        for i, obj in enumerate(objects):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            adr = model.jnt_dofadr[joint_id]
            data.qvel[adr: adr + 6] = [*obj["velocity"], *obj["angular_velocity"]]

            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_obj{i}")
            objects[i]["geom_id"] = geom_id
            objects[i]["id"] = f"geom_obj{i}"

        physics = PhysicsTaxonomy(objects)
        prev_frame_data = {
            obj["id"]: {
                "velocity": [float(x) for x in obj["velocity"]],
                "angular_velocity": [float(x) for x in obj["angular_velocity"]],
                "position": [obj["init_possition_x"], obj["init_possition_y"], obj["base_z"]],
            }
            for obj in objects
        }

        normal_frames = []
        segmentation_frames = []
        annotation_frames = []
        prev_time = data.time

        with mujoco.Renderer(model, self.height, self.width) as renderer:
            renderer.enable_segmentation_rendering()
            while data.time < duration:

                mujoco.mj_step(model, data)

                if len(normal_frames) < data.time * framerate:
                    if camera["mode"] == 1:
                        self.camera_settings.update_camera(num_objects, renderer)

                    # Render normal frame
                    renderer.disable_segmentation_rendering()
                    renderer.update_scene(data, camera=self.camera_settings.camera)
                    normal_frames.append(renderer.render())

                    # Render segmentation frame
                    renderer.enable_segmentation_rendering()
                    seg_frame = renderer.render()
                    segmentation_frames.append(self.__convert_segmentation_to_mask(seg_frame))

                    # Generate annotations
                    annotation = self.annotator.get_annotation(seg_frame, objects, data, model)
                    pairs = self.__detect_collisions(model, data)
                    current_time = data.time
                    dt = current_time - prev_time
                    # df = 1.0 / framerate
                    prev_time = current_time
                    events = physics.get_taxonomy(model, data, dt, prev_frame_data, annotation["objects"])

                    annotation_all = {
                        obj_id: {
                            **annotation["objects"].get(obj_id, {}),
                            "taxonomy": events.get(obj_id, {}),
                            "prev_data": prev_frame_data.get(obj_id, {}),
                        }
                        for obj_id in set(annotation["objects"].keys()).union(events.keys())
                    }

                    annotation_frames.append({
                        "time": current_time,
                        "objects": annotation_all,
                        "interactions": pairs,
                    })

                    # Update previous frame data
                    for i, obj in enumerate(objects):
                        obj_id = obj["id"]
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
                        adr = model.jnt_dofadr[joint_id]
                        prev_frame_data[obj_id] = {
                            "velocity": [float(x) for x in data.qvel[adr: adr + 3].tolist()],
                            "angular_velocity": [float(x) for x in data.qvel[adr + 3: adr + 6].tolist()],
                            "position": data.qpos[adr: adr + 3].tolist(),
                        }

        normal_video_filename = f"{path}simulation_objects.mp4"
        segmentation_video_filename = f"{path}simulation_objects_segmentation.mp4"
        imageio.mimsave(normal_video_filename, normal_frames, fps=framerate, codec='libx264')
        imageio.mimsave(segmentation_video_filename, segmentation_frames, fps=framerate)

        data = {
            "camera": camera_init_config,
            "objects": objects,
            "frames": annotation_frames,
        }
        file_path = f"{path}obj.json"
        save_file(file_path, data)

        return {
            "video_file": normal_video_filename,
            "segmentation_video_filename": segmentation_video_filename,
            "file_path": file_path,
        }
