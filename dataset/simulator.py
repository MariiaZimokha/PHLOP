import random
import numpy as np
import mujoco
import imageio
import json
import cv2

from dataset.utils import save_file, set_physics_properties, set_position_and_velocity
from dataset.camera import CameraSettings


class Simulation:
    def __init__(self, world_object, width=1440, height=1024, annotator=None):
        self.obj = world_object
        self.width, self.height = width, height
        self.modes = ["collision", "sliding", "stationary", "offset"]
        self.annotator = annotator
        self.camera_settings = CameraSettings()

        self.header = f"""
<mujoco model="dynamic_objects">
    <size nconmax="200" njmax="200"/>
    <option timestep="0.0005" gravity="0 0 -9.81"/>
    <visual>
        <global offwidth="{self.width}" offheight="{self.height}"/>
    </visual>"""

        # Give the floor group="0" so that it is excluded from the object segmentation mask.
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

        self.seg_color_map = {}

    def __get_mode(self, weights=(0.4, 0.2, 0.2, 0.2)):
        return random.choices(self.modes, weights=weights, k=1)[0]

    def __convert_segmentation_to_mask(self, seg_frame):
        # The red channel
        seg_ids = seg_frame[:, :, 0]

        mask = np.zeros((seg_frame.shape[0], seg_frame.shape[1], 3), dtype=np.uint8)
        unique_ids = np.unique(seg_ids)

        for geom_id in unique_ids:
            # If we haven't assigned a color yet, do so:
            if geom_id not in self.seg_color_map:
                if geom_id == 0:
                    self.seg_color_map[geom_id] = [0, 0, 0]  # floor
                else:
                    self.seg_color_map[geom_id] = np.random.randint(
                        0, 255, size=3
                    ).tolist()

            color = self.seg_color_map[geom_id]
            mask[seg_ids == geom_id] = color

        return mask

    def __get_world_objects(self, num_objects=3):
        objects = []
        for i in range(num_objects):
            # Retrieve random object params
            obj = self.obj.get_object()
            obj["mode"] = self.__get_mode()

            # Position and velocity initialization based on mode
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

        # data.ncon is the number of contacts at this moment
        for i in range(data.ncon):
            c = data.contact[i]
            # print("contact", c)
            g1 = c.geom1
            g2 = c.geom2
            pair = tuple(sorted((g1, g2)))
            colliding_pairs.add(pair)

        return colliding_pairs

    def run_simulation(
        self, num_objects=3, objects=[], duration=5.0, framerate=30, camera={}, path=""
    ):
        """
        camera:
            mode:
                0 - static
                1 - dynamic
        """
        cam = camera.get("mode", 0)
        cam_init = camera.get("init", {})

        if not len(objects):
            objects = self.__get_world_objects(num_objects)

        num_objects = len(objects)

        asset_defs, bodies_xml = self.__build_assets_and_bodies(objects)

        simulation_xml = (
            f"{self.header}"
            f"<asset>{asset_defs}</asset>"
            f"{self.world_body_start}"
            f"{bodies_xml}"
            f"{self.world_body_end}"
        )

        # Build and run the MuJoCo simulation
        model = mujoco.MjModel.from_xml_string(simulation_xml)
        data = mujoco.MjData(model)

        # camera_settings = CameraSettings(model, data)
        print(self.camera_settings)
        self.camera_settings.set_model(model, data)
        self.camera_settings.set_init_settings(cam_init)
        camera_init_config = self.camera_settings.get_init_settings()

        # set init velocity and map object for segmentation
        for i, obj in enumerate(objects):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free"
            )
            adr = model.jnt_dofadr[joint_id]
            data.qvel[adr : adr + 6] = [*obj["velocity"], *obj["angular_velocity"]]

            geom_name = f"geom_obj{i}"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            objects[i]["geom_id"] = geom_id
            objects[i]["id"] = f"geom_obj{i}"

        physics = PhysicsTaxonomy(objects)

        normal_frames = []
        segmentation_frames = []
        annotation_frames = []
        prev_time = data.time

        with mujoco.Renderer(model, self.height, self.width) as renderer:
            renderer.enable_segmentation_rendering()
            while data.time < duration:
                dt = data.time - prev_time
                prev_time = data.time

                mujoco.mj_step(model, data)

                if len(normal_frames) < data.time * framerate:
                    if cam == 1:
                        self.camera_settings.update_camera(num_objects, renderer)

                    # Render the normal frame (disable segmentation).
                    renderer.disable_segmentation_rendering()
                    renderer.update_scene(data, camera=self.camera_settings.camera)
                    # renderer.update_scene(data, camera="top")
                    normal_frame = renderer.render()
                    normal_frames.append(normal_frame)

                    # Render the segmentation frame (enable segmentation).
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(data, camera=self.camera_settings.camera)
                    # renderer.update_scene(data, camera="top")
                    seg_frame = renderer.render()

                    mask = self.__convert_segmentation_to_mask(seg_frame)
                    segmentation_frames.append(mask)

                    annotation = self.annotator.get_annotation(
                        seg_frame, objects, data, model
                    )
                    # print(annotation)
                    pairs = self.__detect_collisions(model, data)
                    events = physics.get_taxonomy(model, data, dt)
                    # print("taxonomy ",events)
                    # print(annotation["objects"].keys())
                    # print(events.keys())
                    # print(events)
                    annotation_all = {}
                    for object_id in set(annotation["objects"].keys()).union(
                        events.keys()
                    ):
                        # print(object_id)
                        # print('events ', events[object_id])
                        annotation_all[object_id] = {
                            **annotation["objects"].get(object_id, {}),
                            "taxonomy": events.get(object_id, {}),
                        }

                    annotation_frames.append(
                        {
                            "time": data.time,
                            "objects": annotation_all,
                            "interations": pairs,
                        }
                    )

        # Save the normal and segmentation videos.
        normal_video_filename = f"{path}simulation_{num_objects}_objects.mp4"
        segmentation_video_filename = (
            f"{path}simulation_{num_objects}_objects_segmentation.mp4"
        )

        imageio.mimsave(normal_video_filename, normal_frames, fps=framerate)
        print(f"Normal video saved as {normal_video_filename}")

        imageio.mimsave(segmentation_video_filename, segmentation_frames, fps=framerate)
        print(f"Segmentation video saved as {segmentation_video_filename}")

        data = {
            "camera": camera_init_config,
            "objects": objects,
            "frames": annotation_frames,
        }

        file_path = f"{path}obj_{num_objects}.json"
        save_file(file_path, data)
        return {
            "video_file": normal_video_filename,
            "segmentation_video_filename": segmentation_video_filename,
            "file_path": file_path,
        }
