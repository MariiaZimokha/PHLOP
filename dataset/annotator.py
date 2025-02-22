import numpy as np
import mujoco
import cv2


class Annotator:
    def __init__(self) -> None:
        pass

    def extract_polygons(self, seg_frame):
        seg_ids = seg_frame[:, :, 0]
        polygons_dict = {}
        unique_ids = np.unique(seg_ids)

        for geom_id in unique_ids:
            if geom_id == 0:  # skip floor
                continue

            # create a binary mask per object
            obj_mask = (seg_ids == geom_id).astype(np.uint8)

            # findContours requires 8-bit single-channel images
            contours, _ = cv2.findContours(
                obj_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # filter out very small contours to prevent noise
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
            # convert each contour to Nx2 coords
            simple_contours = [cnt.reshape(-1, 2) for cnt in filtered_contours]
            polygons_dict[geom_id] = simple_contours

        return polygons_dict

    def get_bbox(self, seg_polygons=None):
        if seg_polygons:
            # Ensure bounding boxes are created separately for each polygon (avoiding overlaps)
            bounding_boxes = [cv2.boundingRect(np.array(pg)) for pg in seg_polygons]

            # compute individual bbox per polygon
            bbox = []
            for x, y, w, h in bounding_boxes:
                bbox.append([[int(x), int(y)], [int(x + w), int(y + h)]])

            # select the bounding box - largest
            if bbox:
                bbox = max(
                    bbox, key=lambda b: (b[1][0] - b[0][0]) * (b[1][1] - b[0][1])
                )
                return bbox

        return [[0, 0], [0, 0]]

    def get_annotation(self, seg_frame, objects=[], data=None, model=None):
        STOP_LIN_SPEED_THRESHOLD = 0.05  # m/s
        STOP_ANG_SPEED_THRESHOLD = 0.05  # rad/s
        TIP_ANGLE_THRESHOLD_DEG = 45.0  # degrees

        polygons_dict = self.extract_polygons(seg_frame)

        frame_annotation = {"time": data.time, "objects": {}}
        for i, obj in enumerate(objects):
            object_id = obj.get("id", f"geom_{obj['geom_id']}")
            gid = obj["geom_id"]
            seg_polygons = polygons_dict.get(gid, [])
            bbox = self.get_bbox(seg_polygons)

            joint_name = f"obj{i}_free"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            adr = model.jnt_dofadr[joint_id]

            velocity = data.qvel[adr : adr + 3].tolist()
            angular_velocity = data.qvel[adr + 3 : adr + 6].tolist()
            position = data.qpos[adr : adr + 3].tolist()

            active_labels = []

            # Stopped detection
            speed_lin = np.linalg.norm(velocity)
            speed_ang = np.linalg.norm(angular_velocity)
            is_stopped = (
                speed_lin < STOP_LIN_SPEED_THRESHOLD
                and speed_ang < STOP_ANG_SPEED_THRESHOLD
            )
            if is_stopped:
                active_labels.append("Stopped")

            # Tilt detection (for a free joint, orientation is in qpos[adr+3:adr+7] as a quaternion)
            quat = data.qpos[adr + 3 : adr + 7]
            rot_mat = np.zeros((3, 3))
            mujoco.mju_quat2Mat(rot_mat.ravel(), quat)
            # local z-axis is rot_mat[:,2], dot with global z-axis => cos(tilt)
            dot_val = np.clip(np.dot(rot_mat[:, 2], [0, 0, 1]), -1.0, 1.0)
            tilt_deg = np.degrees(np.arccos(dot_val))
            is_tipped = tilt_deg > TIP_ANGLE_THRESHOLD_DEG
            if is_tipped:
                active_labels.append("Tipped")

            frame_annotation["objects"][object_id] = {
                "velocity": velocity,
                "angular_velocity": angular_velocity,
                "active_labels": active_labels,
                "position": position,
                "bbox": bbox,
                "segment_polygons": [contour.tolist() for contour in seg_polygons],
            }

        return frame_annotation
