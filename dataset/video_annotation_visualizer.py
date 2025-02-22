import cv2
import json
import numpy as np


class VideoAnnotationVisualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        pass

    def __read_json(self, path):
        with open(path, "r") as f:
            annotations = json.load(f)
        return annotations

    def annotate(self, file_path="", video_path="", annotated_video_path=""):
        if not video_path or not file_path or not annotated_video_path:
            return None
        cap = cv2.VideoCapture(video_path)
        annotations = self.__read_json(file_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(
            annotated_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index < len(annotations["frames"]):
                frame_data = annotations["frames"][frame_index]["objects"]

                for obj_id, obj_info in frame_data.items():
                    bbox = obj_info.get("bbox", [(0, 0), (0, 0)])
                    (x1, y1), (x2, y2) = bbox

                    obj_data = next(
                        (obj for obj in annotations["objects"] if obj["id"] == obj_id),
                        None,
                    )
                    if not obj_data:
                        continue

                    mass = obj_data["mass"]
                    velocity = obj_info["velocity"]
                    material = obj_data["material"]
                    color = obj_data["visual"]["rgba"]
                    taxonomies = obj_info.get("taxonomy", [])
                    taxonomy_labels = [t["labels"] for t in taxonomies]
                    # print(taxonomy_labels)
                    if taxonomy_labels:
                        taxonomy_labels = (
                            np.concatenate(taxonomy_labels).ravel().tolist()
                        )

                    # Convert RGBA to BGR color format for OpenCV
                    color_rgb = tuple([int(float(c) * 255) for c in color.split()[:3]])

                    # bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_rgb, 2)

                    # text color
                    text_color = (255, 255, 255)  # White text for visibility
                    bg_color = (0, 0, 0)  # Black background for contrast

                    #  description
                    text = f"Mass: {mass:.2f} kg\nVelocity: {[round(v, 2) for v in velocity]}\nMaterial: {material}\nTaxonomy: {','.join(taxonomy_labels)}"
                    y_offset = 20

                    for line in text.split("\n"):
                        text_size = cv2.getTextSize(line, self.font, 0.5, 1)[0]
                        text_x, text_y = x1, y1 - y_offset

                        # background
                        cv2.rectangle(
                            frame,
                            (text_x - 5, text_y - text_size[1] - 5),
                            (text_x + text_size[0] + 5, text_y + 5),
                            bg_color,
                            thickness=cv2.FILLED,
                        )

                        # text with shadow
                        cv2.putText(
                            frame,
                            line,
                            (text_x + 1, text_y + 1),
                            self.font,
                            0.5,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )  # Black shadow
                        cv2.putText(
                            frame,
                            line,
                            (text_x, text_y),
                            self.font,
                            0.5,
                            text_color,
                            1,
                            cv2.LINE_AA,
                        )  # White text

                        y_offset += 20
            out.write(frame)
            frame_index += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Annotated video - {annotated_video_path}")
