import json
import random
import matplotlib.colors as mcolors
from typing import List, Dict


class QuestionAnswers:
    def __init__(self, file_path: str, fps: int = 25):
        self.fps = fps
        self.data = self._load_json(file_path)
        self.frames = self.data.get("frames", [])
        self.objects = self.data.get("objects", [])
        self.props = self._get_physical_props(self.objects)

    def _load_json(self, path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    def _get_physical_props(self, objects: List[Dict]) -> Dict:
        props = {}
        for obj in objects:
            rgba = obj.get("visual", {}).get("rgba", "")
            color = [float(x) for x in rgba.split()] if rgba else []
            color_name = self._rgba_to_name(color)
            props[obj["id"]] = {
                "mass": float(obj.get("mass", 1.0)),
                "friction": [float(x) for x in obj.get("friction", "").split()] if obj.get("friction") else [0.4],
                "shape": obj.get("geom_type", "object"),
                "material": obj.get("material", "unknown"),
                "color": color_name,
            }
        return props

    def _rgba_to_name(self, rgba):
        if not rgba or len(rgba) < 3:
            return "unknown color"
        rgb = tuple(rgba[:3])
        min_dist = float("inf")
        best_name = "unknown color"
        for name, hex_val in mcolors.CSS4_COLORS.items():
            named_rgb = mcolors.to_rgb(hex_val)
            dist = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, named_rgb))
            if dist < min_dist:
                min_dist = dist
                best_name = name
        return best_name.replace("grey", "gray").replace("gray", "grey")

    def _describe_obj(self, p):
        return f"{p.get('color', 'unknown color')} {p.get('shape', 'object')}"

    def get_questions_answers(self) -> List[Dict]:
        questions = []

        # Question: Number of objects
        questions.append({
            "question": "How many distinct physical objects appear during the video?",
            "answer": str(len(self.props))
        })

        # Question: Highest friction object
        if self.props:
            max_fr_obj = max(self.props.items(), key=lambda kv: kv[1]["friction"][0])[0]
            correct = self._describe_obj(self.props[max_fr_obj])
            options = [self._describe_obj(p) for p in self.props.values()]
            random.shuffle(options)
            questions.append({
                "question": "Which object had the highest friction coefficient?",
                "options": options,
                "answer": correct
            })

        # Per-object stationary duration and start time
        for obj_id, p in self.props.items():
            desc = self._describe_obj(p)
            count_stat = 0
            first_stat_frame = None
            for i, fr in enumerate(self.frames):
                obj = fr.get("objects", {}).get(obj_id)
                if not obj:
                    continue
                labels = []
                for tax in obj.get("taxonomy", []):
                    labels += tax.get("labels", [])
                if any("stationary" in l.lower() for l in labels):
                    count_stat += 1
                    if first_stat_frame is None:
                        first_stat_frame = i

            # Duration stationary
            true_seconds = round(count_stat / self.fps, 2)
            dur_cands = {
                round(true_seconds, 2),
                round(max(true_seconds * 0.8, 0), 2),
                round(true_seconds * 1.2, 2),
                round(abs(true_seconds - 1.0), 2)
            }
            while len(dur_cands) < 4:
                dur_cands.add(round(true_seconds + random.uniform(0.25, 1.0), 2))
            opts = [f"{v:.2f}s" for v in dur_cands]
            random.shuffle(opts)

            questions.append({
                "question": f"How many seconds did the {desc} spend stationary?",
                "options": opts,
                "answer": f"{true_seconds:.2f}s",
                "explanation": (
                    f"Count the number of video frames labelled ‘stationary’ for this object, "
                    f"then divide by the frame-rate ({self.fps} fps)."
                ),
                # "details": {
                #     "object_id": obj_id,
                #     "stationary_frames": count_stat,
                #     "fps": self.fps,
                #     "computed_duration": true_seconds
                # }
            })

            # Start time of stationary
            if first_stat_frame is not None:
                start_time = round(first_stat_frame / self.fps, 2)
                ts_cands = {
                    round(start_time, 2),
                    round(max(start_time - 0.4, 0), 2),
                    round(start_time + 0.4, 2),
                    round(abs(start_time - 1.0), 2)
                }
                while len(ts_cands) < 4:
                    ts_cands.add(round(start_time + random.uniform(0.25, 1.0), 2))
                time_opts = [f"{v:.2f}s" for v in ts_cands]
                random.shuffle(time_opts)

                questions.append({
                    "question": f"At what time in the video does the {desc} first become stationary?",
                    "options": time_opts,
                    "answer": f"{start_time:.2f}s",
                    "explanation": (
                        f"Find the first frame where the object is labeled ‘stationary’, "
                        f"then divide the frame index by the frame-rate ({self.fps} fps)."
                    ),
                    # "details": {
                    #     "object_id": obj_id,
                    #     "first_stationary_frame": first_stat_frame,
                    #     "fps": self.fps,
                    #     "computed_time": start_time
                    # }
                })

        random.shuffle(questions)
        return questions
