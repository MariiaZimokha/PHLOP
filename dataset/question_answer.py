import json
import random
import matplotlib.colors as mcolors
from typing import List, Dict
from collections import defaultdict
import re


class QuestionAnswers:
    def __init__(self, file_path: str, fps: int = 25):
        self.fps = fps
        self.data = self._load_json(file_path)
        self.frames = self.data.get("frames", [])
        self.objects = self.data.get("objects", [])
        self.props = self._get_physical_props(self.objects)
        self.stationary_re = re.compile(r"stationary", re.IGNORECASE)
        self.motion_re = re.compile(r"sliding|rolling|accelerating|decelerating", re.IGNORECASE)

    def _load_json(self, path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

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

    def _describe_obj(self, p):
        return f"{p.get('color', 'unknown color')} {p.get('shape', 'object')}"

    def _get_taxonomy_sequences(self) -> Dict[str, List[List[str]]]:
        taxonomy = defaultdict(list)
        for fr in self.frames:
            for obj_id, obj_state in fr.get("objects", {}).items():
                labels = []
                for tax in obj_state.get("taxonomy", []):
                    labels.extend(tax.get("labels", []))
                taxonomy[obj_id].append(labels)
        return taxonomy

    def _get_state_transitions(self, taxonomy) -> Dict[str, set]:
        transitions = {
            'stopped_objects': set(),
            'moving_to_stationary': set(),
            'stationary_to_moving': set(),
            'rolling': set(),
        }

        for obj_id, label_seq in taxonomy.items():
            prev = None
            for labels in label_seq:
                current = labels[-1].lower() if labels else ""
                if "rolling" in current:
                    transitions["rolling"].add(obj_id)
                if prev:
                    if any(s in prev for s in ["moving", "accelerating", "decelerating"]) and "stationary" in current:
                        transitions["moving_to_stationary"].add(obj_id)
                    if "stationary" in prev and "accelerating" in current:
                        transitions["stationary_to_moving"].add(obj_id)
                prev = current
            if any("stationary" in s.lower() for labels in label_seq for s in labels):
                transitions["stopped_objects"].add(obj_id)

        return transitions

    def _identify_collisions(self):
        collision_pairs = set()
        for frame in self.frames:
            for interaction in frame.get("interactions", []):
                ids = [f"geom_obj{int(i)-1}" for i in interaction]
                if len(ids) >= 2:
                    pair = tuple(sorted(ids[:2]))
                    collision_pairs.add(pair)
        return list(collision_pairs)

    def _get_most_collided_object(self):
        count = defaultdict(int)
        for frame in self.frames:
            objects = frame.get("objects", {})
            for interaction in frame.get("interactions", []):
                for i in interaction:
                    oid = f"geom_obj{int(i)-1}"
                    if oid in objects:
                        count[oid] += 1
        if not count:
            return []
        max_val = max(count.values())
        return [k for k, v in count.items() if v == max_val]

    def _get_kinematic_loss(self, collision_pairs, props, annotations):
        questions = []

        for obj1_id, obj2_id in collision_pairs:
            if obj1_id not in props or obj2_id not in props:
                continue

            p1, p2 = props[obj1_id], props[obj2_id]
            desc1 = self._describe_obj(p1)
            desc2 = self._describe_obj(p2)

            valid_frames = [
                (i, frame) for i, frame in enumerate(annotations['frames'])
                if obj1_id in frame['objects'] and obj2_id in frame['objects']
            ]
            if not valid_frames:
                continue

            peak_frame_idx = 0
            max_delta_v = 0
            for i, frame in valid_frames:
                v1 = frame['objects'][obj1_id].get('velocity', [0, 0, 0])
                v2 = frame['objects'][obj2_id].get('velocity', [0, 0, 0])
                delta_v = sum((v1[i] - v2[i]) ** 2 for i in range(3))
                if delta_v > max_delta_v:
                    max_delta_v = delta_v
                    peak_frame_idx = i

            pre_idx = max(0, peak_frame_idx - 1)
            post_idx = min(len(annotations['frames']) - 1, peak_frame_idx + 1)
            pre_frame = annotations['frames'][pre_idx]
            post_frame = annotations['frames'][post_idx]

            if obj1_id not in pre_frame['objects'] or obj2_id not in post_frame['objects']:
                continue

            v1_before = pre_frame['objects'][obj1_id].get('velocity', [0, 0, 0])
            v2_before = pre_frame['objects'][obj2_id].get('velocity', [0, 0, 0])
            v1_after = post_frame['objects'][obj1_id].get('velocity', [0, 0, 0])
            v2_after = post_frame['objects'][obj2_id].get('velocity', [0, 0, 0])

            def kinetic_energy(v, m):
                return 0.5 * m * sum(vi ** 2 for vi in v)

            ke1_before = kinetic_energy(v1_before, p1['mass'])
            ke1_after = kinetic_energy(v1_after, p1['mass'])
            ke2_before = kinetic_energy(v2_before, p2['mass'])
            ke2_after = kinetic_energy(v2_after, p2['mass'])

            total_ke_before = ke1_before + ke2_before
            total_ke_after = ke1_after + ke2_after
            percent_ke_loss = (
                100 * (total_ke_before - total_ke_after) / total_ke_before
                if total_ke_before > 0 else 0
            )

            contact_frames = []
            for i, frame in enumerate(annotations['frames']):
                for interaction in frame.get('interactions', []):
                    involved = {f"geom_obj{int(oid)-1}" for oid in interaction}
                    if {obj1_id, obj2_id}.issubset(involved):
                        contact_frames.append(i)
                        break

            if contact_frames:
                duration = (max(contact_frames) - min(contact_frames) + 1) / self.fps
            else:
                duration = 0.0

            def make_opts(true_val, is_percent=True):
                a = round(true_val * 0.8, 1 if is_percent else 2)
                b = round(min(true_val + (10 if is_percent else 0.2), 100), 1 if is_percent else 2)
                c = round(abs(true_val - (50 if is_percent else 0.5)), 1 if is_percent else 2)
                opts = list({true_val, a, b, c})
                random.shuffle(opts)
                suffix = "%" if is_percent else "s"
                return [f"{v:.1f}{suffix}" if is_percent else f"{v:.2f}{suffix}" for v in opts]

            questions.extend([
                {
                    "question": f"What percentage of the system’s kinetic energy was lost when the {desc1} collided with the {desc2}?",
                    "answer": f"{round(percent_ke_loss, 1):.1f}%",
                    "options": make_opts(round(percent_ke_loss, 1)),
                    "explanation": (
                        "For each object, kinetic energy KE = 0.5·m·|v|². "
                        "Compute before and after collision, sum them, then compute the percentage lost."
                    )
                },
                {
                    "question": f"How long did the collision between {desc1} and {desc2} last (video fps is {self.fps})?",
                    "answer": f"{round(duration, 2):.2f}s",
                    "options": make_opts(round(duration, 2), is_percent=False),
                    "explanation": (
                        "Count how many frames the two objects are interacting, then divide by the frame-rate."
                    )
                }
            ])

        return questions

    def get_questions_answers(self) -> List[Dict]:
        questions = []

        # Base QAs
        questions.append({
            "question": "How many distinct physical objects appear during the video?",
            "answer": str(len(self.props))
        })

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

        taxonomy = self._get_taxonomy_sequences()
        transitions = self._get_state_transitions(taxonomy)
        questions.append({
            "question": "How many objects come to a complete stop during the video that we can see?",
            "answer": str(len(transitions["stopped_objects"]))
        })
        questions.append({
            "question": "How many objects display rolling motion at any point in the video?",
            "answer": str(len(transitions["rolling"]))
        })

        collision_pairs = self._identify_collisions()
        has_collisions = bool(collision_pairs)
        questions.append({
            "question": "Are there any moments in the video where two or more objects collide or make physical contact?",
            "answer": "Yes" if has_collisions else "No"
        })
        if has_collisions:
            involved = set()
            for a, b in collision_pairs:
                involved.update([a, b])
            questions.append({
                "question": "How many unique objects were involved in collision throughout the video?",
                "answer": str(len(involved))
            })

            most_collided = self._get_most_collided_object()
            if most_collided:
                options = [self._describe_obj(self.props[o]) for o in self.props]
                answers = [self._describe_obj(self.props[o]) for o in most_collided if o in self.props]
                random.shuffle(options)
                questions.append({
                    "question": "Which object was involved in the most collisions with other objects?",
                    "options": options,
                    "answer": answers,
                    "multiple_answers": len(answers) > 1
                })

            # Append kinetic energy + duration questions
            questions.extend(self._get_kinematic_loss(collision_pairs, self.props, self.data))

        # Stationary duration & start time
        for obj_id, p in self.props.items():
            desc = self._describe_obj(p)
            count_stat = 0
            first_stat_frame = None
            for i, fr in enumerate(self.frames):
                obj = fr.get("objects", {}).get(obj_id)
                if not obj:
                    continue
                for tax in obj.get("taxonomy", []):
                    if any(self.stationary_re.search(lbl) for lbl in tax.get("labels", [])):
                        count_stat += 1
                        if first_stat_frame is None:
                            first_stat_frame = i

            if count_stat == 0:
                continue

            true_seconds = round(count_stat / self.fps, 2)
            dur_opts = list({true_seconds, true_seconds * 0.8, true_seconds * 1.2, abs(true_seconds - 1.0)})
            while len(dur_opts) < 4:
                dur_opts.append(round(true_seconds + random.uniform(0.1, 1.0), 2))
            opts = [f"{v:.2f}s" for v in dur_opts]
            random.shuffle(opts)

            questions.append({
                "question": f"How many seconds did the {desc} spend stationary?",
                "options": opts,
                "answer": f"{true_seconds:.2f}s",
                "explanation": f"Count the number of video frames labelled ‘stationary’ for this object, then divide by the frame-rate ({self.fps} fps)."
            })

            if first_stat_frame is not None:
                start_time = round(first_stat_frame / self.fps, 2)
                ts_opts = list({start_time, max(start_time - 0.4, 0), start_time + 0.4, abs(start_time - 1.0)})
                while len(ts_opts) < 4:
                    ts_opts.append(round(start_time + random.uniform(0.1, 1.0), 2))
                time_opts = [f"{v:.2f}s" for v in ts_opts]
                random.shuffle(time_opts)
                questions.append({
                    "question": f"At what time in the video does the {desc} first become stationary?",
                    "options": time_opts,
                    "answer": f"{start_time:.2f}s",
                    "explanation": f"Find the first frame where the object is labeled ‘stationary’, then divide the frame index by the frame-rate ({self.fps} fps)."
                })

        random.shuffle(questions)
        return questions
