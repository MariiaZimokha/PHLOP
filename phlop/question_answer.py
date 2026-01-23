import random
from typing import List, Dict
from collections import defaultdict
import re
from phlop.utils import (
    describe_object_unique,
    rgba_to_name,
    load_json,
    get_appeared_object_ids,
)


class QuestionAnswers:
    def __init__(self, file_path: str, fps: int = 25):
        self.fps = fps
        self.data = load_json(file_path)
        self.frames = self.data.get("frames", [])
        self.objects = self.data.get("objects", [])

        self.appeared_obj_ids = get_appeared_object_ids(self.frames)

        # Physical properties ONLY for appeared objects
        self.props = self._get_physical_props(self.objects)

        self.stationary_re = re.compile(r"stationary", re.IGNORECASE)
        self.motion_re = re.compile(
            r"sliding|rolling|accelerating|decelerating", re.IGNORECASE
        )

    def _get_physical_props(self, objects: List[Dict]) -> Dict:
        props = {}
        for obj in objects:
            obj_id = obj["id"]
            if obj_id not in self.appeared_obj_ids:
                continue

            rgba = obj.get("visual", {}).get("rgba", "")
            color = [float(x) for x in rgba.split()] if rgba else []

            # Parse friction as list
            friction_str = obj.get("friction", "0.4 0 0")
            friction = (
                [float(x) for x in friction_str.split()] if friction_str else [0.4]
            )

            props[obj_id] = {
                "mass": float(obj.get("mass", 1.0)),
                "friction": friction,
                "shape": obj.get("geom_type", "object"),
                "material": obj.get("material", "unknown"),
                "color": rgba_to_name(color),
            }
        return props

    def _describe_obj(self, p):
        return f"{p.get('color', 'unknown color')} {p.get('shape', 'object')}"

    def _describe_obj_unique(self, obj_id: str) -> str:
        """
        Generates a unique description using the shared utility function.
        """
        return describe_object_unique(
            target_id=obj_id,
            objects=self.objects,
            frames=self.frames,
            appeared_obj_ids=self.appeared_obj_ids,
            rgba_to_name_func=rgba_to_name,
        )

    def _get_taxonomy_sequences(self) -> Dict[str, List[List[str]]]:
        taxonomy = defaultdict(list)

        for frame in self.frames:
            for obj_id, obj_state in frame.get("objects", {}).items():
                if obj_id not in self.appeared_obj_ids:
                    continue

                bbox = obj_state.get("bbox", [[0, 0], [0, 0]])
                if bbox == [[0, 0], [0, 0]]:
                    taxonomy[obj_id].append([])
                    continue

                labels = []
                for tax in obj_state.get("taxonomy", []):
                    labels.extend(tax.get("labels", []))
                taxonomy[obj_id].append(labels)

        return taxonomy

    def _get_state_transitions(self, taxonomy) -> Dict[str, set]:
        transitions = {
            "stopped_objects": set(),
            "moving_to_stationary": set(),
            "stationary_to_moving": set(),
            "rolling": set(),
        }

        for obj_id, label_seq in taxonomy.items():
            prev_labels = []
            for current_labels in label_seq:
                curr_lower = [l.lower() for l in current_labels]
                prev_lower = [l.lower() for l in prev_labels]

                if any("rolling" in l for l in curr_lower):
                    transitions["rolling"].add(obj_id)

                if any("stationary" in l for l in curr_lower):
                    transitions["stopped_objects"].add(obj_id)

                if prev_lower and curr_lower:
                    is_prev_moving = any(
                        s in l
                        for l in prev_lower
                        for s in ["moving", "accelerating", "decelerating", "sliding"]
                    )
                    is_curr_stat = any("stationary" in l for l in curr_lower)

                    if is_prev_moving and is_curr_stat:
                        transitions["moving_to_stationary"].add(obj_id)

                    if any("stationary" in l for l in prev_lower) and any(
                        "accelerating" in l for l in curr_lower
                    ):
                        transitions["stationary_to_moving"].add(obj_id)

                prev_labels = current_labels

        return transitions

    def _identify_collisions(self):
        """
        Identify collisions from taxonomy (collision events) instead of interactions list.
        """
        collisions = []
        seen_collisions = set()  # Track (time, obj_pair) to avoid duplicates

        for frame in self.frames:
            frame_time = frame.get("time", 0)

            for obj_id, obj_state in frame.get("objects", {}).items():
                if obj_id not in self.appeared_obj_ids:
                    continue

                # Search taxonomy for collision events
                for tax in obj_state.get("taxonomy", []):
                    # Check if this is a collision event
                    if (
                        tax.get("category") == "Interaction Events"
                        and tax.get("subcategory") == "Collision"
                    ):
                        collision_type = tax.get("labels", [None])[0]
                        if not collision_type:
                            continue

                        # Extract collision data
                        energy_analysis = tax.get("energy_analysis")
                        context = tax.get("context", {})
                        momentum_check = tax.get("momentum_check", {})
                        other_obj_id = context.get("other_obj_id", "unknown")

                        collision_key = (round(frame_time, 3), obj_id, other_obj_id)

                        # Skip if already processed
                        if collision_key in seen_collisions:
                            continue
                        seen_collisions.add(collision_key)

                        obj_pair = None
                        if other_obj_id != "unknown":
                            obj_pair = tuple(sorted([obj_id, other_obj_id]))
                        else:
                            # fallback: try to find from interactions list
                            for interaction in frame.get("interactions", []):
                                if len(interaction) >= 2:
                                    g1, g2 = interaction[0], interaction[1]
                                    o1 = f"geom_obj{g1 - 1}"
                                    o2 = f"geom_obj{g2 - 1}"

                                    if obj_id in [o1, o2]:
                                        other = o2 if o1 == obj_id else o1
                                        if other in self.appeared_obj_ids:
                                            obj_pair = tuple(sorted([obj_id, other]))
                                            break
                        if not obj_pair:
                            continue
                        if any(oid not in self.appeared_obj_ids for oid in obj_pair):
                            continue

                        collisions.append(
                            {
                                "obj_ids": obj_pair,
                                "time": frame_time,
                                "labels": [collision_type],
                                "context": context,
                                "energy_analysis": energy_analysis,
                                "momentum_check": momentum_check,
                            }
                        )

        return collisions

    def _collision_questions(self, collisions):
        questions = []

        for col in collisions[:3]:
            o1, o2 = col["obj_ids"]
            if o1 not in self.props or o2 not in self.props:
                continue

            p1, p2 = self.props[o1], self.props[o2]
            desc1, desc2 = self._describe_obj_unique(o1), self._describe_obj_unique(o2)
            t = col["time"]

            momentum = col.get("momentum_check", {})

            if momentum:
                ratio = momentum.get("ratio", 1.0)
                conserved = momentum.get("conserved", True)

                questions.append(
                    {
                        "question": (
                            f"During the collision at t={t:.2f}s, "
                            f"the momentum ratio is {ratio:.2f}. "
                            f"What does this indicate?"
                        ),
                        "options": [
                            "Momentum is approximately conserved.",
                            "Momentum is not conserved.",
                            "Momentum always increases in collisions.",
                            "Momentum is irrelevant here.",
                        ],
                        "answer": (
                            "Momentum is approximately conserved."
                            if conserved
                            else "Momentum is not conserved."
                        ),
                        "answer_type": "multiple_choice",
                        "difficulty": "hard",
                        "category": "Collision Physics",
                        "question_type": "momentum_conservation",
                        "rationale": (
                            "Momentum conservation is evaluated numerically from velocities."
                        ),
                        "physics_signals": {
                            "momentum_ratio": ratio,
                            "conserved": conserved,
                        },
                    }
                )

        return questions

    def get_questions_answers(self) -> List[Dict]:
        questions = []

        # Object count (only appeared objects)
        questions.append(
            {
                "question": "How many distinct physical objects appear in the video?",
                "options": None,
                "answer": str(len(self.props)),
                "answer_type": "numerical",
                "difficulty": "easy",
                "category": "Counting",
                "question_type": "object_count",
                "rationale": "Only objects with visible bounding boxes are counted.",
                "physics_signals": {"num_objects": len(self.props)},
            }
        )

        taxonomy = self._get_taxonomy_sequences()
        transitions = self._get_state_transitions(taxonomy)

        rolling = [
            oid
            for oid, seq in taxonomy.items()
            if any("Rolling" in label for frame in seq for label in frame)
        ]

        questions.append(
            {
                "question": "How many objects exhibit rolling motion at any point?",
                "options": None,
                "answer": str(len(rolling)),
                "answer_type": "numerical",
                "difficulty": "easy",
                "category": "Motion Analysis",
                "question_type": "rolling_detection",
                "rationale": "Rolling motion is detected from physics-based taxonomy labels.",
                "physics_signals": {"rolling_objects": rolling},
            }
        )

        questions.append(
            {
                "question": "How many objects come to a complete stop during the video that we can see?",
                "options": None,
                "answer": str(len(transitions["stopped_objects"])),
                "answer_type": "numerical",
                "difficulty": "easy",
                "category": "Motion Analysis",
                "question_type": "stopped_objects_count",
                "rationale": "Objects that are labeled as stationary at any point are counted.",
                "physics_signals": {
                    "stopped_objects": list(transitions["stopped_objects"])
                },
            }
        )

        # Highest friction coefficient question
        if self.props:
            max_friction_obj = max(
                self.props.items(),
                key=lambda kv: kv[1]["friction"][0] if kv[1]["friction"] else 0.4,
            )[0]
            correct = self._describe_obj(self.props[max_friction_obj])
            options = [self._describe_obj(p) for p in self.props.values()]
            random.shuffle(options)
            questions.append(
                {
                    "question": "Which object had the highest friction coefficient?",
                    "options": options,
                    "answer": correct,
                    "answer_type": "multiple_choice",
                    "difficulty": "medium",
                    "category": "Physical Properties",
                    "question_type": "friction_comparison",
                    "rationale": "Friction coefficient is extracted from object properties.",
                    "physics_signals": {"max_friction_obj": max_friction_obj},
                }
            )

        collisions = self._identify_collisions()

        questions.append(
            {
                "question": "Are there any collisions between objects in the video?",
                "options": ["Yes", "No"],
                "answer": "Yes" if collisions else "No",
                "answer_type": "yes_no",
                "difficulty": "easy",
                "category": "Collision Detection",
                "question_type": "collision_presence",
                "rationale": "Collisions are detected from engine contact events.",
                "physics_signals": {"num_collisions": len(collisions)},
            }
        )

        if collisions:
            questions.extend(self._collision_questions(collisions))

        # Stationary duration & start time questions
        # Find objects visible in all frames
        fully_visible_objects = []
        for obj_id in self.props.keys():
            visible_in_all = all(
                obj_id in frame.get("objects", {}) for frame in self.frames
            )
            if visible_in_all:
                fully_visible_objects.append(obj_id)

        # Pick one object randomly if there are multiple
        if fully_visible_objects:
            selected_obj_id = random.choice(fully_visible_objects)
            p = self.props[selected_obj_id]
            desc = self._describe_obj_unique(selected_obj_id)

            count_stat = 0
            first_stat_frame = None
            for i, fr in enumerate(self.frames):
                obj = fr.get("objects", {}).get(selected_obj_id)
                if not obj:
                    continue
                for tax in obj.get("taxonomy", []):
                    if any(
                        self.stationary_re.search(lbl) for lbl in tax.get("labels", [])
                    ):
                        count_stat += 1
                        if first_stat_frame is None:
                            first_stat_frame = i

            if count_stat > 0:
                true_seconds = round(count_stat / self.fps, 2)
                dur_opts = list(
                    {
                        true_seconds,
                        round(true_seconds * 0.8, 2),
                        round(true_seconds * 1.2, 2),
                        round(abs(true_seconds - 1.0), 2),
                    }
                )
                while len(dur_opts) < 4:
                    dur_opts.append(round(true_seconds + random.uniform(0.1, 1.0), 2))
                opts = [f"{v:.2f}s" for v in dur_opts]
                random.shuffle(opts)

                questions.append(
                    {
                        "question": f"How many seconds did the {desc} spend stationary?",
                        "options": opts,
                        "answer": f"{true_seconds:.2f}s",
                        "answer_type": "multiple_choice",
                        "difficulty": "medium",
                        "category": "Motion Analysis",
                        "question_type": "stationary_duration",
                        "rationale": f"Count the number of video frames labelled 'stationary' for this object, then divide by the frame-rate ({self.fps} fps).",
                        "physics_signals": {
                            "stationary_frames": count_stat,
                            "fps": self.fps,
                        },
                    }
                )

                if first_stat_frame is not None:
                    start_time = round(first_stat_frame / self.fps, 2)
                    ts_opts = list(
                        {
                            start_time,
                            round(max(start_time - 0.4, 0), 2),
                            round(start_time + 0.4, 2),
                            round(abs(start_time - 1.0), 2),
                        }
                    )
                    while len(ts_opts) < 4:
                        ts_opts.append(round(start_time + random.uniform(0.1, 1.0), 2))
                    time_opts = [f"{v:.2f}s" for v in ts_opts]
                    random.shuffle(time_opts)
                    questions.append(
                        {
                            "question": f"At what time in the video does the {desc} first become stationary?",
                            "options": time_opts,
                            "answer": f"{start_time:.2f}s",
                            "answer_type": "multiple_choice",
                            "difficulty": "medium",
                            "category": "Motion Analysis",
                            "question_type": "stationary_start_time",
                            "rationale": f"Find the first frame where the object is labeled 'stationary', then divide the frame index by the frame-rate ({self.fps} fps).",
                            "physics_signals": {
                                "first_stationary_frame": first_stat_frame,
                                "fps": self.fps,
                            },
                        }
                    )

        # remove duplicates
        seen = set()
        unique_questions = []
        for q in questions:
            question_key = q.get("question", "")
            if question_key and question_key not in seen:
                seen.add(question_key)
                unique_questions.append(q)
        random.shuffle(unique_questions)
        return unique_questions
