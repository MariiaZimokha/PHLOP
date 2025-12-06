import json
import random
import matplotlib.colors as mcolors
from typing import List, Dict
from collections import defaultdict
import re
import numpy as np


class QuestionAnswers:
    def __init__(
        self, file_path: str, fps: int = 25, include_hard_questions: bool = True
    ):
        self.fps = fps
        self.data = self._load_json(file_path)
        self.frames = self.data.get("frames", [])
        self.objects = self.data.get("objects", [])
        self.props = self._get_physical_props(self.objects)
        self.stationary_re = re.compile(r"stationary", re.IGNORECASE)
        self.motion_re = re.compile(
            r"sliding|rolling|accelerating|decelerating", re.IGNORECASE
        )
        self.include_hard_questions = include_hard_questions

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
                "friction": obj.get("friction", "0.4 0 0"),
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
            "stopped_objects": set(),
            "moving_to_stationary": set(),
            "stationary_to_moving": set(),
            "rolling": set(),
            "spinning": set(),
        }

        for obj_id, label_seq in taxonomy.items():
            prev = None
            for labels in label_seq:
                current = labels[-1].lower() if labels else ""
                if "rolling" in current:
                    transitions["rolling"].add(obj_id)
                if "spinning" in current:
                    transitions["spinning"].add(obj_id)
                if prev:
                    if (
                        any(
                            s in prev
                            for s in ["moving", "accelerating", "decelerating"]
                        )
                        and "stationary" in current
                    ):
                        transitions["moving_to_stationary"].add(obj_id)
                    if "stationary" in prev and "accelerating" in current:
                        transitions["stationary_to_moving"].add(obj_id)
                prev = current
            if any("stationary" in s.lower() for labels in label_seq for s in labels):
                transitions["stopped_objects"].add(obj_id)

        return transitions

    def _identify_collisions(self):
        """Extract collision information with analysis."""
        collisions = []
        for frame in self.frames:
            for interaction in frame.get("interactions", []):
                if len(interaction) >= 2:
                    g1, g2 = interaction[0], interaction[1]
                    obj_ids = [f"geom_obj{g1 - 1}", f"geom_obj{g2 - 1}"]

                    # Extract collision analysis from taxonomy
                    for obj_id in obj_ids:
                        obj_data = frame["objects"].get(obj_id, {})
                        for tax in obj_data.get("taxonomy", []):
                            if "Collision" in tax.get("labels", [""])[0]:
                                collisions.append(
                                    {
                                        "obj_ids": tuple(sorted(obj_ids)),
                                        "time": frame.get("time", 0),
                                        "frame": frame.get("frame_index", 0),
                                        "collision_type": tax.get("labels", [""])[0],
                                        "context": tax.get("context", {}),
                                        "energy_analysis": tax.get("energy_analysis"),
                                        "momentum_check": tax.get("momentum_check"),
                                    }
                                )

        return collisions

    def _get_most_collided_object(self):
        count = defaultdict(int)
        for frame in self.frames:
            objects = frame.get("objects", {})
            for interaction in frame.get("interactions", []):
                for i in interaction:
                    oid = f"geom_obj{int(i) - 1}"
                    if oid in objects:
                        count[oid] += 1
        if not count:
            return []
        max_val = max(count.values())
        return [k for k, v in count.items() if v == max_val]

    def _get_collision_questions(self, collisions):
        """Generate questions from collision analysis data."""
        questions = []

        for collision in collisions[:3]:  # Limit to first 3 collisions
            obj_ids = collision["obj_ids"]
            if obj_ids[0] not in self.props or obj_ids[1] not in self.props:
                continue

            p1, p2 = self.props[obj_ids[0]], self.props[obj_ids[1]]
            desc1 = self._describe_obj(p1)
            desc2 = self._describe_obj(p2)
            collision_type = collision.get("collision_type", "Unknown")
            time_sec = collision.get("time", 0)

            # Q1: Collision type classification
            questions.append(
                {
                    "question": f"At t={time_sec:.2f}s, a collision occurred between a {desc1} and a {desc2}. The collision was classified as '{collision_type}'. What does this tell us about energy conservation?",
                    "answer": "Elastic collisions conserve kinetic energy; inelastic collisions dissipate some energy as heat/sound/deformation."
                    if collision_type == "Elastic Collision"
                    else "This collision dissipated energy to heat, sound, and deformation.",
                    "options": [
                        "Elastic collisions conserve kinetic energy; inelastic collisions dissipate some energy as heat/sound/deformation.",
                        "All collisions conserve energy equally.",
                        "Elastic and inelastic collisions are the same thing.",
                        "Energy is always lost in collisions.",
                    ]
                    if collision_type == "Elastic Collision"
                    else [
                        "This collision dissipated energy to heat, sound, and deformation.",
                        "No energy was lost in this collision.",
                        "This collision conserved all kinetic energy.",
                        "The collision type doesn't affect energy.",
                    ],
                    "answer_type": "multiple_choice",
                    "difficulty": "medium",
                    "category": "Collision Physics",
                }
            )

            # Q2: Momentum conservation
            momentum_check = collision.get("momentum_check", {})
            if momentum_check:
                is_conserved = momentum_check.get("conserved", True)
                ratio = momentum_check.get("ratio", 1.0)

                questions.append(
                    {
                        "question": f"During the collision at t={time_sec:.2f}s, momentum was {'conserved (ratio ≈ 1.0)' if is_conserved else f'not perfectly conserved (ratio = {ratio:.2f}). Why might this be?'}",
                        "answer": "Momentum is always conserved in isolated collisions (Newton's third law)."
                        if is_conserved
                        else "External forces (like friction from the floor) may reduce momentum.",
                        "options": [
                            "Momentum is always conserved in isolated collisions (Newton's third law).",
                            "Momentum is never conserved in collisions.",
                            "Only energy is conserved, not momentum.",
                            "Momentum conservation only applies to elastic collisions.",
                        ]
                        if is_conserved
                        else [
                            "External forces (like friction from the floor) may reduce momentum.",
                            "Momentum is never conserved.",
                            "The heavier object always loses momentum.",
                            "Momentum was violating the laws of physics.",
                        ],
                        "answer_type": "multiple_choice",
                        "difficulty": "medium",
                        "category": "Collision Physics",
                    }
                )

            # Q3: Energy transfer analysis
            energy_analysis = collision.get("energy_analysis")
            if energy_analysis:
                questions.append(
                    {
                        "question": f"The collision at t={time_sec:.2f}s showed '{energy_analysis}' energy behavior. What does this classification mean?",
                        "answer": self._energy_classification_explanation(
                            energy_analysis
                        ),
                        "options": [
                            self._energy_classification_explanation(energy_analysis),
                            "All objects conserve the same amount of energy in collisions.",
                            "Energy classification depends only on object color.",
                            "This classification is random and meaningless.",
                        ],
                        "answer_type": "multiple_choice",
                        "difficulty": "hard",
                        "category": "Energy Transfer",
                    }
                )

            # Q4: Relative velocity from context
            context = collision.get("context", {})
            if context:
                rel_vel = context.get("relative_velocity_magnitude", 0)
                is_head_on = context.get("is_head_on", False)

                questions.append(
                    {
                        "question": f"Before collision, the relative velocity between objects was {rel_vel:.2f} m/s. The collision was {'head-on' if is_head_on else 'not head-on'}. How would this affect impact severity?",
                        "answer": "Higher relative velocity means more severe impact and greater energy transfer."
                        if rel_vel > 1.0
                        else "Lower relative velocity means gentler impact with less energy transfer.",
                        "options": [
                            "Higher relative velocity means more severe impact and greater energy transfer.",
                            "Lower relative velocity means more severe impact.",
                            "Relative velocity doesn't affect impact severity.",
                            "Only object mass affects impact severity.",
                        ],
                        "answer_type": "multiple_choice",
                        "difficulty": "hard",
                        "category": "Collision Physics",
                    }
                )

        return questions

    def _energy_classification_explanation(self, classification):
        """Get explanation for energy classification."""
        explanations = {
            "Elastic (Energy Conserved)": "Kinetic energy is conserved; objects bounce apart with minimal energy loss.",
            "Partially Inelastic": "Some kinetic energy is lost, but objects don't stick together.",
            "Highly Inelastic": "Most kinetic energy is dissipated; objects may stick together or move slowly after collision.",
            "Negligible Initial Energy": "Objects had very low kinetic energy before collision.",
        }
        return explanations.get(classification, f"Energy behavior: {classification}")

    def get_questions_answers(self) -> List[Dict]:
        questions = []

        # ==================== BASIC QUESTIONS ====================
        questions.append(
            {
                "question": "How many distinct physical objects appear during the video?",
                "answer": str(len(self.props)),
                "answer_type": "numerical",
                "difficulty": "easy",
                "category": "Counting",
            }
        )

        if self.props:
            # Find object with max friction
            max_fr_obj = max(
                self.props.items(),
                key=lambda kv: float(
                    kv[1]["friction"].split()[0]
                    if isinstance(kv[1]["friction"], str)
                    else kv[1]["friction"]
                ),
            )[0]
            correct = self._describe_obj(self.props[max_fr_obj])
            options = [self._describe_obj(p) for p in self.props.values()]
            random.shuffle(options)
            questions.append(
                {
                    "question": "Which object had the highest friction coefficient?",
                    "options": options,
                    "answer": correct,
                    "answer_type": "multiple_choice",
                    "difficulty": "easy",
                    "category": "Physical Properties",
                }
            )

        taxonomy = self._get_taxonomy_sequences()
        transitions = self._get_state_transitions(taxonomy)

        questions.append(
            {
                "question": "How many objects come to a complete stop during the video?",
                "answer": str(len(transitions["stopped_objects"])),
                "answer_type": "numerical",
                "difficulty": "easy",
                "category": "State Tracking",
            }
        )

        questions.append(
            {
                "question": "How many objects display rolling motion at any point?",
                "answer": str(len(transitions["rolling"])),
                "answer_type": "numerical",
                "difficulty": "easy",
                "category": "Motion Detection",
            }
        )

        questions.append(
            {
                "question": "How many objects display spinning motion at any point?",
                "answer": str(len(transitions["spinning"])),
                "answer_type": "numerical",
                "difficulty": "easy",
                "category": "Motion Detection",
            }
        )

        # ==================== COLLISION QUESTIONS ====================
        collisions = self._identify_collisions()
        has_collisions = bool(collisions)

        questions.append(
            {
                "question": "Are there any collisions between objects in this video?",
                "answer": "Yes" if has_collisions else "No",
                "answer_type": "yes_no",
                "difficulty": "easy",
                "category": "Collision Detection",
            }
        )

        if has_collisions:
            # Add collision-specific questions
            questions.extend(self._get_collision_questions(collisions))

            # Unique objects in collisions
            involved = set()
            for collision in collisions:
                involved.update(collision["obj_ids"])

            questions.append(
                {
                    "question": "How many unique objects were involved in collisions?",
                    "answer": str(len(involved)),
                    "answer_type": "numerical",
                    "difficulty": "medium",
                    "category": "Collision Detection",
                }
            )

        # ==================== TEMPORAL QUESTIONS ====================
        for obj_id, p in self.props.items():
            desc = self._describe_obj(p)
            count_stat = 0
            first_stat_frame = None

            for i, fr in enumerate(self.frames):
                obj = fr.get("objects", {}).get(obj_id)
                if not obj:
                    continue
                for tax in obj.get("taxonomy", []):
                    if any(
                        self.stationary_re.search(lbl) for lbl in tax.get("labels", [])
                    ):
                        count_stat += 1
                        if first_stat_frame is None:
                            first_stat_frame = i

            if count_stat == 0:
                continue

            true_seconds = round(count_stat / self.fps, 2)
            dur_opts = [
                true_seconds,
                true_seconds * 0.8,
                true_seconds * 1.2,
                abs(true_seconds - 1.0),
            ]
            dur_opts = [round(v, 2) for v in dur_opts if v > 0]
            while len(dur_opts) < 4:
                dur_opts.append(round(true_seconds + random.uniform(0.1, 1.0), 2))
            opts = [f"{v:.2f}s" for v in dur_opts]
            random.shuffle(opts)

            questions.append(
                {
                    "question": f"How many seconds did the {desc} spend stationary?",
                    "options": opts,
                    "answer": f"{true_seconds:.2f}s",
                    "answer_type": "numerical",
                    "difficulty": "medium",
                    "category": "Temporal Reasoning",
                }
            )

            if first_stat_frame is not None:
                start_time = round(first_stat_frame / self.fps, 2)
                ts_opts = [
                    start_time,
                    max(start_time - 0.4, 0),
                    start_time + 0.4,
                    abs(start_time - 1.0),
                ]
                ts_opts = [round(v, 2) for v in ts_opts if v >= 0]
                while len(ts_opts) < 4:
                    ts_opts.append(round(start_time + random.uniform(0.1, 1.0), 2))
                time_opts = [f"{v:.2f}s" for v in ts_opts]
                random.shuffle(time_opts)

                questions.append(
                    {
                        "question": f"At what time does the {desc} first become stationary?",
                        "options": time_opts,
                        "answer": f"{start_time:.2f}s",
                        "answer_type": "numerical",
                        "difficulty": "medium",
                        "category": "Temporal Reasoning",
                    }
                )

        random.shuffle(questions)
        return questions
