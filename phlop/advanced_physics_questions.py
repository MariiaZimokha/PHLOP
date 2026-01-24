import random
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from phlop.utils import (
    describe_object_unique,
    rgba_to_name,
    load_json,
    get_appeared_object_ids,
)


class AdvancedPhysicsQuestions:
    def __init__(self, file_path: str, fps: int = 25, split: str = "train"):
        self.data = load_json(file_path)
        self.frames = self.data.get("frames", [])
        self.objects = self.data.get("objects", [])
        self.fps = fps
        self.split = split  # "train", "val", or "test"
        self.appeared_obj_ids = get_appeared_object_ids(self.frames)

        # Cache frequently accessed data
        self._object_descriptions_cache = {}
        self._object_properties_cache = {}
        self._peak_velocities_cache = None

    def _describe_object_unique(self, obj_id: str) -> str:
        """Get unique object description with caching."""
        if obj_id not in self._object_descriptions_cache:
            self._object_descriptions_cache[obj_id] = describe_object_unique(
                target_id=obj_id,
                objects=self.objects,
                frames=self.frames,
                appeared_obj_ids=self.appeared_obj_ids,
                rgba_to_name_func=rgba_to_name,
            )
        return self._object_descriptions_cache[obj_id]

    def _get_object(self, obj_id: str) -> Optional[Dict]:
        return next((o for o in self.objects if o.get("id") == obj_id), None)

    def _describe_object(self, obj_id: str) -> str:
        """Get human-readable description of object."""
        obj = self._get_object(obj_id)
        if not obj:
            return obj_id

        shape = obj.get("geom_type", "object")

        # Try to get color from visual properties
        rgba_str = obj.get("visual", {}).get("rgba", "")
        if rgba_str:
            try:
                rgba = [float(x) for x in rgba_str.split()]
                color = rgba_to_name(rgba)
            except (ValueError, AttributeError):
                color = "unknown color"
        else:
            color = "unknown color"

        return f"{color} {shape} ({obj_id})"

    def _get_taxonomy(
        self, obj_data: Dict, category: str = None, subcategory: str = None
    ) -> List[Dict]:
        """Extract taxonomy entries, optionally filtered by category/subcategory."""
        out = []
        for tax in obj_data.get("taxonomy", []):
            if category and tax.get("category") != category:
                continue
            if subcategory and tax.get("subcategory") != subcategory:
                continue
            out.append(tax)
        return out

    def _should_mask_labels(self) -> bool:
        """Return True if taxonomy labels should be masked (for val/test splits)."""
        return self.split in ["val", "test"]

    def _get_velocity_from_frame(self, obj_id: str, frame_idx: int) -> np.ndarray:
        """Get velocity from frame data (for label masking)."""
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return np.array([0, 0, 0])
        frame = self.frames[frame_idx]
        obj_state = frame.get("objects", {}).get(obj_id, {})
        return np.array(obj_state.get("velocity", [0, 0, 0]))

    def _infer_motion_state_from_velocity(
        self, obj_id: str, frame_idx: int
    ) -> Optional[str]:
        """Infer motion state from velocity data (for label masking)."""
        if self._should_mask_labels():
            if frame_idx < 1 or frame_idx >= len(self.frames):
                return None
            prev_vel = self._get_velocity_from_frame(obj_id, frame_idx - 1)
            curr_vel = self._get_velocity_from_frame(obj_id, frame_idx)

            prev_speed = np.linalg.norm(prev_vel)
            curr_speed = np.linalg.norm(curr_vel)

            if prev_speed < 0.05 and curr_speed < 0.05:
                return "Stationary"
            elif curr_speed > prev_speed + 0.1:
                return "Accelerating"
            elif curr_speed < prev_speed - 0.1:
                return "Decelerating"
            elif abs(curr_speed - prev_speed) < 0.1:
                return "Constant Velocity"
        return None

    def _shuffle_options(self, options: List[str]) -> List[str]:
        shuffled = options.copy()
        random.shuffle(shuffled)
        return shuffled

    def _geom_id_to_obj_id(self, geom_id: int) -> str:
        """Convert geometry ID to object ID string."""
        return f"geom_obj{geom_id - 1}"

    def _get_velocity_from_obj_state(self, obj_state: Dict) -> np.ndarray:
        """Extract velocity vector from object state dictionary."""
        return np.array(obj_state.get("velocity", [0, 0, 0]))

    def _get_speed_from_velocity(self, velocity) -> float:
        """Calculate speed (magnitude) from velocity vector."""
        if isinstance(velocity, (list, tuple)):
            velocity = np.array(velocity)
        return np.linalg.norm(velocity)

    def _get_object_properties(self, obj_id: str) -> Optional[Dict]:
        """Get cached object properties (mass, friction, description)."""
        if obj_id not in self._object_properties_cache:
            obj = self._get_object(obj_id)
            if not obj or obj_id not in self.appeared_obj_ids:
                return None

            mass = float(obj.get("mass", 1.0))
            friction_str = obj.get("friction", "0.4")
            if isinstance(friction_str, str):
                friction = float(friction_str.split()[0])
            else:
                friction = float(friction_str)

            self._object_properties_cache[obj_id] = {
                "mass": mass,
                "friction": friction,
                "desc": self._describe_object_unique(obj_id),
            }
        return self._object_properties_cache[obj_id]

    def _get_peak_velocities(self) -> Dict[str, float]:
        """Calculate and cache peak velocities for all objects."""
        if self._peak_velocities_cache is None:
            self._peak_velocities_cache = {}
            for obj_id in self.appeared_obj_ids:
                max_speed = 0
                for frame in self.frames:
                    obj_state = frame.get("objects", {}).get(obj_id)
                    if obj_state:
                        vel = self._get_velocity_from_obj_state(obj_state)
                        speed = self._get_speed_from_velocity(vel)
                        max_speed = max(max_speed, speed)
                self._peak_velocities_cache[obj_id] = max_speed
        return self._peak_velocities_cache

    def _iter_collisions(self):
        """Generator that yields collision information (frame_idx, obj1_id, obj2_id, frame)."""
        for i in range(1, len(self.frames) - 1):
            cur_f = self.frames[i]
            if not cur_f.get("interactions"):
                continue

            for g1, g2 in cur_f["interactions"]:
                obj1_id = self._geom_id_to_obj_id(g1)
                obj2_id = self._geom_id_to_obj_id(g2)

                if (
                    obj1_id not in self.appeared_obj_ids
                    or obj2_id not in self.appeared_obj_ids
                ):
                    continue

                yield i, obj1_id, obj2_id, cur_f

    def _get_collision_velocities(
        self, frame_idx: int, obj1_id: str, obj2_id: str
    ) -> Optional[Dict]:
        """Get velocities before and after collision for both objects."""
        if frame_idx < 1 or frame_idx >= len(self.frames) - 1:
            return None

        prev_f = self.frames[frame_idx - 1]
        cur_f = self.frames[frame_idx]
        next_f = self.frames[frame_idx + 1]

        o1_prev = prev_f.get("objects", {}).get(obj1_id, {})
        o2_prev = prev_f.get("objects", {}).get(obj2_id, {})
        o1_next = next_f.get("objects", {}).get(obj1_id, {})
        o2_next = next_f.get("objects", {}).get(obj2_id, {})

        return {
            "v1_before": self._get_velocity_from_obj_state(o1_prev),
            "v2_before": self._get_velocity_from_obj_state(o2_prev),
            "v1_after": self._get_velocity_from_obj_state(o1_next),
            "v2_after": self._get_velocity_from_obj_state(o2_next),
        }

    def _infer_collision_type_from_energy_loss(self, ke_loss: float) -> tuple:
        """Infer collision type and answer from energy loss (matching physics_engine.py thresholds)."""
        if ke_loss < 10:  # ke_ratio > 0.9, energy loss < 10%
            return (
                "Elastic Collision",
                "Kinetic energy is conserved; objects bounce apart.",
            )
        elif ke_loss < 50:  # ke_ratio > 0.5, energy loss 10-50%
            return (
                "Partially Inelastic Collision",
                "Some kinetic energy is lost, but not all.",
            )
        else:  # ke_ratio <= 0.5, energy loss >= 50%
            return (
                "Highly Inelastic Collision",
                "Most kinetic energy is dissipated to heat and sound.",
            )

    def _calculate_kinetic_energy_loss(
        self, obj1_id: str, obj2_id: str, collision_frame_idx: int
    ) -> Optional[float]:
        """Calculate percentage of kinetic energy lost during collision."""
        if collision_frame_idx < 1 or collision_frame_idx >= len(self.frames) - 1:
            return None

        prev_frame = self.frames[collision_frame_idx - 1]
        post_frame = self.frames[collision_frame_idx + 1]

        obj1 = self._get_object(obj1_id)
        obj2 = self._get_object(obj2_id)

        if not obj1 or not obj2:
            return None

        m1 = float(obj1.get("mass", 1.0))
        m2 = float(obj2.get("mass", 1.0))

        v1_before = (
            prev_frame.get("objects", {}).get(obj1_id, {}).get("velocity", [0, 0, 0])
        )
        v2_before = (
            prev_frame.get("objects", {}).get(obj2_id, {}).get("velocity", [0, 0, 0])
        )
        v1_after = (
            post_frame.get("objects", {}).get(obj1_id, {}).get("velocity", [0, 0, 0])
        )
        v2_after = (
            post_frame.get("objects", {}).get(obj2_id, {}).get("velocity", [0, 0, 0])
        )

        def kinetic_energy(v, m):
            return 0.5 * m * sum(vi**2 for vi in v)

        ke1_before = kinetic_energy(v1_before, m1)
        ke2_before = kinetic_energy(v2_before, m2)
        ke1_after = kinetic_energy(v1_after, m1)
        ke2_after = kinetic_energy(v2_after, m2)

        total_ke_before = ke1_before + ke2_before
        total_ke_after = ke1_after + ke2_after

        if total_ke_before <= 0:
            return None

        percent_loss = 100 * (total_ke_before - total_ke_after) / total_ke_before
        return max(0, percent_loss)

    def generate_collision_geometry_questions(self) -> List[Dict]:
        """1.3 Collision Geometry & Impact - 1-2 questions"""
        questions = []

        for i, obj1, obj2, cur_f in self._iter_collisions():
            prev_f = self.frames[i - 1]
            o1_data = prev_f["objects"].get(obj1, {})
            o2_data = prev_f["objects"].get(obj2, {})

            v1 = self._get_velocity_from_obj_state(o1_data)
            v2 = self._get_velocity_from_obj_state(o2_data)
            rel_vel = v1 - v2
            rel_vel_mag = self._get_speed_from_velocity(rel_vel)

            if rel_vel_mag < 0.1:
                continue

            t = cur_f.get("time", 0)
            desc1 = self._describe_object_unique(obj1)
            desc2 = self._describe_object_unique(obj2)

            # Convert 30% of numeric questions to decision questions
            use_decision = random.random() < 0.3

            if use_decision:
                # Decision question: Is relative velocity high enough for highly inelastic collision?
                # Note: This threshold is a heuristic for high-energy collisions.
                # physics_engine.py uses energy ratio (ke_ratio <= 0.5) to classify highly inelastic collisions,
                # not relative velocity. This threshold is used for decision-making questions only.
                threshold = 1.5  # m/s - heuristic threshold for high-energy collision decision questions
                is_high_energy = rel_vel_mag > threshold

                options = [
                    "Yes, the relative velocity is high enough to cause a highly inelastic collision.",
                    "No, the relative velocity is too low for a highly inelastic collision.",
                    "Relative velocity doesn't affect collision type.",
                    "Cannot determine from the data provided.",
                ]
                shuffled = self._shuffle_options(options)

                answer = (
                    "Yes, the relative velocity is high enough to cause a highly inelastic collision."
                    if is_high_energy
                    else "No, the relative velocity is too low for a highly inelastic collision."
                )

                questions.append(
                    {
                        "question": (
                            f"At t={t:.2f}s, was the relative velocity between {desc1} and {desc2} "
                            f"high enough (above {threshold} m/s) to cause a highly inelastic collision?"
                        ),
                        "options": shuffled,
                        "answer": answer,
                        "answer_type": "multiple_choice",
                        "difficulty": "medium",
                        "category": "Collision Geometry",
                        "question_type": "relative_velocity_decision",
                        "rationale": f"Relative velocity magnitude is {round(rel_vel_mag, 2)} m/s. Threshold for high-energy collision is {threshold} m/s.",
                        "physics_signals": {
                            "relative_velocity": round(rel_vel_mag, 2),
                            "threshold": threshold,
                        },
                    }
                )
            else:
                # Original numeric question
                options = [
                    f"{round(rel_vel_mag, 2):.2f} m/s",
                    f"{round(rel_vel_mag * 0.7, 2):.2f} m/s",
                    f"{round(rel_vel_mag * 1.3, 2):.2f} m/s",
                    f"{round(rel_vel_mag * 0.5, 2):.2f} m/s",
                ]
                shuffled = self._shuffle_options(options)

                questions.append(
                    {
                        "question": (
                            f"At t={t:.2f}s, what was the relative velocity magnitude "
                            f"between {desc1} and {desc2} just before collision?"
                        ),
                        "options": shuffled,
                        "answer": f"{round(rel_vel_mag, 2):.2f} m/s",
                        "answer_type": "multiple_choice",
                        "difficulty": "medium",
                        "category": "Collision Geometry",
                        "question_type": "relative_velocity_magnitude",
                        "rationale": "Relative velocity is calculated as |v1 - v2|.",
                        "physics_signals": {"relative_velocity": round(rel_vel_mag, 2)},
                    }
                )

            if len(questions) >= 1:
                return questions

        return questions

    def generate_post_collision_motion_questions(self) -> List[Dict]:
        """1.5 Post-Collision Motion - 1-2 questions"""
        questions = []

        for i, obj1, obj2, cur_f in self._iter_collisions():
            velocities = self._get_collision_velocities(i, obj1, obj2)
            if not velocities:
                continue

            v1_before = velocities["v1_before"]
            v1_after = velocities["v1_after"]

            speed1_before = self._get_speed_from_velocity(v1_before)
            speed1_after = self._get_speed_from_velocity(v1_after)

            t = cur_f.get("time", 0)
            desc1 = self._describe_object_unique(obj1)
            desc2 = self._describe_object_unique(obj2)

            # Q: Direction reversal
            dot_product = np.dot(v1_before, v1_after)
            reversed = (
                "Yes"
                if dot_product < 0 and speed1_before > 0.1 and speed1_after > 0.1
                else "No"
            )

            questions.append(
                {
                    "question": (
                        f"At t={t:.2f}s, after the collision, did {desc1} "
                        f"reverse its direction of motion?"
                    ),
                    "options": ["Yes", "No"],
                    "answer": reversed,
                    "answer_type": "yes_no",
                    "difficulty": "medium",
                    "category": "Post-Collision Motion",
                    "question_type": "direction_reversal",
                    "rationale": (
                        "Direction reversal occurs when the velocity vectors point in opposite directions "
                        "(negative dot product)."
                    ),
                    "physics_signals": {
                        "speed_before": round(speed1_before, 2),
                        "speed_after": round(speed1_after, 2),
                    },
                }
            )

            if len(questions) >= 1:
                return questions

        return questions

    def generate_mass_effects_questions(self) -> List[Dict]:
        """Mass Effects - 1-2 questions"""
        questions = []

        obj_masses = {}
        for obj in self.objects:
            obj_id = obj["id"]
            if obj_id in self.appeared_obj_ids:
                obj_masses[obj_id] = float(obj.get("mass", 1.0))

        if len(obj_masses) < 2:
            return questions

        sorted_objs = sorted(obj_masses.items(), key=lambda x: x[1], reverse=True)
        heaviest_id, heaviest_mass = sorted_objs[0]
        lightest_id, lightest_mass = sorted_objs[-1]

        desc_heavy = self._describe_object_unique(heaviest_id)
        desc_light = self._describe_object_unique(lightest_id)

        # Q: Mass ratio
        mass_ratio = round(heaviest_mass / lightest_mass, 2)
        options = [
            f"{mass_ratio:.2f}",
            f"{round(mass_ratio * 0.8, 2):.2f}",
            f"{round(mass_ratio * 1.2, 2):.2f}",
            f"{round(mass_ratio * 0.5, 2):.2f}",
        ]
        shuffled = self._shuffle_options(options)

        questions.append(
            {
                "question": (
                    f"What is the mass ratio between {desc_heavy} (heaviest) "
                    f"and {desc_light} (lightest)?"
                ),
                "options": shuffled,
                "answer": f"{mass_ratio:.2f}",
                "answer_type": "multiple_choice",
                "difficulty": "medium",
                "category": "Mass & Density",
                "question_type": "mass_ratio",
                "rationale": "Mass ratio = heaviest_mass / lightest_mass.",
                "physics_signals": {
                    "heaviest_mass": round(heaviest_mass, 2),
                    "lightest_mass": round(lightest_mass, 2),
                    "mass_ratio": mass_ratio,
                },
            }
        )

        return questions

    def generate_friction_coefficient_questions(self) -> List[Dict]:
        """Friction Coefficient - 1-2 questions"""
        questions = []

        friction_data = {}
        for obj_id in self.appeared_obj_ids:
            props = self._get_object_properties(obj_id)
            if props:
                friction_data[obj_id] = props["friction"]

        if len(friction_data) < 2:
            return questions

        sorted_friction = sorted(
            friction_data.items(), key=lambda x: x[1], reverse=True
        )
        highest_id, highest_friction = sorted_friction[0]
        lowest_id, lowest_friction = sorted_friction[-1]

        desc_high = self._describe_object_unique(highest_id)
        desc_low = self._describe_object_unique(lowest_id)

        # Q: Which material has higher friction (and difference)
        diff = round(highest_friction - lowest_friction, 3)
        options = [
            f"{desc_high} (difference: {diff:.3f})",
            f"{desc_low} (difference: {diff:.3f})",
            f"{desc_high} (difference: {round(diff * 0.5, 3):.3f})",
            "They have equal friction coefficients",
        ]
        shuffled = self._shuffle_options(options)

        questions.append(
            {
                "question": (
                    f"Between {desc_high} and {desc_low}, which has a higher "
                    f"friction coefficient, and by approximately how much?"
                ),
                "options": shuffled,
                "answer": f"{desc_high} (difference: {diff:.3f})",
                "answer_type": "multiple_choice",
                "difficulty": "medium",
                "category": "Material Properties",
                "question_type": "friction_coefficient_comparison",
                "rationale": "Friction coefficient is extracted from object properties.",
                "physics_signals": {
                    "highest_friction": round(highest_friction, 3),
                    "lowest_friction": round(lowest_friction, 3),
                    "difference": diff,
                },
            }
        )

        return questions

    def generate_shape_distribution_questions(self) -> List[Dict]:
        """Object Shapes - 1-2 questions"""
        questions = []

        shape_count = defaultdict(int)
        for obj in self.objects:
            obj_id = obj["id"]
            if obj_id in self.appeared_obj_ids:
                shape = obj.get("geom_type", "unknown")
                shape_count[shape] += 1

        if not shape_count:
            return questions

        # Q: Shape distribution
        shapes_list = ", ".join(
            [f"{count} {shape}(s)" for shape, count in sorted(shape_count.items())]
        )
        options = [
            shapes_list,
            f"{len(self.appeared_obj_ids)} mixed shapes",
            f"Only {max(shape_count, key=shape_count.get)}'s",
            "Insufficient data",
        ]
        shuffled = self._shuffle_options(options)

        questions.append(
            {
                "question": (
                    "What is the distribution of object shapes in the simulation?"
                ),
                "options": shuffled,
                "answer": shapes_list,
                "answer_type": "multiple_choice",
                "difficulty": "easy",
                "category": "Geometry & Shape",
                "question_type": "shape_distribution",
                "rationale": "Count the geometric types of all appeared objects.",
                "physics_signals": {"shape_distribution": dict(shape_count)},
            }
        )

        return questions

    def generate_velocity_comparison_questions(self) -> List[Dict]:
        """Object Comparisons - 1-2 questions"""
        questions = []

        if len(self.appeared_obj_ids) < 2:
            return questions

        # Use cached peak velocities
        peak_velocities = self._get_peak_velocities()
        sorted_objs = sorted(peak_velocities.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_objs) < 2:
            return questions

        fastest_id, fastest_speed = sorted_objs[0]
        second_id, second_speed = sorted_objs[1]

        desc_fastest = self._describe_object_unique(fastest_id)
        desc_second = self._describe_object_unique(second_id)

        # Q: Fastest object
        options = [
            f"{desc_fastest} ({round(fastest_speed, 2):.2f} m/s)",
            f"{desc_second} ({round(second_speed, 2):.2f} m/s)",
            "They moved at equal speeds",
            "Cannot determine from data",
        ]
        shuffled = self._shuffle_options(options)

        questions.append(
            {
                "question": (
                    "Which object reached the highest peak velocity during the simulation?"
                ),
                "options": shuffled,
                "answer": f"{desc_fastest} ({round(fastest_speed, 2):.2f} m/s)",
                "answer_type": "multiple_choice",
                "difficulty": "medium",
                "category": "Comparative Questions",
                "question_type": "fastest_object",
                "rationale": "Peak velocity is the maximum speed achieved during motion.",
                "physics_signals": {
                    "fastest_object": fastest_id,
                    "fastest_speed": round(fastest_speed, 2),
                },
            }
        )

        return questions

    def generate_velocity_scaling_counterfactual_questions(self) -> List[Dict]:
        """Velocity Counterfactuals - 1-2 questions"""
        questions = []

        for obj_id in self.appeared_obj_ids:
            # Find sliding event with high initial velocity
            for i, frame in enumerate(self.frames):
                obj_state = frame.get("objects", {}).get(obj_id)
                if not obj_state:
                    continue

                vel = self._get_velocity_from_obj_state(obj_state)
                speed = self._get_speed_from_velocity(vel)

                if speed > 1.0:  # Significant speed
                    # Check if object eventually stops
                    total_distance = 0
                    stopped_frame = None
                    for j in range(i, min(i + 30, len(self.frames))):
                        future_frame = self.frames[j]
                        future_state = future_frame.get("objects", {}).get(obj_id)
                        if future_state:
                            future_vel = self._get_velocity_from_obj_state(future_state)
                            future_speed = self._get_speed_from_velocity(future_vel)
                            if future_speed < 0.05:
                                stopped_frame = j
                                break

                    if stopped_frame and stopped_frame > i:
                        desc = self._describe_object_unique(obj_id)
                        t = frame.get("time", 0)

                        # Q: Velocity scaling (quadratic relationship)
                        options = [
                            "4 times farther (quadratic)",
                            "2 times farther (linear)",
                            "Same distance (independent)",
                            "Half the distance",
                        ]
                        shuffled = self._shuffle_options(options)

                        questions.append(
                            {
                                "question": (
                                    f"At t={t:.2f}s, {desc} has velocity {speed:.2f} m/s. "
                                    f"If the initial velocity doubled, how much farther would it slide "
                                    f"(assuming same friction)?"
                                ),
                                "options": shuffled,
                                "answer": "4 times farther (quadratic)",
                                "answer_type": "multiple_choice",
                                "difficulty": "hard",
                                "category": "Counterfactual Reasoning",
                                "question_type": "velocity_scaling",
                                "rationale": (
                                    "Stopping distance d = v^2/(2*μ*g). If v doubles, d becomes (2v)^2/(2*μ*g) = 4v^2/(2*μ*g) = 4d."
                                ),
                                "physics_signals": {
                                    "initial_velocity": round(speed, 2),
                                    "scaling_factor": 4,
                                },
                            }
                        )

                        if len(questions) >= 1:
                            return questions

        return questions

    def generate_physics_principle_questions(self) -> List[Dict]:
        """Physics Principles - 1-2 questions"""
        questions = []

        # Q: Newton's Second Law
        questions.append(
            {
                "question": (
                    "According to Newton's Second Law (F = m*a), if an object's mass doubles "
                    "but the applied force remains the same, what happens to its acceleration?"
                ),
                "options": [
                    "Acceleration halves",
                    "Acceleration doubles",
                    "Acceleration stays the same",
                    "Acceleration becomes zero",
                ],
                "answer": "Acceleration halves",
                "answer_type": "multiple_choice",
                "difficulty": "medium",
                "category": "Conceptual Physics",
                "question_type": "newtons_second_law",
                "rationale": "From F = m*a, if m doubles and F stays constant, a = F/(2m) = (1/2)*(F/m).",
                "physics_signals": {"principle": "newtons_second_law"},
            }
        )

        return questions

    def generate_temporal_sequence_questions(self) -> List[Dict]:
        questions = []

        if not self.frames or len(self.appeared_obj_ids) < 1:
            return questions

        # Track events: collisions, motion starts, motion stops
        events = []
        seen_descriptions = set()  # Track unique event descriptions to avoid duplicates

        for i, frame in enumerate(self.frames):
            t = frame.get("time", 0)

            # Collision events
            if frame.get("interactions"):
                for g1, g2 in frame["interactions"]:
                    obj1 = self._geom_id_to_obj_id(g1)
                    obj2 = self._geom_id_to_obj_id(g2)
                    if obj1 in self.appeared_obj_ids and obj2 in self.appeared_obj_ids:
                        description = f"{self._describe_object_unique(obj1)} collides with {self._describe_object_unique(obj2)}"
                        # Only add if we haven't seen this exact description before
                        if description not in seen_descriptions:
                            seen_descriptions.add(description)
                            events.append(
                                {
                                    "time": t,
                                    "type": "collision",
                                    "obj1": obj1,
                                    "obj2": obj2,
                                    "description": description,
                                }
                            )

        if len(events) < 2:
            return questions

        # Sort events by time
        events.sort(key=lambda e: e["time"])

        # Get unique event descriptions (up to 3)
        event_descriptions = [e["description"] for e in events[:3]]

        # Ensure we have at least 2 unique events
        if len(event_descriptions) < 2:
            return questions

        # Create shuffled version for the question
        shuffled_events = event_descriptions.copy()
        random.shuffle(shuffled_events)

        # Generate distinct options
        correct_order = ", ".join(event_descriptions)
        reversed_order = ", ".join(reversed(event_descriptions))

        # Generate alternative orders that are different from correct and reversed
        options = [correct_order, reversed_order]

        # If we have 3 events, create more permutations
        if len(event_descriptions) == 3:
            # Create alternative permutations (middle, first, last) and (last, first, middle)
            alt1 = ", ".join(
                [event_descriptions[1], event_descriptions[0], event_descriptions[2]]
            )
            alt2 = ", ".join(
                [event_descriptions[2], event_descriptions[0], event_descriptions[1]]
            )
            options.extend([alt1, alt2])
        elif len(event_descriptions) == 2:
            # For 2 events, we only have 2 possible orders, so add generic alternatives
            options.append("The events occurred simultaneously")
            options.append("Cannot determine from data")

        # Remove duplicates and ensure we have exactly 4 options
        unique_options = []
        for opt in options:
            if opt not in unique_options:
                unique_options.append(opt)

        # If we still don't have 4 unique options, add generic ones
        while len(unique_options) < 4:
            if "Cannot determine from data" not in unique_options:
                unique_options.append("Cannot determine from data")
            elif "The events occurred simultaneously" not in unique_options:
                unique_options.append("The events occurred simultaneously")
            else:
                # Create a random permutation that's different from existing ones
                random_order = event_descriptions.copy()
                random.shuffle(random_order)
                new_option = ", ".join(random_order)
                if new_option not in unique_options:
                    unique_options.append(new_option)
                else:
                    # If we can't create a unique option, just add a generic one
                    unique_options.append("Insufficient information to determine order")
                    break

        # Shuffle options but keep track of the correct answer
        shuffled_options = unique_options.copy()
        random.shuffle(shuffled_options)

        questions.append(
            {
                "question": (
                    f"What is the correct chronological order of the following events: "
                    f"{', '.join(shuffled_events)}"
                ),
                "options": shuffled_options,
                "answer": correct_order,
                "answer_type": "multiple_choice",
                "difficulty": "medium",
                "category": "Temporal Reasoning",
                "question_type": "event_sequence",
                "rationale": "Events are ordered by their timestamps in the simulation.",
                "physics_signals": {"num_events": len(events)},
            }
        )

        return questions

    def generate_causal_questions(self) -> List[Dict]:
        """
        Direct cause-effect questions grounded in collision taxonomy.
        """
        questions = []
        seen_collisions = set()

        for i in range(1, len(self.frames) - 1):
            cur_f = self.frames[i]
            if not cur_f.get("interactions"):
                continue

            t = cur_f.get("time", 0)

            # Iterate over interactions in the current frame
            for g1, g2 in cur_f["interactions"]:
                obj1 = self._geom_id_to_obj_id(g1)
                obj2 = self._geom_id_to_obj_id(g2)

                # Deduplication key
                collision_key = (round(t, 3), tuple(sorted([obj1, obj2])))
                if collision_key in seen_collisions:
                    continue
                seen_collisions.add(collision_key)

                if (
                    obj1 not in self.appeared_obj_ids
                    or obj2 not in self.appeared_obj_ids
                ):
                    continue

                # get collision taxonomy data specificallyeven
                o1_data = cur_f["objects"].get(obj1, {})
                col_tax = self._get_taxonomy(o1_data, "Interaction Events", "Collision")

                if not col_tax:
                    continue

                collision_data = col_tax[0]
                collision_type = collision_data.get("labels", [None])[0]

                energy_analysis_data = collision_data.get("energy_analysis", {})

                desc1 = self._describe_object_unique(obj1)
                desc2 = self._describe_object_unique(obj2)

                # Generate Question based on the label directly
                if collision_type:
                    answer = ""
                    options = []

                    if "Elastic" in collision_type:
                        answer = "Kinetic energy is conserved; objects bounce apart."
                        options = [
                            "Kinetic energy is conserved; objects bounce apart.",
                            "All energy is lost to heat and sound.",
                            "Energy increases during the collision.",
                            "Energy conservation doesn't apply.",
                        ]
                    elif "Highly Inelastic" in collision_type:
                        answer = "Most kinetic energy is dissipated."
                        options = [
                            "Most kinetic energy is dissipated.",
                            "Kinetic energy is completely conserved.",
                            "Objects bounce perfectly.",
                            "Energy increases.",
                        ]
                    else:
                        # Partially Inelastic / Default
                        answer = "Some kinetic energy is lost."
                        options = [
                            "Some kinetic energy is lost.",
                            "Kinetic energy is completely conserved.",
                            "All energy is lost.",
                            "Energy increases.",
                        ]

                    shuffled_options = self._shuffle_options(options)

                    questions.append(
                        {
                            "question": (
                                f"At t={t:.2f}s, {desc1} collides with {desc2}. "
                                f"The collision is classified as '{collision_type}'. "
                                f"What does this imply about the system's energy?"
                            ),
                            "options": shuffled_options,
                            "answer": answer,
                            "answer_type": "multiple_choice",
                            "difficulty": "hard",
                            "category": "Causal Reasoning",
                            "question_type": "energy_analysis_taxonomy",
                            "rationale": f"Based on physics engine classification: {collision_type}",
                            "physics_signals": {
                                "collision_type": collision_type,
                                "ground_truth_data": energy_analysis_data,
                            },
                        }
                    )

        return questions

    def generate_counterfactual_questions(self) -> List[Dict]:
        """
        Counterfactual reasoning grounded in friction taxonomy.
        """
        questions = []

        for obj in self.objects:
            obj_id = obj["id"]

            # Only process objects that appeared in frames
            if obj_id not in self.appeared_obj_ids:
                continue

            slide_start = None

            for frame in self.frames:
                obj_state = frame["objects"].get(obj_id)
                if not obj_state:
                    continue

                friction_tax = self._get_taxonomy(
                    obj_state, "Environmental Interactions", "Friction"
                )

                labels = [l for t in friction_tax for l in t.get("labels", [])]

                if "Sliding with Friction" in labels and slide_start is None:
                    slide_start = frame["time"]

                if "Friction Stop" in labels and slide_start is not None:
                    stop_time = frame["time"]
                    duration = stop_time - slide_start

                    if duration > 0.1:
                        # Get object description (e.g., "red ball", "blue cube")
                        obj_desc = self._describe_object_unique(obj_id)

                        options = [
                            "It would stop in roughly half the time.",
                            "It would slide for the same duration.",
                            "It would slide longer.",
                            "It would never stop.",
                        ]
                        shuffled_options = self._shuffle_options(options)

                        questions.append(
                            {
                                "question": (
                                    f"A {obj_desc} slides for {duration:.2f}s before stopping. "
                                    f"If the friction coefficient were doubled, "
                                    f"what would most likely happen?"
                                ),
                                "options": shuffled_options,
                                "answer": "It would stop in roughly half the time.",
                                "answer_type": "multiple_choice",
                                "difficulty": "hard",
                                "category": "Counterfactual Reasoning",
                                "question_type": "friction_scaling",
                                "rationale": (
                                    "Stopping time under kinetic friction scales inversely "
                                    "with the friction coefficient: t = v / (μ * g)"
                                ),
                                "physics_signals": {
                                    "slide_duration": duration,
                                    "friction_event": True,
                                },
                            }
                        )

                    slide_start = None

        return questions

    def generate_property_competition_questions(self) -> List[Dict]:
        """
        Property competition questions: pit conflicting properties against each other.
        Example: "Despite being lighter, why did X travel farther?"
        """
        questions = []

        if len(self.appeared_obj_ids) < 2:
            return questions

        # Collect object properties (using cache)
        obj_properties = {}
        for obj_id in self.appeared_obj_ids:
            props = self._get_object_properties(obj_id)
            if props:
                obj_properties[obj_id] = props

        if len(obj_properties) < 2:
            return questions

        # Find collisions and track post-collision distances
        for i in range(1, len(self.frames) - 1):
            cur_f = self.frames[i]
            if not cur_f.get("interactions"):
                continue

            for g1, g2 in cur_f["interactions"]:
                obj1_id = self._geom_id_to_obj_id(g1)
                obj2_id = self._geom_id_to_obj_id(g2)

                if obj1_id not in obj_properties or obj2_id not in obj_properties:
                    continue

                # Calculate post-collision distances traveled
                post_collision_dist1 = 0
                post_collision_dist2 = 0

                # Track positions after collision
                if i + 1 < len(self.frames):
                    pos1_start = np.array(
                        cur_f["objects"].get(obj1_id, {}).get("position", [0, 0, 0])
                    )
                    pos2_start = np.array(
                        cur_f["objects"].get(obj2_id, {}).get("position", [0, 0, 0])
                    )

                    # Track for next 20 frames
                    for j in range(i + 1, min(i + 21, len(self.frames))):
                        frame = self.frames[j]
                        pos1_curr = np.array(
                            frame["objects"].get(obj1_id, {}).get("position", [0, 0, 0])
                        )
                        pos2_curr = np.array(
                            frame["objects"].get(obj2_id, {}).get("position", [0, 0, 0])
                        )

                        dist1 = np.linalg.norm(pos1_curr - pos1_start)
                        dist2 = np.linalg.norm(pos2_curr - pos2_start)
                        post_collision_dist1 = max(post_collision_dist1, dist1)
                        post_collision_dist2 = max(post_collision_dist2, dist2)

                props1 = obj_properties[obj1_id]
                props2 = obj_properties[obj2_id]

                # Find property competition scenarios
                # Scenario 1: Lighter object travels farther despite lower mass
                if (
                    props1["mass"] < props2["mass"]
                    and post_collision_dist1 > post_collision_dist2 * 1.2
                ):
                    # Check if friction or other properties compensate
                    if props1["friction"] < props2["friction"]:
                        options = [
                            f"Lower friction on {props1['desc']} allowed it to travel farther despite lower mass.",
                            f"Higher mass on {props2['desc']} caused it to stop sooner.",
                            "Measurement error in the simulation.",
                            "Violation of physics laws.",
                        ]
                        shuffled = self._shuffle_options(options)

                        questions.append(
                            {
                                "question": (
                                    f"After colliding, {props1['desc']} (mass: {props1['mass']:.2f} kg, friction: {props1['friction']:.3f}) "
                                    f"traveled farther than {props2['desc']} (mass: {props2['mass']:.2f} kg, friction: {props2['friction']:.3f}), "
                                    f"despite being lighter. Why did this happen?"
                                ),
                                "options": shuffled,
                                "answer": f"Lower friction on {props1['desc']} allowed it to travel farther despite lower mass.",
                                "answer_type": "multiple_choice",
                                "difficulty": "very_hard",
                                "category": "Property Competition",
                                "question_type": "property_competition",
                                "rationale": (
                                    "Multiple properties interact: lower friction compensates for lower mass, "
                                    "allowing the lighter object to travel farther."
                                ),
                                "physics_signals": {
                                    "obj1_mass": props1["mass"],
                                    "obj1_friction": props1["friction"],
                                    "obj2_mass": props2["mass"],
                                    "obj2_friction": props2["friction"],
                                    "obj1_distance": round(post_collision_dist1, 2),
                                    "obj2_distance": round(post_collision_dist2, 2),
                                },
                            }
                        )
                        return questions

                # Scenario 2: Higher friction but travels farther (unusual case)
                elif (
                    props1["friction"] > props2["friction"]
                    and post_collision_dist1 > post_collision_dist2 * 1.2
                ):
                    # This could happen if mass difference is significant
                    if props1["mass"] > props2["mass"] * 1.5:
                        options = [
                            f"Higher mass on {props1['desc']} provided more momentum, overcoming higher friction.",
                            f"Lower friction on {props2['desc']} caused it to stop sooner.",
                            "Measurement error in the simulation.",
                            "Friction doesn't affect post-collision distance.",
                        ]
                        shuffled = self._shuffle_options(options)

                        questions.append(
                            {
                                "question": (
                                    f"After colliding, {props1['desc']} (mass: {props1['mass']:.2f} kg, friction: {props1['friction']:.3f}) "
                                    f"traveled farther than {props2['desc']} (mass: {props2['mass']:.2f} kg, friction: {props2['friction']:.3f}), "
                                    f"despite having higher friction. Why did this happen?"
                                ),
                                "options": shuffled,
                                "answer": f"Higher mass on {props1['desc']} provided more momentum, overcoming higher friction.",
                                "answer_type": "multiple_choice",
                                "difficulty": "very_hard",
                                "category": "Property Competition",
                                "question_type": "property_competition",
                                "rationale": (
                                    "Higher mass provides more momentum (p = mv), which can overcome "
                                    "the retarding effect of higher friction."
                                ),
                                "physics_signals": {
                                    "obj1_mass": props1["mass"],
                                    "obj1_friction": props1["friction"],
                                    "obj2_mass": props2["mass"],
                                    "obj2_friction": props2["friction"],
                                    "obj1_distance": round(post_collision_dist1, 2),
                                    "obj2_distance": round(post_collision_dist2, 2),
                                },
                            }
                        )
                        return questions

        return questions

    def generate_contradictory_questions(self) -> List[Dict]:
        """
        Conceptual contradiction checks derived from kinematic taxonomy.
        Expanded to include temporal consistency and label vs observation mismatches.
        """
        questions = []
        found_contradictions = {
            "stationary_spinning": False,
            "rolling_slipping": False,
            "temporal_consistency": False,
            "label_observation_mismatch": False,
        }

        for frame in self.frames:
            t = frame.get("time", 0)

            for obj_id, obj_state in frame.get("objects", {}).items():
                # Only process objects that appeared in frames
                if obj_id not in self.appeared_obj_ids:
                    continue

                labels = [
                    l
                    for tax in obj_state.get("taxonomy", [])
                    for l in tax.get("labels", [])
                ]

                # Contradiction 1: Stationary + Pure Rotation
                if not found_contradictions["stationary_spinning"]:
                    if "Stationary" in labels and "Pure Rotation" in labels:
                        found_contradictions["stationary_spinning"] = True

                        obj_desc = self._describe_object_unique(obj_id)

                        options = [
                            "Yes, an object can spin in place.",
                            "No, stationary means no motion at all.",
                            "Only in zero gravity.",
                            "Only for deformable objects.",
                        ]
                        shuffled_options = self._shuffle_options(options)

                        questions.append(
                            {
                                "question": (
                                    f"At t={t:.2f}s, a {obj_desc} is stationary but rotating. "
                                    f"Is this physically possible?"
                                ),
                                "options": shuffled_options,
                                "answer": "Yes, an object can spin in place.",
                                "answer_type": "multiple_choice",
                                "difficulty": "medium",
                                "category": "Conceptual Physics",
                                "question_type": "apparent_contradiction",
                                "rationale": (
                                    "Stationary refers to zero linear velocity; angular velocity "
                                    "can be non-zero simultaneously (pure rotation in place)."
                                ),
                                "physics_signals": {
                                    "linear_velocity": 0,
                                    "angular_velocity": "> 0",
                                },
                            }
                        )

                # Contradiction 2: Rolling with Slipping
                if not found_contradictions["rolling_slipping"]:
                    if "Rolling Motion with Slipping" in labels:
                        found_contradictions["rolling_slipping"] = True

                        obj_desc = self._describe_object_unique(obj_id)

                        options = [
                            "Yes, rolling with slipping means both occur simultaneously.",
                            "No, you're either rolling OR sliding.",
                            "No, physics forbids simultaneous rolling and sliding.",
                            "Only with friction coefficient = 0.",
                        ]
                        shuffled_options = self._shuffle_options(options)

                        questions.append(
                            {
                                "question": (
                                    f"At t={t:.2f}s, a {obj_desc} is both rolling and sliding. "
                                    f"Can an object do both at the same time?"
                                ),
                                "options": shuffled_options,
                                "answer": "Yes, rolling with slipping means both occur simultaneously.",
                                "answer_type": "multiple_choice",
                                "difficulty": "medium",
                                "category": "Conceptual Physics",
                                "question_type": "apparent_contradiction",
                                "rationale": (
                                    "Rolling with slipping occurs when v ≠ r*ω, meaning the object "
                                    "both rotates AND slides at the same time."
                                ),
                                "physics_signals": {
                                    "rolling_with_slip": True,
                                },
                            }
                        )

                # Contradiction 3: Temporal consistency - Stationary and Accelerating in adjacent frames
                if not found_contradictions["temporal_consistency"]:
                    # Check current and next frame
                    frame_idx = self.frames.index(frame)
                    if frame_idx < len(self.frames) - 1:
                        next_frame = self.frames[frame_idx + 1]
                        next_obj_state = next_frame.get("objects", {}).get(obj_id)

                        if next_obj_state:
                            next_labels = [
                                l
                                for tax in next_obj_state.get("taxonomy", [])
                                for l in tax.get("labels", [])
                            ]

                            # Check for contradictory labels across frames
                            if "Stationary" in labels and "Accelerating" in next_labels:
                                question_text = ""
                                question_text = []
                                found_contradictions["temporal_consistency"] = True

                                obj_desc = self._describe_object_unique(obj_id)
                                next_t = next_frame.get("time", 0)

                                options = [
                                    "Yes, this is consistent - an object can transition from stationary to accelerating.",
                                    "No, this is inconsistent - stationary objects cannot accelerate.",
                                    "Only if an external force is applied.",
                                    "This violates conservation of energy.",
                                ]
                                question_text = (
                                    f"At t={t:.2f}s, {obj_desc} is labeled as 'Stationary'. "
                                    f"At t={next_t:.2f}s, it is labeled as 'Accelerating'. "
                                    f"Is this transition consistent?"
                                )

                                # For test/val splits, mask labels and use velocity inference
                                if self._should_mask_labels():
                                    inferred_state_curr = (
                                        self._infer_motion_state_from_velocity(
                                            obj_id, frame_idx
                                        )
                                    )
                                    inferred_state_next = (
                                        self._infer_motion_state_from_velocity(
                                            obj_id, frame_idx + 1
                                        )
                                    )

                                    if (
                                        inferred_state_curr == "Stationary"
                                        and inferred_state_next == "Accelerating"
                                    ):
                                        options = [
                                            "Yes, this is consistent - an object can transition from stationary to accelerating.",
                                            "No, this is inconsistent - stationary objects cannot accelerate.",
                                            "Only if an external force is applied.",
                                            "This violates conservation of energy.",
                                        ]
                                        curr_vel = self._get_velocity_from_obj_state(
                                            obj_state
                                        )
                                        next_vel = self._get_velocity_from_obj_state(
                                            next_obj_state
                                        )
                                        question_text = (
                                            f"At t={t:.2f}s, {obj_desc} has velocity magnitude {self._get_speed_from_velocity(curr_vel):.2f} m/s. "
                                            f"At t={next_t:.2f}s, its velocity magnitude is {self._get_speed_from_velocity(next_vel):.2f} m/s. "
                                            f"Is this transition from stationary to accelerating physically consistent?"
                                        )
                                shuffled_options = self._shuffle_options(options)

                                questions.append(
                                    {
                                        "question": question_text,
                                        "options": shuffled_options,
                                        "answer": "Yes, this is consistent - an object can transition from stationary to accelerating.",
                                        "answer_type": "multiple_choice",
                                        "difficulty": "hard",
                                        "category": "Conceptual Physics",
                                        "question_type": "temporal_consistency",
                                        "rationale": (
                                            "An object can transition from stationary to accelerating when a force is applied. "
                                            "This is consistent with Newton's laws."
                                        ),
                                        "physics_signals": {
                                            "frame1_time": t,
                                            "frame2_time": next_t,
                                            "transition": "Stationary → Accelerating",
                                        },
                                    }
                                )

                # Contradiction 4: Label vs observation mismatch
                if not found_contradictions["label_observation_mismatch"]:
                    # Check if object is labeled as "Accelerating" but velocity magnitude decreases
                    if "Accelerating" in labels:
                        vel = self._get_velocity_from_obj_state(obj_state)
                        vel_mag = self._get_speed_from_velocity(vel)

                        # Check previous frame velocity
                        frame_idx = self.frames.index(frame)
                        if frame_idx > 0:
                            prev_frame = self.frames[frame_idx - 1]
                            prev_obj_state = prev_frame.get("objects", {}).get(obj_id)
                            if prev_obj_state:
                                prev_vel = self._get_velocity_from_obj_state(
                                    prev_obj_state
                                )
                                prev_vel_mag = self._get_speed_from_velocity(prev_vel)

                                # Velocity magnitude decreased but labeled as accelerating
                                if prev_vel_mag > vel_mag + 0.1:  # Significant decrease
                                    found_contradictions[
                                        "label_observation_mismatch"
                                    ] = True

                                    obj_desc = self._describe_object_unique(obj_id)

                                    options = [
                                        "Yes, this is possible - acceleration is a vector; speed can decrease while accelerating.",
                                        "No, this is inconsistent - accelerating means speed must increase.",
                                        "Only in non-inertial reference frames.",
                                        "This violates Newton's laws.",
                                    ]
                                    shuffled_options = self._shuffle_options(options)

                                    questions.append(
                                        {
                                            "question": (
                                                f"At t={t:.2f}s, {obj_desc} is labeled as 'Accelerating', but its velocity magnitude "
                                                f"decreased from {prev_vel_mag:.2f} m/s to {vel_mag:.2f} m/s. Is this possible?"
                                            ),
                                            "options": shuffled_options,
                                            "answer": "Yes, this is possible - acceleration is a vector; speed can decrease while accelerating.",
                                            "answer_type": "multiple_choice",
                                            "difficulty": "hard",
                                            "category": "Conceptual Physics",
                                            "question_type": "label_observation_mismatch",
                                            "rationale": (
                                                "Acceleration is a vector quantity. An object can be accelerating "
                                                "in a direction opposite to its velocity, causing speed to decrease "
                                                "(e.g., deceleration is negative acceleration)."
                                            ),
                                            "physics_signals": {
                                                "prev_velocity_mag": round(
                                                    prev_vel_mag, 2
                                                ),
                                                "curr_velocity_mag": round(vel_mag, 2),
                                                "label": "Accelerating",
                                            },
                                        }
                                    )

                # Early exit if all contradictions found
                if all(found_contradictions.values()):
                    break

            if all(found_contradictions.values()):
                break

        return questions

    def generate_multihop_questions(self) -> List[Dict]:
        """
        Multi-step causal chains: A → B → C and A → B → C → D.
        """
        questions = []

        # Try to find 4-hop chain first (A → B → C → D)
        for i, frame in enumerate(self.frames):
            if not frame.get("interactions"):
                continue

            g1, g2 = frame["interactions"][0]
            a = f"geom_obj{g1 - 1}"
            b = f"geom_obj{g2 - 1}"

            if a not in self.appeared_obj_ids or b not in self.appeared_obj_ids:
                continue

            # Find B → C collision
            for j in range(i + 1, min(i + 15, len(self.frames))):
                future_frame = self.frames[j]
                if not future_frame.get("interactions"):
                    continue

                for fg1, fg2 in future_frame.get("interactions", []):
                    ids = [self._geom_id_to_obj_id(fg1), self._geom_id_to_obj_id(fg2)]
                    if b in ids:
                        c = ids[0] if ids[1] == b else ids[1]

                        if c not in self.appeared_obj_ids:
                            continue

                        # Find C → D collision
                        for k in range(j + 1, min(j + 15, len(self.frames))):
                            future_frame2 = self.frames[k]
                            if not future_frame2.get("interactions"):
                                continue

                            for fg3, fg4 in future_frame2.get("interactions", []):
                                ids2 = [
                                    self._geom_id_to_obj_id(fg3),
                                    self._geom_id_to_obj_id(fg4),
                                ]
                                if c in ids2:
                                    d = ids2[0] if ids2[1] == c else ids2[1]

                                    if d not in self.appeared_obj_ids:
                                        continue

                                    # Found 4-hop chain: A → B → C → D
                                    desc_a = self._describe_object_unique(a)
                                    desc_b = self._describe_object_unique(b)
                                    desc_c = self._describe_object_unique(c)
                                    desc_d = self._describe_object_unique(d)

                                    options = [
                                        "Yes, through a chain of momentum transfers: A→B→C→D.",
                                        "Partially, but the chain is too long to establish direct causation.",
                                        "No, each collision is independent.",
                                        "Only if all objects have the same mass.",
                                    ]
                                    shuffled_options = self._shuffle_options(options)

                                    questions.append(
                                        {
                                            "question": (
                                                f"{desc_a} hits {desc_b}. Then {desc_b} hits {desc_c}. "
                                                f"Finally, {desc_c} hits {desc_d}. "
                                                f"Is {desc_a} indirectly responsible for the collision between {desc_c} and {desc_d}?"
                                            ),
                                            "options": shuffled_options,
                                            "answer": "Yes, through a chain of momentum transfers: A→B→C→D.",
                                            "answer_type": "multiple_choice",
                                            "difficulty": "very_hard",
                                            "category": "Multi-Hop Reasoning",
                                            "question_type": "four_hop_causation",
                                            "rationale": (
                                                "Momentum transfers through the chain: A imparts momentum to B, "
                                                "B to C, and C to D. While each transfer reduces the causal link, "
                                                "the initial collision (A→B) is still part of the causal chain."
                                            ),
                                            "physics_signals": {
                                                "chain": [a, b, c, d],
                                                "chain_length": 4,
                                            },
                                        }
                                    )
                                    return questions

        # Fallback to 3-hop chain (A → B → C) if 4-hop not found
        for i, frame in enumerate(self.frames):
            if not frame.get("interactions"):
                continue

            g1, g2 = frame["interactions"][0]
            a = f"geom_obj{g1 - 1}"
            b = f"geom_obj{g2 - 1}"

            if a not in self.appeared_obj_ids or b not in self.appeared_obj_ids:
                continue

            for future in self.frames[i + 1 : i + 10]:
                for fg1, fg2 in future.get("interactions", []):
                    ids = [self._geom_id_to_obj_id(fg1), self._geom_id_to_obj_id(fg2)]
                    if b in ids:
                        c = ids[0] if ids[1] == b else ids[1]

                        if c not in self.appeared_obj_ids:
                            continue

                        desc_a = self._describe_object_unique(a)
                        desc_b = self._describe_object_unique(b)
                        desc_c = self._describe_object_unique(c)

                        options = [
                            "Partially, via momentum transfer.",
                            "Yes, directly.",
                            "No, events are independent.",
                            "Impossible to tell.",
                        ]
                        shuffled_options = self._shuffle_options(options)

                        questions.append(
                            {
                                "question": (
                                    f"{desc_a} hits {desc_b}. Later, {desc_b} hits {desc_c}. "
                                    f"Is {desc_a} indirectly responsible for the second collision?"
                                ),
                                "options": shuffled_options,
                                "answer": "Partially, via momentum transfer.",
                                "answer_type": "multiple_choice",
                                "difficulty": "very_hard",
                                "category": "Multi-Hop Reasoning",
                                "question_type": "indirect_causation",
                                "rationale": (
                                    "A caused B to move, enabling a later collision, "
                                    "but B's later trajectory is not fully determined by A."
                                ),
                                "physics_signals": {
                                    "chain": [a, b, c],
                                    "chain_length": 3,
                                },
                            }
                        )
                        return questions

        return questions

    def generate_all_advanced_questions(self) -> List[Dict]:
        """Generate all advanced question types."""
        questions = []

        questions.extend(self.generate_causal_questions())
        questions.extend(self.generate_counterfactual_questions())
        questions.extend(self.generate_contradictory_questions())
        questions.extend(self.generate_multihop_questions())
        questions.extend(self.generate_property_competition_questions())
        questions.extend(self.generate_collision_geometry_questions())
        questions.extend(self.generate_post_collision_motion_questions())
        questions.extend(self.generate_mass_effects_questions())
        questions.extend(self.generate_friction_coefficient_questions())
        questions.extend(self.generate_shape_distribution_questions())
        questions.extend(self.generate_velocity_comparison_questions())
        questions.extend(self.generate_velocity_scaling_counterfactual_questions())
        questions.extend(self.generate_physics_principle_questions())
        questions.extend(self.generate_temporal_sequence_questions())

        # Remove duplicates by converting to set of JSON strings, then back to list
        seen = set()
        unique_questions = []
        for q in questions:
            # Use question text as the key for deduplication
            question_key = q.get("question", "")
            if question_key and question_key not in seen:
                seen.add(question_key)
                unique_questions.append(q)

        random.shuffle(unique_questions)
        return unique_questions
