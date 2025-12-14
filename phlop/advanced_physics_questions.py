import json
import random
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Tuple
from phlop.utils import describe_object_unique, rgba_to_name


class AdvancedPhysicsQuestions:
    def __init__(self, file_path: str, fps: int = 25):
        self.data = self._load_json(file_path)
        self.frames = self.data.get("frames", [])
        self.objects = self.data.get("objects", [])
        self.fps = fps

        self.appeared_obj_ids = self._get_appeared_object_ids()

    def _load_json(self, path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    def _get_appeared_object_ids(self):
        """Get set of object IDs that actually appear in frames (have visible bounding boxes)."""
        appeared = set()
        for frame in self.frames:
            for obj_id, obj_state in frame.get("objects", {}).items():
                bbox = obj_state.get("bbox", [[0, 0], [0, 0]])
                if bbox != [[0, 0], [0, 0]]:
                    appeared.add(obj_id)
        return appeared

    def _get_object(self, obj_id: str) -> Optional[Dict]:
        return next((o for o in self.objects if o.get("id") == obj_id), None)

    def _rgba_to_name(self, rgba):
        """Convert RGBA tuple to closest CSS color name."""
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
        return best_name.replace("grey", "gray")

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
                color = self._rgba_to_name(rgba)
            except (ValueError, AttributeError):
                color = "unknown color"
        else:
            color = "unknown color"
        
        return f"{color} {shape} ({obj_id})"

    def _describe_object_unique(self, target_id: str) -> str:
        """
        Generates a unique description using the shared utility function.
        """
        return describe_object_unique(
            target_id=target_id,
            objects=self.objects,
            frames=self.frames,
            appeared_obj_ids=self.appeared_obj_ids,
            rgba_to_name_func=self._rgba_to_name
        )

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

    def _shuffle_options(self, options: List[str]) -> List[str]:
        shuffled = options.copy()
        random.shuffle(shuffled)
        return shuffled

    def _calculate_kinetic_energy_loss(self, obj1_id: str, obj2_id: str, collision_frame_idx: int) -> Optional[float]:
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
        
        v1_before = prev_frame.get("objects", {}).get(obj1_id, {}).get("velocity", [0, 0, 0])
        v2_before = prev_frame.get("objects", {}).get(obj2_id, {}).get("velocity", [0, 0, 0])
        v1_after = post_frame.get("objects", {}).get(obj1_id, {}).get("velocity", [0, 0, 0])
        v2_after = post_frame.get("objects", {}).get(obj2_id, {}).get("velocity", [0, 0, 0])
        
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

    def generate_causal_questions(self) -> List[Dict]:
        """
        Direct cause-effect questions grounded in collision taxonomy.
        """
        questions = []
        seen_collisions = set()  # Track (time, obj_pair) to avoid duplicates

        for i in range(1, len(self.frames) - 1):
            prev_f = self.frames[i - 1]
            cur_f = self.frames[i]
            next_f = self.frames[i + 1]

            if not cur_f.get("interactions"):
                continue

            t = cur_f.get("time", 0)

            for g1, g2 in cur_f["interactions"]:
                obj1 = f"geom_obj{g1 - 1}"
                obj2 = f"geom_obj{g2 - 1}"

                # Only process objects that appeared in frames
                if obj1 not in self.appeared_obj_ids or obj2 not in self.appeared_obj_ids:
                    continue

                # for traking if the same collision has been processed already
                collision_key = (round(t, 3), tuple(sorted([obj1, obj2])))
                
                # Skip if already processed
                if collision_key in seen_collisions:
                    continue
                seen_collisions.add(collision_key)

                o1_data = cur_f["objects"].get(obj1, {})
                o2_data = cur_f["objects"].get(obj2, {})

                col_tax = (
                    self._get_taxonomy(o1_data, "Interaction Events", "Collision")
                    or self._get_taxonomy(o2_data, "Interaction Events", "Collision")
                )
                if not col_tax:
                    continue

                collision_data = col_tax[0]
                collision_type = collision_data.get("labels", [None])[0]
                energy_analysis = collision_data.get("energy_analysis")

                # Get object descriptions
                desc1 = self._describe_object_unique(obj1)
                desc2 = self._describe_object_unique(obj2)

                # Q1: Motion change causation
                v_prev = prev_f["objects"].get(obj2, {}).get("velocity", [0, 0, 0])
                v_next = next_f["objects"].get(obj2, {}).get("velocity", [0, 0, 0])

                prev_speed = sum(v * v for v in v_prev) ** 0.5
                next_speed = sum(v * v for v in v_next) ** 0.5

                if prev_speed < 0.05 and next_speed > 0.3:
                    options = [
                        "Yes, momentum transfer caused the motion.",
                        "No, the object was already moving.",
                        "No, collisions do not cause motion.",
                        "Cannot be determined.",
                    ]
                    shuffled_options = self._shuffle_options(options)
                    
                    questions.append(
                        {
                            "question": (
                                f"At t={t:.2f}s, {desc1} collides with {desc2}. "
                                f"The second object was stationary before the collision "
                                f"but moves afterward. Is the collision the cause?"
                            ),
                            "options": shuffled_options,
                            "answer": "Yes, momentum transfer caused the motion.",
                            "answer_type": "multiple_choice",
                            "difficulty": "hard",
                            "category": "Causal Reasoning",
                            "question_type": "direct_causation",
                            "rationale": (
                                "The physics engine shows a transition from zero to non-zero "
                                "velocity immediately after the collision."
                            ),
                            "physics_signals": {
                                "prev_speed": prev_speed,
                                "next_speed": next_speed,
                                "collision": collision_type,
                            },
                        }
                    )

                # Q2: Collision type and energy conservation
                if collision_type:
                    if "Elastic" in collision_type:
                        answer = "Kinetic energy is conserved; objects bounce apart."
                        options = [
                            "Kinetic energy is conserved; objects bounce apart.",
                            "All energy is lost to heat and sound.",
                            "Energy increases during the collision.",
                            "Energy conservation doesn't apply to collisions.",
                        ]
                    elif "Partially Inelastic" in collision_type:
                        answer = "Some kinetic energy is lost, but not all."
                        options = [
                            "Some kinetic energy is lost, but not all.",
                            "Kinetic energy is completely conserved.",
                            "All energy is lost to heat and sound.",
                            "Energy increases during the collision.",
                        ]
                    elif "Highly Inelastic" in collision_type:
                        answer = "Most kinetic energy is dissipated to heat and sound."
                        options = [
                            "Most kinetic energy is dissipated to heat and sound.",
                            "Kinetic energy is completely conserved.",
                            "Some energy is created during collision.",
                            "Energy conservation doesn't apply here.",
                        ]
                    else:
                        continue

                    shuffled_options = self._shuffle_options(options)
                    
                    questions.append(
                        {
                            "question": (
                                f"At t={t:.2f}s, {desc1} collides with {desc2}. "
                                f"The collision is classified as '{collision_type}'. "
                                f"What does this tell us about energy conservation?"
                            ),
                            "options": shuffled_options,
                            "answer": answer,
                            "answer_type": "multiple_choice",
                            "difficulty": "hard",
                            "category": "Causal Reasoning",
                            "question_type": "energy_analysis",
                            "rationale": (
                                f"Energy classification: {energy_analysis}"
                            ),
                            "physics_signals": {
                                "collision_type": collision_type,
                                "energy_analysis": energy_analysis,
                            },
                        }
                    )

                # Q3: Kinetic energy loss percentage
                ke_loss = self._calculate_kinetic_energy_loss(obj1, obj2, i)
                if ke_loss is not None and ke_loss >= 0:
                    def make_opts(true_val):
                        a = round(true_val * 0.8, 1)
                        b = round(min(true_val + 10, 100), 1)
                        c = round(abs(true_val - 50), 1)
                        opts = list({round(true_val, 1), a, b, c})
                        random.shuffle(opts)
                        return [f"{v:.1f}%" for v in opts]
                    
                    options = make_opts(round(ke_loss, 1))
                    
                    questions.append(
                        {
                            "question": (
                                f"What percentage of the system's kinetic energy was lost "
                                f"when the {desc1} collided with the {desc2}?"
                            ),
                            "options": options,
                            "answer": f"{round(ke_loss, 1):.1f}%",
                            "answer_type": "multiple_choice",
                            "difficulty": "very_hard",
                            "category": "Energy Analysis",
                            "question_type": "kinetic_energy_loss",
                            "rationale": (
                                "For each object, kinetic energy KE = 0.5·m·|v|². "
                                "Compute before and after collision, sum them, then compute the percentage lost."
                            ),
                            "physics_signals": {
                                "percent_ke_lost": round(ke_loss, 1),
                                "collision_type": collision_type,
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

    def generate_contradictory_questions(self) -> List[Dict]:
        """
        Conceptual contradiction checks derived from kinematic taxonomy.
        """
        questions = []
        found_contradictions = {
            "stationary_spinning": False,
            "rolling_slipping": False,
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

                # Early exit if both contradictions found
                if all(found_contradictions.values()):
                    break

            if all(found_contradictions.values()):
                break

        return questions

    def generate_multihop_questions(self) -> List[Dict]:
        """
        Multi-step causal chains: A → B → C.
        """
        questions = []

        for i, frame in enumerate(self.frames):
            if not frame.get("interactions"):
                continue

            if not frame["interactions"]:
                continue

            g1, g2 = frame["interactions"][0]
            a = f"geom_obj{g1 - 1}"
            b = f"geom_obj{g2 - 1}"

            # Only process objects that appeared in frames
            if a not in self.appeared_obj_ids or b not in self.appeared_obj_ids:
                continue

            for future in self.frames[i + 1 : i + 10]:
                for fg1, fg2 in future.get("interactions", []):
                    ids = [f"geom_obj{fg1 - 1}", f"geom_obj{fg2 - 1}"]
                    if b in ids:
                        c = ids[0] if ids[1] == b else ids[1]

                        # Only process objects that appeared in frames
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