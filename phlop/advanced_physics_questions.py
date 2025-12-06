import json
import random
from typing import List, Dict, Tuple
from pathlib import Path


class AdvancedPhysicsQuestions:
    def __init__(self, meta_data: Dict):
        self.meta = meta_data
        self.frames = meta_data.get("frames", [])
        self.objects = meta_data.get("objects", [])
        self.fps = 25

    def generate_causal_questions(self) -> List[Dict]:
        questions = []

        # find collisions
        for frame_idx, frame in enumerate(self.frames):
            if not frame.get("interactions"):
                continue

            time = frame.get("time", 0)
            interactions = frame.get("interactions", [])

            for interaction in interactions:
                if len(interaction) < 2:
                    continue

                g1, g2 = interaction[0], interaction[1]
                obj_ids = [f"geom_obj{g1 - 1}", f"geom_obj{g2 - 1}"]

                # Get object descriptions
                obj_descs = []
                for obj_id in obj_ids:
                    obj = next((o for o in self.objects if o["id"] == obj_id), None)
                    if obj:
                        shape = obj.get("geom_type", "object")
                        color = obj.get("visual", {}).get("rgba", "")
                        obj_descs.append(f"{shape} ({obj_id})")

                if len(obj_descs) < 2:
                    continue

                desc1, desc2 = obj_descs

                # if there is motion before and after collision
                if frame_idx > 0 and frame_idx < len(self.frames) - 1:
                    prev_frame = self.frames[frame_idx - 1]
                    next_frame = self.frames[frame_idx + 1]

                    vel_before = (
                        prev_frame["objects"]
                        .get(obj_ids[1], {})
                        .get("velocity", [0, 0, 0])
                    )
                    vel_after = (
                        next_frame["objects"]
                        .get(obj_ids[1], {})
                        .get("velocity", [0, 0, 0])
                    )

                    vel_before_mag = sum(v**2 for v in vel_before) ** 0.5
                    vel_after_mag = sum(v**2 for v in vel_after) ** 0.5

                    # Causal Q1: Direct causality
                    if vel_before_mag < 0.1 and vel_after_mag > 0.5:
                        questions.append(
                            {
                                "question": f"At t={time:.1f}s, {desc1} collides with {desc2}. "
                                f"Before collision, {desc2} is stationary. "
                                f"After collision, {desc2} is moving. "
                                f"Is the collision the CAUSE of {desc2}'s motion?",
                                "answer": "Yes, the collision transferred momentum to the stationary object.",
                                "answer_type": "multiple_choice",
                                "options": [
                                    "Yes, the collision transferred momentum to the stationary object.",
                                    "No, the object was already moving.",
                                    "No, collisions don't cause motion.",
                                    "It's just correlation, not causation.",
                                ],
                                "difficulty": "hard",
                                "category": "Causal Reasoning",
                                "question_type": "causal",
                            }
                        )

                    # # Causal Q2: Counterfactual causality
                    # if vel_after_mag < 0.1:
                    #     questions.append({
                    #         "question": f"If {desc1} had NOT collided with {desc2} at t={time:.1f}s, "
                    #                    f"would {desc2} have stopped at the same time?",
                    #         "answer": "Unknown - we can't know what would have happened without the collision.",
                    #         "answer_type": "multiple_choice",
                    #         "options": [
                    #             "Yes, it would have stopped anyway due to friction.",
                    #             "No, the collision was necessary for it to stop.",
                    #             "Unknown - we can't know what would have happened without the collision.",
                    #             "The collision doesn't affect stopping time."
                    #         ],
                    #         "difficulty": "hard",
                    #         "category": "Causal Reasoning",
                    #         "question_type": "causal",
                    #     })

        return questions

    def generate_counterfactual_questions(self) -> List[Dict]:
        questions = []

        for obj in self.objects:
            obj_id = obj.get("id", "")
            if not obj_id:
                continue

            shape = obj.get("geom_type", "object")
            mass = float(obj.get("mass", 1.0))
            friction = obj.get("friction", "0.4 0 0")
            if isinstance(friction, str):
                friction = float(friction.split()[0])

            #  when object is sliding
            for frame_idx, frame in enumerate(self.frames):
                if obj_id not in frame["objects"]:
                    continue

                vel = frame["objects"][obj_id].get("velocity", [0, 0, 0])
                vel_mag = sum(v**2 for v in vel) ** 0.5

                #  stopping point
                if 0.5 < vel_mag < 2.0:
                    for stop_idx in range(frame_idx + 1, len(self.frames)):
                        stop_vel = self.frames[stop_idx]["objects"][obj_id].get(
                            "velocity", [0, 0, 0]
                        )
                        stop_vel_mag = sum(v**2 for v in stop_vel) ** 0.5

                        if stop_vel_mag < 0.05:
                            stop_time = self.frames[stop_idx].get("time", 0)
                            start_time = frame.get("time", 0)

                            duration = stop_time - start_time

                            # CF Q1: Friction doubling
                            if friction > 0:
                                # Physics: t = v / (μ * g)
                                # If μ doubles: t_new = v / (2μ * g) = t_old / 2
                                new_duration = duration / 2

                                questions.append(
                                    {
                                        "question": f"A {shape} is sliding and takes {duration:.2f}s to stop. "
                                        f"If friction coefficient were doubled from {friction:.2f} to {friction * 2:.2f}, "
                                        f"approximately how long would it take to stop?",
                                        "answer": f"{new_duration:.2f} seconds (halved)",
                                        "answer_type": "multiple_choice",
                                        "options": [
                                            f"{new_duration:.2f} seconds (halved)",
                                            f"{duration:.2f} seconds (unchanged)",
                                            f"{duration * 2:.2f} seconds (doubled)",
                                            f"{duration * 0.25:.2f} seconds (quartered)",
                                        ],
                                        "difficulty": "hard",
                                        "category": "Counterfactual Reasoning",
                                        "question_type": "counterfactual",
                                    }
                                )

                            # CF Q2: Mass doubling
                            # Physics: v ∝ (1/m), so heavier object keeps more momentum
                            questions.append(
                                {
                                    "question": f"If this {shape} had DOUBLE the mass ({mass * 2:.1f}kg vs {mass:.1f}kg), "
                                    f"how would its sliding distance change?",
                                    "answer": "It would slide further because heavier objects lose kinetic energy slower.",
                                    "answer_type": "multiple_choice",
                                    "options": [
                                        "It would slide further because heavier objects lose kinetic energy slower.",
                                        "It would slide the same distance - mass doesn't matter.",
                                        "It would slide shorter - heavier objects experience more friction.",
                                        "It depends on the color of the object.",
                                    ],
                                    "difficulty": "hard",
                                    "category": "Counterfactual Reasoning",
                                    "question_type": "counterfactual",
                                }
                            )

                            break

        return questions

    # ==================== CONTRADICTORY SCENARIOS ====================

    def generate_contradictory_questions(self) -> List[Dict]:
        """
        Generate contradictory scenario questions
        Tests: Understanding of physics category definitions
        """
        questions = []

        # Spinning vs stationary contradiction
        questions.append(
            {
                "question": "Can an object be BOTH stationary AND spinning at the same time?",
                "answer": "Yes, if v ≈ 0 but ω > 0 (spinning in place like a top).",
                "answer_type": "multiple_choice",
                "options": [
                    "Yes, if v ≈ 0 but ω > 0 (spinning in place like a top).",
                    "No, stationary means completely motionless.",
                    "No, physics forbids this.",
                    "Only for quantum objects.",
                ],
                "difficulty": "medium",
                "category": "Contradictory Scenarios",
                "question_type": "contradictory",
            }
        )

        # Rolling vs sliding contradiction
        questions.append(
            {
                "question": "Can an object be BOTH rolling AND sliding at the same time?",
                "answer": "Yes, in rolling with slipping: v ≠ r*ω means both rotation and sliding occur.",
                "answer_type": "multiple_choice",
                "options": [
                    "Yes, in rolling with slipping: v ≠ r*ω means both rotation and sliding occur.",
                    "No, you're either rolling OR sliding.",
                    "No, physics forbids simultaneous rolling and sliding.",
                    "Only with friction coefficient = 0.",
                ],
                "difficulty": "medium",
                "category": "Contradictory Scenarios",
                "question_type": "contradictory",
            }
        )

        # Elastic collision contradiction
        questions.append(
            {
                "question": "In an elastic collision, is kinetic energy conserved or lost?",
                "answer": "Conserved - elastic means energy is preserved (not lost to deformation).",
                "answer_type": "multiple_choice",
                "options": [
                    "Conserved - elastic means energy is preserved (not lost to deformation).",
                    "Lost - all collisions dissipate energy.",
                    "Gained - collisions create energy.",
                    "Depends on the season.",
                ],
                "difficulty": "easy",
                "category": "Contradictory Scenarios",
                "question_type": "contradictory",
            }
        )

        return questions

    # ==================== NEGATIVE QUESTIONS ====================

    def generate_negative_questions(self) -> List[Dict]:
        """
        Generate negative versions of basic questions
        Tests: Negation understanding
        """
        questions = []

        for frame_idx, frame in enumerate(self.frames):
            if not frame.get("interactions"):
                continue

            # Positive: "Did objects collide?"
            # Negative: "Did objects NOT collide?"
            questions.append(
                {
                    "question": f"At t={frame.get('time', 0):.1f}s, did the objects NOT collide?",
                    "answer": "No, they did collide.",
                    "answer_type": "yes_no",
                    "difficulty": "easy",
                    "category": "Negation",
                    "question_type": "negative",
                }
            )

        # Find frames WITHOUT collisions
        for frame_idx, frame in enumerate(self.frames):
            if frame.get("interactions"):
                continue

            time = frame.get("time", 0)

            if random.random() < 0.1:  # Sample 10% of non-collision frames
                questions.append(
                    {
                        "question": f"At t={time:.1f}s, did objects collide?",
                        "answer": "No",
                        "answer_type": "yes_no",
                        "difficulty": "easy",
                        "category": "Negation",
                        "question_type": "negative",
                    }
                )

        return questions

    # ==================== MULTI-HOP REASONING ====================

    def generate_multihop_questions(self) -> List[Dict]:
        """
        Generate multi-hop reasoning questions
        Tests: Chain of reasoning through multiple steps
        """
        questions = []

        for frame_idx, frame in enumerate(self.frames):
            if not frame.get("interactions"):
                continue

            time = frame.get("time", 0)

            # Multi-hop: A → B collision, then B → C interaction
            interaction = frame.get("interactions", [[]])[0]
            if len(interaction) < 2:
                continue

            g1, g2 = interaction[0], interaction[1]
            obj_b_id = f"geom_obj{g2 - 1}"
            obj_a_id = f"geom_obj{g1 - 1}"

            # Find if object B later collides with C
            for future_frame_idx in range(
                frame_idx + 1, min(frame_idx + 10, len(self.frames))
            ):
                future_frame = self.frames[future_frame_idx]
                future_interactions = future_frame.get("interactions", [])

                for future_inter in future_interactions:
                    if len(future_inter) >= 2:
                        fobj1, fobj2 = future_inter[0], future_inter[1]
                        fobj1_id = f"geom_obj{fobj1 - 1}"
                        fobj2_id = f"geom_obj{fobj2 - 1}"

                        # Check if B is involved
                        if obj_b_id in [fobj1_id, fobj2_id]:
                            other_id = fobj2_id if fobj1_id == obj_b_id else fobj1_id

                            questions.append(
                                {
                                    "question": f"At t={time:.1f}s, {obj_a_id} hits {obj_b_id}. "
                                    f"Then at t={future_frame.get('time', 0):.1f}s, {obj_b_id} hits {other_id}. "
                                    f"Can we say {obj_a_id} is INDIRECTLY responsible for the second collision?",
                                    "answer": "Partially - A caused B to move, which made B collide with C, "
                                    "but B could have collided with C anyway.",
                                    "answer_type": "multiple_choice",
                                    "options": [
                                        "Yes - A directly caused both collisions.",
                                        "No - they're independent events.",
                                        "Partially - A caused B to move, which made B collide with C, "
                                        "but B could have collided with C anyway.",
                                        "Impossible to determine from video.",
                                    ],
                                    "difficulty": "very_hard",
                                    "category": "Multi-Hop Reasoning",
                                    "question_type": "multi_hop",
                                }
                            )
                            break

        return questions

    def generate_all_advanced_questions(self) -> List[Dict]:
        """Generate all advanced question types"""
        all_questions = []

        print("Generating causal questions...")
        all_questions.extend(self.generate_causal_questions())

        print("Generating counterfactual questions...")
        all_questions.extend(self.generate_counterfactual_questions())

        print("Generating contradictory questions...")
        all_questions.extend(self.generate_contradictory_questions())

        print("Generating negative questions...")
        all_questions.extend(self.generate_negative_questions())

        print("Generating multi-hop questions...")
        all_questions.extend(self.generate_multihop_questions())

        return all_questions


# ==================== MAIN USAGE ====================


def augment_qa_dataset(meta_json_path: str, qa_json_path: str, output_path: str = None):
    """
    Load existing QA dataset and augment with advanced questions

    Args:
        meta_json_path: Path to meta.json
        qa_json_path: Path to existing qa.json
        output_path: Path to save augmented qa.json (default: append "_augmented")
    """

    # Load metadata
    with open(meta_json_path, "r") as f:
        metadata = json.load(f)

    # Load existing questions
    with open(qa_json_path, "r") as f:
        existing_questions = json.load(f)

    print(f"Loaded {len(existing_questions)} existing questions")

    # Generate advanced questions
    generator = AdvancedPhysicsQuestions(metadata)
    advanced_questions = generator.generate_all_advanced_questions()

    print(f"Generated {len(advanced_questions)} advanced questions")

    # Combine
    all_questions = existing_questions + advanced_questions

    # Save
    output_file = output_path or str(qa_json_path).replace(".json", "_augmented.json")
    with open(output_file, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\n✅ Saved {len(all_questions)} total questions to {output_file}")

    # Statistics
    print("\nQuestion Type Distribution:")
    question_types = {}
    for q in all_questions:
        qtype = q.get("question_type", "basic")
        question_types[qtype] = question_types.get(qtype, 0) + 1

    for qtype, count in sorted(question_types.items()):
        pct = 100 * count / len(all_questions)
        print(f"  {qtype}: {count} ({pct:.1f}%)")

    return all_questions


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python script.py <meta.json> <qa.json> [output.json]")
        sys.exit(1)

    meta_file = sys.argv[1]
    qa_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    augment_qa_dataset(meta_file, qa_file, output_file)
