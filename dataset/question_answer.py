import json
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Tuple, Dict, Set


class QuestionAnswers:
    def __init__(self, file_path: str):
        self.data = self.read_file(file_path)

    def read_file(self, path: str) -> Dict:
        with open(path) as f:
            return json.load(f)

    def rgba_to_text(self, rgba_str: str) -> Tuple[str, str]:
        """Convert an RGBA string to the closest named color and opacity description."""
        rgba = tuple(map(float, rgba_str.split()))

        # Find the closest named color
        closest_color = min(
            mcolors.CSS4_COLORS.items(),
            key=lambda item: sum((a - b) ** 2 for a, b in zip(rgba, mcolors.to_rgba(item[1])))
        )[0]

        # Describe the opacity
        opacity_text = (
            "fully opaque" if rgba[3] == 1.0 else
            "mostly opaque" if rgba[3] > 0.5 else
            "mostly transparent"
        )

        return closest_color, opacity_text

    def calculate_kinetic_energy(self, mass: float, velocity: List[float]) -> float:
        velocity_magnitude = np.linalg.norm(velocity)
        return round(0.5 * mass * (velocity_magnitude ** 2), 3)

    def get_questions_answers(self) -> List[Tuple[str, str]]:
        questions_answers = []
        collisions = []
        decelerating_objects = set()
        accelerating_objects = set()

        heaviest_object = ""
        max_mass = 0
        highest_friction_object = None
        max_friction = 0
        highest_density_object = None
        max_density = 0

        objects_data = {obj["id"]: obj for obj in self.data.get("objects", [])}

        for frame_idx, frame in enumerate(self.data.get("frames", [])):
            time = round(float(frame.get("time", "0")), 3)
            objects = frame.get("objects", {})

            for obj_id, obj in objects.items():
                object_data = objects_data.get(obj_id)
                if not object_data:
                    continue

                velocity = obj.get("velocity")
                mass = round(float(object_data.get("mass", "0")), 3)
                friction = round(float(object_data.get("friction", "0")), 3)
                density = round(float(object_data.get("density", "0")), 3)

                closest_color, _ = self.rgba_to_text(object_data.get("visual", {}).get("rgba", ""))
                obj_description = f"{closest_color} {object_data.get('material', 'unknown_material')} {object_data.get('geom_type', 'unknown_geom_type')}"

                if mass > max_mass:
                    max_mass = mass
                    heaviest_object = obj_description

                if friction > max_friction:
                    max_friction = friction
                    highest_friction_object = obj_description

                if density > max_density:
                    max_density = density
                    highest_density_object = obj_description

                for item in obj.get("taxonomy", []):
                    if item.get("subcategory") == "Collision":
                        self._process_collision(frame_idx, obj_id, obj_description, mass,
                                                velocity, time, collisions, questions_answers)

                    for label in item.get("labels", []):
                        if label == "Decelerating":
                            decelerating_objects.add(obj_description)
                        elif label == "Accelerating":
                            accelerating_objects.add(obj_description)

        self._add_general_questions(questions_answers, collisions, decelerating_objects, accelerating_objects,
                                    heaviest_object, max_mass, highest_friction_object, max_friction,
                                    highest_density_object, max_density)

        return questions_answers

    def _process_collision(self, frame_idx: int, obj_id: str, obj_description: str, mass: float, velocity: List[float],
                           time: float, collisions: List, questions_answers: List[Tuple[str, str]]):
        prev_frame = self.data.get("frames")[frame_idx - 1].get("objects", {})
        previous_velocity = prev_frame.get(obj_id, {}).get("velocity", [])

        if previous_velocity and velocity:
            current_velocity = np.array(velocity)
            previous_velocity = np.array(previous_velocity)
            delta_v = np.linalg.norm(current_velocity - previous_velocity)

            collisions.append((obj_id, obj_description, mass, previous_velocity, velocity, time, delta_v))

            # Add kinetic energy question
            questions_answers.append(
                (f"What was the kinetic energy of the {obj_description} before the collision at time {time} s?",
                 f"The kinetic energy of the {obj_description} before the collision was {self.calculate_kinetic_energy(mass, previous_velocity)} J (Joules).")
            )

            # Add collision type questions
            for label in self.data.get("frames")[frame_idx].get("objects", {}).get(obj_id, {}).get("taxonomy", []):
                if label.get("subcategory") == "Collision":
                    for collision_label in label.get("labels", []):
                        if collision_label == "Elastic Collision":
                            questions_answers.append(
                                (f"What happened during the elastic collision involving {obj_description} at time {time} s?",
                                 f"{obj_description} is conserving both momentum and kinetic energy.")
                            )
                        elif collision_label == "Inelastic Collision":
                            questions_answers.append(
                                (f"What happened during the inelastic collision involving {obj_description} at time {time} s?",
                                 f"{obj_description} is losing some kinetic energy.")
                            )

    def _add_general_questions(self, questions_answers: List[Tuple[str, str]], collisions: List,
                               decelerating_objects: Set[str], accelerating_objects: Set[str],
                               heaviest_object: str, max_mass: float, highest_friction_object: str, max_friction: float,
                               highest_density_object: str, max_density: float):
        if collisions:
            self._add_collision_questions(questions_answers, collisions)

        if decelerating_objects:
            questions_answers.append(
                ("Which objects decelerated during the simulation?",
                 f"The following objects decelerated: {', '.join(decelerating_objects)}.")
            )

        if accelerating_objects:
            questions_answers.append(
                ("Which objects accelerated during the simulation?",
                 f"The following objects accelerated: {', '.join(accelerating_objects)}.")
            )

        if heaviest_object:
            questions_answers.append(
                ("What was the heaviest object in the simulation?",
                 f"{heaviest_object} with a mass of {max_mass} kg.")
            )

        if highest_friction_object:
            questions_answers.append(
                ("Which object had the highest friction?",
                 f"{highest_friction_object} with a friction value of {max_friction}.")
            )

        if highest_density_object:
            questions_answers.append(
                ("Which object had the highest density?",
                 f"{highest_density_object} with a density of {max_density} kg/m³.")
            )

    def _add_collision_questions(self, questions_answers: List[Tuple[str, str]], collisions: List):
        colliding_objects = {obj_desc for _, obj_desc, _, _, _, _, _ in collisions}

        questions_answers.extend([
            ("How many objects were involved in collisions?",
             f"{len(colliding_objects)} objects were involved in collisions."),
            ("Which objects were involved in collisions during the simulation?",
             f"The following objects were involved in collisions: {', '.join(colliding_objects)}.")
        ])

        if collisions:
            most_velocity_loss_collision = max(collisions, key=lambda x: x[-1])
            obj_id, obj_desc, mass, prev_velocity, velocity, time, delta_v = most_velocity_loss_collision

            questions_answers.extend([
                ("Which object lost the most velocity after the collisions?",
                 f"The object {obj_desc} lost the most velocity with a change of {delta_v:.3f} m/s."),
                (f"At time {time} s, what was the velocity change during collision for object {obj_desc}?",
                 f"The object {obj_desc} velocity changed by {delta_v:.3f} m/s at time {time} s.")
            ])

