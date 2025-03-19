import json
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Tuple, Dict, Set


class QuestionAnswers:
    def __init__(self, file_path: str):
        self.data = self.read_file(file_path)
        self.taxonomy_labels = self._extract_all_taxonomy_labels()
        self.object_states = self._calculate_object_states()

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

    def _extract_all_taxonomy_labels(self) -> Set[str]:
        labels = set()
        for frame in self.data.get("frames", []):
            for obj_id, obj in frame.get("objects", {}).items():
                for item in obj.get("taxonomy", []):
                    labels.update(item.get("labels", []))
        return labels

    def _calculate_object_states(self) -> Dict[str, Dict[str, float]]:
        object_states = {}
        prev_time = 0.0

        for frame in self.data.get("frames", []):
            current_time = frame.get("time", 0.0)
            time_delta = current_time - prev_time
            prev_time = current_time

            for obj_id, obj in frame.get("objects", {}).items():
                if obj_id not in object_states:
                    object_states[obj_id] = {
                        "accelerating": 0.0,
                        "decelerating": 0.0,
                        "stationary": 0.0,
                        # "tipped": 0.0,
                        "sliding": 0.0,
                        "collision": 0.0,
                    }

                for item in obj.get("taxonomy", []):
                    for label in item.get("labels", []):
                        if label == "Accelerating":
                            object_states[obj_id]["accelerating"] += time_delta
                        elif label == "Decelerating":
                            object_states[obj_id]["decelerating"] += time_delta
                        elif label == "Stationary":
                            object_states[obj_id]["stationary"] += time_delta
                        # elif label == "Tipped":
                        #     object_states[obj_id]["tipped"] += time_delta
                        elif label == "Sliding":
                            object_states[obj_id]["sliding"] += time_delta
                        elif label == "Collision":
                            object_states[obj_id]["collision"] += time_delta

        return object_states

    def _compare_object_states(self) -> List[Tuple[str, str]]:
        """Generate questions comparing the time objects spent in different states."""
        questions_answers = []

        # the total time of the simulation
        total_time = self.data.get("frames", [])[-1].get("time", 0.0)

        # compare time spent in different states
        for obj_id, states in self.object_states.items():
            obj_description = self._get_object_description(obj_id)
            if not obj_description:
                continue

            # compare time spent in each state
            for state, time_spent in states.items():
                percentage = (time_spent / total_time) * 100
                questions_answers.append(
                    (f"What percentage of the simulation did {obj_description} spend {state}?",
                     f"{obj_description} spent {round(percentage, 2)}% of the simulation {state}.")
                )

            # compare relative time spent in different states
            if states["accelerating"] > states["decelerating"]:
                questions_answers.append(
                    (f"Did {obj_description} spend more time accelerating or decelerating?",
                     f"{obj_description} spent more time accelerating than decelerating.")
                )
            elif states["decelerating"] > states["accelerating"]:
                questions_answers.append(
                    (f"Did {obj_description} spend more time accelerating or decelerating?",
                     f"{obj_description} spent more time decelerating than accelerating.")
                )

            if states["stationary"] > 0:
                questions_answers.append(
                    (f"Did {obj_description} spend any time stationary?",
                     f"Yes, {obj_description} spent {round(states['stationary'], 2)} seconds stationary.")
                )
            else:
                questions_answers.append(
                    (f"Did {obj_description} spend any time stationary?",
                     f"No, {obj_description} did not spend any time stationary.")
                )

        return questions_answers

    def _get_object_description(self, obj_id: str) -> str:
        """Get a description of the object based on its ID."""
        for obj in self.data.get("objects", []):
            if obj.get("id") == obj_id:
                closest_color, _ = self.rgba_to_text(obj.get("visual", {}).get("rgba", ""))
                return f"{closest_color} {obj.get('material', 'unknown_material')} {obj.get('geom_type', 'unknown_geom_type')}"
        return ""

    def get_questions_answers(self) -> List[Tuple[str, str]]:
        questions_answers = []
        collisions = []
        decelerating_objects = set()
        accelerating_objects = set()
        taxonomy_events = {}

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
                friction = obj.get("friction", '0 0 0')
                friction = [float(f) for f in friction.split(' ')]
                friction = round(friction[0], 3)
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
                    category = item.get("category", "")
                    subcategory = item.get("subcategory", "")
                    for label in item.get("labels", []):
                        if (category, subcategory, label) not in taxonomy_events:
                            taxonomy_events[(category, subcategory, label)] = set()
                        taxonomy_events[(category, subcategory, label)].add(obj_description)

                        if subcategory == "Collision":
                            questions_answers.extend(self._process_collision(
                                frame_idx, obj_id, obj_description, mass, velocity, collisions))

                        if label == "Decelerating":
                            decelerating_objects.add(obj_description)
                        elif label == "Accelerating":
                            accelerating_objects.add(obj_description)

        questions_answers.extend(self._add_general_questions(collisions, decelerating_objects, accelerating_objects,
                                                             heaviest_object, max_mass, highest_friction_object, max_friction, highest_density_object, max_density, taxonomy_events))

        # add questions comparing object states
        questions_answers.extend(self._compare_object_states())

        # add taxonomy-based questions
        questions_answers.extend(self._add_taxonomy_based_questions(taxonomy_events))

        # add what-if type of questions
        questions_answers.extend(self._add_what_if_questions(objects_data, collisions))

        return questions_answers

    def _process_collision(self, frame_idx: int, obj_id: str, obj_description: str, mass: float, velocity: List[float],
                           collisions: List):
        questions_answers = []
        prev_frame = self.data.get("frames")[frame_idx - 1].get("objects", {})
        previous_velocity = prev_frame.get(obj_id, {}).get("velocity", [])

        if previous_velocity and velocity:
            current_velocity = np.array(velocity)
            previous_velocity = np.array(previous_velocity)
            delta_v = np.linalg.norm(current_velocity - previous_velocity)

            collisions.append((obj_id, obj_description, mass, previous_velocity, velocity, delta_v))

            # Add kinetic energy question
            questions_answers.append(
                (f"What was the kinetic energy of the {obj_description} before the collision?",
                 f"The kinetic energy of the {obj_description} before the collision was {self.calculate_kinetic_energy(mass, previous_velocity)} J (Joules).")
            )

            # Add collision type questions
            for item in self.data.get("frames")[frame_idx].get("objects", {}).get(obj_id, {}).get("taxonomy", []):
                if item.get("subcategory") == "Collision":
                    for label in item.get("labels", []):
                        if label == "Elastic Collision":
                            questions_answers.append(
                                (f"What happened during the elastic collision involving {obj_description}?",
                                 f"{obj_description} conserved both momentum and kinetic energy.")
                            )
                        elif label == "Inelastic Collision":
                            questions_answers.append(
                                (f"What happened during the inelastic collision involving {obj_description}?",
                                 f"{obj_description} lost some kinetic energy.")
                            )
        return questions_answers

    def _add_general_questions(self, collisions: List,
                               decelerating_objects: Set[str], accelerating_objects: Set[str],
                               heaviest_object: str, max_mass: float, highest_friction_object: str, max_friction: float,
                               highest_density_object: str, max_density: float, taxonomy_events: Dict):
        questions_answers = []
        if collisions:
            questions_answers.extend(self._add_collision_questions(collisions))

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
        return questions_answers

    def _add_collision_questions(self, collisions: List):
        colliding_objects = {obj_desc for _, obj_desc, _, _, _, _ in collisions}
        questions_answers = []
        questions_answers.extend([
            ("How many objects were involved in collisions?",
             f"{len(colliding_objects)} objects were involved in collisions."),
            ("Which objects were involved in collisions during the simulation?",
             f"The following objects were involved in collisions: {', '.join(colliding_objects)}.")
        ])

        if collisions:
            most_velocity_loss_collision = max(collisions, key=lambda x: x[-1])
            obj_id, obj_desc, mass, prev_velocity, velocity, delta_v = most_velocity_loss_collision

            questions_answers.extend([
                ("Which object lost the most velocity after the collisions?",
                 f"The object {obj_desc} lost the most velocity with a change of {delta_v:.3f} m/s."),
            ])

        return questions_answers

    def _add_taxonomy_based_questions(self, taxonomy_events: Dict) -> List[Tuple[str, str]]:
        """Generate questions based on the extracted taxonomy events (category, subcategory, and labels)."""
        questions_answers = []
        for (category, subcategory, label), objects in taxonomy_events.items():
            objects_str = ", ".join(objects)
            questions_answers.append(
                (f"Which objects experienced the {category} event '{label}' under the subcategory '{subcategory}'?",
                 f"The following objects experienced the {category} event '{label}' under the subcategory '{subcategory}': {objects_str}.")
            )
        return questions_answers

    def _add_what_if_questions(self, objects_data: Dict, collisions: List) -> List[Tuple[str, str]]:
        questions_answers = []

        if collisions:
            obj_id, obj_desc, mass, prev_velocity, velocity, delta_v = collisions[0]

            # mass doubled
            questions_answers.append(
                (f"What if the mass of {obj_desc} was doubled?",
                 f"If the mass of {obj_desc} was doubled, its kinetic energy before the collision would be {self.calculate_kinetic_energy(2 * mass, prev_velocity)} J, and the collision impact would be greater.")
            )

            # velocity halved
            questions_answers.append(
                (f"What if the velocity of {obj_desc} was halved before the collision?",
                 f"If the velocity of {obj_desc} was halved, its kinetic energy before the collision would be {self.calculate_kinetic_energy(mass, [v / 2 for v in prev_velocity])} J, and the collision impact would be significantly reduced.")
            )

            # elastic collision turned inelastic
            questions_answers.append(
                (f"What if the collision involving {obj_desc} was inelastic instead of elastic?",
                 f"If the collision was inelastic, {obj_desc} would lose some kinetic energy, and the objects might stick together after the collision.")
            )

        if objects_data:
            obj_id = next(iter(objects_data))
            obj_data = objects_data[obj_id]
            obj_desc = f"{self.rgba_to_text(obj_data.get('visual', {}).get('rgba', ''))[0]} {obj_data.get('material', 'unknown_material')} {obj_data.get('geom_type', 'unknown_geom_type')}"

            # friction increased by 50%
            questions_answers.append(
                (f"What if the friction of {obj_desc} was increased by 50%?",
                 f"If the friction of {obj_desc} was increased by 50%, it would decelerate faster and potentially reduce the distance it travels.")
            )

            # density doubled
            questions_answers.append(
                (f"What if the density of {obj_desc} was doubled?",
                 f"If the density of {obj_desc} was doubled, its mass would increase proportionally (assuming volume remains constant), making it harder to accelerate or decelerate.")
            )

            # material changed to a more elastic one
            questions_answers.append(
                (f"What if {obj_desc} was made of a more elastic material?",
                 f"If {obj_desc} was made of a more elastic material, it would rebound more after collisions, conserving more kinetic energy.")
            )

            # gravity was doubled
            questions_answers.append(
                (f"What if gravity was doubled in the simulation?",
                 f"If gravity was doubled, all objects would accelerate faster toward the ground, and collisions would likely be more forceful.")
            )

            # air resistance was introduced
            questions_answers.append(
                (f"What if air resistance was introduced in the simulation?",
                 f"If air resistance was introduced, lighter objects would decelerate more quickly, and their trajectories would be significantly affected.")
            )

        return questions_answers
