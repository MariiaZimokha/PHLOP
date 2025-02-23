import json
import matplotlib.colors as mcolors


class QuestionAnswers:
    def __init__(self, file_path):
        self.data = self.read_file(file_path)

    def read_file(self, path):
        with open(path) as f:
            d = json.load(f)
            return d

    def rgba_to_text(self, rgba_str):
        rgba = tuple(map(float, rgba_str.split()))

        # Find the closest named color
        closest_color = None
        min_distance = float("inf")
        for name, hex_value in mcolors.CSS4_COLORS.items():
            rgb = mcolors.to_rgba(hex_value)
            distance = sum((a - b) ** 2 for a, b in zip(rgba, rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = name

        # Describe the opacity
        if rgba[3] == 1.0:
            opacity_text = "fully opaque"
        elif rgba[3] > 0.5:
            opacity_text = "mostly opaque"
        else:
            opacity_text = "mostly transparent"

        return (closest_color, opacity_text)

    def get_questions_answers(self):
        questions_answers = []

        collisions = []
        decelerating_objects = set()
        accelerating_objects = set()

        heaviest_object = None
        max_mass = 0
        highest_friction_object = None
        max_friction = 0
        highest_density_object = None
        max_density = 0

        for frame in self.data.get("frames", []):
            time = frame.get("time", "unknown_time")
            objects = frame.get("objects", {})

            for obj_id, obj in objects.items():
                velocity = obj.get("velocity", None)
                angular_velocity = obj.get("angular_velocity", None)
                position = obj.get("position", [])
                taxonomy = obj.get("taxonomy", [])
                active_labels = obj.get("active_labels", [])

                object_data = [
                    x for x in self.data["objects"] if x.get("id") == obj_id
                ][0]
                mass = object_data.get("mass", "unknown_mass")
                friction = object_data.get("friction", "unknown_friction")
                material = object_data.get("material", "unknown_material")
                elasticity = object_data.get("elasticity", "unknown_elasticity")
                geom_type = object_data.get("geom_type", "unknown_geom_type")
                density = object_data.get("density", "unknown_density")
                visual = object_data.get("visual", "unknown_visual")
                rgba = visual.get("rgba", "")

                (closest_color, opacity_text) = self.rgba_to_text(rgba)
                obj_description = f"{closest_color} {material} {geom_type}"

                # Track the heaviest object
                if mass != "unknown_mass" and float(mass) > max_mass:
                    max_mass = float(mass)
                    heaviest_object = obj_description

                # Track the object with the highest friction
                if friction != "unknown_friction" and float(friction) > max_friction:
                    max_friction = float(friction)
                    highest_friction_object = obj_description

                # Track the object with the highest density
                if density != "unknown_density" and float(density) > max_density:
                    max_density = float(density)
                    highest_density_object = obj_description

                # Track collisions and other taxonomy labels
                for category in taxonomy:
                    for label in category.get("labels", []):
                        if label == "Collision":
                            collisions.append(
                                (obj_id, obj_description, mass, velocity, time)
                            )
                            questions_answers.append(
                                (
                                    f"What happened during the collision involving {obj_description} at time {time}?",
                                    f"{obj_description} was involved in a collision at time {time}.",
                                )
                            )
                        if label == "Elastic Collision":
                            questions_answers.append(
                                (
                                    f"What happened during the elastic collision involving {obj_description} at time {time}?",
                                    f"{obj_description} was involved in an elastic collision at time {time}, conserving both momentum and kinetic energy.",
                                )
                            )
                        if label == "Inelastic Collision":
                            questions_answers.append(
                                (
                                    f"What happened during the inelastic collision involving {obj_description} at time {time}?",
                                    f"{obj_description} was involved in an inelastic collision at time {time}, losing some kinetic energy.",
                                )
                            )
                        if label == "Decelerating":
                            decelerating_objects.add(obj_description)
                        if label == "Accelerating":
                            accelerating_objects.add(obj_description)

        if collisions:
            # Question 1: Which objects were involved in collisions?
            colliding_objects = set([obj_desc for _, obj_desc, _, _, _ in collisions])
            questions_answers.append(
                (
                    "Which objects were involved in collisions during the simulation?",
                    f"The following objects were involved in collisions: {', '.join(colliding_objects)}.",
                )
            )

            # Question 2: Which object lost the most velocity after collisions?
            if len(collisions) >= 2:
                obj1_id, obj1_desc, obj1_mass, obj1_velocity, _ = collisions[0]
                obj2_id, obj2_desc, obj2_mass, obj2_velocity, _ = collisions[1]
                questions_answers.append(
                    (
                        "After the collisions, which object lost the most velocity?",
                        f"The object with lower mass ({obj1_desc if float(obj1_mass) < float(obj2_mass) else obj2_desc}) lost more velocity due to momentum conservation.",
                    )
                )

        if decelerating_objects:
            # Question 3: Which objects decelerated during the simulation?
            questions_answers.append(
                (
                    "Which objects decelerated during the simulation?",
                    f"The following objects decelerated: {', '.join(decelerating_objects)}.",
                )
            )

        if accelerating_objects:
            # Question 4: Which objects accelerated during the simulation?
            questions_answers.append(
                (
                    "Which objects accelerated during the simulation?",
                    f"The following objects accelerated: {', '.join(accelerating_objects)}.",
                )
            )

        if heaviest_object:
            # Question 5: What was the heaviest object in the simulation?
            questions_answers.append(
                (
                    "What was the heaviest object in the simulation?",
                    f"The heaviest object was {heaviest_object} with a mass of {max_mass}.",
                )
            )

        if highest_friction_object:
            # Question 6: Which object had the highest friction?
            questions_answers.append(
                (
                    "Which object had the highest friction?",
                    f"The object with the highest friction was {highest_friction_object} with a friction value of {max_friction}.",
                )
            )

        if highest_density_object:
            # Question 7: Which object had the highest density?
            questions_answers.append(
                (
                    "Which object had the highest density?",
                    f"The object with the highest density was {highest_density_object} with a density of {max_density}.",
                )
            )

        return questions_answers
