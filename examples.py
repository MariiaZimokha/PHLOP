import os
import random
import numpy as np
from dataset.world.object import Object
from dataset.annotator import Annotator
from dataset.video_annotation_visualizer import VideoAnnotationVisualizer
from dataset.simulator import Simulation
from dataset.utils import save_file, set_position_and_velocity, set_physics_properties
from dataset.video_qa import add_qa_to_video
from dataset.question_answer import QuestionAnswers

# Initialize necessary objects
obj = Object()
annotator = Annotator()
video_annotator = VideoAnnotationVisualizer()

# Simulation settings
width = 1600
height = 912
width = 1024
height = 768
sim = Simulation(obj, annotator=annotator, width=width, height=height)

# Function to generate objects based on improved physics demonstration


def generate_obj_json_like(shape,
                           material,
                           mode,
                           velocity,
                           angular_velocity,
                           elasticity,
                           friction,
                           x_position=None,
                           y_position=None):
    obj_data = obj.get_object(shape=shape, material=material)
    obj_data["mode"] = mode if mode else "stationary"
    obj_data = set_physics_properties(obj_data)
    obj_data = set_position_and_velocity(obj_data)
    obj_data["velocity"] = velocity
    obj_data["angular_velocity"] = angular_velocity
    obj_data["elasticity"] = elasticity
    obj_data["friction"] = friction
    if x_position:
        print(obj_data["init_possition_x"])
        obj_data["init_possition_x"] = x_position
        print(obj_data["init_possition_x"])
    if y_position:
        obj_data["init_possition_y"] = y_position
    return obj_data


def generate_constant_velocity_objects():
    constant_vel = [-0.30071041457826514, 1.212283498931664, -0.06523486597772561]
    constant_ang_vel = [-3.334549536739062, 1.2509361202748748, -4.838985965705394]
    return [
        generate_obj_json_like("cube", "glass", "sliding", [2.0, 2.0, 2.0], [0.0, 0.0, 0.0], 0.9, "0.0"),
        generate_obj_json_like(
            "ball", "glass", "sliding", constant_vel, constant_ang_vel, 0.661, "0.24", 0.6668, -2.6882
        ),
        generate_obj_json_like("ball", "rubber", "sliding",
                               [0.3, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               0.4,
                               "0.5",
                               -0.8525, 0.4002),

    ]


def generate_stationary_objects():
    return [
        generate_obj_json_like("cube", "glass", "stationary", [0, 0, 0], [0, 0, 0], 0.1, 0.6, 3.0, 0),
        generate_obj_json_like("sphere", "glass", "stationary", [0, 0, 0], [0, 0, 0], 0.05, 0.4, 3.0, 2),
        generate_obj_json_like("cylinder", "metal", "stationary", [0, 0, 0], [0, 0, 0], 0.2, 0.5, 0, 0)
    ]


def generate_pure_rotation_objects():
    return [
        generate_obj_json_like("ball", "metal", "sliding", [0, 0, 0], [0.0, 0.0, 20.0], 0.9, 0.0, -1.4587, -0.6467),
        generate_obj_json_like("ball", "wood", "sliding", [0, 0, 0], [0, 0, 10], 0.8, 0.0, -0.5261, 0.8233)
    ]


def generate_rolling_motion_objects():
    return [
        generate_obj_json_like("ball", "metal", "sliding", [0, 0, 0], [20.0, 20.0, 20.0], 0.9, 0.0, 1.4587, -0.6467),
        generate_obj_json_like("ball", "glass", "sliding", [0, 0, 0], [0, 10.5, 0], 0.8, 0.0, -0.5261, 0.8233)
    ]


def generate_decelerating_objects():
    return [
        generate_obj_json_like("cube", "metal", "sliding", [3.5, 0, 0], [0, 0, 0], 0.1, 0.6, -0.5261, 0.8233),
        generate_obj_json_like("cube", "glass", "sliding", [-3.5, 0, 0], [0, 0, 0], 0.1, 0.6,  1.4587, -0.6467)
    ]


def generate_inelastic_collision_objects():
    return [
        generate_obj_json_like(
            shape="ball",
            material="plastic",
            mode="collision",
            velocity=[
                0.16871682536676982,
                1.2085505065767836,
                0.01039912189402277
            ],
            angular_velocity=[
                2.4002006700429055,
                -4.408637078310188,
                3.322123085408281
            ],
            elasticity=0.1,
            friction=0.3,
            x_position=-0.2113,
            y_position=-1.5136
        ),
        generate_obj_json_like(
            shape="ball",
            material="rubber",
            mode="collision",
            velocity=[
                1.1757714900812308,
                -0.021501278922900633,
                -0.08831802324698271
            ],
            angular_velocity=[
                2.687100087471629,
                -1.5428969058136266,
                -3.6053503279078503
            ],
            elasticity=0.1,
            friction=0.6,
            x_position=-1.6099,
            y_position=0.0294
        ),
    ]


def generate_elastic_collision_objects():
    return [
        generate_obj_json_like(
            shape="ball",
            material="glass",
            mode="collision",
            velocity=[
                -2.715869033823274,
                -2.773728290865599,
                0.1953383072039434
            ],
            angular_velocity=[
                0.5614961036775568,
                0.4810220863413195,
                1.0239316560435674
            ],
            elasticity=0.537,
            friction=0.16,
            x_position=1.0161568949251112,
            y_position=1.0378052447706247
        ),
        generate_obj_json_like(
            shape="ball",
            material="glass",
            mode="collision",
            velocity=[
                0.0,
                0.0,
                0.0
            ],
            angular_velocity=[
                0.8815920380476594,
                0.6845507292370079,
                2.7108616527921834
            ],
            elasticity=0.677,
            friction=0.19,
            x_position=-0.9870242114298601,
            y_position=-0.9776275321723126
        ),
    ]


def generate_stationary_to_moving_objects():
    return [
        generate_obj_json_like(
            shape="ball",
            material="glass",
            mode="collision",
            velocity=[
                -2.715869033823274,
                -2.773728290865599,
                0.1953383072039434
            ],
            angular_velocity=[
                0.5614961036775568,
                0.4810220863413195,
                1.0239316560435674
            ],
            elasticity=0.537,
            friction=0.16,
            x_position=1.0161568949251112,
            y_position=1.0378052447706247
        ),
        generate_obj_json_like(
            shape="ball",
            material="glass",
            mode="collision",
            velocity=[
                0.0,
                0.0,
                0.0
            ],
            angular_velocity=[
                0.0,
                0.0,
                0.0
            ],
            elasticity=0.677,
            friction=0.19,
            x_position=-0.9870242114298601,
            y_position=-0.9776275321723126
        ),
    ]


def generate_moving_to_stopping_objects():
    return [
        generate_obj_json_like("sphere", "rubber", "sliding", [6.0, 0, 0], [0, 0, 0], 0.4, 0.85),
        generate_obj_json_like("cube", "cork", "sliding", [5.5, 0, 0], [0, 0, 0], 0.3, 0.75)
    ]


def generate_sliding_with_friction():
    return [
        generate_obj_json_like("cube", "metal", "sliding", [4.5, 0, 0], [0, 0, 0], 0.1, 0.6, -0.5261, 0.8233),
        generate_obj_json_like("cube", "glass", "sliding", [-4.5, 0, 0], [0, 0, 0], 0.1, 0.6,  1.4587, -0.6467)
    ]


event_labels = {
    "Constant Velocity": generate_constant_velocity_objects,
    "Stationary": generate_stationary_objects,
    "Rolling Motion": generate_rolling_motion_objects,
    "Pure Rotation": generate_pure_rotation_objects,
    "Decelerating": generate_decelerating_objects,  # friction stop
    "Inelastic Collision": generate_inelastic_collision_objects,

    # # not done yet
    # # "Elastic Collision": generate_elastic_collision_objects,

    "Stationary to Moving": generate_stationary_to_moving_objects,
    "Moving to Stopping": generate_decelerating_objects,
    "Friction Stop": generate_decelerating_objects,
    "Sliding with Friction": generate_sliding_with_friction
}


for event_label, generate_objects in event_labels.items():
    objects = generate_objects()
    path = f"generated/{event_label.replace(' ', '_')}/"
    os.makedirs(path, exist_ok=True)
    print("objects", objects)
    print(f"Running simulation for: {event_label}")
    out = sim.run_simulation(objects=objects, duration=5, path=path)

    video_annotator.annotate(
        file_path=out["file_path"],
        video_path=out["video_file"],
        annotated_video_path=f"{path}output_video.mp4"
    )

    q = QuestionAnswers(out["file_path"])
    questions_answers = q.get_questions_answers()
    save_file(f"{path}questions_answers.json", questions_answers)
    add_qa_to_video(questions_answers, out["file_path"], path, num_objects=len(objects))

    print(f"Completed: {event_label}")

print("All simulations and annotations are done!")
