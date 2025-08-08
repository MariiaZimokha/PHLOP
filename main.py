import argparse
import os
import random
from tqdm import tqdm


from dataset.question_answer import QuestionAnswers
from dataset.world.object import Object
from dataset.annotator import Annotator
from dataset.video_annotation_visualizer import VideoAnnotationVisualizer
from dataset.simulator import Simulation
from dataset.utils import save_file
from dataset.video_qa import add_qa_to_video
from tqdm import tqdm


def generate_dataset(output_dir: str,
                     num_videos: int,
                     duration: int = 15,
                     seed: int = 0) -> None:
    if seed:
        random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    width, height = 1024, 768
    obj = Object()
    annotator = Annotator()
    video_annotator = VideoAnnotationVisualizer()
    sim = Simulation(obj, annotator=annotator, width=width, height=height)

    for i in tqdm(range(num_videos), desc="Generating simulations"):
        num_objects = random.randrange(2, 8)

        scene_dir = os.path.join(output_dir, f"{i}/")
        os.makedirs(scene_dir, exist_ok=True)
        print('scene_dir ', scene_dir)

        sim_out = sim.run_simulation(
            num_objects=num_objects,
            duration=duration,  # seconds
            path=scene_dir,
        )
        print('sim_out,', sim_out)
        video_file = sim_out["video_file"]
        file_path = sim_out["file_path"]

        video_annotator.annotate(
            file_path=file_path,
            video_path=video_file,
            annotated_video_path=os.path.join(scene_dir, "output_video.mp4"),
        )

        qa_generator = QuestionAnswers(file_path)
        questions_answers = qa_generator.get_questions_answers()
        qa_file = os.path.join(scene_dir, "questions_answers.json")
        save_file(qa_file, questions_answers)

        # add_qa_to_video(questions_answers, file_path, scene_dir, num_objects)


def parse_args() -> argparse.Namespace:
    """Parse command–line arguments."""
    parser = argparse.ArgumentParser(description="Generate the PHLOP dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory in which to store generated simulations.",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=50,
        help="Number of simulation videos to generate (default: 50)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help=(
            "Video duration is seconds (default: 15)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed for reproducibility (default: 0 means no fixed seed)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        args.output_dir,
        args.num_videos,
        args.duration,
        args.seed)


if __name__ == "__main__":
    main()
