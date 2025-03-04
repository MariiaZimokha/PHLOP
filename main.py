from dataset.question_answer import QuestionAnswers
from dataset.object import Object
from dataset.annotator import Annotator
from dataset.video_annotation_visualizer import VideoAnnotationVisualizer
from dataset.simulator import Simulation
from dataset.utils import save_file
from dataset.video_qa import add_qa_to_video
from tqdm import tqdm

import random
import os

obj = Object()
annotator = Annotator()
video_annotator = VideoAnnotationVisualizer()
width = 1600
height = 912
sim = Simulation(obj, annotator=annotator, width=width, height=height)

for i in tqdm(range(20)):
    num_objects = random.randrange(2, 15)
    path = f"generated/{i}/"

    if not os.path.exists(path):
        os.makedirs(path)

    out = sim.run_simulation(num_objects=num_objects, duration=5, path=path)
    video_file, file_path = out["video_file"], out["file_path"]
    video_annotator.annotate(
        file_path=file_path,
        video_path=video_file,
        annotated_video_path=f"{path}output_video.mp4",
    )

    q = QuestionAnswers(file_path)
    questions_answers = q.get_questions_answers()
    qa_file = f"{path}questions_answers.json"
    save_file(qa_file, questions_answers)

    add_qa_to_video(questions_answers, file_path, path, num_objects)
