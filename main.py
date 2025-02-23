from dataset.object import Object
from dataset.annotator import Annotator
from dataset.video_annotation_visualizer import VideoAnnotationVisualizer
from dataset.simulator import Simulation

import random
import os

obj = Object()
annotator = Annotator()
video_annotator = VideoAnnotationVisualizer()
sim = Simulation(obj, annotator=annotator)

num_objects = random.randrange(2, 15)
path = f"generated/{num_objects}/"

if not os.path.exists(path):
    os.makedirs(path)

out = sim.run_simulation(num_objects=num_objects, duration=5, path=path)
video_file, file_path = out["video_file"], out["file_path"]
video_annotator.annotate(
    file_path=file_path,
    video_path=video_file,
    annotated_video_path=f"{path}output_video_{num_objects}.mp4",
)
