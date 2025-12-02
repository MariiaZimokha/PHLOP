from pathlib import Path
import os

HF_REPO = "zimmari-ai/phlop"
HF_TOKEN = os.environ.get("HF_TOKEN")
# TOTAL_VIDEOS =10
TRAIN_COUNT = 10000
VAL_COUNT = 2000
TEST_COUNT = 2000
# TOTAL_VIDEOS = 10000

SHARD_SIZE = 10  # Upload every 50 videos
OUTPUT_DIR = Path("phlop_generation_buffer")


WIDTH = 512
HEIGHT = 384
FPS = 25

# --- CAMERA CONSTANTS ---
# --- CAMERA CONSTANTS ---

# Training: Close, intimate views, standard angles
# Distance reduced to 1.8m - 3.5m (Objects fill the screen)
TRAIN_CAM_AZIMUTH = (-60.0, 60.0)
TRAIN_CAM_ELEVATION = (-40.0, -10.0)  # Look DOWN at objects
TRAIN_CAM_DISTANCE = (1.8, 3.5)  # MUCH CLOSER

# Validation: Side views, slightly zoomed out
# Distance: 3.0m - 4.5m
VAL_CAM_AZIMUTH = (70.0, 110.0)
VAL_CAM_ELEVATION = (-50.0, -20.0)
VAL_CAM_DISTANCE = (3.0, 4.5)

# Test: Back views, Extreme Closeups OR Extreme Far
# Distance: 1.5m (Macro) to 5.5m (Far)
TEST_CAM_AZIMUTH = (130.0, 170.0)
TEST_CAM_ELEVATION = (-60.0, -5.0)
TEST_CAM_DISTANCE = (1.5, 5.5)
