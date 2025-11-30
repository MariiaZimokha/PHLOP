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
# Training: Standard frontal views
TRAIN_CAM_AZIMUTH = (-30.0, 30.0)
TRAIN_CAM_ELEVATION = (-30.0, -10.0)
TRAIN_CAM_DISTANCE = (2.0, 3.0)

# Validation: Slightly shifted, lower angles
VAL_CAM_AZIMUTH = (35.0, 45.0)
VAL_CAM_ELEVATION = (-10.0, 0.0)
VAL_CAM_DISTANCE = (2.0, 3.5)

# Test: Extreme angles (OOD generalization)
TEST_CAM_AZIMUTH = (45.0, 75.0)
TEST_CAM_ELEVATION = (5.0, 20.0)
TEST_CAM_DISTANCE = (2.5, 4.0)
