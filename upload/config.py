from pathlib import Path
import os

HF_REPO = "zimmari-ai/phlop"
HF_TOKEN = os.environ.get("HF_TOKEN")

# TOTAL_VIDEOS =10
TRAIN_COUNT = 16000
# TRAIN_COUNT = 100
VAL_COUNT = 2000
TEST_COUNT = 2000
# VAL_COUNT = 100
# TEST_COUNT = 100
# TOTAL_VIDEOS = 10000

SHARD_SIZE = 50  # Upload every 50 videos
OUTPUT_DIR = Path("phlop_generation_buffer")

WIDTH, HEIGHT = 1024, 768

# WIDTH = 512
# HEIGHT = 384
FPS = 25
