from pathlib import Path
import os

HF_REPO = "zimmari-ai/phlop"
HF_TOKEN = os.environ.get("HF_TOKEN")

# TOTAL_VIDEOS =10
# TRAIN_COUNT = 10000
TRAIN_COUNT = 30
# VAL_COUNT = 2000
# TEST_COUNT = 2000
VAL_COUNT = 10
TEST_COUNT = 10
# TOTAL_VIDEOS = 10000

SHARD_SIZE = 10  # Upload every 50 videos
OUTPUT_DIR = Path("phlop_generation_buffer")

WIDTH, HEIGHT = 1024, 768

# WIDTH = 512
# HEIGHT = 384
FPS = 25
