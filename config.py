import os

# ------------------ video download and segmentation configuration ------------------ #
VIDEO_DATABASE_FOLDER = "./video_database/"
VIDEO_RESOLUTION = "360" # denotes the height of the video 
VIDEO_FPS = 2 # frames per second
CLIP_SECS = 10 # seconds

# ------------------ model configuration ------------------ #
# API Keys - Please set your API keys here
ALi_API_KEY = ""  # Alibaba Cloud API key for deepseek
HuoShan_API_KEY = ""  # Volcano Engine API key for doubao_vl

# Model configuration - Note: model_name and base_url are hardcoded in api.py functions

AOAI_TOOL_VLM_MAX_FRAME_NUM = 30

# ------------------ Dataset path configuration ------------------ #
# Modify these paths according to your actual situation
LONGVIDEO_DATA_PATH = ".../videomme/test-00000-of-00001.json"
VIDEOMME_DATA_PATH = "..."
MLVU_DATA_PATH = "..."
LVBENCH_DATA_PATH = "..."

# ------------------ languagebind_image model weights path ------------------ #
LANGUAGEBIND_MODEL_PATH = ""

# ------------------ agent and tool setting ------------------ #
OVERWRITE_CLIP_SEARCH_TOPK = 0 # 0 means no overwrite and let agent decide

SINGLE_CHOICE_QA = True  # Design for benchmark test. If True, the agent will only return options for single-choice questions.
MAX_ITERATIONS = 5  # Maximum number of iterations for the agent to run