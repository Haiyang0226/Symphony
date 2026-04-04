import os

# ------------------ video download and segmentation configuration ------------------ #
VIDEO_DATABASE_FOLDER = "./video_database/"
VIDEO_RESOLUTION = "360" # denotes the height of the video 
VIDEO_FPS = 2 # frames per second
CLIP_SECS = 10 # seconds

# ------------------ model configuration ------------------ #
ALi_API_KEY = ""
HuoShan_API_KEY = ""

AOAI_CAPTION_VLM_ENDPOINT_LIST = [""]
AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST = [""]

AOAI_ORCHESTRATOR_LLM_MODEL_NAME = "deepseek-r1"
AOAI_TOOL_VLM_MODEL_NAME = "doubao-seed-1-6-vision-250815"

AOAI_TOOL_VLM_MAX_FRAME_NUM = 30

AOAI_EMBEDDING_RESOURCE_LIST = [""]
AOAI_EMBEDDING_LARGE_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
AOAI_EMBEDDING_LARGE_DIM = 2560

# ------------------ agent and tool setting ------------------ #
OVERWRITE_CLIP_SEARCH_TOPK = 0 # 0 means no overwrite and let agent decide

SINGLE_CHOICE_QA = True  # Design for benchmark test. If True, the agent will only return options for single-choice questions.
MAX_ITERATIONS = 5  # Maximum number of iterations for the agent to run