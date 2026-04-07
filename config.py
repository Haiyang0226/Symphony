import os

# ------------------ video download and segmentation configuration ------------------ #
VIDEO_DATABASE_FOLDER = "./video_database/"
VIDEO_RESOLUTION = "360" # denotes the height of the video 
VIDEO_FPS = 2 # frames per second
CLIP_SECS = 10 # seconds

# ------------------ model configuration ------------------ #
# API Keys - 请在此处设置您的API密钥
ALi_API_KEY = ""  # 阿里云API密钥 for deepseek
HuoShan_API_KEY = ""  # 火山引擎API密钥 for doubao_vl

# 模型配置 - 注意：model_name和base_url已在api.py的函数中写死

AOAI_TOOL_VLM_MAX_FRAME_NUM = 30

# ------------------ 数据集路径配置 ------------------ #
# 根据实际情况修改这些路径
LONGVIDEO_DATA_PATH = ".../videomme/test-00000-of-00001.json"
VIDEOMME_DATA_PATH = "..."
MLVU_DATA_PATH = "..."
LVBENCH_DATA_PATH = "..."

# ------------------ languagebind_image模型权重路径 ------------------ #
LANGUAGEBIND_MODEL_PATH = ""

# ------------------ agent and tool setting ------------------ #
OVERWRITE_CLIP_SEARCH_TOPK = 0 # 0 means no overwrite and let agent decide

SINGLE_CHOICE_QA = True  # Design for benchmark test. If True, the agent will only return options for single-choice questions.
MAX_ITERATIONS = 5  # Maximum number of iterations for the agent to run