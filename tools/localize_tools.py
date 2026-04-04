import os
import math
import json
from typing import List, Dict, Any
from tqdm import tqdm
import config as config
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Annotated as A
from tools.func_call_shema import doc as D
from api import call_seed_vl_with_tools_wanqin as call_openai_model_with_tools #seed
from concurrent.futures import ThreadPoolExecutor
import time

# 系统提示语
SYSTEM_PROMPT = "You are a helpful assistant skilled in video understanding and question analysis."

# 用户提示语模板


# JUDGEMENT_PROMPT = """
# You are given a sequence of video frames sampled from a 1-minute video clip. 
# User Question: USER_QUESTION

# Your task is to:
# 1. Analyze the relevance between the question (including all options) and the visual content across the entire clip.
# 2. Output a global relevance score and description.

# Please output your analysis in the following JSON format:

# {
#     "relevance_score": integer,  // Relevance score from 1 to 4
#     "clip_caption": "string",     // Concise description of main people (with distinguishing features), key events, actions, and relationships. Focus on elements related to the question.
#     "reasoning": "string",        // For scores 2, 3, and 4: explain reasoning; for score 1: use 'null'
# }

# ### Instructions for clip_caption:
# - Describe **main people, objects, events, actions, and their relationships** that are visually confirmable.
# - If the question involves specific individuals or objects (especially in options), compare them clearly by appearance, position, or behavior to avoid confusion.
# - If there are multiple scenarios, describe them respectively. Pay attention to the sequence of events!
# - Only describe what is **directly observable**. Do **not** infer, imagine, or fabricate scenes beyond the visual evidence.

# ### Scoring Criteria:
# 4 points: Key elements of the question and options are clearly visible, sufficient to directly answer the question.
# 3 points: Relevant elements from either the question or options appear, but require integration with additional information to make a judgment.
# 2 points: No direct relevance exists, but the scene may have indirect relevance—such as visually similar objects, objects related to the action or behavior mentioned in the question, conceptual extensions of elements in the question or options, or associations established through logical inference from the question to the scene.
# 1 point: Completely unrelated scene.

# ### Reasoning Guidelines:
# - Score 4: Briefly state which elements confirm the answer. Output the answer.
# - Score 3: Explain what is missing or ambiguous (e.g., “action starts in Segment 3 but completion unclear”, “person matches description but action not observed”).
# - Score 2: Explain how you decomposed or extended the question (e.g., “question asks about ‘a musician’, and a person holding a guitar appears”).
# - Score 1: Set reasoning to 'null'.

# Be thorough, precise, and strictly grounded in visual evidence. Avoid temporal phrases like 'the first time'.
# """



JUDGEMENT_PROMPT = """
You are given a sequence of video frames sampled from a 1-minute video clip. 
User Question: {USER_QUESTION}

Your task is to:
1. Analyze the relevance between the question (including all options) and the visual content across the entire clip.
2. Output a global relevance score and description.

Please output your analysis in the following JSON format:

{
    "relevance_score": integer,  // Relevance score from 1 to 4
    "clip_caption": "string",     // Concise description of main people (with distinguishing features), key events, actions, and relationships. Focus on elements related to the question.
    "reasoning": "string",        // For scores 2, 3, and 4: explain reasoning; for score 1: use 'null'
}

### Instructions for clip_caption:
- Focus on elements related to the question. Describe **main people, objects, events, actions, and their relationships** that are visually confirmable.
- If there are multiple scenarios, describe them respectively. Pay attention to the sequence of events!
- Only describe what is **directly observable**. Do **not** infer, imagine, or fabricate scenes beyond the visual evidence.
- If the question is about counting (e.g. 'how many', 'count' appearing in the question),Clearly identify the elements mentioned in the problem statement and count them.

### Scoring Criteria:
4 points: Key elements of the question and options are clearly visible, sufficient to directly answer the question.
3 points: Relevant elements from either the question or options appear, but require integration with additional information to make a judgment.
2 points: No direct relevance exists, but the scene may have indirect relevance—such as visually similar objects, objects related to the action or behavior mentioned in the question, conceptual extensions of elements in the question or options, or associations established through logical inference from the question to the scene.
1 point: Completely unrelated scene.

### Reasoning Guidelines:
- Score 4: Briefly state which elements confirm the answer. Output the answer.
- Score 3: Explain what is missing or ambiguous (e.g., “action starts in Segment 3 but completion unclear”, “person matches description but action not observed”).
- Score 2: Explain how you decomposed or extended the question (e.g., “question asks about ‘a musician’, and a person holding a guitar appears”).
- Score 1: Set reasoning to 'null'.

Be thorough, precise, and strictly grounded in visual evidence. Avoid temporal phrases like 'the first time'.
"""


# JUDGEMENT_PROMPT = """
# You are given a sequence of video frames sampled from a 1-minute video clip. This segment is extracted from a longer video, specifically from START_TIME to END_TIME mark of the original video.

# Question: USER_QUESTION

# Read the Question carefully, Your task is to:
# 1. Analyze the relevance between the Question and the visual content across the entire clip.
# 2. Output a global relevance score and description.


# ### Instructions for clip_caption:
# - Describe **main people, objects, events, actions, and their relationships** that are visually confirmable.
# - If the question involves specific individuals or objects (especially in options), compare them clearly by appearance, position, or behavior to avoid confusion.
# - If there are multiple scenarios, describe them respectively. Pay attention to the sequence of events!
# - Only describe what is **directly observable**. Do **not** infer, imagine, or fabricate scenes beyond the visual evidence.

# ### Scoring Criteria:
# 4 points: Key elements of the query and options are clearly visible, sufficient to directly answer the question.
# 3 points: Relevant elements from either the query or options appear, but require integration with additional information to make a judgment.
# 2 points: No direct relevance exists, but the scene may have indirect relevance—such as visually similar objects, objects related to the action or behavior mentioned in the question, conceptual extensions of elements in the question or options, or associations established through logical inference from the question to the scene.
# 1 point: Completely unrelated scene.


# Carefully examine each frame and relate it to the question and the options. Do not overlook any clues related to the question.
# Please output your analysis in the following JSON format:

# {
#     "clip_caption": "string",    // Please organize and recount the main content of the video based on its timeline. The description should be coherent and rich in detail, resembling a detailed shot-by-shot record.
#     "reasoning": "string",      // Provide the reasoning and justification for the score based on the content description in clip_caption.
#     "relevance_score": integer,  // Relevance score from 1 to 4
# }

# If referring to 'left' or 'right,' be aware that the video is mirrored. The judgment should be based on the perspective of the person or object in the frame.
# Be thorough, precise, and strictly grounded in visual evidence. Avoid temporal phrases like 'the first time'.
# """

fenzu_time = 30
fps = 0.5
sample_interval = int(2/fps)

def format_time(seconds: int) -> str:
    """将秒数转换为 HH:MM:SS 格式"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_frame_paths(frame_folder: str) -> List[str]:
    """获取所有帧路径，并按名称排序"""
    frame_files = [f for f in os.listdir(frame_folder) if f.endswith(".jpg")]
    frame_files.sort()
    return [os.path.join(frame_folder, f) for f in frame_files]


def group_frames(frame_paths: List[str]) -> List[List[str]]:
    """按分钟分组，每组采样30帧（每4帧采样一次）"""
    frames_per_minute = 2 * fenzu_time  # 2fps * 60秒 = 120帧/分钟
    sampled_per_minute = frames_per_minute // sample_interval  # 120/4 = 30帧
    
    groups = []
    total_frames = len(frame_paths)
    
    # 计算有多少个完整的分钟段
    num_minutes = total_frames // frames_per_minute +1
    
    for minute in range(num_minutes):
        start_idx = minute * frames_per_minute
        end_idx = start_idx + frames_per_minute

        end_idx =min(end_idx,total_frames)
            
        # 在当前分钟内每隔4帧采样一次
        group = [frame_paths[i] for i in range(start_idx, end_idx, sample_interval)]
        groups.append(group)
    
    return groups


def judge_question_relevance(frames: List[str], question: str, idx: int):
    """判断问题与一组帧的相关性，返回结果数据"""
    
    # 每组代表1分钟（60秒），起始时间为 idx * 60 秒
    start_sec = idx * fenzu_time
    end_sec = start_sec + fenzu_time - 1  # 每组覆盖60秒（0-59秒）

    time_range = f"{format_time(start_sec)}-{format_time(end_sec)}"

    prompt = JUDGEMENT_PROMPT.replace("{USER_QUESTION}", question)
    if "START_TIME" in prompt:
        prompt = prompt.replace("START_TIME", format_time(start_sec))
        prompt = prompt.replace("END_TIME", format_time(end_sec))
    
    send_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    try:
        resp = call_openai_model_with_tools(
            send_messages,
            model_name=config.AOAI_CAPTION_VLM_MODEL_NAME,
            return_json=True,  # 要求返回JSON格式
            image_paths=frames,  # 传递图像路径列表
            down_sample_frame=True
        )["content"]
        #print(resp)

        # 解析响应
        if isinstance(resp, str):
            result = json.loads(resp)
        else:
            result = resp
        # result = resp

        # 构造输出数据
        output_data = {
            "time": time_range,
            "judgement": result
        }
        
        return output_data
            
    except Exception as e:
        print(f"Error processing group {idx}: {e}")
        return None


def localize_tool(
    question: A[str, D("The question and options")],
    frame_path: A[str, D("The frames path")],
) -> List[str]:
    # 获取所有帧的路径
    frame_paths = get_frame_paths(frame_path)
    
    # 按时间分组（假设每组代表1分钟）
    frame_groups = group_frames(frame_paths)
    print(f"Split into {len(frame_groups)} groups (1 minute each).")

    # 存储所有相关性较高的结果
    all_results = []


    start_time = time.time()
    with ThreadPoolExecutor(max_workers=20) as executor:
        # 提交所有任务到线程池
        future_to_group = {executor.submit(judge_question_relevance, group, question, idx): idx for idx, group in enumerate(frame_groups)}
        
        # 创建一个列表来按顺序存储结果
        results_in_order = [None] * len(frame_groups)
        
        for future in future_to_group:
            idx = future_to_group[future]
            try:
                result = future.result()
                results_in_order[idx] = result
            except Exception as exc:
                print(f'Group {idx} generated an exception: {exc}')
    end_time = time.time()
    duration_time = end_time - start_time

    # 过滤并收集结果
    for result in results_in_order:
        if result and result["judgement"]["relevance_score"] > 1:  # 仅保留相关性较高的结果
            all_results.append(result)


    return 'The relevance segment:' + str(all_results)