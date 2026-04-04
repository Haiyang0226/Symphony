import os
from typing import Annotated as A
from typing import Tuple, List
import numpy as np
import re
import config
from tools.func_call_shema import doc as D

from tools.bind import VideoQAProcessor
from api import call_seed_vl_with_tools_huoshan as call_openai_model_with_tools

import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

_video_qa_processor = None
_processor_initialized = False

def _initialize_video_qa_processor(device: str = 'cuda:0', cache_dir: str = './cache_dir'):
    global _video_qa_processor, _processor_initialized
    
    if not _processor_initialized:
        print("Initializing VideoQAProcessor...")
        _video_qa_processor = VideoQAProcessor()
        _video_qa_processor.initialize_models()
        _processor_initialized = True
        print("VideoQAProcessor initialized successfully!")
    
    return _video_qa_processor

def get_video_qa_processor():
    global _video_qa_processor, _processor_initialized
    if not _processor_initialized:
        return _initialize_video_qa_processor()
    return _video_qa_processor

def extract_frame_number(frame_path: str) -> int:
    match = re.search(r"frame_n(\d+)", os.path.basename(frame_path))
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Invalid frame path format: {frame_path}")

def extract_frame_seconds(retrieved_frame_paths, fps=2):
    fps = 2
    frame_seconds = []
    
    for frame_path in retrieved_frame_paths:
        match = re.search(r'(?:frame_n|image_)(\d+)\.(?:jpg|png)$', os.path.basename(frame_path))
        if match:
            frame_number = int(match.group(1))
            seconds = frame_number / fps
            frame_seconds.append(convert_seconds_to_hhmmss(seconds))
        else:
            raise ValueError(f"error: {frame_path}")
    
    return frame_seconds

def retrieve_tool(
    cue: A[str, D("The specific Description of Scene/objects/people contained in the question, used to retrieve relevant frames.")],
    frame_path: A[str, D("The frames path")],

) -> List[str]:
    
    processor = get_video_qa_processor()
    retrieval_frame_num = 15

    frame_list = [os.path.join(frame_path, d) for d in os.listdir(frame_path) ]
    retrieved_frame_paths = processor.process_video_qa(
                frames_list=frame_list,
                questions_list=[cue],
                top_k=retrieval_frame_num
        )

    frame_seconds=extract_frame_seconds(retrieved_frame_paths)
    print('The most similar time point:', frame_seconds)
    return 'The most similar time point:'+str(frame_seconds)


def retrieve_and_ans_tool(
    question: A[str, D("The specific question (including options) about the video content.")],
    cue: A[str, D("The specific Description of Scene/objects/people contained in the question, used to retrieve relevant frames")],
    frame_path: A[str, D("The frames path")],

) -> List[str]:
    
    processor = get_video_qa_processor()
    retrieval_frame_num = 10
    frame_list = [os.path.join(frame_path, d) for d in os.listdir(frame_path) ]

    retrieved_frame_paths = processor.process_video_qa(
        frames_list=frame_list,
        questions_list=[cue],
        top_k=retrieval_frame_num
    )

    all_framepaths = sorted(retrieved_frame_paths, key=extract_frame_number)
    prompt = """
    You are given a question and a sequence of video frame images. 
    Your task is: 
    First, determine whether the described scenario and scene in the question match. If they do not match, do not answer the question but only describe the scene.
    If match, use the visual content from these images to address the question.
    \nQuestion: {question}\n

    Carefully examine the visual details (e.g., objects, actions, context) across the entire sequence to infer the most plausible scenario. 
    These video frames may not be sufficient to answer the question, so the first step is to determine whether more information is needed.
    If very confident (There is direct evidence in the picture), output the answer. 
    If not confident (e.g., frames are irrelevant, incomplete, or contradictory), Briefly describe the scene, output only the information that is certain and relevant to the question.
    Answer solely based on the observable information, rather than through reasoning or fabrication!
    """

    input_msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant to answer questions."
        },
        {
            "role": "user",
            "content": prompt.format(question=question)
        },
    ]

    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
        api_key=config.HuoShan_API_KEY,
        image_paths=all_framepaths,
        temperature=0,
        max_tokens=512,
    )
    if msgs is None:
        raise ValueError("No response from the model")
    return msgs["content"]

def frame_associate_tool(
    question: A[str, D("The specific question (including options) containing multi sence.")],
    cue: A[List[str], D("The list of descriptions, each element is a str of description for one scene/object/people contained in the question or options, used to retrieve relevant frames")],
    frame_path: A[str, D("The frames path")],
) -> List[str]:
    
    processor = get_video_qa_processor()
    retrieval_frame_num = 10
    frame_list = [os.path.join(frame_path, d) for d in os.listdir(frame_path)]
    
    all_retrieved_frames = set()
    
    for cue_item in cue:
        retrieved_frame_paths = processor.process_video_qa(
            frames_list=frame_list,
            questions_list=[cue_item],
            top_k=retrieval_frame_num
        )
        all_retrieved_frames.update(retrieved_frame_paths)
    
    all_framepaths = sorted(list(all_retrieved_frames), key=extract_frame_number)

    prompt = """
    You are given a question and a sequence of video frame images. These segments appear in chronological order, but they may not be continuous in terms of time.
    Your task is: 
    First, determine whether the described scenario and scene in the question match. If they do not match, do not answer the question but only describe the scene.
    If match, use the visual content from these images to address the question.
    \nQuestion: {question}\n

    Carefully examine the visual details (e.g., objects, actions, context) across the entire sequence to infer the most plausible scenario. 
    These video frames may not be sufficient to answer the question, so the first step is to determine whether more information is needed.
    If very confident (There is direct evidence in the picture), output the answer. 
    If not confident (e.g., frames are irrelevant, incomplete, or contradictory), Briefly describe the scene, output only the information that is certain and relevant to the question.
    Answer solely based on the observable information, rather than through reasoning or fabrication!
    pay attention to the sequence in which the events occur and happen.
    """

    input_msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant to answer questions."
        },
        {
            "role": "user",
            "content": prompt.format(question=question)
        },
    ]

    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        image_paths=all_framepaths,
        temperature=0,
        max_tokens=512,
    )
    if msgs is None:
        raise ValueError("No response from the model")
    return msgs["content"]


def frame_inspect_tool(
    question: A[str, D("The specific question (including options) about the video content during the specified time range. Do not add time ranges and subtitle in the question.")],
    time_range: A[Tuple[str, str], D("A tuple containing start and end time in HH:MM:SS format. ")],
    cue: A[str, D("The specific objects contained in the question and options, used to retrieve relevant frames")],
    frame_path: A[str, D("The frames path")],
) -> List[str]:
    
    processor = get_video_qa_processor()
    fps = 2
    start_secs = convert_hhmmss_to_seconds(time_range[0])
    end_secs = convert_hhmmss_to_seconds(time_range[1])
    total_time = end_secs - start_secs
    
    uniform_frame_num = min(70, total_time)
    retrieval_frame_num = 20

    start_frame_idx = int(start_secs * fps)
    end_frame_idx = int(end_secs * fps)
    file_count = len([f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))])
    end_frame_idx = min(end_frame_idx,file_count-1)
    all_indices = list(range(start_frame_idx, end_frame_idx + 1))

    frames_list = [
        os.path.join(frame_path, f"frame_n{fn:06d}.jpg")
        for fn in all_indices
    ]

    retrieved_frame_paths = processor.process_video_qa(
        frames_list=frames_list,
        questions_list=[cue],
        top_k=retrieval_frame_num
    )
    
    uniform_indices = [
        start_frame_idx + i * (end_frame_idx - start_frame_idx) // max(uniform_frame_num - 1, 1)
        for i in range(uniform_frame_num)
    ]
    uniform_frames = [
        os.path.join(frame_path, f"frame_n{fn:06d}.jpg")
        for fn in uniform_indices
    ]

    all_framepaths=set(uniform_frames + retrieved_frame_paths)
    all_framepaths = sorted(all_framepaths, key=extract_frame_number)

    prompt = """You are given a question and a sequence of video frame images relevant to the question. Your task is to reason exclusively using the visual content from these images to address the question.
    \nQuestion: {question}\n
Analyze all provided frames: Carefully examine the visual details (e.g., objects, actions, context) across the entire sequence to infer the most plausible scenario.  

**Critical Rules:**
1. If confident, output the answer. 
2. If not confident (e.g., frames are blurry, irrelevant, incomplete, or contradictory), Briefly describe the scene, output only the information that is certain and relevant to the question.
3. The concepts in the question may not match the objects that appear in the scene. At this point, it is necessary to check if there are any other object match the options. Return the most matching option and explain the gap between this option and the problem description. 
4. If the question contains any information about subtitles, ignore them as there is no subtitle information in the frames. Just return the content of the scene in chronological order!
"""

    input_msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant to answer questions."
        },
        {
            "role": "user",
            "content": prompt.format(question=question)
        },
    ]

    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        image_paths=all_framepaths,
        temperature=0,
        max_tokens=512,
    )

    if msgs is None:
        raise ValueError("No response from the model")

    print(msgs["content"])
    return msgs["content"]



def interval_summary_tool(
    question: A[str, D("A question (including options) about an overview of the video.")],
    time_range: A[tuple, D("A tuple containing start and end time in HH:MM:SS format.")],
    frame_path: A[str, D("The frames path")],

) -> str:

    fps = 2
    time_ranges_secs = []
    
    start_secs = convert_hhmmss_to_seconds(time_range[0])
    end_secs = convert_hhmmss_to_seconds(time_range[1])
    time_ranges_secs.append((start_secs, end_secs))

    # Calculate total time across all ranges
    total_time = sum(end - start for start, end in time_ranges_secs)
    
    # Maximum number of timepoints to sample
    max_timepoints = config.AOAI_TOOL_VLM_MAX_FRAME_NUM
    timepoints = []

    assert total_time > 0 and max_timepoints > 0 

    # ① Uniformly sample on the flattened timeline
    #    endpoint=False ensures the last sample point < total_time
    offsets = np.linspace(
        0, total_time,
        num=max_timepoints,
        endpoint=False,
        dtype=float
    )

    # ② Calculate prefix sums for each segment, used to map offsets back to actual timestamps
    prefix_len = []          # (cumulative length, segment start, segment length)
    acc = 0
    for start, end in time_ranges_secs:
        seg_len   = end - start
        prefix_len.append((acc, start, seg_len))
        acc += seg_len

    # ③ Complete the mapping
    for off in offsets:
        # off = int(round(off))          # Ensure it is an integer
        for base, seg_start, seg_len in prefix_len:
            if off < base + seg_len:   # Find the corresponding segment
                timepoints.append(seg_start + (off - base))
                break
    max_frame_idx =999999

    framepoints = [
        min(max(int(round(ts * fps)), 0), max_frame_idx)  # clamp to [0, max_frame_idx]
        for ts in timepoints
    ]
    framepoints = sorted(set(framepoints))[:max_timepoints]

    prompt= """
    You are given a question and a sequence of video frame images relevant to the question. Your task is to reason exclusively using the visual content from these images to address the question.
    \nQuestion: {question}\n
    Analyze all provided frames: Carefully examine the visual details (e.g., objects, actions, context) across the entire sequence to infer the most plausible scenario.  
    If confident, output the answer. 
    If not confident (e.g., frames are blurry, irrelevant, incomplete, or contradictory), Briefly describe the scene, output only the information that is certain and relevant to the question.

    The concepts in the question may not match the objects that appear in the scene. At this point, it is necessary to check if there are any other object match the options. Return the most matching option and explain the gap between this option and the problem description. 
    """

    input_msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant to answer questions."
        },
        {
            "role": "user",
            "content": prompt
        },
    ]

    input_msgs[1]['content'] = input_msgs[1]['content'].format(question=question)

    files = [
        os.path.join(frame_path,  f"frame_n{fn:06d}.jpg") for fn in framepoints
    ]
    
    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        image_paths=files,
        temperature=0,
        max_tokens=512,
    )
    if msgs is None:
        raise ValueError("No response from the model")
    print (msgs["content"])
    return msgs["content"]




def associate(
    question: A[str, D("The question to be answered within the video's time_range using visual perception. Do not add time ranges and subtitle in the question.")],
    time_ranges_hhmmss: A[list[tuple], D("A list of tuples containing start and end times in HH:MM:SS format. It is necessary to cover a specific point in time and be as concise as possible. ")],
    frame_path: A[str, D("The frames path")],
 
) -> str:
    """
    Crop the video frames based on the time ranges and ask the model a detailed question about the cropped video clips.
    Returns:
        str: The model's response to the question. If no relevant content is found within the time range,
             returns an error message: "Error: Cannot find corresponding result in the given time range."
    """

    fps = 2
    time_ranges_secs = []
    for time_range in time_ranges_hhmmss:
        start_secs = convert_hhmmss_to_seconds(time_range[0])
        end_secs = convert_hhmmss_to_seconds(time_range[1])
        time_ranges_secs.append((start_secs, end_secs))

    time_ranges_secs.sort(key=lambda x: x[0])  # Sort by start time
    # Calculate total time across all ranges
    total_time = sum(end - start for start, end in time_ranges_secs)
    
    # Maximum number of timepoints to sample
    max_timepoints = config.AOAI_TOOL_VLM_MAX_FRAME_NUM
    timepoints = []

    assert total_time > 0 and max_timepoints > 0 

    # ① Uniformly sample on the flattened timeline
    #    endpoint=False ensures the last sample point < total_time
    offsets = np.linspace(
        0, total_time,
        num=max_timepoints,
        endpoint=False,
        dtype=float
    )

    # ② Calculate prefix sums for each segment, used to map offsets back to actual timestamps
    prefix_len = []          # (cumulative length, segment start, segment length)
    acc = 0
    for start, end in time_ranges_secs:
        seg_len   = end - start
        prefix_len.append((acc, start, seg_len))
        acc += seg_len

    # ③ Complete the mapping
    for off in offsets:
        # off = int(round(off))          # Ensure it is an integer
        for base, seg_start, seg_len in prefix_len:
            if off < base + seg_len:   # Find the corresponding segment
                timepoints.append(seg_start + (off - base))
                break
   
    framepoints = [int(round(ts * fps))  for ts in timepoints]
    framepoints = sorted(set(framepoints))[:max_timepoints]

    prompt= """
    You are given a question and a sequence of video frame images relevant to the question.These frames may come from temporally discontinuous segments, but they have certain correlations (such as the appearance of the same object, person, etc.).
    Your task is to reason exclusively using the visual content from these images to address the question.
    1. Analyze all provided frames: Carefully examine the visual details (e.g., objects, actions, context) across the entire sequence to infer the most plausible scenario.  
    2. Assess confidence: Determine if the visual evidence is sufficient, clear, and unambiguous.  
       - If confident, output the answer.  
       - If not confident (e.g., frames are blurry, irrelevant, incomplete, or contradictory), output only: `Reason: [specific explanation]`.Never guess or use external knowledge—base reasoning solely on the input images.  
    \nQuestion: {question}\n
    """

    input_msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant to answer questions."
        },
        {
            "role": "user",
            "content": prompt
        },
    ]

    input_msgs[1]['content'] = input_msgs[1]['content'].format(question=question)

    files = [
        os.path.join(frame_path, f"frame_n{fn:06d}.jpg") for fn in framepoints
    ]
    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        image_paths=files,
        temperature=0,
        max_tokens=512,
    )
    if msgs is None:
        raise ValueError("No response from the model")
    print (msgs["content"])
    return msgs["content"]


import json

def subtitle_tool(
    subtitle_path: A[str, D("The subtitle path")],
    start_time: A[int, D("set to 0")],

) -> List[str]:
    if not os.path.exists(subtitle_path):
        print('no subtitle')
        return 'no subtitle!'

    with open(subtitle_path, 'r', encoding='utf-8') as f:
        subtitles = json.load(f)
    
    result = []
    for subtitle in subtitles:
        start_seconds = convert_hhmmss_to_seconds(subtitle["start"])
        if start_seconds >= start_time:
            relative_start = start_seconds - start_time
            relative_end = convert_hhmmss_to_seconds(subtitle["end"]) - start_time
            
            start_time_str = convert_seconds_to_hhmmss(relative_start)
            end_time_str = convert_seconds_to_hhmmss(relative_end)
            
            formatted_subtitle = f"{start_time_str}-{end_time_str}:{subtitle['line']}"
            result.append(formatted_subtitle)
    
    res= " ".join(result)
    return res



def convert_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def convert_hhmmss_to_seconds(hhmmss):
    hhmmss = hhmmss.split('.')[0]
    parts = hhmmss.split(":")
    if len(parts) < 2:
        raise ValueError("Invalid time format. Expected HH:MM:SS.")
    elif len(parts) == 2:
        parts = ["00"] + parts
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds

def is_covered(d,N):
    i=sorted((int(a),int(b))for a,b in(map(lambda x:x.split('_'),d)));c=0
    return all(s==c and not (c:=e) for s,e in i) and c==N


def associate(
    question: A[str, D("The question (including options) about the video content during the specified time ranges.")],
    time_ranges_hhmmss: A[list[tuple], D("A list of tuples containing start and end times in HH:MM:SS format.")],
    frame_path: A[str, D("The frames path")],
) -> str:
    """
    Modified version that explicitly tells the model which time range each frame belongs to.
    Returns:
        str: Model's response with time-range-aware analysis.
    """
    fps = 2
    time_ranges_secs = []
    
    # Convert HH:MM:SS to seconds and sort ranges
    for time_range in time_ranges_hhmmss:
        start_secs = convert_hhmmss_to_seconds(time_range[0])
        end_secs = convert_hhmmss_to_seconds(time_range[1])
        time_ranges_secs.append((start_secs, end_secs, time_range[0], time_range[1]))  # Store original HH:MM:SS
    
    time_ranges_secs.sort(key=lambda x: x[0])  # Sort by start time

    # Generate frame paths with time range tags
    framed_timepoints = []
    for start_secs, end_secs, start_hhmmss, end_hhmmss in time_ranges_secs:
        # Calculate frame indices for this range
        start_frame = int(start_secs * fps)
        end_frame = int(end_secs * fps)
        
        # Uniformly sample frames within this range
        num_frames = min(20, end_frame - start_frame)  # Max 20 frames per range
        frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int, endpoint=False)
        
        # Add frame paths with time range metadata
        for idx in frame_indices:
            frame_time_sec = idx / fps
            frame_time_hhmmss = convert_seconds_to_hhmmss(frame_time_sec)
            framed_timepoints.append({
                "path": os.path.join(frame_path, f"frame_n{idx:06d}.jpg"),
                "time_range": f"{start_hhmmss}-{end_hhmmss}",
                "frame_time": frame_time_hhmmss
            })

    # Construct prompt with time range awareness
    prompt = """
    You will analyze a video question based on frames from SPECIFIC TIME RANGES. 
    Each frame is annotated with:
    - Time range it belongs to (e.g. "00:01:00-00:02:00")
    - Exact timestamp of that frame (e.g. "00:01:15")

    TASK REQUIREMENTS:
    1. First identify which time range contains key evidence
    2. Describe events separately for each time range if needed
    3. For comparisons, explicitly state which range has which feature

    FRAME METADATA:
    {frame_metadata}

    QUESTION: {question}
    """

    # Prepare frame metadata for prompt
    range_groups = {}
    for item in framed_timepoints:
        if item["time_range"] not in range_groups:
            range_groups[item["time_range"]] = []
        range_groups[item["time_range"]].append(item["frame_time"])
    
    metadata_str = "\n".join(
        f"Time Range [{range_}]: Frames at {', '.join(times)}"
        for range_, times in range_groups.items()
    )

    # Prepare image paths (sorted by frame number)
    image_paths = sorted(
        [item["path"] for item in framed_timepoints],
        key=lambda x: extract_frame_number(x)
    )

    input_msgs = [
        {"role": "system", "content": "You are a video analysis expert."},
        {"role": "user", "content": prompt.format(
            question=question,
            frame_metadata=metadata_str
        )}
    ]

    # Call the model
    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        image_paths=image_paths,
        temperature=0,
        max_tokens=512,
    )

    return msgs["content"] if msgs else "Error: No response from model"


if __name__ == "__main__":
    question='is there a man in the frames?'
    aa=frame_inspect_tool(question,['00:01:00', '00:01:20'])
