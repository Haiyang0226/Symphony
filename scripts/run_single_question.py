import os
import sys
import json
import time
import datetime
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_understanding import VideoUnderstandingSystem

def log_to_file(message, log_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def get_frame_count(frame_path):
    """Estimate video duration based on file count in frame directory (assuming 2fps)"""
    frame_count = len([f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))])
    return frame_count / 1


def process_single(question, frame_path, subtitle_path, log_file, data_name="lv_bench", max_retries=2):
    """
    Process a single video QA task.

    Args:
        question: Question text (with options)
        frame_path: Video frame directory path
        subtitle_path: Subtitle file path
        log_file: Path to save intermediate process logs
        data_name: Dataset name
        max_retries: Maximum number of retries

    Returns:
        final_result: Final answer returned by the model
    """
    if not os.path.exists(frame_path):
        log_to_file(f"[Error] Frame path does not exist: {frame_path}", log_file)
        return None

    video_duration = get_frame_count(frame_path)
    log_to_file(f"Question: {question}", log_file)
    log_to_file(f"Frame path: {frame_path}, Duration: {video_duration}s", log_file)

    for attempt in range(1, max_retries + 1):
        try:
            vus = VideoUnderstandingSystem(
                video_duration=video_duration,
                question=question,
                frame_path=frame_path,
                sub_path=subtitle_path,
                log_path=log_file,
                data_name=data_name
            )
            final_result = vus.run()
            log_to_file(f"Final answer: {final_result}", log_file)
            return final_result
        except Exception as e:
            log_to_file(f"Attempt {attempt} failed: {e}", log_file)
            if attempt < max_retries:
                time.sleep(1)

    log_to_file("[Failed] Maximum retries reached", log_file)
    return None


if __name__ == "__main__":
    # ===== Modify inputs here =====
    question = ""
    frame_path = "./clips"
    subtitle_path = ""
    log_file = "./MAS_result/output_log.txt"
    # ==========================

    result = process_single(question, frame_path, subtitle_path, log_file)
    print(f"Final answer: {result}")
