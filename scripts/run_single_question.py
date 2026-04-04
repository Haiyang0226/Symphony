import os
import json
import time
import datetime
import re

from video_understanding import VideoUnderstandingSystem

def log_to_file(message, log_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def get_frame_count(frame_path):
    """根据帧目录中的文件数量估算视频时长（假设2fps）"""
    frame_count = len([f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))])
    return frame_count / 1


def process_single(question, frame_path, subtitle_path, log_file, data_name="lv_bench", max_retries=2):
    """
    处理单个视频问答任务。

    Args:
        question: 问题文本（含选项）
        frame_path: 视频帧目录路径
        subtitle_path: 字幕文件路径
        log_file: 中间过程日志保存路径
        data_name: 数据集名称
        max_retries: 最大重试次数

    Returns:
        final_result: 模型返回的最终答案
    """
    if not os.path.exists(frame_path):
        log_to_file(f"[错误] 帧路径不存在: {frame_path}", log_file)
        return None

    video_duration = get_frame_count(frame_path)
    log_to_file(f"问题: {question}", log_file)
    log_to_file(f"帧路径: {frame_path}, 时长: {video_duration}s", log_file)

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
            log_to_file(f"第 {attempt} 次尝试失败: {e}", log_file)
            if attempt < max_retries:
                time.sleep(1)

    log_to_file("[失败] 已达最大重试次数", log_file)
    return None


if __name__ == "__main__":
    # ===== 在这里修改输入 =====
    question = "这个视频中是否出现了毛泽东主席"
    frame_path = "/home/web_server/antispam/project/zhouhongyun/kuai_midvideo/tools/photo/1442462243/clips"
    subtitle_path = ""
    log_file = "/home/web_server/antispam/project/zhouhongyun/kuai_midvideo/MAS_result/output_log.txt"
    # ==========================

    result = process_single(question, frame_path, subtitle_path, log_file)
    print(f"最终答案: {result}")
