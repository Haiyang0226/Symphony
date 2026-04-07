import os
import json
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import datetime
import re

import config
from video_understanding import VideoUnderstandingSystem
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 设置帧、字幕路径
frame_root = './video_database/frames'
subtitle_root = './video_database/subtitles'

def log_to_file(message, log_file='process_log.txt'):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"日志写入失败: {e}")

def get_duration(folder_path):
    files = os.listdir(folder_path)
    numbers = [int(re.search(r'n(\d+)\.jpg', f).group(1)) for f in files if re.search(r'n(\d+)\.jpg', f)]
    max_number = max(numbers) if numbers else None
    return int(max_number/2)


def process_item(item, idx):
    video_path = item['video_path']
    video_key = os.path.splitext(video_path)[0]
    question = item['question']
    candidates = item['candidates']
    question_with_options = question + "\nOptions:\n"
    for i, candidate in enumerate(candidates):
        question_with_options += f"{i}. {candidate}\n"

    subtitle_path = os.path.join(subtitle_root, item['subtitle_path'])
    frame_path = os.path.join(frame_root, video_key, 'frames')
    duration = item['duration']
    correct_answer = item['correct_choice']

    result_log_file = f"./long_video/{idx}_{video_key}"

    log_to_file(question_with_options, result_log_file)    
    if os.path.exists(result_log_file):
        with open(result_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "Final answer:{'status':" in content:
                print(f"[{idx}] 项目已处理，跳过: {video_key}")
                return  None

    
    start_msg = f"[{idx}] 开始处理项目: {video_key}"
    log_to_file(start_msg, result_log_file)

    
    # 检查帧路径是否存在
    if not os.path.exists(frame_path):
        error_msg = f"[{idx}] [错误] 帧路径不存在: {frame_path}"
        print(error_msg)
        log_to_file(error_msg, result_log_file)
        return None
    
    try:
        num_frames = len([f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))])
        estimated_time = num_frames / 2.0
    except Exception as e:
        error_msg = f"[{idx}] [错误] 无法计算帧数: {str(e)}"
        print(error_msg)
        log_to_file(error_msg, result_log_file)
        return None
    
    # 重试机制
    max_retries = 2
    retry_count = 0
    success = False
    result = None
    
    while not success and retry_count < max_retries:
        try:            
            # 初始化 Agent
            dur = f"计算得到时长：{get_duration(frame_path)}   数据集原本时长：{duration}"
            log_to_file(dur, result_log_file)
        
            Vus = VideoUnderstandingSystem(
                video_duration = get_duration(frame_path),
                question = question_with_options,
                frame_path = frame_path,
                sub_path = subtitle_path,
                log_path = result_log_file,
                data_name = "longvideo"
            )
            final_result = Vus.run()
            # 运行推理
            
            success_msg = f"[{idx}] 问题处理成功! 正确答案: {correct_answer}"
            log_to_file(success_msg, result_log_file)
            log_to_file(f"Final answer:{final_result}", result_log_file)
            
            result = {
                'video_key': video_key,
                'question': question,
                'correct_answer': correct_answer,
            }
            
            success = True
        except Exception as e:
            retry_count += 1
            error_msg = f"[{idx}] 处理错误 - 重试 {retry_count}/{max_retries}: {str(e)}"
            print(error_msg)
            log_to_file(error_msg, result_log_file)
            if retry_count < max_retries:
                time.sleep(1)  # 重试前等待1秒
    
    if not success:
        fail_msg = f"[{idx}] [失败] 项目处理失败，已达最大重试次数"
        print(fail_msg)
        log_to_file(fail_msg, result_log_file)
    
    # 记录分隔线
    separator = "-" * 60
    log_to_file(separator, result_log_file)
    
    return result

# 全局配置
json_file_path = config.LONGVIDEO_DATA_PATH

# 主函数 - 使用线程池处理所有项目
def main():
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {str(e)}")
        return
    
    # 收集所有项目
    items = []
    for idx, item in enumerate(data):
        items.append((item, idx))
    
    # 创建线程池 - 根据系统资源调整线程数
    max_workers = 50
    all_results = []
    
    start_msg = f"开始处理 {len(items)} 个项目，使用 {max_workers} 个线程"
    print(start_msg)
    
    start_time = time.time()
    
    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {}
        for item, idx in items:
            future = executor.submit(process_item, item, idx)
            futures[future] = idx
        
        # 等待所有任务完成并处理结果
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                    success_msg = f"[{idx}] 项目处理完成!"
                    print(success_msg)
                else:
                    error_msg = f"[{idx}] 项目处理失败!"
                    print(error_msg)
            except Exception as e:
                error_msg = f"[{idx}] 线程异常: {str(e)}"
                print(error_msg)
    
    # 保存所有结果到文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    end_time = time.time()
    duration = end_time - start_time
    avg_time = duration / len(items) if items else 0
    
    summary = (
        f"所有任务处理完成! 总耗时: {duration:.2f}秒\n"
        f"处理项目数量: {len(all_results)}/{len(items)}\n"
        f"平均每个项目耗时: {avg_time:.2f}秒"
    )    
    print(summary)

if __name__ == "__main__":
    main()
