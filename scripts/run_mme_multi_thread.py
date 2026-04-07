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

# 设置帧、字幕路径
frame_root = './video_database/frames'
subtitle_root = './video_database/subtitle'

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
    if idx not in mme_random100:
        return None

    video_key = item.get('videoID')
    question_id = item.get('question_id')
    task_type = item.get('task_type')
    question = item.get('question')
    options = item.get('options', [])
    correct_answer = item.get('answer')

    api_key = get_next_api_key()
    
    # 为每个项目创建独立日志文件
    result_log_file = f"./test_logs/mme/{idx}_{video_key}"

    if os.path.exists(result_log_file):
        with open(result_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "Final answer:{'status':" in content:
                print(f"[{idx}] 项目已处理，跳过: {video_key} - {question_id}")
                return  None
    
    start_msg = f"[{idx}] 开始处理项目: {video_key} - {question_id}"
    log_to_file(start_msg, result_log_file)
    
    
    # 构造问题文本
    option_lines = "\n".join([f"{opt}" for opt in options])
    question_with_options = question + "\n" + option_lines
    log_to_file(question_with_options, result_log_file)
    
    frame_path = os.path.join(frame_root, video_key, 'frames')
    subtitle_path = os.path.join(subtitle_root, f"{video_key}.json")
    
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
            Vus = VideoUnderstandingSystem(
                video_duration = get_duration(frame_path),
                question = question_with_options,
                frame_path = frame_path,
                sub_path = subtitle_path,
                log_path = result_log_file,
                data_name = "video_mme"
            )
            final_result = Vus.run()
    
            
            success_msg = f"[{idx}] 问题处理成功! 正确答案: {correct_answer}"
            log_to_file(success_msg, result_log_file)
            log_to_file(f"Final answer:{final_result}", result_log_file)
            
            result = {
                'video_key': video_key,
                'question_id': question_id,
                'task_type': task_type,
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
            }
            
            success = True
        except Exception as e:
            retry_count += 1
            error_msg = f"[{idx}] 处理错误 - 重试 {retry_count}/{max_retries}: {str(e)}"
            print(error_msg)
            log_to_file(error_msg, result_log_file)
            if retry_count < max_retries:
                time.sleep(1) 
    
    if not success:
        fail_msg = f"[{idx}] [失败] 项目处理失败，已达最大重试次数"
        print(fail_msg)
        log_to_file(fail_msg, result_log_file)
    
    # 记录分隔线
    separator = "-" * 60
    log_to_file(separator, result_log_file)
    
    return result

# 全局配置
jsonl_file = config.VIDEOMME_DATA_PATH

# 主函数 - 使用线程池处理所有项目
def main():
    # 读取 JSONL 文件
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取JSONL文件失败: {str(e)}")
        return
    
    # 收集所有项目
    items = []
    for idx, line in enumerate(lines):
        try:
            item = json.loads(line.strip())
            items.append((item, idx))
        except json.JSONDecodeError:
            print(f"跳过无效行: {line}")
    
    # 创建线程池 - 根据系统资源调整线程数
    max_workers = 30
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
