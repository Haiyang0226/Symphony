import os
import json
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import datetime
import re

from video_understanding import VideoUnderstandingSystem

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


def process_item(item, json_file_name, idx):

    if idx not in [i for i in range(100)]:
        return None


    video_filename = item.get('video')
    video_key = os.path.splitext(video_filename)[0]
    duration = item.get('duration')
    question = item.get('question')
    candidates = item.get('candidates', [])
    correct_answer = item.get('answer')
    question_type = item.get('question_type')

    i_to_options = ['A','B','C','D','E','F','G','H','I']

    
    # 为每个项目创建独立日志文件
    result_log_file = f"/home/web_server/antispam/project/zhouhongyun/long_video/MAS/MAS_new_for_lv/test_logs/1102/mlvu/{idx}_{video_key}"

    if os.path.exists(result_log_file):
        with open(result_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "Final answer:{'status':" in content:
                print(f"[{idx}] 项目已处理，跳过: {video_key} - {json_file_name}")
                return  None
    
    start_msg = f"[{idx}] 开始处理项目: {video_key} - {json_file_name}"
    log_to_file(start_msg, result_log_file)
    
    
    # 构造问题文本
    question_with_options = question + '\n'
    for i, candidate in enumerate(candidates):
        question_with_options += f"{i_to_options[i]}. {candidate}\n"
        if correct_answer == candidate:
            correct_answer_op = i_to_options[i]

    log_to_file(question_with_options, result_log_file)
    
    # 构造路径
    frame_root = './video_database/frames'
    subtitle_root = './video_database/subtitles'
    
    frame_path = os.path.join(frame_root, video_key, 'frames')
    subtitle_path = os.path.join(subtitle_root, f"{video_key}.json")
    
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
            Vus = VideoUnderstandingSystem(
                video_duration = get_duration(frame_path),
                question = question_with_options,
                frame_path = frame_path,
                sub_path = subtitle_path,
                log_path = result_log_file,
                data_name = "lv_bench"
            )
            final_result = Vus.run()
            # 运行推理
            
            success_msg = f"[{idx}] 问题处理成功! 正确答案: {correct_answer_op}"
            log_to_file(success_msg, result_log_file)
            log_to_file(f"Final answer:{final_result}", result_log_file)
            
            result = {
                'video_key': video_key,
                'question_id': f"{json_file_name}_{idx}",
                'question_type': question_type,
                'question': question,
                'options': candidates,
                'correct_answer': correct_answer_op,
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

# 主函数 - 使用线程池处理所有项目
def main():
    json_dir = '/MLVU/json'
    
    items_to_process = []
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

    for json_file in json_files:

        if json_file in ['8_sub_scene.json', '9_summary.json']:
            continue

        json_path = os.path.join(json_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for idx, item in enumerate(data):
                items_to_process.append((item, json_file, idx))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"无法读取或解析JSON文件: {json_path}, 错误: {e}")

    max_workers = 40
    all_results = []
    
    start_msg = f"开始处理 {len(items_to_process)} 个项目，使用 {max_workers} 个线程"
    print(start_msg)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item, json_file, idx): (json_file, idx) for item, json_file, idx in items_to_process}
        
        for future in as_completed(futures):
            json_file, idx = futures[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                    success_msg = f"[{idx}] 项目 {json_file} 处理完成!"
                    print(success_msg)
                else:
                    error_msg = f"[{idx}] 项目 {json_file} 处理失败!"
                    print(error_msg)
            except Exception as e:
                error_msg = f"[{idx}] 线程异常于项目 {json_file}: {str(e)}"
                print(error_msg)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/mlvu_results_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    end_time = time.time()
    duration = end_time - start_time
    avg_time = duration / len(items_to_process) if items_to_process else 0
    
    summary = (
        f"所有任务处理完成! 总耗时: {duration:.2f}秒\n"
        f"处理项目数量: {len(all_results)}/{len(items_to_process)}\n"
        f"平均每个项目耗时: {avg_time:.2f}秒"
    )
    
    print(summary)
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
