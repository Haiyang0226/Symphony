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
# Set frame and subtitle paths. Frames can be obtained via video2frames.py; subtitles were converted using Whisper large
frame_root = './video_database/frames'
subtitle_root = './video_database/subtitles'

def log_to_file(message, log_file='process_log.txt'):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed to write log: {e}")

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

    
    # Create independent log file for each item (using relative path)
    result_log_file = f"./test_logs/mlvu/{idx}_{video_key}"

    if os.path.exists(result_log_file):
        with open(result_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "Final answer:{'status':" in content:
                print(f"[{idx}] Item already processed, skipping: {video_key} - {json_file_name}")
                return  None
    
    start_msg = f"[{idx}] Starting to process item: {video_key} - {json_file_name}"
    log_to_file(start_msg, result_log_file)
    
    
    # Construct question text
    question_with_options = question + '\n'
    for i, candidate in enumerate(candidates):
        question_with_options += f"{i_to_options[i]}. {candidate}\n"
        if correct_answer == candidate:
            correct_answer_op = i_to_options[i]

    log_to_file(question_with_options, result_log_file)
    
    frame_path = os.path.join(frame_root, video_key, 'frames')
    subtitle_path = os.path.join(subtitle_root, f"{video_key}.json")
    
    # Check if frame path exists
    if not os.path.exists(frame_path):
        error_msg = f"[{idx}] [Error] Frame path does not exist: {frame_path}"
        print(error_msg)
        log_to_file(error_msg, result_log_file)
        return None
    
    try:
        num_frames = len([f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))])
        estimated_time = num_frames / 2.0
    except Exception as e:
        error_msg = f"[{idx}] [Error] Unable to count frames: {str(e)}"
        print(error_msg)
        log_to_file(error_msg, result_log_file)
        return None
    
    # Retry mechanism
    max_retries = 2
    retry_count = 0
    success = False
    result = None

    
    while not success and retry_count < max_retries:
        try:            
            # Initialize Agent
            Vus = VideoUnderstandingSystem(
                video_duration = get_duration(frame_path),
                question = question_with_options,
                frame_path = frame_path,
                sub_path = subtitle_path,
                log_path = result_log_file,
                data_name = "lv_bench"
            )
            final_result = Vus.run()
            # Run inference
            
            success_msg = f"[{idx}] Question processed successfully! Correct answer: {correct_answer_op}"
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
            error_msg = f"[{idx}] Processing error - retry {retry_count}/{max_retries}: {str(e)}"
            print(error_msg)
            log_to_file(error_msg, result_log_file)
            if retry_count < max_retries:
                time.sleep(1)  # Wait 1 second before retry
    
    if not success:
        fail_msg = f"[{idx}] [Failed] Item processing failed, maximum retries reached"
        print(fail_msg)
        log_to_file(fail_msg, result_log_file)
    
    # Record separator line
    separator = "-" * 60
    log_to_file(separator, result_log_file)
    
    return result

# Main function - process all items using thread pool
def main():
    json_dir = config.MLVU_DATA_PATH
    
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
            print(f"Unable to read or parse JSON file: {json_path}, error: {e}")

    max_workers = 40
    all_results = []
    
    start_msg = f"Starting to process {len(items_to_process)} items using {max_workers} threads"
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
                    success_msg = f"[{idx}] Item {json_file} processed successfully!"
                    print(success_msg)
                else:
                    error_msg = f"[{idx}] Item {json_file} processing failed!"
                    print(error_msg)
            except Exception as e:
                error_msg = f"[{idx}] Thread exception in item {json_file}: {str(e)}"
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
        f"All tasks completed! Total time: {duration:.2f} seconds\n"
        f"Processed items: {len(all_results)}/{len(items_to_process)}\n"
        f"Average time per item: {avg_time:.2f} seconds"
    )
    
    print(summary)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
