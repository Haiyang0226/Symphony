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


# Function to process individual items
def process_item(item, idx):


    video_key = item.get('key')
    task_type = item.get('type')
    question_with_options = item.get('question')
    correct_answer = item.get('answer')
    uid = int(item.get('uid'))
    
    # Create independent log file for each item
    result_log_file = f"LVBench/{idx}_{video_key}"
    if os.path.exists(result_log_file):
        with open(result_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "Final answer:{'status':" in content:
                print(f"[{idx}] Item already processed, skipping: {video_key}")
                return  None
    
    start_msg = f"[{idx}] Starting to process item: {video_key}"
    log_to_file(start_msg, result_log_file)

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
        frame_count = len([f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))])
        time_val = frame_count / 2  # Assume 2 frames per second
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
                video_duration = time_val,
                question = question_with_options,
                frame_path = frame_path,
                sub_path = subtitle_path,
                log_path = result_log_file,
                data_name = "lv_bench"
            )
            final_result = Vus.run()

            
            # Run inference
            
            success_msg = f"[{idx}] Question processed successfully! Correct answer: {correct_answer}"
            log_to_file(success_msg, result_log_file)
            log_to_file(f"Final answer:{final_result}", result_log_file)
            
            result = {
                'video_key': video_key,
                'task_type': task_type,
                'question': question_with_options,
                'correct_answer': correct_answer,
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

# Global configuration
json_file_path = config.LVBENCH_DATA_PATH

# Main function - process all items using thread pool
def main():
    
    # Collect all items
    items = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 直接加载整个JSON文件
            
            # Assume JSON file is a list containing all items
            for idx, item in enumerate(data):
                items.append((item, idx))
    except Exception as e:
        print(f"Failed to read JSON file: {str(e)}")
        return
    
    # Create thread pool - adjust thread count based on system resources
    max_workers = 40
    all_results = []
    
    start_msg = f"Starting to process {len(items)} items using {max_workers} threads"
    print(start_msg)
    
    start_time = time.time()
    
    # Process using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {}
        for item, idx in items:
            future = executor.submit(process_item, item, idx)
            futures[future] = idx
        
        # Wait for all tasks to complete and process results
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                    success_msg = f"[{idx}] Item processed successfully!"
                    print(success_msg)
                else:
                    error_msg = f"[{idx}] Item processing failed!"
                    print(error_msg)
            except Exception as e:
                error_msg = f"[{idx}] Thread exception: {str(e)}"
                print(error_msg)
    
    # Save all results to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    end_time = time.time()
    duration = end_time - start_time
    avg_time = duration / len(items) if items else 0
    
    summary = (
        f"All tasks completed! Total time: {duration:.2f} seconds\n"
        f"Processed items: {len(all_results)}/{len(items)}\n"
        f"Average time per item: {avg_time:.2f} seconds"
    )
    
    print(summary)
if __name__ == "__main__":
    main()
