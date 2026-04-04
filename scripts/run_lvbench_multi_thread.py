import os
import json
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import datetime
import re

from video_understanding import VideoUnderstandingSystem



current_api_key_index = 0

def get_next_api_key():
    global current_api_key_index
    api_key = api_keys[current_api_key_index]
    current_api_key_index = (current_api_key_index + 1) % len(api_keys)
    return api_key


# 日志记录函数
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


# 处理单个项目的函数
def process_item(item, idx):
    # if idx not in [i for i in range(100)]:
    #     return None
    # if idx not in ours_wrong_idx_lvbench[:30]:
    #     return None
    # if idx not in lv_random100:
    #     return None


    # if idx not in  [5, 7, 8, 11, 15, 19, 21, 29, 31, 32, 34, 49, 52, 55, 65, 66, 67, 68, 74, 81, 83, 85, 90, 93, 95, 105, 106, 107, 110, 114, 123, 124, 125, 135, 136, 137, 141, 142, 145, 146, 147, 149, 150, 151, 154, 160, 167, 170, 172, 173, 174, 176, 180, 189, 198, 204, 207, 215, 218, 223, 235, 236, 237, 240, 244, 245, 246, 247, 249, 250, 251, 274, 278, 280, 284, 285, 294, 296, 299, 300, 303, 309, 311, 312, 314, 316, 317, 320, 322, 323, 325, 326, 333, 340, 350, 352, 356, 358, 360, 379, 380, 385, 386, 389, 390, 394, 395, 398, 403, 405, 406, 409, 412, 417, 418, 420, 421, 422, 423, 425, 426, 432, 435, 436, 439, 442, 443, 444, 445, 451, 460, 461, 465, 466, 474, 476, 483, 486, 487, 489, 491, 492, 495, 496, 498, 499, 505, 507, 514, 515, 516, 519, 520, 521, 529, 531, 533, 539, 543, 544, 548, 549, 554, 555, 558, 559, 574, 577, 578, 579, 581, 583, 593, 600, 601, 604, 605, 607, 609, 610, 612, 615, 618, 637, 641, 642, 646, 647, 648, 650, 657, 659, 662, 664, 665, 666, 667, 669, 672, 677, 678, 680, 681, 682, 686, 687, 694, 695, 697, 700, 701, 703, 704, 707, 708, 710, 711, 712, 713, 718, 720, 721, 722, 723, 726, 727, 728, 731, 733, 737, 756, 758, 759, 760, 763, 764, 768, 774, 778, 790, 792, 814, 819, 825, 833, 838, 851, 859, 884] + [16, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 75, 111, 206, 210, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 275, 346, 431, 446, 456, 469, 470, 502, 511, 522, 530, 566, 572, 573, 575, 576, 582, 584, 585, 590, 591, 592, 594, 598, 599, 620, 628, 673, 676, 691, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 788, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 826, 836, 837, 841, 853, 855, 864, 866, 867, 868, 874, 875, 877, 879, 881, 882, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 926, 943, 954, 974, 997, 1009, 1045, 1046, 1079, 1182, 1210, 1224, 1289, 1298, 1322, 1341, 1362, 1393, 1395, 1413, 1455, 1458, 1492, 1547]:
    #     return None

    if idx not in range(500,700):
        return None


    # if idx not in [389]:
    #     return None

    video_key = item.get('key')
    task_type = item.get('type')
    question_with_options = item.get('question')
    correct_answer = item.get('answer')
    uid = int(item.get('uid'))

    # if uid not in lv_hard:
    #     return None
    # if idx != 317:
    #     return None

    api_key = get_next_api_key()
    
    # 为每个项目创建独立日志文件
    result_log_file = f"/home/web_server/antispam/project/zhouhongyun/long_video/MAS/MAS_new_for_lv/test_logs/1102/lv_500_700_perceptionnew_with_reflex/{idx}_{video_key}"
    if os.path.exists(result_log_file):
        with open(result_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "Final answer:{'status':" in content:
                print(f"[{idx}] 项目已处理，跳过: {video_key}")
                return  None
    
    start_msg = f"[{idx}] 开始处理项目: {video_key}"
    log_to_file(start_msg, result_log_file)

    log_to_file(question_with_options, result_log_file)
    
    
    # 构造路径
    frame_root = '/home/web_server/antispam/project/zhouhongyun/long_video/DeepVideoDiscovery-main/video_database'
    subtitle_root = ''
    
    frame_path = os.path.join(frame_root, video_key, 'frames')
    subtitle_path = os.path.join(subtitle_root, f"{video_key}.json")
    
    # 检查帧路径是否存在
    if not os.path.exists(frame_path):
        error_msg = f"[{idx}] [错误] 帧路径不存在: {frame_path}"
        print(error_msg)
        log_to_file(error_msg, result_log_file)
        return None

    try:
        frame_count = len([f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))])
        time_val = frame_count / 2  # 假设每秒2帧
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
                video_duration = time_val,
                question = question_with_options,
                frame_path = frame_path,
                sub_path = subtitle_path,
                log_path = result_log_file,
                data_name = "lv_bench"
            )
            final_result = Vus.run()

            
            # 运行推理
            
            success_msg = f"[{idx}] 问题处理成功! 正确答案: {correct_answer}"
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
json_file_path = '/home/web_server/antispam/project/zhouhongyun/long_video/LVBench/data/video_info_all.json'

# 主函数 - 使用线程池处理所有项目
def main():
    
    # 收集所有项目
    items = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 直接加载整个JSON文件
            
            # 假设JSON文件是一个包含所有项目的列表
            for idx, item in enumerate(data):
                items.append((item, idx))
    except Exception as e:
        print(f"读取JSON文件失败: {str(e)}")
        return
    
    # 创建线程池 - 根据系统资源调整线程数
    max_workers = 40
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
