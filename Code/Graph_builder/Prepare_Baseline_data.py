import json
import os
import re
import glob

# ================= 配置区域 =================
INPUT_FOLDER = 'minidata'           # 原始数据源
OUTPUT_FOLDER = 'ready_data'    # 专门存放给LLM跑的数据
INTERACTION_THRESHOLD = 100         # 互动数筛选阈值
MAX_LENGTH = 500                    # 截断长度
MIN_LENGTH = 10                     # 最短长度
# ===========================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def clean_text(text):
    """文本清洗：去除URL、HTML、干扰符"""
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、：；“”（）《》,.!?:;\'"()%]+', ' ', text)
    return text.strip()

def check_high_interaction(time_frames_dict):
    """检查互动数"""
    if not time_frames_dict or not isinstance(time_frames_dict, dict):
        return False
    for day_metrics in time_frames_dict.values():
        if isinstance(day_metrics, dict):
            val = day_metrics.get("interaction_count", 0)
            try:
                if int(val) > INTERACTION_THRESHOLD: return True
            except: continue
    return False

def process_file_for_llm(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return

    topic_name = data.get('topic_name', 'Unknown')
    media_info = data.get('media_info', {})

    # === 核心改变：创建一个简单的列表，而不是复杂的字典 ===
    slim_data_list = []
    seen_hashes = set()

    for hash_id, item in media_info.items():
        # 1. 筛选高热度
        if not check_high_interaction(item.get('time_frames')):
            continue

        # 2. 获取并清洗文本
        raw_content = item.get('desc', '') or item.get('content', '') or item.get('title', '')
        text = clean_text(raw_content)

        # 3. 长度与去重检查
        if len(text) < MIN_LENGTH: continue

        # 简单去重 (基于文本内容的哈希，防止同一段话被不同账号发)
        text_hash = hash(text)
        if text_hash in seen_hashes: continue
        seen_hashes.add(text_hash)

        # 4. === 构造数据包 ===
        # 只保留需要的字段
        slim_item = {
            "text": text[:MAX_LENGTH] # 预先截断
        }
        slim_data_list.append(slim_item)

    # 如果该文件提取出了数据，则保存
    if slim_data_list:
        # 文件名保持一致，方便追踪
        base_name = os.path.basename(file_path)
        out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}")

        final_output = {
            "topic_name": topic_name,
            "data_count": len(slim_data_list),
            "items": slim_data_list # 这是一个扁平的列表
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print(f"处理完成: {topic_name[:10]}... -> 提取出 {len(slim_data_list)} 条精简数据")

def main():
    files = glob.glob(os.path.join(INPUT_FOLDER, '*.json'))
    print(f"开始生成 baseline 专用数据 (来源: {INPUT_FOLDER})...")
    for f in files:
        process_file_for_llm(f)
    print(f"\n全部完成！数据已保存在 '{OUTPUT_FOLDER}' 文件夹。")

if __name__ == "__main__":
    main()