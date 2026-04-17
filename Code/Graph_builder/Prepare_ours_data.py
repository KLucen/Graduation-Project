import json
import os
import re
import glob
import hashlib

# ================= 配置区域 =================
INPUT_FOLDER = 'minidata'                # 原始数据源
OUTPUT_FOLDER = 'ready_data_cognitive'   # 存放清洗后数据的文件夹
INTERACTION_THRESHOLD = 100               # 互动数筛选阈值
MAX_LENGTH = 500                          # 截断长度
MIN_LENGTH = 10                            # 最短长度
# ============================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def clean_text(text):
    """文本清洗：移除URL、HTML标签，保留常用标点，去除多余空格"""
    if not isinstance(text, str):
        return ""
    # 移除URL
    text = re.sub(r'http\S+', '', text)
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 将非中文字符、非英文字母、非数字的符号替换为空格（保留常用中文标点）
    # 常用标点保留：，。！？、：；“”‘’（）《》【】……
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、：；“”‘’（）《》【】… ]+', ' ', text)
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def check_high_interaction(time_frames_dict):
    """检查互动数是否大于阈值（任意一天满足即可）"""
    if not time_frames_dict or not isinstance(time_frames_dict, dict):
        return False
    for day_metrics in time_frames_dict.values():
        if isinstance(day_metrics, dict):
            val = day_metrics.get("interaction_count", 0)
            try:
                if int(val) > INTERACTION_THRESHOLD:
                    return True
            except:
                continue
    return False

def get_text_from_item(item):
    """按优先级获取文本：desc > content > title，返回第一个非空字符串"""
    for field in ['desc', 'content', 'title']:
        val = item.get(field, '')
        if val and isinstance(val, str):
            return val
    return ""

def process_file_for_cognitive(file_path):
    """处理单个话题文件，生成清洗后的数据列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return

    topic_name = data.get('topic_name', 'Unknown')
    media_info = data.get('media_info', {})

    items_out = []
    seen_text_hashes = set()  # 用于去重（基于清洗后的文本）

    for hash_id, item in media_info.items():
        # 1. 互动数筛选
        if not check_high_interaction(item.get('time_frames')):
            continue

        # 2. 获取平台信息（作为用户群体标签）
        platform = item.get('platform', 'unknown')

        # 3. 获取原始文本
        raw_text = get_text_from_item(item)
        if not raw_text:
            continue

        # 4. 清洗文本
        cleaned = clean_text(raw_text)

        # 5. 长度检查
        if len(cleaned) < MIN_LENGTH:
            continue

        # 6. 截断
        truncated = cleaned[:MAX_LENGTH]

        # 7. 内容去重（不同用户发布相同内容）
        text_hash = hashlib.md5(truncated.encode('utf-8')).hexdigest()
        if text_hash in seen_text_hashes:
            continue
        seen_text_hashes.add(text_hash)

        # 8. 构造输出项（只保留 text 和 platform）
        items_out.append({
            "text": truncated,
            "platform": platform
        })

    # 如果该话题有数据，保存到输出文件夹
    if items_out:
        base_name = os.path.basename(file_path)
        out_path = os.path.join(OUTPUT_FOLDER, base_name)

        final_output = {
            "topic_name": topic_name,
            "data_count": len(items_out),
            "items": items_out
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print(f"处理完成: {topic_name[:20]}... -> 提取出 {len(items_out)} 条数据")

def main():
    files = glob.glob(os.path.join(INPUT_FOLDER, '*.json'))
    print(f"开始清洗数据 (互动数 > {INTERACTION_THRESHOLD})...")
    for f in files:
        process_file_for_cognitive(f)
    print(f"\n全部完成！数据已保存在 '{OUTPUT_FOLDER}' 文件夹。")

if __name__ == "__main__":
    main()