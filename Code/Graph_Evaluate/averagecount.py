import json
import os

def compute_average(file_path):
    """读取 JSON 文件，计算所有话题的五个指标平均值"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化累加器
    sums = {
        'faithfulness': 0.0,
        'answer_relevance': 0.0,
        'context_relevance': 0.0,
        'multi_hop_score': 0.0,
        'community_score': 0.0
    }
    topic_count = 0

    for topic, scores in data.items():
        # 检查是否所有指标都存在
        if all(k in scores for k in sums):
            for k in sums:
                sums[k] += scores[k]
            topic_count += 1
        else:
            print(f"警告: 话题 '{topic}' 缺少指标，已跳过")

    if topic_count == 0:
        print(f"文件 {file_path} 中没有有效话题")
        return None, 0

    averages = {k: sums[k] / topic_count for k in sums}
    return averages, topic_count

# 文件路径配置（请根据实际存放位置调整）
files = [
    ("ZeroShot", "ZeroShot_Evaluation_Results/all_topics_zeroshot_evaluation.json"),
    ("OpenIE", "OpenIE_Evaluation_Results/all_topics_baseline_evaluation.json"),
    ("Cognitive", "Cognitive_Evaluation_Results/all_topics_evaluation.json")
]

for name, path in files:
    if not os.path.exists(path):
        print(f"文件 {path} 不存在，跳过。")
        continue

    avg, count = compute_average(path)
    if avg:
        print(f"\n========== {name} 图谱评估平均值 ==========")
        print(f"话题数量: {count}")
        for metric, value in avg.items():
            print(f"  {metric}: {value:.4f}")