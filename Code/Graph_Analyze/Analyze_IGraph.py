import pandas as pd
import igraph as ig
from neo4j import GraphDatabase
import time
import re

# ================= 配置 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"  # 请确认密码正确

METHODS = {
    "OpenIE (Baseline)": {"node_label": "Entity_Baseline", "rel_type": "RELATION_OPENIE"},
    "LLM Zero-shot": {"node_label": "Entity_ZeroShot", "rel_type": "RELATION_ZEROSHOT"}
}
# =======================================

class GraphDiagnoser:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def fetch_edges(self, node_label, rel_type, topic_label=None):
        """
        从 Neo4j 提取所有边用于构建网络。
        如果指定 topic_label，则只返回该话题标签下的边。
        """
        if topic_label:
            query = f"""
            MATCH (s:{node_label}:{topic_label})-[r:{rel_type}]->(t:{node_label}:{topic_label})
            RETURN s.name AS source, t.name AS target
            """
        else:
            query = f"""
            MATCH (s:{node_label})-[r:{rel_type}]->(t:{node_label})
            RETURN s.name AS source, t.name AS target
            """
        with self.driver.session() as session:
            result = session.run(query)
            edges = [(r["source"], r["target"]) for r in result]
        return edges

    def get_topic_labels(self, node_label):
        """获取该节点标签下所有存在的话题标签（以 Topic_ 开头）"""
        query = f"""
        MATCH (n:{node_label})
        WHERE ANY(l IN labels(n) WHERE l STARTS WITH 'Topic_')
        RETURN DISTINCT [l IN labels(n) WHERE l STARTS WITH 'Topic_'][0] AS topic_label
        """
        with self.driver.session() as session:
            result = session.run(query)
            topics = [record["topic_label"] for record in result if record["topic_label"]]
        return topics

    def close(self):
        self.driver.close()


def analyze_graph(edges, method_name, topic_name="ALL"):
    """
    给定边列表，计算各项图指标，并打印详细结果（含 PageRank top10）。
    返回一个字典，包含方法名、话题名、节点数、边数、最大度、一度节点比例、
    聚集系数、最大连通子图比例、平均最短路径长度和分析耗时。
    """
    print(f"\n{'='*60}")
    print(f"分析: {method_name} - {topic_name}")
    print(f"{'='*60}")
    start_time = time.time()

    if len(edges) == 0:
        print("  边列表为空，跳过")
        return {
            "method": method_name,
            "topic": topic_name,
            "nodes": 0,
            "edges": 0,
            "max_degree": None,
            "degree_1_ratio": None,
            "clustering_coeff": None,
            "lcc_nodes_ratio": None,
            "avg_path_length": None,
            "analysis_time": 0
        }

    # 构建有向图
    G = ig.Graph.TupleList(edges, directed=True)
    nodes_count = G.vcount()
    edges_count = G.ecount()
    print(f"✓ 图谱规模: {nodes_count} 节点, {edges_count} 边")

    # 维度一：度分布
    degrees = G.degree(mode="all")
    degree_1_count = sum(1 for d in degrees if d == 1)
    degree_1_ratio = degree_1_count / nodes_count if nodes_count > 0 else 0
    max_degree = max(degrees)
    print(f"✓ [维度1] 度分布:")
    print(f"  - 最高度数 (超级节点): {max_degree}")
    print(f"  - 度为1的长尾节点占比: {degree_1_count}/{nodes_count} ({degree_1_ratio:.2%})")

    # 维度二：聚集系数（转为无向图计算）
    G_undirected = G.as_undirected()
    clustering_coeff = G_undirected.transitivity_undirected() if G_undirected.ecount() > 0 else 0.0
    print(f"✓ [维度2] 聚集系数 (Clustering Coefficient): {clustering_coeff:.6f}")

    # 维度三：PageRank 核心节点识别
    pagerank_scores = G.pagerank(directed=True)
    pr_ranking = sorted(zip(G.vs["name"], pagerank_scores), key=lambda x: x[1], reverse=True)
    print(f"✓ [维度3] Top 10 核心节点 (PageRank):")
    for i in range(min(10, len(pr_ranking))):
        print(f"  {i+1}. {pr_ranking[i][0]} (PR值: {pr_ranking[i][1]:.6f})")

    # 维度四：平均最短路径（基于最大连通子图）
    components = G.connected_components(mode='weak')
    lcc = components.giant()
    lcc_nodes = lcc.vcount()
    lcc_ratio = lcc_nodes / nodes_count if nodes_count > 0 else 0
    print(f"✓ [维度4] 推理潜力 (基于最大连通子图 LCC):")
    print(f"  - LCC 包含节点: {lcc_nodes} ({lcc_ratio:.2%})")
    print(f"  正在计算 LCC 平均最短路径...")

    if lcc_nodes > 1:
        avg_path_length = lcc.average_path_length(directed=False)
        print(f"  - 平均最短路径长度: {avg_path_length:.4f} 步")
    else:
        avg_path_length = float('nan')
        print(f"  - 平均最短路径长度: 无法计算 (LCC 节点数 ≤1)")

    elapsed = time.time() - start_time
    print(f"\n{method_name} - {topic_name} 分析完成，耗时: {elapsed:.2f} 秒")

    return {
        "method": method_name,
        "topic": topic_name,
        "nodes": nodes_count,
        "edges": edges_count,
        "max_degree": max_degree,
        "degree_1_ratio": degree_1_ratio,
        "clustering_coeff": clustering_coeff,
        "lcc_nodes_ratio": lcc_ratio,
        "avg_path_length": avg_path_length,
        "analysis_time": elapsed
    }


def main():
    diagnoser = GraphDiagnoser(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    all_results = []

    for name, config in METHODS.items():
        node_label = config["node_label"]
        rel_type = config["rel_type"]

        # 1. 整体分析（不限制话题）
        edges_all = diagnoser.fetch_edges(node_label, rel_type)
        if edges_all:
            result_all = analyze_graph(edges_all, name, "ALL")
            all_results.append(result_all)
        else:
            print(f"警告: {name} 整体图谱为空")

        # 2. 按话题分析
        topic_labels = diagnoser.get_topic_labels(node_label)
        print(f"发现 {len(topic_labels)} 个话题: {topic_labels}")
        for topic_label in topic_labels:
            edges_topic = diagnoser.fetch_edges(node_label, rel_type, topic_label)
            if edges_topic:
                result_topic = analyze_graph(edges_topic, name, topic_label)
                all_results.append(result_topic)
            else:
                print(f"  话题 {topic_label} 边为空，跳过")

    diagnoser.close()

    # 转换为 DataFrame 并保存 CSV
    df = pd.DataFrame(all_results)
    # 重新排序列
    cols = ["method", "topic", "nodes", "edges", "max_degree", "degree_1_ratio",
            "clustering_coeff", "lcc_nodes_ratio", "avg_path_length", "analysis_time"]
    df = df[cols]
    csv_path = "graph_analysis_results.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至 {csv_path}")

    # 打印表格摘要
    print("\n=== 分析摘要（前20行） ===")
    print(df.head(20).to_string())


if __name__ == "__main__":
    main()