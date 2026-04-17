import pandas as pd
import igraph as ig
from neo4j import GraphDatabase
import time

# ================= 配置 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# 节点标签和关系类型需与构建时一致
METHODS = {
    "Ours (Cognitive)": {"node_label": "Entity_Cognitive", "rel_type": "RELATION_Cognitive"},
}
# =======================================

class GraphDiagnoser:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def fetch_edges_all(self, node_label, rel_type):
        """获取所有概念之间的边（整体图谱）"""
        query = f"""
        MATCH (s:{node_label})-[r:{rel_type}]->(t:{node_label})
        RETURN s.name AS source, t.name AS target
        """
        with self.driver.session() as session:
            result = session.run(query)
            edges = [(r["source"], r["target"]) for r in result]
        return edges

    def fetch_edges_for_topic(self, node_label, rel_type, topic_name):
        """
        获取指定话题下的边：两个概念节点都必须通过 IN_TOPIC 关系连接到该话题节点
        topic_name 是话题的名称（字符串），例如 "A股下跌沪指跌破3400点"
        """
        query = f"""
        MATCH (t:Topic {{name: $topic_name}})<-[:IN_TOPIC]-(c1:{node_label})
        MATCH (c1)-[r:{rel_type}]->(c2:{node_label})
        WHERE (c2)-[:IN_TOPIC]->(t)
        RETURN c1.name AS source, c2.name AS target
        """
        with self.driver.session() as session:
            result = session.run(query, topic_name=topic_name)
            edges = [(r["source"], r["target"]) for r in result]
        return edges

    def get_all_topics(self):
        """获取所有话题节点的名称列表"""
        query = "MATCH (t:Topic) RETURN t.name AS topic_name"
        with self.driver.session() as session:
            result = session.run(query)
            topics = [r["topic_name"] for r in result]
        return topics

    def close(self):
        self.driver.close()


def analyze_graph(edges, method_name, topic_name="ALL"):
    """计算图指标（与原函数相同）"""
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

    G = ig.Graph.TupleList(edges, directed=True)
    nodes_count = G.vcount()
    edges_count = G.ecount()
    print(f"✓ 图谱规模: {nodes_count} 节点, {edges_count} 边")

    # 度分布
    degrees = G.degree(mode="all")
    degree_1_count = sum(1 for d in degrees if d == 1)
    degree_1_ratio = degree_1_count / nodes_count if nodes_count > 0 else 0
    max_degree = max(degrees)
    print(f"✓ [维度1] 度分布:")
    print(f"  - 最高度数: {max_degree}")
    print(f"  - 度为1的节点占比: {degree_1_ratio:.2%}")

    # 聚集系数
    G_undirected = G.as_undirected()
    clustering_coeff = G_undirected.transitivity_undirected() if G_undirected.ecount() > 0 else 0.0
    print(f"✓ [维度2] 聚集系数: {clustering_coeff:.6f}")

    # PageRank
    pagerank_scores = G.pagerank(directed=True)
    pr_ranking = sorted(zip(G.vs["name"], pagerank_scores), key=lambda x: x[1], reverse=True)
    print(f"✓ [维度3] Top 10 核心节点 (PageRank):")
    for i in range(min(10, len(pr_ranking))):
        print(f"  {i+1}. {pr_ranking[i][0]} (PR值: {pr_ranking[i][1]:.6f})")

    # 最大连通子图
    components = G.connected_components(mode='weak')
    lcc = components.giant()
    lcc_nodes = lcc.vcount()
    lcc_ratio = lcc_nodes / nodes_count if nodes_count > 0 else 0
    print(f"✓ [维度4] 最大连通子图 (LCC):")
    print(f"  - LCC 节点数: {lcc_nodes} ({lcc_ratio:.2%})")
    print(f"  正在计算 LCC 平均最短路径...")

    if lcc_nodes > 1:
        avg_path_length = lcc.average_path_length(directed=False)
        print(f"  - 平均最短路径长度: {avg_path_length:.4f}")
    else:
        avg_path_length = float('nan')
        print(f"  - 平均最短路径长度: 无法计算")

    elapsed = time.time() - start_time
    print(f"\n耗时: {elapsed:.2f} 秒")

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

    with diagnoser.driver.session() as session:
        # 查询所有话题
        topics_result = session.run("MATCH (t:Topic) RETURN t.name AS name")
        topics = [record["name"] for record in topics_result]
        print(f"[DEBUG] 话题列表: {topics}")

        # 查询所有 Entity_Cognitive 节点数
        node_count = session.run("MATCH (n:Entity_Cognitive) RETURN count(n) AS cnt").single()["cnt"]
        print(f"[DEBUG] Entity_Cognitive 节点总数: {node_count}")

        # 查询所有 RELATION_Cognitive 关系数
        rel_count = session.run("MATCH ()-[r:RELATION_Cognitive]->() RETURN count(r) AS cnt").single()["cnt"]
        print(f"[DEBUG] RELATION_Cognitive 关系总数: {rel_count}")

    for name, config in METHODS.items():
        node_label = config["node_label"]
        rel_type = config["rel_type"]

        # 整体分析
        edges_all = diagnoser.fetch_edges_all(node_label, rel_type)
        if edges_all:
            all_results.append(analyze_graph(edges_all, name, "ALL"))
        else:
            print(f"警告: {name} 整体图谱为空")

        # 按话题分析
        topics = diagnoser.get_all_topics()
        print(f"\n发现 {len(topics)} 个话题")
        for topic_name in topics:
            edges_topic = diagnoser.fetch_edges_for_topic(node_label, rel_type, topic_name)
            if edges_topic:
                all_results.append(analyze_graph(edges_topic, name, topic_name))
            else:
                print(f"  话题 '{topic_name}' 边为空，跳过")

    diagnoser.close()

    # 保存结果
    df = pd.DataFrame(all_results)
    cols = ["method", "topic", "nodes", "edges", "max_degree", "degree_1_ratio",
            "clustering_coeff", "lcc_nodes_ratio", "avg_path_length", "analysis_time"]
    df = df[cols]
    csv_path = "graph_analysis_results_Cognitive.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至 {csv_path}")
    print(df.to_string())


if __name__ == "__main__":
    main()