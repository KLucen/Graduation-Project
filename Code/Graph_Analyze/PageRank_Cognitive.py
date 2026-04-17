import pandas as pd
import igraph as ig
from neo4j import GraphDatabase
import time

# ================= 配置 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# Cognitive 方法配置
METHOD_NAME = "Ours (Cognitive)"
NODE_LABEL = "Entity_Cognitive"
REL_TYPE = "RELATION_Cognitive"
# ========================================

class GraphAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def fetch_edges_all(self):
        """获取所有 Entity_Cognitive 节点之间的边（整体图谱）"""
        query = f"""
        MATCH (s:{NODE_LABEL})-[r:{REL_TYPE}]->(t:{NODE_LABEL})
        RETURN s.name AS source, t.name AS target
        """
        with self.driver.session() as session:
            result = session.run(query)
            edges = [(r["source"], r["target"]) for r in result]
        return edges

    def fetch_edges_for_topic(self, topic_name):
        """
        获取指定话题下的边：
        两个概念节点都必须通过 IN_TOPIC 关系连接到该话题节点
        """
        query = f"""
        MATCH (t:Topic {{name: $topic_name}})<-[:IN_TOPIC]-(c1:{NODE_LABEL})
        MATCH (c1)-[r:{REL_TYPE}]->(c2:{NODE_LABEL})
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


def compute_pagerank_top10(edges, method_name, topic_name):
    """从边列表计算 PageRank，返回 Top 10 记录列表"""
    if not edges:
        return []
    # 创建有向图
    G = ig.Graph.TupleList(edges, directed=True)
    # 计算 PageRank
    pagerank = G.pagerank(directed=True)
    # 排序取前10
    ranking = sorted(zip(G.vs["name"], pagerank), key=lambda x: x[1], reverse=True)
    return [(method_name, topic_name, i+1, name, score)
            for i, (name, score) in enumerate(ranking[:10])]


def main():
    analyzer = GraphAnalyzer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    all_rows = []

    # 1. 整体图谱 PageRank
    print("正在获取整体图谱边...")
    edges_all = analyzer.fetch_edges_all()
    print(f"整体边数: {len(edges_all)}")
    all_rows.extend(compute_pagerank_top10(edges_all, METHOD_NAME, "ALL"))

    # 2. 按话题分别计算 PageRank
    topics = analyzer.get_all_topics()
    print(f"\n发现 {len(topics)} 个话题")
    for idx, topic in enumerate(topics, 1):
        print(f"正在处理话题 [{idx}/{len(topics)}]: {topic}")
        edges_topic = analyzer.fetch_edges_for_topic(topic)
        if edges_topic:
            print(f"  边数: {len(edges_topic)}")
            all_rows.extend(compute_pagerank_top10(edges_topic, METHOD_NAME, topic))
        else:
            print(f"  无关系，跳过")

    analyzer.close()

    # 保存结果
    df = pd.DataFrame(all_rows, columns=["method", "topic", "rank", "node", "pagerank"])
    csv_path = "pagerank_cognitive.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至 {csv_path}")

    # 打印摘要
    print("\n=== PageRank Top 10 (整体) ===")
    overall = df[df["topic"] == "ALL"]
    if not overall.empty:
        print(overall[["rank", "node", "pagerank"]].to_string(index=False))
    else:
        print("无数据")


if __name__ == "__main__":
    main()