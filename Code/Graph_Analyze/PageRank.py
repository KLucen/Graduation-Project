import pandas as pd
import igraph as ig
from neo4j import GraphDatabase

# ================= 配置 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

METHODS = {
    "OpenIE (Baseline)": {"node_label": "Entity_Baseline", "rel_type": "RELATION_OPENIE"},
    "LLM Zero-shot": {"node_label": "Entity_ZeroShot", "rel_type": "RELATION_ZEROSHOT"}
}
# =======================================

class GraphDiagnoser:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def fetch_edges(self, node_label, rel_type, topic_label=None):
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


def compute_pagerank_top10(edges, method_name, topic_name):
    """从边列表计算 PageRank，返回 Top 10 记录列表"""
    if not edges:
        return []
    G = ig.Graph.TupleList(edges, directed=True)
    pagerank = G.pagerank(directed=True)
    ranking = sorted(zip(G.vs["name"], pagerank), key=lambda x: x[1], reverse=True)
    return [(method_name, topic_name, i+1, name, score)
            for i, (name, score) in enumerate(ranking[:10])]


def main():
    diagnoser = GraphDiagnoser(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    all_rows = []

    for name, config in METHODS.items():
        node_label = config["node_label"]
        rel_type = config["rel_type"]

        # 整体
        edges_all = diagnoser.fetch_edges(node_label, rel_type)
        all_rows.extend(compute_pagerank_top10(edges_all, name, "ALL"))

        # 各话题
        topic_labels = diagnoser.get_topic_labels(node_label)
        for topic_label in topic_labels:
            edges_topic = diagnoser.fetch_edges(node_label, rel_type, topic_label)
            all_rows.extend(compute_pagerank_top10(edges_topic, name, topic_label))

    diagnoser.close()

    # 保存结果
    df = pd.DataFrame(all_rows, columns=["method", "topic", "rank", "node", "pagerank"])
    df.to_csv("pagerank_top10.csv", index=False, encoding='utf-8-sig')
    print("PageRank Top 10 已保存至 pagerank_top10.csv")


if __name__ == "__main__":
    main()