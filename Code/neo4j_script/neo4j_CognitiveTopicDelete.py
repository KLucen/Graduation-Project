from neo4j import GraphDatabase
import re

# ================= 配置区域 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# 实验标签配置
NODE_LABEL = "Entity_Cognitive"          # 概念节点的主标签（应与构建时一致）
RELATION_TYPE = "RELATION_Cognitive"      # 关系类型（仅用于提示，实际删除节点时会自动删除关系）
TOPIC_NAME = "A股 下跌 沪指 跌破 3400点"              # 话题名，例如 "微信提醒木马病毒"；若为 None 则删除所有该主标签的数据以及所有 Topic 节点
# ========================================================

class Neo4jDeleter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def delete_all(self, node_label):
        """删除所有带有指定主标签的节点以及所有 Topic 节点"""
        with self.driver.session() as session:
            # 删除所有概念节点（自动删除其关系，包括 IN_TOPIC 和 RELATION_Ours）
            query1 = f"MATCH (n:{node_label}) DETACH DELETE n"
            print(f"执行: {query1}")
            result1 = session.run(query1)
            summary1 = result1.consume()
            nodes_deleted1 = summary1.counters.nodes_deleted
            rels_deleted1 = summary1.counters.relationships_deleted

            # 删除所有 Topic 节点
            query2 = "MATCH (t:Topic) DETACH DELETE t"
            print(f"执行: {query2}")
            result2 = session.run(query2)
            summary2 = result2.consume()
            nodes_deleted2 = summary2.counters.nodes_deleted
            rels_deleted2 = summary2.counters.relationships_deleted

            total_nodes = nodes_deleted1 + nodes_deleted2
            total_rels = rels_deleted1 + rels_deleted2
            return total_nodes, total_rels

    def delete_topic(self, node_label, topic_name):
        """删除指定话题的数据：
        1. 删除话题节点
        2. 删除仅属于该话题的概念节点（即 IN_TOPIC 度数为0的概念）
        """
        with self.driver.session() as session:
            # 检查话题是否存在
            topic_result = session.run("MATCH (t:Topic {name: $topic_name}) RETURN t", topic_name=topic_name)
            if not topic_result.single():
                print(f"话题 '{topic_name}' 不存在")
                return 0, 0

            # 1. 删除该话题节点（自动删除所有指向它的 IN_TOPIC 关系）
            query_topic = "MATCH (t:Topic {name: $topic_name}) DETACH DELETE t"
            print(f"执行: {query_topic}")
            session.run(query_topic, topic_name=topic_name)

            # 2. 删除那些不再属于任何话题的概念节点（孤立概念）
            query_orphans = f"""
            MATCH (n:{node_label})
            WHERE NOT EXISTS {{
                MATCH (n)-[:IN_TOPIC]->(:Topic)
            }}
            DETACH DELETE n
            """
            print(f"执行: {query_orphans}")
            result_orphans = session.run(query_orphans)
            summary_orphans = result_orphans.consume()
            nodes_deleted = summary_orphans.counters.nodes_deleted
            rels_deleted = summary_orphans.counters.relationships_deleted

            return nodes_deleted, rels_deleted

    def delete_by_labels(self, node_label, topic_name=None):
        """根据 topic_name 决定删除策略"""
        if topic_name is None:
            return self.delete_all(node_label)
        else:
            return self.delete_topic(node_label, topic_name)

def main():
    print("⚠️  警告：此操作将永久删除 Neo4j 数据库中指定标签的数据！")
    print("当前配置：")
    print(f"  概念节点主标签 (NODE_LABEL): {NODE_LABEL}")
    print(f"  关系类型 (RELATION_TYPE): {RELATION_TYPE}")
    if TOPIC_NAME:
        print(f"  话题名称: '{TOPIC_NAME}'")
    else:
        print("  话题名称: 无 (将删除所有概念节点和所有话题节点)")

    confirm = input("\n请输入 'YES' 以确认执行：")
    if confirm != "YES":
        print("❌ 操作已取消。")
        return

    deleter = Neo4jDeleter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        # 测试连接
        with deleter.driver.session() as session:
            session.run("RETURN 1")
        print("✅ 连接成功，开始删除...")

        nodes, rels = deleter.delete_by_labels(NODE_LABEL, TOPIC_NAME)
        print(f"✅ 删除完成。")
        print(f"   删除节点数：{nodes}")
        print(f"   删除关系数：{rels}")
    except Exception as e:
        print(f"❌ 操作失败：{e}")
    finally:
        deleter.close()

if __name__ == "__main__":
    main()