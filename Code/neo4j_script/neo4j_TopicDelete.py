from neo4j import GraphDatabase
import re

# ================= 配置区域 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# 主标签（请根据实际数据标签填写）
NODE_LABEL = "Entity_ZeroShot"      # 例如 Entity_Baseline / Entity_ZeroShot

# 要删除的话题列表（每个话题名将生成 Topic_XXX 标签）
TOPIC_NAMES = [
    "洛杉矶兔子博物馆大火摧毁",
    "特朗普 加拿大 关税",
    "特朗普 军事 格陵兰岛",
    "特朗普 马斯克 总统",
    "特朗普 美国 叙利亚 冲突",
    "特朗普 中国 关税 谈判",
    "提高 退休人员 养老金",
    "中国 对美 加征关税",
    "中国 加拿大 商品 关税",
    "中国 美国 关税 打击"
]

# ===========================================

class Neo4jDeleter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    @staticmethod
    def _sanitize_label(topic_name):
        """将话题名转换为 Neo4j 合法的话题标签"""
        clean_name = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', topic_name)
        return f"Topic_{clean_name}"

    def delete_by_labels(self, node_label, topic_label=None):
        """
        删除指定主标签和可选话题标签下的所有节点及其关系。
        若 topic_label 为 None，则删除所有带有 node_label 的节点。
        """
        with self.driver.session() as session:
            if topic_label:
                # 删除同时带有主标签和话题标签的节点
                query = f"MATCH (n:{node_label}:{topic_label}) DETACH DELETE n"
                print(f"执行: {query}")
                result = session.run(query)
            else:
                # 删除所有带有主标签的节点
                query = f"MATCH (n:{node_label}) DETACH DELETE n"
                print(f"执行: {query}")
                result = session.run(query)

            summary = result.consume()
            nodes_deleted = summary.counters.nodes_deleted
            rels_deleted = summary.counters.relationships_deleted
            return nodes_deleted, rels_deleted

def main():
    # 显示所有待删除的话题
    print("⚠️  警告：以下话题的所有数据将被永久删除！")
    for idx, topic in enumerate(TOPIC_NAMES, 1):
        topic_label = Neo4jDeleter._sanitize_label(topic)
        print(f"  {idx}. {topic} -> 标签: {topic_label}")
    print(f"\n主标签: {NODE_LABEL}")
    confirm = input("\n请输入 'YES' 以确认批量删除所有上述话题的数据: ")
    if confirm != "YES":
        print("❌ 操作已取消。")
        return

    deleter = Neo4jDeleter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        # 测试连接
        with deleter.driver.session() as session:
            session.run("RETURN 1")
        print("✅ 连接成功，开始批量删除...\n")

        total_nodes = 0
        total_rels = 0
        for topic in TOPIC_NAMES:
            topic_label = Neo4jDeleter._sanitize_label(topic)
            print(f"正在删除话题: {topic} (标签: {topic_label})")
            nodes, rels = deleter.delete_by_labels(NODE_LABEL, topic_label)
            print(f"  删除节点数: {nodes}, 关系数: {rels}\n")
            total_nodes += nodes
            total_rels += rels

        print(f"✅ 批量删除完成。")
        print(f"   总计删除节点数: {total_nodes}")
        print(f"   总计删除关系数: {total_rels}")
    except Exception as e:
        print(f"❌ 操作失败: {e}")
    finally:
        deleter.close()

if __name__ == "__main__":
    main()