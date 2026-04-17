from neo4j import GraphDatabase

# ================= 配置 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"
# ========================================

class Neo4jCleaner:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """删除数据库中所有节点和关系"""
        with self.driver.session() as session:
            # 执行清空操作
            result = session.run("MATCH (n) DETACH DELETE n")
            summary = result.consume()
            nodes_deleted = summary.counters.nodes_deleted
            rels_deleted = summary.counters.relationships_deleted
            print(f"✅ 数据库已清空。")
            print(f"   删除节点数：{nodes_deleted}")
            print(f"   删除关系数：{rels_deleted}")

def main():
    print("⚠️  警告：此操作将永久删除当前 Neo4j 数据库中的所有数据！")
    confirm = input("请输入 'YES' 以确认执行：")
    if confirm != "YES":
        print("❌ 操作已取消。")
        return

    cleaner = Neo4jCleaner(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        # 可选：测试连接是否成功
        with cleaner.driver.session() as session:
            session.run("RETURN 1")
        print("✅ 连接成功，开始清空...")
        cleaner.clear_database()
    except Exception as e:
        print(f"❌ 操作失败：{e}")
    finally:
        cleaner.close()

if __name__ == "__main__":
    main()