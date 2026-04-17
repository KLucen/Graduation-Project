import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= 配置区域 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# 嵌入模型（使用你已下载的中文模型）
MODEL_NAME = 'BAAI/bge-large-zh-v1.5'
VECTOR_DIMENSION = 1024  # 必须与模型输出维度一致

# 要处理的节点标签列表（可自由增删）
NODE_LABELS = ["Entity_Baseline", "Entity_ZeroShot"]

# 批处理大小
BATCH_SIZE = 100
# ===========================================

model = SentenceTransformer(MODEL_NAME)

def get_embedding(text):
    return model.encode(text).tolist()

def get_all_nodes(tx, label):
    """获取指定标签下所有节点的 elementId 和 name"""
    query = f"MATCH (n:{label}) RETURN elementId(n) AS node_id, n.name AS name"
    result = tx.run(query)
    return [(record["node_id"], record["name"]) for record in result]

def update_node_embedding(tx, node_id, embedding):
    """更新单个节点的 embedding 属性"""
    query = "MATCH (n) WHERE elementId(n) = $id SET n.embedding = $embedding"
    tx.run(query, id=node_id, embedding=embedding)

def create_vector_index(session, label):
    """为指定标签创建向量索引（如果不存在则先删除后创建）"""
    index_name = f"vector_{label.lower()}"
    # 删除旧索引（忽略不存在的情况）
    session.run(f"DROP INDEX {index_name} IF EXISTS")
    # 创建新索引
    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:{label}) ON (n.embedding)
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: {VECTOR_DIMENSION},
      `vector.similarity_function`: 'cosine'
    }}}}
    """
    session.run(query)
    print(f"  索引 {index_name} 创建完成（维度 {VECTOR_DIMENSION}）")

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    for label in NODE_LABELS:
        print(f"\n===== 处理标签: {label} =====")

        # 1. 获取所有节点
        with driver.session() as session:
            nodes = session.execute_read(get_all_nodes, label)
        print(f"获取到 {len(nodes)} 个节点")

        if not nodes:
            print("无节点，跳过")
            continue

        # 2. 分批生成并更新嵌入
        for i in tqdm(range(0, len(nodes), BATCH_SIZE), desc=f"生成嵌入（{label}）"):
            batch = nodes[i:i+BATCH_SIZE]
            with driver.session() as session:
                for node_id, name in batch:
                    emb = get_embedding(name)
                    session.execute_write(update_node_embedding, node_id, emb)

        # 3. 创建向量索引
        with driver.session() as session:
            create_vector_index(session, label)

    driver.close()
    print("\n所有标签处理完毕。")

if __name__ == "__main__":
    main()