import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# 加载新模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

def get_embedding(text):
    return model.encode(text).tolist()

def get_all_concept_nodes(tx):
    # 使用 elementId 替代 id
    query = "MATCH (n:Entity_Cognitive) RETURN elementId(n) AS node_id, n.name AS name"
    result = tx.run(query)
    return [(record["node_id"], record["name"]) for record in result]

def update_node_embedding(tx, node_id, embedding):
    # 使用 elementId 匹配节点
    query = "MATCH (n) WHERE elementId(n) = $id SET n.embedding = $embedding"
    tx.run(query, id=node_id, embedding=embedding)

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # 1. 获取所有节点
    with driver.session() as session:
        nodes = session.execute_read(get_all_concept_nodes)
    print(f"获取到 {len(nodes)} 个节点")

    # 2. 分批更新嵌入向量
    batch_size = 100
    for i in tqdm(range(0, len(nodes), batch_size), desc="处理批次"):
        batch = nodes[i:i+batch_size]
        with driver.session() as session:
            for node_id, name in batch:
                emb = get_embedding(name)
                session.execute_write(update_node_embedding, node_id, emb)

    # 3. 重建向量索引
    with driver.session() as session:
        # 删除旧索引（如果存在）
        session.run("DROP INDEX entity_embeddings IF EXISTS")
        # 创建新索引
        session.run("""
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (n:Entity_Cognitive) ON (n.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 1024,
              `vector.similarity_function`: 'cosine'
            }}
        """)
    print("向量索引重建完成。")

    driver.close()

if __name__ == "__main__":
    main()