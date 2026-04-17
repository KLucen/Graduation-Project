"""
RAG系统：基于认知图谱的检索增强生成（适配 BAAI/bge-large-zh-v1.5 嵌入模型 + Neo4j elementId）
要求：
- Neo4j中已存在Entity_Cognitive节点，且每个节点已通过add_embeddings.py生成了embedding属性（1024维）。
- 已创建向量索引 'entity_embeddings'（维度1024，余弦相似度）。
"""

import os
from typing import List, Dict, Any

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import openai

# ================== 配置区域 ==================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# 嵌入模型（必须与构建节点嵌入时使用的模型一致）
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-zh-v1.5'

# 向量索引名称
VECTOR_INDEX_NAME = 'entity_embeddings'

# 生成模型配置 - 切换为 DeepSeek API
OPENAI_API_KEY = "sk-82ab2070416a4613b249c60e3528ecab"   # 替换为你的 DeepSeek API Key
OPENAI_BASE_URL = "https://api.deepseek.com"             # DeepSeek 官方 API 地址
GENERATION_MODEL = "deepseek-chat"                       # DeepSeek 对话模型

# RAG参数
TOP_K_SEED_NODES =  8              # 种子节点数量（减少以避免子图过大）
MAX_HOPS = 1                          # 图遍历跳数
MAX_CONTEXT_NODES = 70                 # 上下文最多包含的节点数
MAX_CONTEXT_RELS = 70                  # 上下文最多包含的关系数
# ==============================================

class CognitiveRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        openai.api_key = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            openai.base_url = OPENAI_BASE_URL
        self.TOP_K_SEED_NODES = TOP_K_SEED_NODES
        self.MAX_HOPS = MAX_HOPS
        self.MAX_CONTEXT_NODES = MAX_CONTEXT_NODES
        self.MAX_CONTEXT_RELS = MAX_CONTEXT_RELS

    def close(self):
        self.driver.close()

    def _embed_query(self, query: str) -> List[float]:
        """将查询文本转换为向量（与嵌入模型一致）"""
        return self.embedding_model.encode(query).tolist()

    def _vector_search_seed_nodes(self, query_vector: List[float]) -> List[Dict]:
        """使用向量索引检索最相似的概念节点，返回节点ID（elementId）、名称、类型和相似度分数"""
        with self.driver.session() as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
                YIELD node, score
                RETURN elementId(node) AS node_id, node.name AS name, node.type AS type, score
                """,
                index_name=VECTOR_INDEX_NAME,
                k=self.TOP_K_SEED_NODES,
                query_vector=query_vector
            )
            return [dict(record) for record in result]

    def _expand_subgraph(self, seed_node_ids: List[str]) -> Dict[str, Any]:
        """
        从种子节点（elementId）出发，进行受限图遍历，返回子图中的所有节点和关系。
        返回结构：
        {
            'nodes': [{'id': elementId, 'name':..., 'type':..., 'properties': {...}}, ...],
            'relationships': [{'id': elementId, 'from_id':..., 'to_id':..., 'type':..., 'properties': {...}}, ...]
        }
        """
        with self.driver.session() as session:
            nodes_dict = {}
            rels_dict = {}
            for seed_id in seed_node_ids:
                # 将 max_hops 直接嵌入查询字符串（安全，因为是内部整数）
                query = f"""
                MATCH path = (start:Entity_Cognitive)-[rels:RELATION_Cognitive*1..{self.MAX_HOPS}]-(other:Entity_Cognitive)
                WHERE elementId(start) = $seed_id
                UNWIND nodes(path) AS n
                UNWIND relationships(path) AS r
                RETURN COLLECT(DISTINCT n) AS path_nodes, COLLECT(DISTINCT r) AS path_rels
                """
                result = session.run(query, seed_id=seed_id).single()
                if result:
                    for n in result['path_nodes']:
                        nodes_dict[n.element_id] = {
                            'id': n.element_id,
                            'name': n.get('name'),
                            'type': n.get('type'),
                            'properties': dict(n)   # 包含所有属性（不含embedding可节省token）
                        }
                    for r in result['path_rels']:
                        rels_dict[r.element_id] = {
                            'id': r.element_id,
                            'from_id': r.start_node.element_id,
                            'to_id': r.end_node.element_id,
                            'type': r.type,
                            'properties': dict(r)
                        }

            nodes = list(nodes_dict.values())
            rels = list(rels_dict.values())

        return {'nodes': nodes, 'relationships': rels}

    def build_context(self, query: str, subgraph: Dict[str, Any]) -> str:
        """
        将子图组装成上下文，供LLM使用。
        限制显示的节点和关系数量，避免超出token限制。
        """
        lines = []
        lines.append(f"用户查询：{query}")
        lines.append("")
        lines.append("相关认知图谱信息：")

        # 列出节点（限制数量）
        if subgraph['nodes']:
            lines.append("概念节点（部分）：")
            nodes_to_show = subgraph['nodes'][:self.MAX_CONTEXT_NODES]
            for node in nodes_to_show:
                props = node['properties']
                name = props.get('name', '未知')
                typ = props.get('type', '未知')
                count = props.get('total_count', 1)
                stance = props.get('dominant_stance', '中立')
                lines.append(f"  - {name} (类型：{typ}，提及次数：{count}，主要立场：{stance})")
            if len(subgraph['nodes']) > self.MAX_CONTEXT_NODES:
                lines.append(f"  ... 还有 {len(subgraph['nodes']) - self.MAX_CONTEXT_NODES} 个节点未列出")
        else:
            lines.append("未找到相关概念节点。")

        # 列出关系（限制数量）
        if subgraph['relationships']:
            lines.append("\n概念间关系（部分）：")
            rels_to_show = subgraph['relationships'][:self.MAX_CONTEXT_RELS]
            for rel in rels_to_show:
                # 从节点列表中找名称
                from_node = next((n for n in subgraph['nodes'] if n['id'] == rel['from_id']), None)
                to_node = next((n for n in subgraph['nodes'] if n['id'] == rel['to_id']), None)
                from_name = from_node['properties']['name'] if from_node else f"节点{rel['from_id']}"
                to_name = to_node['properties']['name'] if to_node else f"节点{rel['to_id']}"
                rel_type = rel['type']
                props = rel['properties']
                strength = props.get('strength', '未知')
                confidence = props.get('confidence', 0.0)
                count = props.get('count', 1)
                line = f"  - {from_name} -[{rel_type}]-> {to_name} (强度：{strength}，置信度：{confidence}，提及次数：{count})"
                lines.append(line)
            if len(subgraph['relationships']) > self.MAX_CONTEXT_RELS:
                lines.append(f"  ... 还有 {len(subgraph['relationships']) - self.MAX_CONTEXT_RELS} 条关系未列出")
        else:
            lines.append("\n未发现概念间关系。")

        return "\n".join(lines)

    def generate_answer(self, context: str, query: str) -> str:
        """
        调用LLM生成最终答案。
        """
        prompt = f"""你是一个基于认知图谱的问答助手。请根据以下上下文信息回答问题。请遵守以下规则：

1. 回答需要先提取与上下文直接相关的信息信息，再进行合理的推断，最后整合成答案，但必须明确说明哪些是推断，哪些是来自上下文。
2. 避免添加与上下文无关的常识性内容。
3. 如果图谱中没有任何相关内容，请直接指出图谱中没有相关信息。

{context}

请用中文回答。
"""
        try:
            response = openai.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有精通解读认知图谱的专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,
                max_tokens=5000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成答案时出错：{str(e)}"

    def answer_query(self, query: str) -> str:
        """主流程：处理查询，返回答案"""
        print(f"处理查询：{query}")

        # 1. 向量化查询
        query_vector = self._embed_query(query)
        print("查询向量化完成。")

        # 2. 向量检索种子节点
        seed_nodes = self._vector_search_seed_nodes(query_vector)
        if not seed_nodes:
            return "未找到与查询相关的概念节点，无法生成答案。"
        seed_ids = [node['node_id'] for node in seed_nodes]
        print(f"找到 {len(seed_ids)} 个种子节点。")

        # 3. 图遍历获取子图
        subgraph = self._expand_subgraph(seed_ids)
        if not subgraph['nodes']:
            # 如果遍历没有返回节点（例如没有关系），至少包括种子节点本身
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n:Entity_Cognitive)
                    WHERE elementId(n) IN $ids
                    RETURN elementId(n) AS id, n.name AS name, n.type AS type, properties(n) AS props
                    """,
                    ids=seed_ids
                )
                nodes = [{'id': record['id'],
                          'name': record['name'],
                          'type': record['type'],
                          'properties': record['props']} for record in result]
                subgraph['nodes'] = nodes
                subgraph['relationships'] = []

        print(f"子图包含 {len(subgraph['nodes'])} 个节点，{len(subgraph['relationships'])} 条关系。")

        # 4. 构建上下文（已去除来源溯源）
        context = self.build_context(query, subgraph)
        print("上下文构建完成。")

        # 5. 生成答案
        answer = self.generate_answer(context, query)
        return answer

def main():
    rag = CognitiveRAG()
    try:
        while True:
            query = input("\n请输入问题（输入exit退出）：")
            if query.lower() in ['exit', 'quit']:
                break
            answer = rag.answer_query(query)
            print("\n答案：")
            print(answer)
    finally:
        rag.close()

if __name__ == "__main__":
    main()