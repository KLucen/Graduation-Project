"""
RAG 问答系统（供 Flask 调用）
"""
import time
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import openai

# ================== 配置 ==================
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-zh-v1.5'
VECTOR_INDEX_NAME = 'entity_embeddings'

OPENAI_API_KEY = "sk-82ab2070416a4613b249c60e3528ecab"
OPENAI_BASE_URL = "https://api.deepseek.com"
GENERATION_MODEL = "deepseek-chat"

TOP_K_SEED_NODES = 3
MAX_HOPS = 1
MAX_CONTEXT_NODES = 50
MAX_CONTEXT_RELS = 50
# ===========================================

class CognitiveRAG:
    def __init__(self, driver):
        self.driver = driver
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        openai.api_key = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            openai.base_url = OPENAI_BASE_URL

    def _embed_query(self, query: str) -> List[float]:
        return self.embedding_model.encode(query).tolist()

    def _vector_search_seed_nodes(self, query_vector: List[float]) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
                YIELD node, score
                RETURN elementId(node) AS node_id, node.name AS name, node.type AS type, score
                """,
                index_name=VECTOR_INDEX_NAME,
                k=TOP_K_SEED_NODES,
                query_vector=query_vector
            )
            return [dict(record) for record in result]

    def _expand_subgraph(self, seed_node_ids: List[str]) -> Dict[str, Any]:
        with self.driver.session() as session:
            nodes_dict = {}
            rels_dict = {}
            for seed_id in seed_node_ids:
                query = f"""
                MATCH path = (start:Entity_Cognitive)-[rels:RELATION_Cognitive*1..{MAX_HOPS}]-(other:Entity_Cognitive)
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
                            'properties': dict(n)
                        }
                    for r in result['path_rels']:
                        rels_dict[r.element_id] = {
                            'id': r.element_id,
                            'from_id': r.start_node.element_id,
                            'to_id': r.end_node.element_id,
                            'type': r.type,
                            'properties': dict(r)
                        }
            return {'nodes': list(nodes_dict.values()), 'relationships': list(rels_dict.values())}

    def build_context(self, query: str, subgraph: Dict[str, Any]) -> str:
        lines = [f"用户查询：{query}", ""]
        if subgraph['nodes']:
            lines.append("相关认知图谱信息：\n概念节点（部分）：")
            nodes_to_show = subgraph['nodes'][:MAX_CONTEXT_NODES]
            for node in nodes_to_show:
                props = node['properties']
                name = props.get('name', '未知')
                typ = props.get('type', '未知')
                count = props.get('total_count', 1)
                stance = props.get('dominant_stance', '中立')
                lines.append(f"  - {name} (类型：{typ}，提及次数：{count}，主要立场：{stance})")
            if len(subgraph['nodes']) > MAX_CONTEXT_NODES:
                lines.append(f"  ... 还有 {len(subgraph['nodes']) - MAX_CONTEXT_NODES} 个节点未列出")
        else:
            lines.append("未找到相关概念节点。")

        if subgraph['relationships']:
            lines.append("\n概念间关系（部分）：")
            rels_to_show = subgraph['relationships'][:MAX_CONTEXT_RELS]
            for rel in rels_to_show:
                from_node = next((n for n in subgraph['nodes'] if n['id'] == rel['from_id']), None)
                to_node = next((n for n in subgraph['nodes'] if n['id'] == rel['to_id']), None)
                from_name = from_node['properties']['name'] if from_node else f"节点{rel['from_id']}"
                to_name = to_node['properties']['name'] if to_node else f"节点{rel['to_id']}"
                rel_type = rel['type']
                props = rel['properties']
                strength = props.get('strength', '未知')
                confidence = props.get('confidence', 0.0)
                count = props.get('count', 1)
                lines.append(f"  - {from_name} -[{rel_type}]-> {to_name} (强度：{strength}，置信度：{confidence}，提及次数：{count})")
            if len(subgraph['relationships']) > MAX_CONTEXT_RELS:
                lines.append(f"  ... 还有 {len(subgraph['relationships']) - MAX_CONTEXT_RELS} 条关系未列出")
        else:
            lines.append("\n未发现概念间关系。")

        return "\n".join(lines)

    def generate_answer(self, context: str, query: str) -> str:
        prompt = f"""你是一个基于认知图谱的问答助手。请根据以下上下文信息回答问题。请遵守以下规则：

1. 回答需要先提取与上下文直接相关的信息信息，再进行合理的推断，最后整合成答案，但必须明确说明哪些是推断，哪些是来自上下文。
2. 避免添加与上下文无关的常识性内容。
3. 如果图谱中没有任何相关内容，请直接指出图谱中没有相关信息。

{context}

请用中文回答。"""
        try:
            response = openai.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有精通解读认知图谱的专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,
                max_tokens=5000,
                timeout=60
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成答案时出错：{str(e)}"

    def answer_query(self, query: str) -> str:
        answer, _ = self.answer_with_context(query)
        return answer

    def answer_with_context(self, query: str) -> Tuple[str, str]:
        """返回 (answer, context) 元组"""
        query_vector = self._embed_query(query)
        seed_nodes = self._vector_search_seed_nodes(query_vector)
        if not seed_nodes:
            return "未找到相关概念节点，无法生成答案。", ""
        seed_ids = [node['node_id'] for node in seed_nodes]
        subgraph = self._expand_subgraph(seed_ids)
        if not subgraph['nodes']:
            # 至少包含种子节点自身
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
        context = self.build_context(query, subgraph)
        answer = self.generate_answer(context, query)
        return answer, context