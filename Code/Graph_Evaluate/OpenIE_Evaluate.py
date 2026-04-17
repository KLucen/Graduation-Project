"""
OpenIE_Evaluate.py
针对 Entity_Baseline 图谱的RAG系统评估脚本（基于RAGAS + 实际图谱KG扩展）
修改：保存每个问题的详细结果（问题、答案、各项分数）
依赖：pip install networkx sentence-transformers openai numpy neo4j
"""

import json
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from typing import List, Dict, Any, Tuple
import re
import time
import glob
import os
from neo4j import GraphDatabase

# ================== 配置区域 ==================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

OPENAI_API_KEY = "sk-0c9d17c9e4024ff4ab7f166f319033e9"
OPENAI_BASE_URL = "https://api.deepseek.com"
EVALUATION_MODEL = "deepseek-chat"
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-zh-v1.5'

# 知识图谱构建参数
SIMILARITY_THRESHOLD = 0.7          # 实体相似度阈值（用于模糊匹配）

# RAG参数
TOP_K_SEED_NODES =  8              # 种子节点数量（减少以避免子图过大）
MAX_HOPS = 1                          # 图遍历跳数
MAX_CONTEXT_NODES = 70                 # 上下文最多包含的节点数
MAX_CONTEXT_RELS = 70                  # 上下文最多包含的关系数

# =============================================

class BaselineRAG:
    """
    针对 Entity_Baseline 和 RELATION_OPENIE 的简化版 RAG 系统
    与 b1_OPENIE.py 构建的图谱完全匹配
    """
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
        # 修正为构建脚本中实际使用的索引名、节点标签和关系类型
        self.VECTOR_INDEX_NAME = "vector_entity_baseline"   # Embedding_Baseline.py 创建
        self.NODE_LABEL = "Entity_Baseline"                 # b1_OPENIE.py 定义
        self.RELATION_TYPE = "RELATION_OPENIE"              # b1_OPENIE.py 定义

    def close(self):
        self.driver.close()

    def _embed_query(self, query: str) -> List[float]:
        return self.embedding_model.encode(query).tolist()

    def _vector_search_seed_nodes(self, query_vector: List[float]) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
                YIELD node, score
                RETURN elementId(node) AS node_id, node.name AS name, score
                """,
                index_name=self.VECTOR_INDEX_NAME,
                k=self.TOP_K_SEED_NODES,
                query_vector=query_vector
            )
            return [dict(record) for record in result]

    def _expand_subgraph(self, seed_node_ids: List[str]) -> Dict[str, Any]:
        with self.driver.session() as session:
            nodes_dict = {}
            rels_dict = {}
            for seed_id in seed_node_ids:
                query = f"""
                MATCH path = (start:{self.NODE_LABEL})-[rels:{self.RELATION_TYPE}*1..{self.MAX_HOPS}]-(other:{self.NODE_LABEL})
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
                            'name': n.get('name')
                        }
                    for r in result['path_rels']:
                        rels_dict[r.element_id] = {
                            'id': r.element_id,
                            'from_id': r.start_node.element_id,
                            'to_id': r.end_node.element_id,
                            'type': r.type
                        }
            nodes = list(nodes_dict.values())
            rels = list(rels_dict.values())
        return {'nodes': nodes, 'relationships': rels}

    def build_context(self, query: str, subgraph: Dict[str, Any]) -> str:
        lines = []
        lines.append(f"用户查询：{query}")
        lines.append("")
        lines.append("相关认知图谱信息：")

        if subgraph['nodes']:
            lines.append("概念节点：")
            nodes_to_show = subgraph['nodes'][:self.MAX_CONTEXT_NODES]
            for node in nodes_to_show:
                lines.append(f"  - {node['name']}")
            if len(subgraph['nodes']) > self.MAX_CONTEXT_NODES:
                lines.append(f"  ... 还有 {len(subgraph['nodes']) - self.MAX_CONTEXT_NODES} 个节点未列出")
        else:
            lines.append("未找到相关概念节点。")

        if subgraph['relationships']:
            lines.append("\n概念间关系：")
            rels_to_show = subgraph['relationships'][:self.MAX_CONTEXT_RELS]
            for rel in rels_to_show:
                from_node = next((n for n in subgraph['nodes'] if n['id'] == rel['from_id']), None)
                to_node = next((n for n in subgraph['nodes'] if n['id'] == rel['to_id']), None)
                from_name = from_node['name'] if from_node else f"节点{rel['from_id']}"
                to_name = to_node['name'] if to_node else f"节点{rel['to_id']}"
                lines.append(f"  - {from_name} -[{rel['type']}]-> {to_name}")
            if len(subgraph['relationships']) > self.MAX_CONTEXT_RELS:
                lines.append(f"  ... 还有 {len(subgraph['relationships']) - self.MAX_CONTEXT_RELS} 条关系未列出")
        else:
            lines.append("\n未发现概念间关系。")

        return "\n".join(lines)

    def generate_answer(self, context: str, query: str, retries=2) -> str:
        prompt = f"""你是一个基于认知图谱的问答助手。请根据以下上下文信息回答问题。请遵守以下规则：

1. 回答需要先提取与上下文直接相关的信息信息，再进行合理的推断，最后整合成答案，但必须明确说明哪些是推断，哪些是来自上下文。
2. 避免添加与上下文无关的常识性内容。
3. 如果图谱中没有任何相关内容，请直接指出图谱中没有相关信息。

{context}

请用中文回答。"""
        for attempt in range(retries):
            try:
                response = openai.chat.completions.create(
                    model=EVALUATION_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个图谱分析专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.05,
                    max_tokens=6000,
                    timeout=60  # 增加超时时间
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"生成答案时出错 (尝试 {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    return f"生成答案时出错：{str(e)}"
                time.sleep(5)  # 等待后重试


class BaselineEvaluator:
    def __init__(self, rag_system):
        self.rag = rag_system
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        openai.api_key = OPENAI_API_KEY
        openai.base_url = OPENAI_BASE_URL
        self.eval_model = EVALUATION_MODEL

    # ------------------ 辅助函数 ------------------
    def _call_llm(self, prompt: str, max_tokens=256, retries=2) -> str:
        for attempt in range(retries):
            try:
                response = openai.chat.completions.create(
                    model=self.eval_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens,
                    timeout=60  # 增加超时
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM调用失败 (尝试 {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    return ""
                time.sleep(5)

    def _extract_json_from_response(self, text: str) -> dict:
        text = re.sub(r'```json|```', '', text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"JSON解析失败，原始响应: {text}")
            return {}

    # ------------------ RAGAS 指标 ------------------
    def compute_faithfulness(self, question: str, context: str, answer: str) -> float:
        prompt = f"""请判断以下生成的答案是否完全基于提供的上下文信息，并且没有引入上下文之外的事实。
上下文信息：
{context}

用户问题：{question}
生成的答案：{answer}

请仅输出一个0到1之间的分数（保留两位小数），表示答案的忠实度（1表示完全忠实，0表示完全不忠实），高忠实度意味着生成的答案没有造信息(幻觉)，并且能够准确反映检索到的内容。"""
        score_str = self._call_llm(prompt, max_tokens=10)
        try:
            score = float(score_str)
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def compute_answer_relevance(self, question: str, answer: str) -> float:
        emb_q = self.embedding_model.encode(question)
        emb_a = self.embedding_model.encode(answer)
        sim = np.dot(emb_q, emb_a) / (np.linalg.norm(emb_q) * np.linalg.norm(emb_a))
        return float(sim)

    def compute_context_relevance(self, question: str, context: str) -> float:
        emb_q = self.embedding_model.encode(question)
        emb_c = self.embedding_model.encode(context)
        sim = np.dot(emb_q, emb_c) / (np.linalg.norm(emb_q) * np.linalg.norm(emb_c))
        return float(sim)

    # ------------------ 从问题中抽取实体 ------------------
    def extract_triplets(self, text: str, source_tag: str = "in") -> List[Tuple[str, str, str]]:
        prompt = f"""请从以下文本中抽取所有重要的事实三元组 (主语, 关系, 宾语)。关系应尽量简洁（如“导致”、“抑制”、“支持”等）。
以JSON格式输出，例如：{{"triplets": [["主语1", "关系1", "宾语1"], ["主语2", "关系2", "宾语2"]]}}
文本：{text}
"""
        response = self._call_llm(prompt, max_tokens=512)
        data = self._extract_json_from_response(response)
        triplets = data.get("triplets", [])
        tagged_triplets = []
        for h, r, t in triplets:
            tagged_triplets.append((f"{h}_{source_tag}", f"{r}_{source_tag}", f"{t}_{source_tag}"))
        return tagged_triplets

    # ------------------ 基于实际图谱的KG指标计算 ------------------
    def compute_kg_scores_from_subgraph(self, question: str, subgraph: Dict[str, Any]) -> Tuple[float, float]:
        # 1. 从问题中抽取实体
        question_triplets = self.extract_triplets(question, source_tag='in')
        question_entities = set()
        for h, _, t in question_triplets:
            if h and h.strip():
                question_entities.add(h)
            if t and t.strip():
                question_entities.add(t)
        question_entities = {e.replace('_in', '') for e in question_entities if e}

        if not question_entities:
            return 0.0, 0.0

        # 2. 构建节点ID到名称的映射
        node_id_to_name = {}
        node_names = []
        for node in subgraph['nodes']:
            node_id = node['id']
            name = node['name']
            node_id_to_name[node_id] = name
            node_names.append(name)

        # 3. 构建实际图谱的NetworkX无向图
        G = nx.Graph()
        for name in node_names:
            G.add_node(name)

        for rel in subgraph['relationships']:
            from_id = rel['from_id']
            to_id = rel['to_id']
            from_name = node_id_to_name.get(from_id)
            to_name = node_id_to_name.get(to_id)
            if from_name is None or to_name is None:
                continue
            G.add_edge(from_name, to_name)

        context_entities = set(node_names)

        # 4. 匹配问题实体到图节点（精确 + 语义模糊）
        matched_entities = set()
        if node_names:
            node_embeddings = self.embedding_model.encode(node_names)
            node_emb_dict = {name: emb for name, emb in zip(node_names, node_embeddings)}
        else:
            node_emb_dict = {}

        for q_entity in question_entities:
            if q_entity in G:
                matched_entities.add(q_entity)
                continue
            if node_emb_dict:
                q_emb = self.embedding_model.encode(q_entity)
                max_sim = -1
                best_match = None
                for name, emb in node_emb_dict.items():
                    sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
                    if sim > max_sim and sim >= SIMILARITY_THRESHOLD:
                        max_sim = sim
                        best_match = name
                if best_match:
                    matched_entities.add(best_match)

        if not matched_entities:
            return 0.0, 0.0

        # 5. 多跳得分（连通性）
        multi_hop = 0.0
        reachable_count = 0
        for q_entity in matched_entities:
            visited = set()
            stack = [q_entity]
            found = False
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                if node != q_entity and node in context_entities:
                    found = True
                    break
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        stack.append(neighbor)
            if found:
                reachable_count += 1
        multi_hop = reachable_count / len(matched_entities)

        # 6. 社区得分
        community = 0.0
        if G.number_of_nodes() > 1:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G))
            if communities:
                context_entities_non_question = context_entities - matched_entities
                mixed = 0
                for comm in communities:
                    has_question = any(e in comm and e in matched_entities for e in comm)
                    has_other_context = any(e in comm and e in context_entities_non_question for e in comm)
                    if has_question and has_other_context:
                        mixed += 1
                community = mixed / len(communities)

        return multi_hop, community

    # ------------------ 主评估流程（返回详细结果） ------------------
    def evaluate_single_query(self, question: str) -> Dict[str, Any]:
        query_vector = self.rag._embed_query(question)
        seed_nodes = self.rag._vector_search_seed_nodes(query_vector)
        if not seed_nodes:
            return {
                'question': question,
                'context': '',
                'answer': '未找到相关概念节点，无法生成答案。',
                'faithfulness': 0.0,
                'answer_relevance': 0.0,
                'context_relevance': 0.0,
                'multi_hop_score': 0.0,
                'community_score': 0.0
            }
        seed_ids = [node['node_id'] for node in seed_nodes]
        subgraph = self.rag._expand_subgraph(seed_ids)
        context_text = self.rag.build_context(question, subgraph)
        answer = self.rag.generate_answer(context_text, question)

        faithfulness = self.compute_faithfulness(question, context_text, answer)
        answer_relevance = self.compute_answer_relevance(question, answer)
        context_relevance = self.compute_context_relevance(question, context_text)

        multi_hop, community = self.compute_kg_scores_from_subgraph(question, subgraph)

        return {
            'question': question,
            'context': context_text,
            'answer': answer,
            'faithfulness': faithfulness,
            'answer_relevance': answer_relevance,
            'context_relevance': context_relevance,
            'multi_hop_score': multi_hop,
            'community_score': community
        }

    def run_evaluation(self, questions: List[str], sample_size: int = None) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        if sample_size and sample_size < len(questions):
            import random
            questions = random.sample(questions, sample_size)

        detailed_results = []
        for i, q in enumerate(questions):
            print(f"处理问题 {i+1}/{len(questions)}: {q[:50]}...")
            try:
                result = self.evaluate_single_query(q)
                detailed_results.append(result)
            except Exception as e:
                print(f"问题处理失败: {e}")
                continue

        if not detailed_results:
            return {}, []

        avg_scores = {}
        keys = ['faithfulness', 'answer_relevance', 'context_relevance', 'multi_hop_score', 'community_score']
        for key in keys:
            values = [r[key] for r in detailed_results]
            avg_scores[key] = np.mean(values)

        return avg_scores, detailed_results


# ================== 主程序 ==================
if __name__ == "__main__":
    rag = BaselineRAG()
    evaluator = BaselineEvaluator(rag)

    QUESTION_DIR = "Question"
    RESULTS_DIR = "OpenIE_Evaluation_Results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_topics_results = {}

    topic_files = glob.glob(os.path.join(QUESTION_DIR, "*.txt"))
    if not topic_files:
        print(f"在 {QUESTION_DIR} 文件夹下未找到任何 .txt 文件，请检查路径。")
        rag.close()
        exit()

    for file_path in topic_files:
        topic_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n========== 开始评估话题：{topic_name} ==========")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            questions = [line.strip() for line in lines if line.strip()]
            if not questions:
                print(f"  话题 {topic_name} 的问题文件为空，跳过")
                continue
            print(f"  读取到 {len(questions)} 个问题")
        except Exception as e:
            print(f"  读取文件失败：{e}")
            continue

        avg_scores, detailed = evaluator.run_evaluation(questions, sample_size=None)
        if avg_scores:
            all_topics_results[topic_name] = avg_scores

            detailed_file = os.path.join(RESULTS_DIR, f"{topic_name}_OpenIE_detailed.json")
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(detailed, f, ensure_ascii=False, indent=2)
            print(f"详细结果已保存至 {detailed_file}")

            print(f"\n话题 {topic_name} 平均评估结果：")
            for metric, value in avg_scores.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"  话题 {topic_name} 评估无结果")

    if all_topics_results:
        output_file = os.path.join(RESULTS_DIR, "all_topics_baseline_evaluation.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_topics_results, f, ensure_ascii=False, indent=2)
        print(f"\n所有话题评估结果已保存至 {output_file}")

    rag.close()