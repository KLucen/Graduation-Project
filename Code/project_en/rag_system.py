"""
RAG system with English prompts
"""
import time
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import openai

EMBEDDING_MODEL_NAME = 'BAAI/bge-large-zh-v1.5'   # still Chinese model but can be changed
VECTOR_INDEX_NAME = 'entity_embeddings'

OPENAI_API_KEY = "sk-82ab2070416a4613b249c60e3528ecab"
OPENAI_BASE_URL = "https://api.deepseek.com"
GENERATION_MODEL = "deepseek-chat"

TOP_K_SEED_NODES = 3
MAX_HOPS = 1
MAX_CONTEXT_NODES = 50
MAX_CONTEXT_RELS = 50

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
        lines = [f"User query: {query}", ""]
        if subgraph['nodes']:
            lines.append("Relevant cognitive graph information:\nConcept nodes (partial):")
            nodes_to_show = subgraph['nodes'][:MAX_CONTEXT_NODES]
            for node in nodes_to_show:
                props = node['properties']
                name = props.get('name', 'Unknown')
                typ = props.get('type', 'Unknown')
                count = props.get('total_count', 1)
                stance = props.get('dominant_stance', 'Neutral')
                lines.append(f"  - {name} (type: {typ}, mentions: {count}, dominant stance: {stance})")
            if len(subgraph['nodes']) > MAX_CONTEXT_NODES:
                lines.append(f"  ... and {len(subgraph['nodes']) - MAX_CONTEXT_NODES} more nodes not shown")
        else:
            lines.append("No relevant concept nodes found.")

        if subgraph['relationships']:
            lines.append("\nRelationships between concepts (partial):")
            rels_to_show = subgraph['relationships'][:MAX_CONTEXT_RELS]
            for rel in rels_to_show:
                from_node = next((n for n in subgraph['nodes'] if n['id'] == rel['from_id']), None)
                to_node = next((n for n in subgraph['nodes'] if n['id'] == rel['to_id']), None)
                from_name = from_node['properties']['name'] if from_node else f"node_{rel['from_id']}"
                to_name = to_node['properties']['name'] if to_node else f"node_{rel['to_id']}"
                rel_type = rel['type']
                props = rel['properties']
                strength = props.get('strength', 'Unknown')
                confidence = props.get('confidence', 0.0)
                count = props.get('count', 1)
                lines.append(f"  - {from_name} -[{rel_type}]-> {to_name} (strength: {strength}, confidence: {confidence}, mentions: {count})")
            if len(subgraph['relationships']) > MAX_CONTEXT_RELS:
                lines.append(f"  ... and {len(subgraph['relationships']) - MAX_CONTEXT_RELS} more relationships not shown")
        else:
            lines.append("\nNo relationships found between concepts.")

        return "\n".join(lines)

    def generate_answer(self, context: str, query: str) -> str:
        prompt = f"""You are a question-answering assistant based on a cognitive knowledge graph. Use the provided context to answer the user's question.

If the context is insufficient to answer the question, state that information is missing and provide possible inferences based on the context.

Context:
{context}

Answer in English."""
        try:
            response = openai.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in graph-based reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                timeout=60
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def answer_query(self, query: str) -> str:
        answer, _ = self.answer_with_context(query)
        return answer

    def answer_with_context(self, query: str) -> Tuple[str, str]:
        query_vector = self._embed_query(query)
        seed_nodes = self._vector_search_seed_nodes(query_vector)
        if not seed_nodes:
            return "No relevant concept nodes found. Cannot generate answer.", ""
        seed_ids = [node['node_id'] for node in seed_nodes]
        subgraph = self._expand_subgraph(seed_ids)
        if not subgraph['nodes']:
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