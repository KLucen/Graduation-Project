"""
认知图谱社区分析模块（供 Flask 调用）
"""
import time
from collections import defaultdict, Counter
from typing import List, Dict, Any

from neo4j import GraphDatabase
import openai

# ================== 配置（从原文件复制）==================
OPENAI_API_KEY = "sk-82ab2070416a4613b249c60e3528ecab"
OPENAI_BASE_URL = "https://api.deepseek.com"
GENERATION_MODEL = "deepseek-chat"

LOUVAIN_MAX_LEVELS = 10
LOUVAIN_CONCURRENCY = 4

TOP_NODES_PER_COMMUNITY = 10
MAX_RELATIONS_PER_COMMUNITY = 20
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1000
# ========================================================

class TopicAnalyzer:
    def __init__(self, driver):
        self.driver = driver
        openai.api_key = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            openai.base_url = OPENAI_BASE_URL

    def get_topic_node_count(self, topic_name: str) -> int:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (c:Entity_Cognitive)-[:IN_TOPIC]->(t:Topic {name: $topic}) RETURN count(c) AS cnt",
                topic=topic_name
            ).single()
            return result["cnt"] if result else 0

    def extract_topic_subgraph(self, topic_name: str) -> Dict[str, Any]:
        with self.driver.session() as session:
            nodes_result = session.run(
                """
                MATCH (c:Entity_Cognitive)-[:IN_TOPIC]->(t:Topic {name: $topic})
                RETURN elementId(c) AS id, c.name AS name, c.type AS type,
                       properties(c) AS props
                """,
                topic=topic_name
            )
            nodes = []
            node_id_to_info = {}
            for record in nodes_result:
                node_info = {
                    'id': record['id'],
                    'name': record['name'],
                    'type': record['type'],
                    'props': record['props']
                }
                nodes.append(node_info)
                node_id_to_info[record['id']] = node_info

            if not nodes:
                return {'nodes': [], 'relationships': []}

            rels_result = session.run(
                """
                MATCH (c1:Entity_Cognitive)-[r:RELATION_Cognitive]-(c2:Entity_Cognitive)
                WHERE elementId(c1) IN $node_ids AND elementId(c2) IN $node_ids
                RETURN elementId(r) AS id, elementId(c1) AS from_id, elementId(c2) AS to_id,
                       type(r) AS rel_type, properties(r) AS props,
                       c1.name AS from_name, c2.name AS to_name
                """,
                node_ids=list(node_id_to_info.keys())
            )
            relationships = []
            for record in rels_result:
                rel_info = {
                    'id': record['id'],
                    'from_id': record['from_id'],
                    'to_id': record['to_id'],
                    'type': record['rel_type'],
                    'props': record['props'],
                    'from_name': record['from_name'],
                    'to_name': record['to_name']
                }
                relationships.append(rel_info)

            return {'nodes': nodes, 'relationships': relationships}

    def project_to_gds(self, topic_name: str, node_ids: List[str]) -> str:
        safe_topic = ''.join(c if c.isalnum() else '_' for c in topic_name)
        graph_name = f"topic_{safe_topic}_louvain"

        node_ids_str = '[' + ','.join(f'"{nid}"' for nid in node_ids) + ']'

        node_query = f"""
            MATCH (c:Entity_Cognitive)
            WHERE elementId(c) IN {node_ids_str}
            RETURN id(c) AS id
        """
        rel_query = f"""
            MATCH (c1:Entity_Cognitive)-[r:RELATION_Cognitive]->(c2:Entity_Cognitive)
            WHERE elementId(c1) IN {node_ids_str} AND elementId(c2) IN {node_ids_str}
            RETURN id(c1) AS source, id(c2) AS target, type(r) AS type
        """

        with self.driver.session() as session:
            session.run(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")
            result = session.run(
                """
                CALL gds.graph.project.cypher($graph_name, $node_query, $rel_query)
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                """,
                graph_name=graph_name,
                node_query=node_query,
                rel_query=rel_query
            ).single()
            if not result:
                raise Exception("GDS projection failed.")
        return graph_name

    def run_louvain(self, graph_name: str) -> Dict[str, int]:
        with self.driver.session() as session:
            result = session.run(
                f"""
                CALL gds.louvain.stream('{graph_name}', {{
                    maxLevels: {LOUVAIN_MAX_LEVELS},
                    concurrency: {LOUVAIN_CONCURRENCY}
                }})
                YIELD nodeId, communityId
                RETURN elementId(gds.util.asNode(nodeId)) AS element_id, communityId
                """
            )
            return {record['element_id']: record['communityId'] for record in result}

    def analyze_communities(self, subgraph: Dict[str, Any], community_map: Dict[str, int]) -> Dict[int, Any]:
        node_by_id = {n['id']: n for n in subgraph['nodes']}
        degree_counter = defaultdict(int)
        for rel in subgraph['relationships']:
            degree_counter[rel['from_id']] += 1
            degree_counter[rel['to_id']] += 1

        communities = defaultdict(lambda: {
            'nodes': [],
            'relationships': [],
            'node_ids': set(),
            'type_counter': Counter()
        })

        for node_id, comm_id in community_map.items():
            if node_id in node_by_id:
                node_info = node_by_id[node_id]
                communities[comm_id]['nodes'].append(node_info)
                communities[comm_id]['node_ids'].add(node_id)
                node_type = node_info.get('type', '未知')
                communities[comm_id]['type_counter'][node_type] += 1

        for rel in subgraph['relationships']:
            from_id = rel['from_id']
            to_id = rel['to_id']
            from_comm = community_map.get(from_id)
            to_comm = community_map.get(to_id)
            if from_comm is not None and to_comm is not None and from_comm == to_comm:
                communities[from_comm]['relationships'].append(rel)

        result = {}
        for comm_id, data in communities.items():
            node_list = data['nodes']
            for node in node_list:
                node['degree'] = degree_counter.get(node['id'], 0)
            node_list.sort(key=lambda x: x['degree'], reverse=True)
            type_dist = dict(data['type_counter'])
            result[comm_id] = {
                'nodes': node_list,
                'relationships': data['relationships'],
                'node_count': len(node_list),
                'rel_count': len(data['relationships']),
                'type_distribution': type_dist,
                'top_nodes': node_list[:TOP_NODES_PER_COMMUNITY]
            }
        return result

    def summarize_community(self, community_id: int, community_data: Dict) -> str:
        lines = []
        lines.append(f"Please summarize the main content and core cognitive theme of the following community (Community ID: {community_id}).")
        lines.append(f"This community contains {community_data['node_count']} cognitive concept nodes and {community_data['rel_count']} relationships.")
        lines.append("Concept type distribution:")
        for typ, cnt in community_data['type_distribution'].items():
            lines.append(f"  - {typ}: {cnt}")
        lines.append("\nImportant concepts (sorted by connectivity):")
        for node in community_data['top_nodes']:
            name = node['name']
            typ = node['type']
            degree = node.get('degree', 0)
            lines.append(f"  - {name} (type: {typ}, connectivity: {degree})")

        rels = community_data['relationships'][:MAX_RELATIONS_PER_COMMUNITY]
        if rels:
            lines.append("\nExample relationships:")
            for rel in rels:
                from_name = rel['from_name']
                to_name = rel['to_name']
                rel_type = rel['type']
                strength = rel['props'].get('strength', 'Unknown')
                lines.append(f"  - {from_name} -[{rel_type}]-> {to_name} (strength: {strength})")
        else:
            lines.append("\nNo direct relationships within this community.")

        lines.append("\nBased on the above information, provide a one-paragraph summary of the community's core cognitive theme in English.")
        prompt = "\n".join(lines)

        try:
            response = openai.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a cognitive graph analysis expert, skilled at summarizing themes from concepts and relationships."},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"生成总结时出错：{str(e)}"

    def analyze_topic(self, topic_name: str, max_communities: int = 10, min_community_size: int = 5) -> Dict:
        """
        对外提供的分析接口，返回可 JSON 序列化的结果
        """
        # 1. 检查话题是否存在
        node_count = self.get_topic_node_count(topic_name)
        if node_count == 0:
            raise ValueError(f"话题 '{topic_name}' 不存在或没有关联节点")

        # 2. 提取子图
        subgraph = self.extract_topic_subgraph(topic_name)
        node_ids = [n['id'] for n in subgraph['nodes']]

        # 3. 投影到 GDS
        graph_name = self.project_to_gds(topic_name, node_ids)

        # 4. 运行 Louvain
        community_map = self.run_louvain(graph_name)

        # 5. 分析社区
        community_details = self.analyze_communities(subgraph, community_map)

        # 6. 过滤小社区并按大小排序
        filtered = [
            (comm_id, data) for comm_id, data in community_details.items()
            if data['node_count'] >= min_community_size
        ]
        filtered.sort(key=lambda x: x[1]['node_count'], reverse=True)
        top_communities = filtered[:max_communities]

        # 7. 为每个社区生成总结
        result = {
            'topic': topic_name,
            'total_nodes': node_count,
            'total_communities': len(community_details),
            'filtered_communities': len(top_communities),
            'communities': []
        }
        for comm_id, data in top_communities:
            summary = self.summarize_community(comm_id, data)
            # 清理不可序列化的数据
            comm_info = {
                'community_id': comm_id,
                'node_count': data['node_count'],
                'rel_count': data['rel_count'],
                'type_distribution': data['type_distribution'],
                'top_nodes': [
                    {'name': n['name'], 'type': n['type'], 'degree': n.get('degree', 0)}
                    for n in data['top_nodes']
                ],
                'summary': summary
            }
            result['communities'].append(comm_info)

        return result