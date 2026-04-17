"""
认知图谱话题分析脚本（优化版）
- 批量处理所有话题，依次进行 Louvain 社区发现
- 优化 LLM 提示词，更精准总结社区主题
- 输出每个话题的社区分析到 txt 文件（Community_Summary 文件夹）
"""

import os
import json
import time
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

from neo4j import GraphDatabase
import openai

# ================== 配置区域 ==================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# LLM 配置（使用 DeepSeek API）
OPENAI_API_KEY = "sk-82ab2070416a4613b249c60e3528ecab"
OPENAI_BASE_URL = "https://api.deepseek.com"
GENERATION_MODEL = "deepseek-chat"

# 社区总结参数
TOP_NODES_PER_COMMUNITY = 10          # 每个社区展示的重要节点数（按度数排序）
MAX_RELATIONS_PER_COMMUNITY = 20       # 每个社区展示的关系示例数
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 5000
MIN_COMMUNITY_SIZE = 5                 # 最小社区大小（小于此值的社区将被过滤）

# GDS 算法配置
LOUVAIN_MAX_LEVELS = 10
LOUVAIN_CONCURRENCY = 4

# 输出目录
OUTPUT_DIR = "Cognitive_Community_Summary"
# ==============================================

class TopicAnalyzer:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        openai.api_key = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            openai.base_url = OPENAI_BASE_URL
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def close(self):
        self.driver.close()

    def get_all_topics(self) -> List[str]:
        with self.driver.session() as session:
            result = session.run("MATCH (t:Topic) RETURN t.name AS name ORDER BY t.name")
            topics = [record["name"] for record in result]
        return topics

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
        """
        使用 Cypher projection 将话题子图投影到 GDS 内存图中。
        返回投影图名称，并在投影后验证图是否存在。
        """
        safe_topic = ''.join(c if c.isalnum() else '_' for c in topic_name)
        graph_name = f"topic_{safe_topic}_louvain"

        # 将 node_ids 列表转换为 Cypher 列表字符串，例如 ["id1","id2"]
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
            # 删除旧投影（如果存在）
            session.run(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")

            # 执行投影，并获取返回的统计信息
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
                raise Exception(f"GDS projection for graph '{graph_name}' failed (no result returned).")

            print(f"  GDS 投影统计: 图名={result['graphName']}, 节点数={result['nodeCount']}, 关系数={result['relationshipCount']}")

            # 验证图是否存在
            exists = session.run("CALL gds.graph.exists($name) YIELD exists", name=graph_name).single()["exists"]
            if not exists:
                raise Exception(f"GDS graph '{graph_name}' was not created successfully despite projection call.")

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
            community_map = {record['element_id']: record['communityId'] for record in result}
            return community_map

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
        """
        优化后的总结提示词：
        - 明确要求识别核心主题、关键节点及其关系模式
        - 引导LLM输出结构化的总结（主题、核心概念、关系特征）
        """
        lines = []
        lines.append(f"你是一位精通解读认知图谱，解构其中内容的认知图谱分析专家。请根据以下信息，对社区（ID：{community_id}）进行深入总结。")
        lines.append(f"该社区包含 {community_data['node_count']} 个认知概念节点，{community_data['rel_count']} 条关系。")
        lines.append("概念类型分布：")
        for typ, cnt in community_data['type_distribution'].items():
            lines.append(f"  - {typ}: {cnt} 个")
        lines.append("\n重要概念（按关联度排序）：")
        for node in community_data['top_nodes']:
            name = node['name']
            typ = node['type']
            degree = node.get('degree', 0)
            lines.append(f"  - {name} (类型：{typ}，关联度：{degree})")

        rels = community_data['relationships'][:MAX_RELATIONS_PER_COMMUNITY]
        if rels:
            lines.append("\n关键关系示例：")
            for rel in rels:
                from_name = rel['from_name']
                to_name = rel['to_name']
                rel_type = rel['type']
                strength = rel['props'].get('strength', '未知')
                lines.append(f"  - {from_name} -[{rel_type}]-> {to_name} (强度：{strength})")
        else:
            lines.append("\n该社区内没有直接关系。")

        lines.append("""
请基于以上信息，完成以下总结要求：
1. 核心主题：用一句话概括该社区讨论的核心认知主题。
2. 关键节点：指出在社区中起到枢纽作用的概念，并说明其角色。
3. 关系模式：分析社区内主要的关系类型（如因果、意图、相关等），并举例说明这些关系如何构建主题。
4. 综合描述：用一段话总结该社区的整体内容，包括可能的争议、共识或隐含的认知倾向。

陈述语言简洁准确。
""")

        prompt = "\n".join(lines)

        try:
            response = openai.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的认知图谱分析专家，擅长从概念网络中提炼主题和洞察。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            summary = f"生成总结时出错：{str(e)}"

        return summary

    def run_analysis_for_topic(self, topic_name: str) -> Dict[str, Any]:
        """对单个话题执行社区分析，返回结果字典"""
        print(f"\n开始分析话题：{topic_name}")
        node_count = self.get_topic_node_count(topic_name)
        if node_count < MIN_COMMUNITY_SIZE:
            print(f"  话题节点数 {node_count} 小于最小社区大小 {MIN_COMMUNITY_SIZE}，跳过。")
            return None

        print(f"  节点数：{node_count}")

        subgraph = self.extract_topic_subgraph(topic_name)
        node_ids = [n['id'] for n in subgraph['nodes']]
        print(f"  提取子图：{len(subgraph['nodes'])} 节点，{len(subgraph['relationships'])} 关系")

        graph_name = self.project_to_gds(topic_name, node_ids)
        print(f"  投影图：{graph_name}")

        start = time.time()
        community_map = self.run_louvain(graph_name)
        communities = set(community_map.values())
        print(f"  发现 {len(communities)} 个社区（耗时 {time.time()-start:.2f} 秒）")

        community_details = self.analyze_communities(subgraph, community_map)

        # 过滤小社区并按大小排序，取前10个
        filtered = [
            (comm_id, data) for comm_id, data in community_details.items()
            if data['node_count'] >= MIN_COMMUNITY_SIZE
        ]
        filtered.sort(key=lambda x: x[1]['node_count'], reverse=True)
        top_communities = filtered[:10]

        # 生成社区总结
        summaries = {}
        for comm_id, data in top_communities:
            summary = self.summarize_community(comm_id, data)
            summaries[comm_id] = summary

        result = {
            'topic': topic_name,
            'node_count': node_count,
            'community_count': len(communities),
            'filtered_community_count': len(top_communities),
            'min_community_size': MIN_COMMUNITY_SIZE,
            'communities': []
        }
        for comm_id, data in top_communities:
            result['communities'].append({
                'community_id': comm_id,
                'node_count': data['node_count'],
                'rel_count': data['rel_count'],
                'type_distribution': data['type_distribution'],
                'top_nodes': [
                    {'name': n['name'], 'type': n['type'], 'degree': n.get('degree', 0)}
                    for n in data['top_nodes']
                ],
                'summary': summaries.get(comm_id, '')
            })
        return result

    def save_txt_report(self, result: Dict[str, Any], filename: str):
        """将单个话题的分析结果保存为易读的txt文件"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"话题：{result['topic']}")
        lines.append(f"分析时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"总节点数：{result['node_count']}")
        lines.append(f"原始社区数：{result['community_count']}")
        lines.append(f"过滤后保留社区数（≥{result['min_community_size']}节点）：{result['filtered_community_count']}")
        lines.append("=" * 80)

        for idx, comm in enumerate(result['communities'], 1):
            lines.append(f"\n【社区 {comm['community_id']}】（排名 {idx}）")
            lines.append(f"  节点数：{comm['node_count']}，关系数：{comm['rel_count']}")
            lines.append("  类型分布：")
            for typ, cnt in comm['type_distribution'].items():
                lines.append(f"    - {typ}: {cnt}")
            lines.append("  重要节点（按关联度）：")
            for node in comm['top_nodes']:
                lines.append(f"    - {node['name']} ({node['type']}, 关联度 {node['degree']})")
            lines.append("\n  LLM 总结：")
            # 缩进总结内容
            summary_lines = comm['summary'].split('\n')
            for sl in summary_lines:
                lines.append(f"    {sl}")
            lines.append("-" * 60)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def run_all_topics(self, skip_empty=True):
        """批量处理所有话题，将每个话题的报告保存到 OUTPUT_DIR"""
        topics = self.get_all_topics()
        print(f"找到 {len(topics)} 个话题，开始批量分析...")

        for topic in topics:
            try:
                result = self.run_analysis_for_topic(topic)
                if result is None and skip_empty:
                    continue
                # 生成安全的文件名
                safe_name = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in topic)
                safe_name = safe_name.replace(' ', '_')
                filename = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")
                self.save_txt_report(result, filename)
                print(f"  报告已保存至：{filename}")
            except Exception as e:
                print(f"处理话题 '{topic}' 时出错：{e}")
                import traceback
                traceback.print_exc()
                continue

def main():
    analyzer = TopicAnalyzer()
    try:
        analyzer.run_all_topics(skip_empty=True)
        print(f"\n所有话题分析完成，报告保存在 {OUTPUT_DIR} 目录下。")
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()