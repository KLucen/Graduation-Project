"""
认知图谱话题分析脚本（双版本版）
- 同时分析 Entity_Baseline (OpenIE) 和 Entity_ZeroShot 图谱
- 基于节点标签 Topic_xxx 识别话题
- 分别输出到 OpenIE_Community_summary 和 Zeroshot_Community_summary 文件夹
"""

import os
import json
import time
from collections import defaultdict, Counter
from typing import List, Dict, Any

from neo4j import GraphDatabase
import openai

# ================== 配置区域 ==================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# LLM 配置
OPENAI_API_KEY = "sk-82ab2070416a4613b249c60e3528ecab"
OPENAI_BASE_URL = "https://api.deepseek.com"
GENERATION_MODEL = "deepseek-chat"

# 社区总结参数
TOP_NODES_PER_COMMUNITY = 10
MAX_RELATIONS_PER_COMMUNITY = 20
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 5000
MIN_COMMUNITY_SIZE = 5

# GDS 算法配置
LOUVAIN_MAX_LEVELS = 10
LOUVAIN_CONCURRENCY = 4

# ==============================================

class TopicAnalyzer:
    def __init__(self, node_label: str, relation_type: str, output_dir: str):
        """
        :param node_label: 节点标签，例如 Entity_Baseline 或 Entity_ZeroShot
        :param relation_type: 关系类型，例如 RELATION_OPENIE 或 RELATION_ZEROSHOT
        :param output_dir: 输出目录名
        """
        self.node_label = node_label
        self.relation_type = relation_type
        self.output_dir = output_dir
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        openai.api_key = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            openai.base_url = OPENAI_BASE_URL
        os.makedirs(self.output_dir, exist_ok=True)

    def close(self):
        self.driver.close()

    def get_topics_with_nodes(self) -> List[Dict[str, str]]:
        """
        返回话题列表，每个元素包含话题原始名和话题标签。
        通过查找同时包含主标签和 Topic_ 前缀标签的节点来确定话题。
        """
        with self.driver.session() as session:
            # 获取所有以 Topic_ 开头的标签
            result = session.run("CALL db.labels() YIELD label")
            topic_labels = [record["label"] for record in result if record["label"].startswith("Topic_")]
            if not topic_labels:
                return []

            topics = []
            for topic_label in topic_labels:
                # 检查是否有节点同时具有主标签和该话题标签
                check = session.run(
                    f"MATCH (n:{self.node_label}:{topic_label}) RETURN count(n) AS cnt LIMIT 1"
                ).single()
                if check and check["cnt"] > 0:
                    # 原始话题名去掉 "Topic_" 前缀
                    topic_name = topic_label[6:]  # 去掉 "Topic_"
                    topics.append({"name": topic_name, "label": topic_label})
            return topics

    def get_topic_node_count(self, topic_label: str) -> int:
        with self.driver.session() as session:
            result = session.run(
                f"MATCH (n:{self.node_label}:{topic_label}) RETURN count(n) AS cnt"
            ).single()
            return result["cnt"] if result else 0

    def extract_topic_subgraph(self, topic_label: str) -> Dict[str, Any]:
        """提取指定话题标签下的所有节点和关系"""
        with self.driver.session() as session:
            # 获取节点
            nodes_result = session.run(
                f"""
                MATCH (c:{self.node_label}:{topic_label})
                RETURN elementId(c) AS id, c.name AS name, c.type AS type,
                       properties(c) AS props
                """
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

            # 获取关系（仅限节点间的关系，且类型匹配）
            rels_result = session.run(
                f"""
                MATCH (c1:{self.node_label})-[r:{self.relation_type}]-(c2:{self.node_label})
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
        """使用 Cypher projection 将话题子图投影到 GDS 内存图"""
        safe_topic = ''.join(c if c.isalnum() else '_' for c in topic_name)
        graph_name = f"{self.node_label}_{safe_topic}_louvain"

        node_ids_str = '[' + ','.join(f'"{nid}"' for nid in node_ids) + ']'

        # 注意：这里需要同时用节点标签限制，但 node_ids 已经包含了具体的 elementId
        # 我们仍需要用节点标签来辅助 GDS 投影，但主要靠 elementId 过滤
        node_query = f"""
            MATCH (c:{self.node_label})
            WHERE elementId(c) IN {node_ids_str}
            RETURN id(c) AS id
        """
        rel_query = f"""
            MATCH (c1:{self.node_label})-[r:{self.relation_type}]->(c2:{self.node_label})
            WHERE elementId(c1) IN {node_ids_str} AND elementId(c2) IN {node_ids_str}
            RETURN id(c1) AS source, id(c2) AS target, type(r) AS type
        """

        with self.driver.session() as session:
            # 删除旧投影
            session.run(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName")

            # 执行投影
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
                raise Exception(f"GDS projection for graph '{graph_name}' failed.")

            print(f"  GDS 投影统计: 图名={result['graphName']}, 节点数={result['nodeCount']}, 关系数={result['relationshipCount']}")

            # 验证图存在
            exists = session.run("CALL gds.graph.exists($name) YIELD exists", name=graph_name).single()["exists"]
            if not exists:
                raise Exception(f"GDS graph '{graph_name}' was not created successfully.")

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

    def run_analysis_for_topic(self, topic_name: str, topic_label: str) -> Dict[str, Any]:
        print(f"\n开始分析话题：{topic_name} (标签: {topic_label})")
        node_count = self.get_topic_node_count(topic_label)
        if node_count < MIN_COMMUNITY_SIZE:
            print(f"  话题节点数 {node_count} 小于最小社区大小 {MIN_COMMUNITY_SIZE}，跳过。")
            return None

        print(f"  节点数：{node_count}")

        subgraph = self.extract_topic_subgraph(topic_label)
        node_ids = [n['id'] for n in subgraph['nodes']]
        print(f"  提取子图：{len(subgraph['nodes'])} 节点，{len(subgraph['relationships'])} 关系")

        graph_name = self.project_to_gds(topic_name, node_ids)
        print(f"  投影图：{graph_name}")

        start = time.time()
        community_map = self.run_louvain(graph_name)
        communities = set(community_map.values())
        print(f"  发现 {len(communities)} 个社区（耗时 {time.time()-start:.2f} 秒）")

        community_details = self.analyze_communities(subgraph, community_map)

        filtered = [
            (comm_id, data) for comm_id, data in community_details.items()
            if data['node_count'] >= MIN_COMMUNITY_SIZE
        ]
        filtered.sort(key=lambda x: x[1]['node_count'], reverse=True)
        top_communities = filtered[:10]

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
            summary_lines = comm['summary'].split('\n')
            for sl in summary_lines:
                lines.append(f"    {sl}")
            lines.append("-" * 60)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def run_all_topics(self):
        topics = self.get_topics_with_nodes()
        if not topics:
            print(f"未找到任何话题节点（标签 {self.node_label}）。")
            return

        print(f"找到 {len(topics)} 个话题，开始批量分析...")
        for topic_info in topics:
            topic_name = topic_info["name"]
            topic_label = topic_info["label"]
            try:
                result = self.run_analysis_for_topic(topic_name, topic_label)
                if result is None:
                    continue
                safe_name = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in topic_name)
                safe_name = safe_name.replace(' ', '_')
                filename = os.path.join(self.output_dir, f"{safe_name}.txt")
                self.save_txt_report(result, filename)
                print(f"  报告已保存至：{filename}")
            except Exception as e:
                print(f"处理话题 '{topic_name}' 时出错：{e}")
                import traceback
                traceback.print_exc()
                continue


def main():
    # 配置1：OpenIE 图谱
    analyzer_openie = TopicAnalyzer(
        node_label="Entity_Baseline",
        relation_type="RELATION_OPENIE",
        output_dir="OpenIE_Community_summary"
    )
    print("\n========== 分析 OpenIE 图谱 (Entity_Baseline) ==========")
    try:
        analyzer_openie.run_all_topics()
    finally:
        analyzer_openie.close()

    # 配置2：ZeroShot 图谱
    analyzer_zeroshot = TopicAnalyzer(
        node_label="Entity_ZeroShot",
        relation_type="RELATION_ZEROSHOT",
        output_dir="Zeroshot_Community_summary"
    )
    print("\n========== 分析 ZeroShot 图谱 (Entity_ZeroShot) ==========")
    try:
        analyzer_zeroshot.run_all_topics()
    finally:
        analyzer_zeroshot.close()

    print("\n所有话题分析完成。")


if __name__ == "__main__":
    main()