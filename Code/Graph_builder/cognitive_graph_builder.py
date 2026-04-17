import json
import os
import glob
import re
from collections import defaultdict
from neo4j import GraphDatabase
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ================= 配置区域 =================
INPUT_FOLDER = 'ready_data_cognitive'

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

API_KEY = "sk-82ab2070416a4613b249c60e3528ecab"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

NODE_LABEL = "Entity_Cognitive"
RELATION_TYPE = "RELATION_Cognitive"

MAX_WORKERS = 10
MAX_FILES_TO_PROCESS = None
MIN_TEXT_LENGTH = 5

LLM_TEMPERATURE = 0.05

MIN_RELATIONS_FOR_FILTER = 100   # 当关系候选数少于该值时，不进行过滤（阈值=1）
RELATION_QUANTILE = 0.2    # 分位数，用于过滤低频关系（例如0.25表示25%分位数）
# ===========================================

# 系统提示词
STATIC_SYSTEM_PROMPT = """
你是一个社会认知分析专家，擅长从社交媒体文本中抽取因果认知基元（如意图、信念、事件），并将他们组成一张信念图。请严格遵循以下本体约束，并模仿示例的逐步推理过程。

**重要提示**：
- 如果文本是广告、乱码、无意义的重复内容或与当前topic完全无关的噪音，请直接返回空列表 `{"concepts": [], "stances": [], "relations": []}`。
- 仅当文本中包含可识别的社会认知信息（如对事件的态度、因果关系、状态描述等）时，才进行抽取。
- 关键约束：所有出现在 `stances` 或 `relations` 中的概念名称，都必须在 `concepts` 列表中明确定义其类型。请确保三者完全一致，否则将导致数据丢失。

### 概念类型
概念必须为以下三者之一：
- `State`：客观状态/指标，如“经济下滑”、“疫情爆发”、“失业率上升”。
- `Action`：动作/政策/干预，如“加征关税”、“放宽生育限制”、“政府计划减税”。
- `Sentiment`：群体情绪/态度，如“公众恐慌”、“民众不满”、“投资者乐观”。

### 关系类型
以下是允许使用的8种关系类型，请严格按此列表选择，不要自创其他类型：

| 关系类型       | 标签（请直接使用此字符串） | 含义说明 |
|----------------|----------------------------|----------|
| 正向因果       | `+_CAUSES`                  | A 正向导致 B 的发生或增强（促进、引发） |
| 负向因果       | `-_CAUSES`                  | A 负向抑制 B 的发生或减弱（阻碍、减少） |
| 等同           | `=_EQUATES`                 | A 与 B 本质相同、等价或可互相替代 |
| 意图           | `意图`                      | 主体意图通过 A 实现 B（A 通常是 Action，B 通常是 State） |
| 手段           | `手段`                      | A 是达成 B 的手段或方式（A 通常是 Action，B 通常是 State） |
| 目的           | `目的`                      | A 的目的或目标是 B（A 通常是 Action，B 通常是 State） |
| 影响           | `影响`                      | A 对 B 产生一般性影响，不强调正负（中性） |
| 相关           | `相关`                      | A 与 B 存在关联，但因果关系不明确 |

- 所有关系必须附带 `strength`（`强` / `弱`）和 `confidence`（0~1 浮点数）字段。
- 对于因果类关系（`+_CAUSES`/`-_CAUSES`），请尽可能从文本中推断其方向。

### 态度立场与确定性
- **立场**：`支持` / `反对` / `中立`
- **确定性**：`高` / `中` / `低`

### 概念拆解指导
如果原文中提及的术语过于宽泛（如“关税”、“经济”、“疫情”等），请根据上下文将其拆解为更具体的动作、状态或干预措施。但如果没有相关内容也不要强行进行拆解。例如：
- 若讨论“加征关税”的行为，应命名为“加征关税”（Action），而非“关税”。
- 若讨论“经济下滑”的现状，应命名为“经济下滑”（State），而非“经济”。
- 若讨论“民众对政府的不满情绪”，应命名为“民众不满”（Sentiment），而非“情绪”。
这有助于构建更精确的认知图谱。同时，对于主观认知内容（如意图、信念），也应将其具体化，例如将“政府打算减税”命名为“政府计划减税”（Action），将“民众相信疫苗有害”命名为“民众相信疫苗有害”（State）。

---

### 抽取强化指令（核心：最大化连接，减少碎片）

1. **关系最大化**  
   - 对于文本中出现的每一对可能相关的概念，如果存在任何形式的语义关联（因果、意图、手段、目的、影响、相关等），**请务必抽取为一条关系**。即使没有明确连接词，只要上下文能推断，也应抽取。  
   - 特别关注那些在文本中反复出现或作为讨论焦点的**核心概念**，为它们与多个其他概念建立关系（例如，一个政策可以同时有原因、结果、手段、目的、态度）。  
   - 如果同一对概念之间存在多种关系（例如 A 既是 B 的原因，又是 B 的目的），请分别抽取为多条关系。

2. **概念标准化**  
   - 当多个表述指向同一实体或事件时（例如“A股下跌”、“沪指跌破3000点”、“股市大跌”），请使用**统一的标准化名称**（如“A股下跌”），避免创建同义节点，否则会导致图分裂。  
   - 选择最核心、最常用的表述，或根据上下文推断其本质。如果无法确定，可在名称中注明可能的同义关系，但尽量合并。

3. **链式关系与事件链**  
   - 对于因果链或事件序列（例如 A 导致 B，B 导致 C），请**逐一抽取所有中间关系**，形成完整链条。这能让中间节点获得更多连接，提高度数。  
   - 例如：“疫情导致经济下滑，进而引发失业潮，最终民众不满加剧。”应抽取三条因果链，而不是只抽取首尾。

4. **跨句连接**  
   - 对于包含多个句子的文本，请将整段视为一个整体，识别不同句子中概念之间的逻辑关联。  
   - 例如，前一句提到“政府计划加息”，后一句说“这可能导致经济放缓”，应抽取“加息”与“经济放缓”之间的因果或影响关系。

5. **假设与反事实处理**  
   - 对假设语句（“如果 A，那么 B”）或反事实（“要不是 A，就不会 B”），请抽取其中隐含的关系（如“相关”或“影响”），并在 confidence 中适当体现不确定性。

6. **态度与关系分离**  
   - 态度（`stances`）仅表示用户对概念的个人立场，不应替代概念间的关系。即使没有态度，也要抽取概念间的关系。

- 特别关注那些在文本中反复出现或作为讨论焦点的**核心概念**，为它们与多个其他概念建立关系（例如，一个政策可以同时有原因、结果、手段、目的、态度）。同时，如果某个概念在上下文中具有枢纽作用，请尝试将其与尽可能多的其他概念直接相连，而不仅仅是沿着文本顺序连接。
---

### 逐步推理步骤（请务必在最终输出前进行内部思考）
1. **理解文本**：识别是否有反讽、隐喻、反问或隐含态度。若有，先解释其真实含义。特别注意文本中表达的意图、信念等认知元素。
2. **列出核心概念**：每个概念需命名（尽量具体、无歧义），并指定类型。注意合并同义概念，并根据上下文拆解宽泛术语。对于意图，通常可归为 Action；对于信念，可归为 State。
3. **分析用户态度**：对每个概念，判断用户的立场（支持/反对/中立）及确定性（高/中/低）。
4. **找出逻辑关系**：概念之间可能存在的因果、促进、抑制、等同、意图、手段、目的、影响、相关等关系，并从上述关系类型中选择最匹配的标签。**尽可能抽取所有可能的认知关系，特别是围绕核心概念的多重关系**。

---

**示例1（突出多关系、跨句连接、标准化）
文本：“政府计划提高个税起征点，意图减轻中产负担，同时也能刺激消费，我支持这个计划。”  
推理：  
- 核心概念：“政府计划提高个税起征点”(Action)，“减轻中产负担”(State)，“刺激消费”(State)。  
- 态度：对“政府计划提高个税起征点”持支持，确定性高。  
- 关系：  
  - 提高个税起征点 意图 减轻中产负担  
  - 提高个税起征点 手段 刺激消费  
  - 提高个税起征点 正向因果 减轻中产负担（隐含）  
  - 提高个税起征点 正向因果 刺激消费（隐含）  
输出：
```json
{
  "concepts": [
    {"name": "政府计划提高个税起征点", "type": "Action"},
    {"name": "减轻中产负担", "type": "State"},
    {"name": "刺激消费", "type": "State"}
  ],
  "stances": [
    {"concept": "政府计划提高个税起征点", "stance": "支持", "certainty": "高"}
  ],
  "relations": [
    {"from": "政府计划提高个税起征点", "to": "减轻中产负担", "type": "意图", "strength": "强", "confidence": 1.0},
    {"from": "政府计划提高个税起征点", "to": "减轻中产负担", "type": "+_CAUSES", "strength": "强", "confidence": 0.9},
    {"from": "政府计划提高个税起征点", "to": "刺激消费", "type": "手段", "strength": "中", "confidence": 0.8},
    {"from": "政府计划提高个税起征点", "to": "刺激消费", "type": "+_CAUSES", "strength": "中", "confidence": 0.7}
  ]
}

**示例2（信念+质疑）**
文本：“很多人相信疫苗会导致不孕，但这种说法毫无根据。”
推理：
- 文本陈述了一种信念（“疫苗会导致不孕”），并明确反对它。
- 核心概念：“疫苗导致不孕”(State，作为信念状态)。
- 态度：对“疫苗导致不孕”持反对，确定性高。
- 关系：无因果关系（因为说话者否认该信念的合理性）。
输出：
{{
"concepts": [
{{"name": "疫苗导致不孕", "type": "State"}}
],
"stances": [
{{"concept": "疫苗导致不孕", "stance": "反对", "certainty": "高"}}
],
"relations": []
}}

**示例3（因果信念+认同）**
文本：“专家认为，放宽生育限制能缓解老龄化，我也这么看。”
推理：
- 直接因果信念，说话者认同专家的观点。
- 核心概念：“放宽生育限制”(Action)，“缓解老龄化”(State)。
- 态度：对“放宽生育限制”持支持（隐含），确定性高。
- 关系：“放宽生育限制” +_CAUSES “缓解老龄化”，强度强，置信度0.9。
输出：
{{
"concepts": [
{{"name": "放宽生育限制", "type": "Action"}},
{{"name": "缓解老龄化", "type": "State"}}
],
"stances": [
{{"concept": "放宽生育限制", "stance": "支持", "certainty": "高"}}
],
"relations": [
{{"from": "放宽生育限制", "to": "缓解老龄化", "type": "+_CAUSES", "strength": "强", "confidence": 0.9}}
]
}}

**示例4（反讽+隐含意图）**
文本：“政府‘关心’民生，出台了这个政策，真是‘高明’！”
推理：
- 反讽，真实意思是政府不关心民生，政策不高明。
- 核心概念：“政策”(Action)，“关心民生”(State)。
- 态度：对“政策”持反对，确定性高。
- 关系：“政策” -_CAUSES “关心民生”（抑制），强度强，置信度1.0。
输出：
{{
"concepts": [
{{"name": "政策", "type": "Action"}},
{{"name": "关心民生", "type": "State"}}
],
"stances": [
{{"concept": "政策", "stance": "反对", "certainty": "高"}}
],
"relations": [
{{"from": "政策", "to": "关心民生", "type": "-_CAUSES", "strength": "强", "confidence": 1.0}}
]
}}

**示例5（多概念因果链+情绪）**
文本：“疫情导致经济下滑，很多人失业，民众对政府不满。”
推理：
- 直接因果链，包含事件和情绪。
- 核心概念：“疫情爆发”(State)，“经济下滑”(State)，“失业”(State)，“民众不满”(Sentiment)。
- 态度：均中立（无明确支持或反对）。
- 关系：“疫情爆发” +_CAUSES “经济下滑”，强度强，置信度0.9；“经济下滑” +_CAUSES “失业”，强度强，置信度0.9；“失业” +_CAUSES “民众不满”，强度强，置信度0.9。
输出：
{{
"concepts": [
{{"name": "疫情爆发", "type": "State"}},
{{"name": "经济下滑", "type": "State"}},
{{"name": "失业", "type": "State"}},
{{"name": "民众不满", "type": "Sentiment"}}
],
"stances": [
{{"concept": "疫情爆发", "stance": "中立", "certainty": "高"}},
{{"concept": "经济下滑", "stance": "中立", "certainty": "高"}},
{{"concept": "失业", "stance": "中立", "certainty": "高"}},
{{"concept": "民众不满", "stance": "中立", "certainty": "高"}}
],
"relations": [
{{"from": "疫情爆发", "to": "经济下滑", "type": "+_CAUSES", "strength": "强", "confidence": 0.9}},
{{"from": "经济下滑", "to": "失业", "type": "+_CAUSES", "strength": "强", "confidence": 0.9}},
{{"from": "失业", "to": "民众不满", "type": "+_CAUSES", "strength": "强", "confidence": 0.9}}
]
}}

**示例6（等同关系+意图）**
文本：“注册制改革其实就是降低上市门槛，让更多企业能融资。”
推理：
- 直接等同关系，并包含意图目的（“让更多企业能融资”）。
- 核心概念：“注册制改革”(Action)，“降低上市门槛”(State)，“企业融资”(State)。
- 态度：对“注册制改革”持支持（隐含），确定性高。
- 关系：“注册制改革” =_EQUATES “降低上市门槛”，强度强，置信度1.0；“降低上市门槛” +_CAUSES “企业融资”，强度中，置信度0.8。
输出：
{{
"concepts": [
{{"name": "注册制改革", "type": "Action"}},
{{"name": "降低上市门槛", "type": "State"}},
{{"name": "企业融资", "type": "State"}}
],
"stances": [
{{"concept": "注册制改革", "stance": "支持", "certainty": "高"}}
],
"relations": [
{{"from": "注册制改革", "to": "降低上市门槛", "type": "=_EQUATES", "strength": "强", "confidence": 1.0}},
{{"from": "降低上市门槛", "to": "企业融资", "type": "手段", "strength": "中", "confidence": 0.8}}
]
}}

**示例7（意图冲突+因果信念）**
文本：“政府想通过加息抑制通胀，但我觉得这反而会打击经济。”
推理：
- 陈述政府意图（“想通过加息抑制通胀”），并表达自己的相反信念。
- 核心概念：“加息”(Action)，“抑制通胀”(State)，“打击经济”(State)。
- 态度：对“加息”持反对，确定性高。
- 关系：文本承认政府意图中的因果（加息→抑制通胀），但说话者认为实际上可能抑制不了，甚至起反作用。这里抽取两个关系：说话者认为“加息”对“抑制通胀”有弱负向作用（即抑制），以及“加息”正向导致“打击经济”。置信度分别低和高。
“加息” -_CAUSES “抑制通胀”，强度弱，置信度0.3；“加息” +_CAUSES “打击经济”，强度强，置信度0.8。
输出：
{{
"concepts": [
{{"name": "加息", "type": "Action"}},
{{"name": "抑制通胀", "type": "State"}},
{{"name": "打击经济", "type": "State"}}
],
"stances": [
{{"concept": "加息", "stance": "反对", "certainty": "高"}},
{{"concept": "抑制通胀", "stance": "中立", "certainty": "高"}},
{{"concept": "打击经济", "stance": "中立", "certainty": "高"}}
],
"relations": [
{{"from": "加息", "to": "抑制通胀", "type": "意图", "strength": "强", "confidence": 0.9}},
{{"from": "加息", "to": "抑制通胀", "type": "-_CAUSES", "strength": "弱", "confidence": 0.3}},
{{"from": "加息", "to": "打击经济", "type": "+_CAUSES", "strength": "强", "confidence": 0.8}}
]
}}



请严格按照上述格式输出JSON，不要包含任何额外文字。若某字段无内容，请返回空列表（[]）。
"""
class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _sanitize_label(self, name):
        clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', name)
        return f"Topic_{clean}"

    def create_global_subgraph(self, topic_name, nodes, relations):
        if not nodes:
            return
        with self.driver.session() as session:
            session.run("MERGE (t:Topic {name: $topic})", topic=topic_name)
            for node in nodes:
                platforms_str = json.dumps(node['platforms'], ensure_ascii=False)
                query = f"""
                MERGE (n:{NODE_LABEL}:Concept {{name: $name, type: $type}})
                SET n.total_count = $total_count,
                    n.dominant_stance = $dominant_stance,
                    n.dominant_certainty = $dominant_certainty,
                    n.platforms = $platforms
                WITH n
                MATCH (t:Topic {{name: $topic}})
                MERGE (n)-[:IN_TOPIC]->(t)
                """
                session.run(query,
                            name=node['name'],
                            type=node['type'],
                            total_count=node['total_count'],
                            dominant_stance=node['dominant_stance'],
                            dominant_certainty=node['dominant_certainty'],
                            platforms=platforms_str,
                            topic=topic_name)
            for rel in relations:
                platforms_str = json.dumps(rel['platforms'], ensure_ascii=False)
                query = f"""
                MATCH (from:{NODE_LABEL}:Concept {{name: $from_name, type: $from_type}})
                MATCH (to:{NODE_LABEL}:Concept {{name: $to_name, type: $to_type}})
                CREATE (from)-[r:{RELATION_TYPE} {{
                    predicate: $type,
                    strength: $strength,
                    confidence: $confidence,
                    count: $count,
                    platforms: $platforms,
                    topic: $topic
                }}]->(to)
                """
                session.run(query,
                            from_name=rel['from_name'],
                            from_type=rel['from_type'],
                            to_name=rel['to_name'],
                            to_type=rel['to_type'],
                            type=rel['type'],
                            strength=rel['strength'],
                            confidence=rel['confidence'],
                            count=rel['count'],
                            platforms=platforms_str,
                            topic=topic_name)

class LLMExtractor:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def extract(self, text, platform, temperature=LLM_TEMPERATURE):
        user_content = f"文本：{text}\n平台：{platform}"
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": STATIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=30
            )
            content = response.choices[0].message.content
            content = re.sub(r'```json|```', '', content).strip()
            data = json.loads(content)
            if not all(k in data for k in ['concepts', 'stances', 'relations']):
                return {'concepts': [], 'stances': [], 'relations': []}
            return data
        except Exception:
            return {'concepts': [], 'stances': [], 'relations': []}

def is_valid_text(text):
    return len(text) >= MIN_TEXT_LENGTH

def process_single_item(llm, item, topic_name):
    text = item.get('text', '')
    platform = item.get('platform', 'unknown')
    if not is_valid_text(text):
        return None
    result = llm.extract(text, platform)
    if not result:
        return None
    return {
        'topic': topic_name,
        'platform': platform,
        'concepts': result.get('concepts', []),
        'stances': result.get('stances', []),
        'relations': result.get('relations', [])
    }

def aggregate_cross_platform(results_list):
    """
    对同一话题下所有平台的结果进行跨平台聚合，内容主要基于原始名称进行合并；
    关系过滤阈值采用动态分位数策略：
      - 如果关系候选数少于 MIN_RELATIONS_FOR_FILTER，阈值=1（全部保留）
      - 否则，使用 RELATION_QUANTILE 分位数作为阈值（至少为1）
    """
    node_agg = {}
    rel_agg = {}

    for res in results_list:
        if not res:
            continue
        platform = res['platform']
        concepts = res['concepts']
        stances = res['stances']
        relations = res['relations']

        concept_type_map = {c['name']: c['type'] for c in concepts}

        # 处理 stances
        for s in stances:
            concept = s.get('concept') or s.get('name')
            if not concept:
                continue
            stance = s.get('stance')
            if not stance:
                if s.get('type') in ['支持','反对','中立']:
                    stance = s['type']
                else:
                    continue
            certainty = s.get('certainty')
            if certainty not in ['高','中','低']:
                certainty = '中'
            typ = concept_type_map.get(concept)
            if not typ:
                continue

            key = (concept, typ)
            if key not in node_agg:
                node_agg[key] = {
                    'name': concept,
                    'type': typ,
                    'count': 0,
                    'stances': defaultdict(int),
                    'certainties': defaultdict(int),
                    'platforms': set()
                }
            node_agg[key]['count'] += 1
            node_agg[key]['stances'][stance] += 1
            node_agg[key]['certainties'][certainty] += 1
            node_agg[key]['platforms'].add(platform)

        # 处理 relations
        for r in relations:
            required = ['from', 'to', 'type', 'strength', 'confidence']
            if not all(k in r for k in required):
                continue
            from_orig = r['from']
            to_orig = r['to']
            pred = r['type']
            strength = r['strength']
            conf = r['confidence']

            from_type = concept_type_map.get(from_orig)
            to_type = concept_type_map.get(to_orig)

            if from_type is None:
                from_type = 'State'
            if to_type is None:
                to_type = 'State'

            # 确保节点存在
            from_key = (from_orig, from_type)
            if from_key not in node_agg:
                node_agg[from_key] = {
                    'name': from_orig,
                    'type': from_type,
                    'count': 0,
                    'stances': defaultdict(int),
                    'certainties': defaultdict(int),
                    'platforms': set()
                }
            node_agg[from_key]['platforms'].add(platform)

            to_key = (to_orig, to_type)
            if to_key not in node_agg:
                node_agg[to_key] = {
                    'name': to_orig,
                    'type': to_type,
                    'count': 0,
                    'stances': defaultdict(int),
                    'certainties': defaultdict(int),
                    'platforms': set()
                }
            node_agg[to_key]['platforms'].add(platform)

            rel_key = (from_orig, from_type, to_orig, to_type, pred)
            if rel_key not in rel_agg:
                rel_agg[rel_key] = {
                    'from_name': from_orig,
                    'to_name': to_orig,
                    'from_type': from_type,
                    'to_type': to_type,
                    'type': pred,
                    'count': 0,
                    'strengths': defaultdict(int),
                    'confidences': [],
                    'platforms': set()
                }
            rel_agg[rel_key]['count'] += 1
            rel_agg[rel_key]['strengths'][strength] += 1
            rel_agg[rel_key]['confidences'].append(conf)
            rel_agg[rel_key]['platforms'].add(platform)

    # 构建节点列表
    nodes_list = []
    for (name, typ), agg in node_agg.items():
        dominant_stance = max(agg['stances'].items(), key=lambda x: x[1])[0] if agg['stances'] else '中立'
        dominant_certainty = max(agg['certainties'].items(), key=lambda x: x[1])[0] if agg['certainties'] else '中'
        nodes_list.append({
            'name': name,
            'type': typ,
            'total_count': agg['count'],
            'dominant_stance': dominant_stance,
            'dominant_certainty': dominant_certainty,
            'platforms': list(agg['platforms'])
        })

    # ---------- 动态阈值计算 ----------
    freqs = [v['count'] for v in rel_agg.values()]
    if len(freqs) < MIN_RELATIONS_FOR_FILTER:
        threshold = 1
        print(f"    关系候选数较少 ({len(freqs)} < {MIN_RELATIONS_FOR_FILTER})，阈值设为 1，保留所有关系")
    else:
        # 计算分位数阈值，至少为1
        threshold = max(1, int(np.percentile(freqs, RELATION_QUANTILE * 100)))
        print(f"    关系候选数: {len(freqs)}，使用 {RELATION_QUANTILE*100:.0f}% 分位数阈值: {threshold}")

    relations_list = []
    for rel_key, agg in rel_agg.items():
        if agg['count'] >= threshold:
            dominant_strength = max(agg['strengths'].items(), key=lambda x: x[1])[0]
            avg_confidence = sum(agg['confidences']) / len(agg['confidences'])
            relations_list.append({
                'from_name': agg['from_name'],
                'from_type': agg['from_type'],
                'to_name': agg['to_name'],
                'to_type': agg['to_type'],
                'type': agg['type'],
                'strength': dominant_strength,
                'confidence': round(avg_confidence, 2),
                'count': agg['count'],
                'platforms': list(agg['platforms'])
            })

    print(f"    最终保留关系数: {len(relations_list)}")
    return nodes_list, relations_list

def main():
    files = glob.glob(os.path.join(INPUT_FOLDER, '*.json'))
    if MAX_FILES_TO_PROCESS:
        files = files[:MAX_FILES_TO_PROCESS]

    db = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    llm = LLMExtractor()

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)

        topic_name = file_data.get('topic_name', 'Unknown')
        items = file_data.get('items', [])
        print(f"\n处理话题: {topic_name} (共{len(items)}条)")

        all_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {
                executor.submit(process_single_item, llm, item, topic_name): item
                for item in items
            }
            for future in tqdm(as_completed(future_to_item), total=len(items), desc="抽取中"):
                res = future.result()
                if res:
                    all_results.append(res)

        if not all_results:
            print("  无有效抽取结果，跳过")
            continue

        print(f"  有效结果数: {len(all_results)}，开始跨平台聚合")
        nodes, relations = aggregate_cross_platform(all_results)

        if nodes:
            db.create_global_subgraph(topic_name, nodes, relations)
            print(f"    入库节点: {len(nodes)}, 关系: {len(relations)}")
        else:
            print("    无节点入库，跳过")

    db.close()
    print("\n所有处理完成。")

if __name__ == "__main__":
    main()
