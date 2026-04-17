import json
import os
import glob
import time
from neo4j import GraphDatabase
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================
# 1. 输入数据路径 (确保与 LLM 使用相同数据源)
INPUT_FOLDER = 'ready_data'

# 2. Neo4j 数据库配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# 3. API 配置
API_KEY = "sk-82ab2070416a4613b249c60e3528ecab"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 4. 实验标签
NODE_LABEL = "Entity_ZeroShot"
RELATION_TYPE = "RELATION_ZEROSHOT"

# 5. 功能开关
MAX_WORKERS = 10  # 并发设置
MAX_FILES_TO_PROCESS = None
# ===========================================

STATIC_SYSTEM_PROMPT = """
你是一个专业的社交媒体信息抽取专家，专为提取结构化格式信息以构建知识图谱。你的任务是从用户提供的文本中提取关键的“实体-关系-实体”认知三元组。

### 提取规则
1. **实体 (Entity)**：提取文本中涉及的关键主体（如人名、机构、政策、事件）或客体。
2. **关系 (Predicate)**：提取实体之间的动作、态度、因果或属性关系。
3. **准确性**：仅提取明确表达的事实或观点。

### 输出格式要求
1. 必须严格输出为 **JSON 列表** 格式。
2. 列表元素包含 `subject`, `predicate`, `object`。
3. 不包含任何额外文字。
4. 如果未提取到有效信息，请返回空列表 `[]`。

### 示例
输入："股民对A股跌破3400点感到失望。"
输出：[
    {"subject": "股民", "predicate": "态度", "object": "失望"},
    {"subject": "A股", "predicate": "状态", "object": "跌破3400点"}
]
"""

class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _sanitize_label(self, topic_name):
        import re
        clean_name = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', topic_name)
        return f"Topic_{clean_name}"

    def create_triples(self, topic_name, triples):
        if not triples: return
        topic_label = self._sanitize_label(topic_name)
        query = f"""
        UNWIND $batch AS row
        MERGE (s:{NODE_LABEL}:{topic_label} {{name: row.subject}})
        MERGE (o:{NODE_LABEL}:{topic_label} {{name: row.object}})
        CREATE (s)-[r:{RELATION_TYPE} {{predicate: row.predicate}}]->(o)
        SET r.topic = $topic, r.created_at = datetime()
        """
        with self.driver.session() as session:
            try:
                session.run(query, batch=triples, topic=topic_name)
            except Exception as e:
                print(f"Neo4j写入错误: {e}")

class LLMExtractor:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def extract(self, text):
        user_content = f"待分析文本：\n{text}"
        try:
            # 设置 timeout 防止网络卡死
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": STATIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                stream=False,
                timeout=30
            )
            content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            if isinstance(data, dict):
                for k in data:
                    if isinstance(data[k], list): return data[k]
                return []
            return data if isinstance(data, list) else []
        except Exception as e:
            # print(f"Error: {e}")
            return []

# 包装函数：用于多线程调用
def process_single_item(llm, item):
    text = item.get('text', '')
    if len(text) < 5: return []
    return llm.extract(text)

def main():
    files = glob.glob(os.path.join(INPUT_FOLDER, '*.json'))
    if MAX_FILES_TO_PROCESS: files = files[:MAX_FILES_TO_PROCESS]

    db = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    llm = LLMExtractor()

    total_count = 0
    print(f"开始多线程实验，线程数: {MAX_WORKERS}")

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)

        topic_name = file_data.get('topic_name', 'Unknown')
        items = file_data.get('items', [])
        print(f"\n正在处理: {topic_name} (共 {len(items)} 条)")

        # === 核心修改：多线程并发 ===
        file_triples = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_item = {executor.submit(process_single_item, llm, item): item for item in items}

            # 使用 tqdm 显示进度，as_completed 会在任务完成时立即返回
            for future in tqdm(as_completed(future_to_item), total=len(items), desc="并发抽取中"):
                triples = future.result()
                if triples:
                    file_triples.extend(triples)

                # 优化：每收集 500 个结果就写入一次，避免内存占用过大
                # 注意：这里是在主线程写入，绝对安全
                if len(file_triples) >= 500:
                    db.create_triples(topic_name, file_triples)
                    total_count += len(file_triples)
                    file_triples = [] # 清空缓存

        # 写入该文件剩余的数据
        if file_triples:
            db.create_triples(topic_name, file_triples)
            total_count += len(file_triples)

    db.close()
    print(f"\n实验结束。总计入库: {total_count}")

if __name__ == "__main__":
    main()