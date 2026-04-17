import json
import os
import glob
import re
import stanza
from neo4j import GraphDatabase
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 输入数据路径 (确保与 LLM 使用相同数据源)
INPUT_FOLDER = 'ready_data'

# 2. Neo4j 数据库配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

# 3. 实验标签
NODE_LABEL = "Entity_Baseline"      # 主标签 (OpenIE 专用)
RELATION_TYPE = "RELATION_OPENIE"   # 关系类型

# 4. 功能开关
MAX_FILES_TO_PROCESS = None         # None = 处理所有文件
# ===========================================

class ChineseOpenIE:
    def __init__(self):
        print("正在加载 Stanza 中文模型 (将使用 GPU 加速)...")
        try:
            # 首次运行需要下载模型，可通过 stanza.download('zh') 提前下载
            self.nlp = stanza.Pipeline(
                'zh',
                processors='tokenize,pos,lemma,depparse',
                use_gpu=True,                 # 启用 GPU 加速
                tokenize_batch_size=32,       # 适当增大 batch 提升吞吐
                pos_batch_size=32,
                depparse_batch_size=32
            )
            print("Stanza 模型加载成功，GPU 已启用。")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请先运行 stanza.download('zh') 下载模型。")
            exit()

    def extract_triples(self, text):
        """输入文本，返回三元组列表 [{'subject':..., 'predicate':..., 'object':...}]"""
        if not text or len(text) < 2:
            return []

        # 截断长文本 (OpenIE 处理长句较慢)
        doc = self.nlp(text[:500])
        triples = []

        for sent in doc.sentences:
            # 寻找核心动词 (ROOT)
            for word in sent.words:
                # 依存关系为 'root' 且词性为动词
                if word.deprel == 'root' and word.upos in ['VERB', 'AUX']:
                    subject = None
                    obj = None

                    # 查找主语 (nsubj, nsubj:pass)
                    for child in sent.words:
                        if child.head == word.id and child.deprel in ['nsubj', 'nsubj:pass']:
                            subject = child.text
                            break

                    # 查找宾语 (obj, dobj, attr)
                    for child in sent.words:
                        if child.head == word.id and child.deprel in ['obj', 'dobj', 'attr']:
                            obj = child.text
                            break

                    if subject and obj:
                        triples.append({
                            "subject": subject,
                            "predicate": word.text,
                            "object": obj
                        })
        return triples


class Neo4jHandler:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("Neo4j 连接成功。")
        except Exception as e:
            print(f"Neo4j 连接失败: {e}")
            raise e

    def close(self):
        self.driver.close()

    def _sanitize_label(self, topic_name):
        """生成合法的话题标签: Topic_A股下跌..."""
        clean_name = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', topic_name)
        return f"Topic_{clean_name}"

    def create_triples(self, topic_name, triples):
        if not triples:
            return

        topic_label = self._sanitize_label(topic_name)

        # 使用 CREATE 保留碎片化，不聚合
        query = f"""
        UNWIND $batch AS row
        MERGE (s:{NODE_LABEL}:{topic_label} {{name: row.subject}})
        MERGE (o:{NODE_LABEL}:{topic_label} {{name: row.object}})
        
        CREATE (s)-[r:{RELATION_TYPE} {{predicate: row.predicate}}]->(o)
        SET r.topic = $topic,
            r.created_at = datetime()
        """

        with self.driver.session() as session:
            try:
                session.run(query, batch=triples, topic=topic_name)
            except Exception as e:
                print(f"写入失败: {e}")


def main():
    # 1. 检查文件
    files = glob.glob(os.path.join(INPUT_FOLDER, '*.json'))
    if not files:
        print(f"在 {INPUT_FOLDER} 中未找到文件。")
        return

    if MAX_FILES_TO_PROCESS is not None:
        files = files[:MAX_FILES_TO_PROCESS]

    # 2. 初始化
    try:
        extractor = ChineseOpenIE()
        db = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    total_triples = 0
    print(f"开始 Baseline (OpenIE with Stanza) 实验，处理 {len(files)} 个文件...")

    # 3. 循环处理
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)

        topic_name = file_data.get('topic_name', 'Unknown')
        items = file_data.get('items', [])

        if not items:
            continue

        print(f"\n正在提取: {topic_name} (共 {len(items)} 条)")

        batch_triples = []

        for item in tqdm(items):
            text = item.get('text', '')
            triples = extractor.extract_triples(text)

            if triples:
                batch_triples.extend(triples)

            # 批处理写入 (防止内存溢出)
            if len(batch_triples) >= 500:
                db.create_triples(topic_name, batch_triples)
                total_triples += len(batch_triples)
                batch_triples = []

        # 写入剩余
        if batch_triples:
            db.create_triples(topic_name, batch_triples)
            total_triples += len(batch_triples)

    db.close()
    print(f"\nBaseline (OpenIE with Stanza) 实验结束。总入库三元组: {total_triples}")
    print(f"主标签: :{NODE_LABEL}")


if __name__ == "__main__":
    main()