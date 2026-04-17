import json
import atexit
from flask import Flask, render_template, request, jsonify
from neo4j import GraphDatabase
import re

from community_analyzer import TopicAnalyzer as CommunityAnalyzer
from rag_system import CognitiveRAG

app = Flask(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pyp20040318"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def close_driver():
    if driver:
        driver.close()
atexit.register(close_driver)

try:
    rag = CognitiveRAG(driver)
    print("RAG system initialized successfully")
except Exception as e:
    print("RAG system initialization failed:", e)
    rag = None

# ---------- Helper functions (same as before, unchanged) ----------
def get_topics_baseline():
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE any(l IN labels(n) WHERE l STARTS WITH 'Topic_') "
            "RETURN DISTINCT [l IN labels(n) WHERE l STARTS WITH 'Topic_'][0] AS topic_label"
        )
        topics = []
        for record in result:
            label = record["topic_label"]
            if label:
                topic_name = label[6:]
                topics.append(topic_name)
        return sorted(set(topics))

def get_topics_cognitive():
    with driver.session() as session:
        result = session.run("MATCH (t:Topic) RETURN t.name AS name ORDER BY t.name")
        return [record["name"] for record in result]

def get_baseline_graph(label_filter, topic_filter):
    nodes = []
    edges = []
    with driver.session() as session:
        if topic_filter == "all":
            if label_filter == "all":
                node_query = """
                MATCH (n)
                WHERE (n:Entity_Baseline OR n:Entity_ZeroShot)
                AND any(l IN labels(n) WHERE l STARTS WITH 'Topic_')
                RETURN n, [l IN labels(n) WHERE l STARTS WITH 'Topic_'][0] AS topic_label
                """
            else:
                node_query = f"""
                MATCH (n:{label_filter})
                WHERE any(l IN labels(n) WHERE l STARTS WITH 'Topic_')
                RETURN n, [l IN labels(n) WHERE l STARTS WITH 'Topic_'][0] AS topic_label
                """
        else:
            topic_label = f"Topic_{topic_filter}"
            if label_filter == "all":
                node_query = f"""
                MATCH (n:{topic_label})
                WHERE n:Entity_Baseline OR n:Entity_ZeroShot
                RETURN n, '{topic_label}' AS topic_label
                """
            else:
                node_query = f"""
                MATCH (n:{label_filter}:{topic_label})
                RETURN n, '{topic_label}' AS topic_label
                """
        node_result = session.run(node_query)
        node_map = {}
        for record in node_result:
            n = record["n"]
            node_id = n.element_id
            props = dict(n)
            node_map[node_id] = {
                "id": node_id,
                "label": props.get("name", "Unknown"),
                "title": f"Name: {props.get('name')}<br>Topic: {record['topic_label']}",
                "group": record["topic_label"],
            }
            nodes.append(node_map[node_id])

        if nodes:
            if label_filter == "all":
                edge_query = """
                MATCH (s)-[r]->(t)
                WHERE (type(r) = 'RELATION_OPENIE' OR type(r) = 'RELATION_ZEROSHOT')
                AND elementId(s) IN $node_ids AND elementId(t) IN $node_ids
                RETURN s, r, t
                """
            else:
                rel_type = "RELATION_OPENIE" if label_filter == "Entity_Baseline" else "RELATION_ZEROSHOT"
                edge_query = f"""
                MATCH (s)-[r:{rel_type}]->(t)
                WHERE elementId(s) IN $node_ids AND elementId(t) IN $node_ids
                RETURN s, r, t
                """
            node_ids = list(node_map.keys())
            edge_result = session.run(edge_query, node_ids=node_ids)
            for record in edge_result:
                s = record["s"]
                r = record["r"]
                t = record["t"]
                edges.append({
                    "from": s.element_id,
                    "to": t.element_id,
                    "label": r.get("predicate", "Unknown"),
                    "title": f"Relation: {r.get('predicate')}<br>Topic: {r.get('topic')}",
                    "arrows": "to",
                })
    return {"nodes": nodes, "edges": edges}

def get_cognitive_graph(topic_filter):
    nodes = []
    edges = []
    with driver.session() as session:
        if topic_filter == "all":
            node_query = "MATCH (n:Entity_Cognitive) RETURN n"
            edge_query = """
            MATCH (s:Entity_Cognitive)-[r:RELATION_Cognitive]->(t:Entity_Cognitive)
            RETURN s, r, t
            """
        else:
            node_query = """
            MATCH (n:Entity_Cognitive)-[:IN_TOPIC]->(:Topic {name: $topic})
            RETURN n
            """
            edge_query = """
            MATCH (s:Entity_Cognitive)-[r:RELATION_Cognitive]->(t:Entity_Cognitive)
            WHERE (s)-[:IN_TOPIC]->(:Topic {name: $topic})
              AND (t)-[:IN_TOPIC]->(:Topic {name: $topic})
            RETURN s, r, t
            """
        node_result = session.run(node_query, topic=topic_filter)
        node_map = {}
        for record in node_result:
            n = record["n"]
            props = dict(n)
            node_id = n.element_id
            tooltip = f"Name: {props.get('name')}<br>Type: {props.get('type')}<br>Stance: {props.get('dominant_stance')}<br>Certainty: {props.get('dominant_certainty')}<br>Count: {props.get('total_count')}"
            node_map[node_id] = {
                "id": node_id,
                "label": props.get("name", "Unknown"),
                "title": tooltip,
                "group": props.get("type", "State"),
            }
            nodes.append(node_map[node_id])

        if nodes:
            edge_result = session.run(edge_query, topic=topic_filter)
            for record in edge_result:
                r = record["r"]
                s = record["s"]
                t = record["t"]
                props = dict(r)
                tooltip = f"Type: {props.get('predicate')}<br>Strength: {props.get('strength')}<br>Confidence: {props.get('confidence')}<br>Count: {props.get('count')}"
                edges.append({
                    "from": s.element_id,
                    "to": t.element_id,
                    "label": props.get("predicate", "Unknown"),
                    "title": tooltip,
                    "arrows": "to",
                })
    return {"nodes": nodes, "edges": edges}

# ---------- Page routes ----------
@app.route('/')
def index():
    return render_template('baseline.html')

@app.route('/baseline')
def baseline_page():
    return render_template('baseline.html')

@app.route('/cognitive')
def cognitive_page():
    return render_template('cognitive.html')

@app.route('/community')
def community_page():
    return render_template('community.html')

@app.route('/rag')
def rag_page():
    return render_template('rag.html')

# ---------- API routes ----------
@app.route('/api/baseline/topics')
def api_baseline_topics():
    topics = get_topics_baseline()
    return jsonify(topics)

@app.route('/api/baseline/graph')
def api_baseline_graph():
    label = request.args.get('label', 'all')
    topic = request.args.get('topic', 'all')
    data = get_baseline_graph(label, topic)
    return jsonify(data)

@app.route('/api/cognitive/topics')
def api_cognitive_topics():
    topics = get_topics_cognitive()
    return jsonify(topics)

@app.route('/api/cognitive/graph')
def api_cognitive_graph():
    topic = request.args.get('topic', 'all')
    data = get_cognitive_graph(topic)
    return jsonify(data)

@app.route('/api/community/topics')
def api_community_topics():
    with driver.session() as session:
        result = session.run("MATCH (t:Topic) RETURN t.name AS name ORDER BY t.name")
        topics = [record["name"] for record in result]
    return jsonify(topics)

@app.route('/api/community/analyze', methods=['POST'])
def api_community_analyze():
    data = request.get_json()
    topic = data.get('topic')
    max_communities = int(data.get('max_communities', 10))
    min_community_size = int(data.get('min_community_size', 5))

    if not topic:
        return jsonify({'error': 'Topic cannot be empty'}), 400

    try:
        analyzer = CommunityAnalyzer(driver)
        result = analyzer.analyze_topic(topic, max_communities, min_community_size)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/ask', methods=['POST'])
def api_rag_ask():
    if rag is None:
        return jsonify({'error': 'RAG system not initialized, please check configuration'}), 500
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    try:
        answer, context = rag.answer_with_context(question)
        return jsonify({'answer': answer, 'context': context})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)