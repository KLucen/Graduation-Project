// ================== Markdown to HTML ==================
function markdownToHtml(text) {
    if (!text) return '';

    const lines = text.split('\n');
    let inUl = false;
    let inOl = false;
    let result = [];

    for (let line of lines) {
        const ulMatch = line.match(/^-\s+(.*)/);
        if (ulMatch) {
            if (!inUl) {
                if (inOl) { result.push('</ol>'); inOl = false; }
                result.push('<ul>');
                inUl = true;
            }
            result.push(`<li>${markdownToHtml(ulMatch[1])}</li>`);
            continue;
        }

        const olMatch = line.match(/^\d+\.\s+(.*)/);
        if (olMatch) {
            if (!inOl) {
                if (inUl) { result.push('</ul>'); inUl = false; }
                result.push('<ol>');
                inOl = true;
            }
            result.push(`<li>${markdownToHtml(olMatch[1])}</li>`);
            continue;
        }

        if (inUl) {
            result.push('</ul>');
            inUl = false;
        }
        if (inOl) {
            result.push('</ol>');
            inOl = false;
        }

        let processedLine = line
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
        result.push(processedLine ? processedLine : '<br>');
    }

    if (inUl) result.push('</ul>');
    if (inOl) result.push('</ol>');

    return result.join('');
}
// =======================================================

let baselineNetwork = null;
let cognitiveNetwork = null;

document.addEventListener('DOMContentLoaded', function() {
    loadTopics();

    if (document.getElementById('load-baseline')) {
        document.getElementById('load-baseline').addEventListener('click', loadBaselineGraph);
    }
    if (document.getElementById('load-cognitive')) {
        document.getElementById('load-cognitive').addEventListener('click', loadCognitiveGraph);
    }
    if (document.getElementById('analyze-community')) {
        document.getElementById('analyze-community').addEventListener('click', analyzeCommunity);
    }
    if (document.getElementById('ask-rag')) {
        document.getElementById('ask-rag').addEventListener('click', askRag);
    }
});

function loadTopics() {
    if (document.getElementById('topic-select-baseline')) {
        fetch('/api/baseline/topics')
            .then(res => res.json())
            .then(topics => {
                const select = document.getElementById('topic-select-baseline');
                topics.forEach(topic => {
                    const option = document.createElement('option');
                    option.value = topic;
                    option.textContent = topic;
                    select.appendChild(option);
                });
            })
            .catch(err => console.error('Failed to load baseline topics:', err));
    }

    if (document.getElementById('topic-select-cognitive')) {
        fetch('/api/cognitive/topics')
            .then(res => res.json())
            .then(topics => {
                const select = document.getElementById('topic-select-cognitive');
                topics.forEach(topic => {
                    const option = document.createElement('option');
                    option.value = topic;
                    option.textContent = topic;
                    select.appendChild(option);
                });
            })
            .catch(err => console.error('Failed to load cognitive topics:', err));
    }

    if (document.getElementById('community-topic')) {
        fetch('/api/community/topics')
            .then(res => res.json())
            .then(topics => {
                const select = document.getElementById('community-topic');
                select.innerHTML = '<option value="">Select a topic</option>';
                topics.forEach(topic => {
                    const option = document.createElement('option');
                    option.value = topic;
                    option.textContent = topic;
                    select.appendChild(option);
                });
            })
            .catch(err => console.error('Failed to load community topics:', err));
    }
}

function loadBaselineGraph() {
    const label = document.getElementById('label-select').value;
    const topic = document.getElementById('topic-select-baseline').value;
    const url = `/api/baseline/graph?label=${encodeURIComponent(label)}&topic=${encodeURIComponent(topic)}`;

    fetch(url)
        .then(res => res.json())
        .then(data => {
            drawNetwork('baseline-network', data, (network) => { baselineNetwork = network; });
        })
        .catch(err => {
            console.error('Failed to load baseline graph:', err);
            alert('Load failed, check console');
        });
}

function loadCognitiveGraph() {
    const topic = document.getElementById('topic-select-cognitive').value;
    const url = `/api/cognitive/graph?topic=${encodeURIComponent(topic)}`;

    fetch(url)
        .then(res => res.json())
        .then(data => {
            drawNetwork('cognitive-network', data, (network) => { cognitiveNetwork = network; });
        })
        .catch(err => {
            console.error('Failed to load cognitive graph:', err);
            alert('Load failed, check console');
        });
}

function drawNetwork(containerId, data, callback) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const panelType = containerId.includes('baseline') ? 'baseline' : 'cognitive';
    const groups = [...new Set(data.nodes.map(n => n.group))];

    const palette = [
        '#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea',
        '#0891b2', '#ea580c', '#4b5563', '#0d9488', '#b91c1c',
        '#15803d', '#b45309', '#6b7280', '#1e40af', '#86198f'
    ];

    const groupsOptions = {};
    groups.forEach((group, index) => {
        const color = palette[index % palette.length];
        groupsOptions[group] = {
            color: { background: color, border: '#1e293b' },
            font: { color: '#1e293b' }
        };
    });

    const displayNames = {};
    groups.forEach(g => {
        if (panelType === 'baseline' && g.startsWith('Topic_')) {
            displayNames[g] = g.substring(6);
        } else {
            displayNames[g] = g;
        }
    });

    const nodes = new vis.DataSet(data.nodes.map(n => ({
        id: n.id,
        label: n.label,
        title: n.title,
        group: n.group
    })));

    const edges = new vis.DataSet(data.edges.map(e => ({
        from: e.from,
        to: e.to,
        label: e.label,
        title: e.title,
        arrows: e.arrows
    })));

    const options = {
        nodes: { shape: 'dot', size: 20, font: { size: 14, face: 'Segoe UI' }, borderWidth: 2, shadow: true },
        edges: { width: 2, shadow: true, font: { size: 12, align: 'middle', face: 'Segoe UI' }, smooth: { type: 'continuous' }, color: { color: '#94a3b8', highlight: '#2563eb' } },
        groups: groupsOptions,
        physics: { stabilization: false, barnesHut: { gravitationalConstant: -8000, centralGravity: 0.3, springLength: 95, springConstant: 0.04 } },
        interaction: { hover: true, tooltipDelay: 200, navigationButtons: true }
    };

    if (containerId === 'baseline-network' && baselineNetwork) baselineNetwork.destroy();
    else if (containerId === 'cognitive-network' && cognitiveNetwork) cognitiveNetwork.destroy();

    const network = new vis.Network(container, { nodes, edges }, options);
    updateLegend(groups, groupsOptions, displayNames);
    if (callback) callback(network);
}

function updateLegend(groups, groupsOptions, displayNames = {}) {
    const legendItems = document.getElementById('legend-items');
    if (!legendItems) return;
    legendItems.innerHTML = '';
    if (groups.length === 0) {
        legendItems.innerHTML = '<div class="legend-item">No groups</div>';
        return;
    }
    groups.forEach(group => {
        const color = groupsOptions[group]?.color?.background || '#ccc';
        const displayName = displayNames[group] || group;
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `<div class="legend-color" style="background-color: ${color};"></div><span title="${displayName}">${displayName}</span>`;
        legendItems.appendChild(item);
    });
}

function analyzeCommunity() {
    const topic = document.getElementById('community-topic').value;
    const maxCommunities = document.getElementById('max-communities').value;
    if (!topic) {
        alert('Please select a topic');
        return;
    }
    const resultDiv = document.getElementById('community-result');
    resultDiv.innerHTML = '<div style="text-align:center; padding:40px;">⏳ Analyzing, please wait...</div>';

    fetch('/api/community/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, max_communities: maxCommunities })
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                resultDiv.innerHTML = `<div style="color:#b91c1c;">Error: ${data.error}</div>`;
                return;
            }
            renderCommunityResult(data);
        })
        .catch(err => {
            resultDiv.innerHTML = `<div style="color:#b91c1c;">Request failed: ${err.message}</div>`;
        });
}

function renderCommunityResult(data) {
    const resultDiv = document.getElementById('community-result');
    let html = `<h3>Topic: ${data.topic}</h3>`;
    html += `<p style="color:#475569;">Total nodes: ${data.total_nodes}, Communities found: ${data.total_communities}, Showing top ${data.filtered_communities} communities (min size ≥5)</p>`;

    data.communities.forEach(comm => {
        let summary = markdownToHtml(comm.summary);
        html += `
            <div class="community-card">
                <h4>Community ${comm.community_id} (nodes: ${comm.node_count}, relations: ${comm.rel_count})</h4>
                <div class="community-stats">
                    <span>Type distribution:</span>
        `;
        for (let [type, cnt] of Object.entries(comm.type_distribution)) {
            html += `<span class="badge">${type}: ${cnt}</span> `;
        }
        html += `</div><div><strong>Key concepts:</strong>`;
        comm.top_nodes.forEach(node => {
            html += `<span class="badge">${node.name} (${node.type}, deg ${node.degree})</span> `;
        });
        html += `</div><div class="community-summary">${summary}</div></div>`;
    });
    resultDiv.innerHTML = html;
}

function askRag() {
    const question = document.getElementById('rag-question').value.trim();
    if (!question) {
        alert('Please enter a question');
        return;
    }
    const answerDiv = document.getElementById('rag-answer').querySelector('.card-content');
    const contextDiv = document.getElementById('rag-context').querySelector('.card-content');
    answerDiv.innerHTML = '<div style="text-align:center; padding:40px;">⏳ Thinking...</div>';
    contextDiv.innerHTML = '<div style="text-align:center; padding:40px;">⏳ Loading...</div>';

    fetch('/api/rag/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                answerDiv.innerHTML = `<div style="color:#b91c1c;">Error: ${data.error}</div>`;
                contextDiv.innerHTML = '';
                return;
            }
            const formattedAnswer = markdownToHtml(data.answer);
            answerDiv.innerHTML = formattedAnswer;
            const formattedContext = markdownToHtml(data.context);
            contextDiv.innerHTML = formattedContext;
        })
        .catch(err => {
            answerDiv.innerHTML = `<div style="color:#b91c1c;">Request failed: ${err.message}</div>`;
            contextDiv.innerHTML = '';
        });
}

function toggleCard(headerElement) {
    const card = headerElement.closest('.collapsible-card');
    const content = card.querySelector('.card-content');
    const icon = headerElement.querySelector('.toggle-icon');
    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.textContent = '▼';
    } else {
        content.style.display = 'none';
        icon.textContent = '▶';
    }
}