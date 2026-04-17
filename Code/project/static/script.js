// ================== Markdown 转 HTML ==================
function markdownToHtml(text) {
    if (!text) return '';

    const lines = text.split('\n');
    let inUl = false;
    let inOl = false;
    let result = [];

    for (let line of lines) {
        // 检查无序列表项：以 "- " 开头
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

        // 检查有序列表项：以数字. 开头，如 "1. "
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

        // 不是列表项，关闭列表标签（如果有）
        if (inUl) {
            result.push('</ul>');
            inUl = false;
        }
        if (inOl) {
            result.push('</ol>');
            inOl = false;
        }

        // 普通行：处理粗体、斜体，然后保留换行
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

// 全局网络实例
let baselineNetwork = null;
let cognitiveNetwork = null;

// 根据当前页面执行不同初始化
document.addEventListener('DOMContentLoaded', function() {
    const path = window.location.pathname;

    // 加载话题列表（通用）
    loadTopics();

    // 绑定按钮事件
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
    // Baseline话题
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
            .catch(err => console.error('加载Baseline话题失败:', err));
    }

    // 认知话题
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
            .catch(err => console.error('加载认知话题失败:', err));
    }

    // 社区话题
    if (document.getElementById('community-topic')) {
        fetch('/api/community/topics')
            .then(res => res.json())
            .then(topics => {
                const select = document.getElementById('community-topic');
                select.innerHTML = '<option value="">请选择话题</option>';
                topics.forEach(topic => {
                    const option = document.createElement('option');
                    option.value = topic;
                    option.textContent = topic;
                    select.appendChild(option);
                });
            })
            .catch(err => console.error('加载社区话题失败:', err));
    }
}

function loadBaselineGraph() {
    const label = document.getElementById('label-select').value;
    const topic = document.getElementById('topic-select-baseline').value;
    const url = `/api/baseline/graph?label=${encodeURIComponent(label)}&topic=${encodeURIComponent(topic)}`;

    fetch(url)
        .then(res => res.json())
        .then(data => {
            drawNetwork('baseline-network', data, (network) => {
                baselineNetwork = network;
            });
        })
        .catch(err => {
            console.error('加载Baseline图谱失败:', err);
            alert('加载失败，请检查控制台');
        });
}

function loadCognitiveGraph() {
    const topic = document.getElementById('topic-select-cognitive').value;
    const url = `/api/cognitive/graph?topic=${encodeURIComponent(topic)}`;

    fetch(url)
        .then(res => res.json())
        .then(data => {
            drawNetwork('cognitive-network', data, (network) => {
                cognitiveNetwork = network;
            });
        })
        .catch(err => {
            console.error('加载认知图谱失败:', err);
            alert('加载失败，请检查控制台');
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
            color: {
                background: color,
                border: '#1e293b'
            },
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
        nodes: {
            shape: 'dot',
            size: 20,
            font: { size: 14, face: 'Segoe UI' },
            borderWidth: 2,
            shadow: true,
        },
        edges: {
            width: 2,
            shadow: true,
            font: { size: 12, align: 'middle', face: 'Segoe UI' },
            smooth: { type: 'continuous' },
            color: { color: '#94a3b8', highlight: '#2563eb' }
        },
        groups: groupsOptions,
        physics: {
            stabilization: false,
            barnesHut: { gravitationalConstant: -8000, centralGravity: 0.3, springLength: 95, springConstant: 0.04 }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            navigationButtons: true
        }
    };

    if (containerId === 'baseline-network' && baselineNetwork) {
        baselineNetwork.destroy();
    } else if (containerId === 'cognitive-network' && cognitiveNetwork) {
        cognitiveNetwork.destroy();
    }

    const network = new vis.Network(container, { nodes, edges }, options);

    updateLegend(groups, groupsOptions, displayNames);

    if (callback) callback(network);
}

function updateLegend(groups, groupsOptions, displayNames = {}) {
    const legendItems = document.getElementById('legend-items');
    if (!legendItems) return;

    legendItems.innerHTML = '';

    if (groups.length === 0) {
        legendItems.innerHTML = '<div class="legend-item">暂无分组</div>';
        return;
    }

    groups.forEach(group => {
        const color = groupsOptions[group]?.color?.background || '#ccc';
        const displayName = displayNames[group] || group;
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `
            <div class="legend-color" style="background-color: ${color};"></div>
            <span title="${displayName}">${displayName}</span>
        `;
        legendItems.appendChild(item);
    });
}

function analyzeCommunity() {
    const topic = document.getElementById('community-topic').value;
    const maxCommunities = document.getElementById('max-communities').value;
    if (!topic) {
        alert('请选择话题');
        return;
    }
    const resultDiv = document.getElementById('community-result');
    resultDiv.innerHTML = '<div style="text-align:center; padding:40px;">⏳ 分析中，请稍候...</div>';

    fetch('/api/community/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, max_communities: maxCommunities })
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                resultDiv.innerHTML = `<div style="color:#b91c1c;">错误：${data.error}</div>`;
                return;
            }
            renderCommunityResult(data);
        })
        .catch(err => {
            resultDiv.innerHTML = `<div style="color:#b91c1c;">请求失败：${err.message}</div>`;
        });
}

function renderCommunityResult(data) {
    const resultDiv = document.getElementById('community-result');
    let html = `<h3>话题：${data.topic}</h3>`;
    html += `<p style="color:#475569;">总节点数：${data.total_nodes}，发现社区数：${data.total_communities}，展示前 ${data.filtered_communities} 个社区</p>`;

    data.communities.forEach(comm => {
        // 使用 markdownToHtml 转换总结内容
        let summary = markdownToHtml(comm.summary);

        html += `
            <div class="community-card">
                <h4>社区 ${comm.community_id}（节点数：${comm.node_count}，关系数：${comm.rel_count}）</h4>
                <div class="community-stats">
                    <span>类型分布：</span>
        `;
        for (let [type, cnt] of Object.entries(comm.type_distribution)) {
            html += `<span class="badge">${type}: ${cnt}</span> `;
        }
        html += `</div><div><strong>重要概念：</strong>`;
        comm.top_nodes.forEach(node => {
            html += `<span class="badge">${node.name} (${node.type}, 度${node.degree})</span> `;
        });
        html += `</div><div class="community-summary">${summary}</div></div>`;
    });
    resultDiv.innerHTML = html;
}

function askRag() {
    const question = document.getElementById('rag-question').value.trim();
    if (!question) {
        alert('请输入问题');
        return;
    }
    const answerDiv = document.getElementById('rag-answer').querySelector('.card-content');
    const contextDiv = document.getElementById('rag-context').querySelector('.card-content');
    answerDiv.innerHTML = '<div style="text-align:center; padding:40px;">⏳ 思考中...</div>';
    contextDiv.innerHTML = '<div style="text-align:center; padding:40px;">⏳ 加载中...</div>';

    fetch('/api/rag/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                answerDiv.innerHTML = `<div style="color:#b91c1c;">错误：${data.error}</div>`;
                contextDiv.innerHTML = '';
                return;
            }
            // 渲染答案（支持 Markdown）
            const formattedAnswer = markdownToHtml(data.answer);
            answerDiv.innerHTML = formattedAnswer;

            // 渲染上下文（同样支持 Markdown）
            const formattedContext = markdownToHtml(data.context);
            contextDiv.innerHTML = formattedContext;
        })
        .catch(err => {
            answerDiv.innerHTML = `<div style="color:#b91c1c;">请求失败：${err.message}</div>`;
            contextDiv.innerHTML = '';
        });
}

// 卡片折叠/展开函数
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