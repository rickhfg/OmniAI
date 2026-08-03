let historyData = [];
let recentLogsData = [];
let lastStatsSeq = 0;
let receivedStatsSnapshot = false;

const REQUEST_METRIC_FIELDS = [
    'id', 'timestamp', 'model', 'input_tokens', 'output_tokens', 'ttft', 'duration'
];
const SAFE_LOG_PREFIX = /^\[(?:WARNING|ERROR)\]/i;
const UNSAFE_LOG_CONTENT = /\b(?:chunk|content|input|message|output|payload|prompt|reasoning|response|text)\b/i;
const authKeyInput = document.getElementById('proxy-auth-key');
const connectButton = document.getElementById('connect-button');

function getAuthHeaders() {
    const key = authKeyInput?.value.trim();
    return key ? { Authorization: `Bearer ${key}` } : {};
}

function resetStatsCursor() {
    lastStatsSeq = 0;
    receivedStatsSnapshot = false;
    historyData = [];
    recentLogsData = [];
}

function sanitizeRequestMetric(request) {
    if (!request || typeof request !== 'object') return null;
    const safeRequest = {};
    REQUEST_METRIC_FIELDS.forEach(field => {
        if (Object.prototype.hasOwnProperty.call(request, field)) {
            safeRequest[field] = request[field];
        }
    });
    if (Number.isInteger(request.seq)) safeRequest.seq = request.seq;
    return safeRequest;
}

function sanitizeRequestMetrics(requests) {
    return (Array.isArray(requests) ? requests : [])
        .map(sanitizeRequestMetric)
        .filter(Boolean);
}

function sanitizeLogEntry(log) {
    if (!log || typeof log !== 'object') return null;
    const message = typeof log.msg === 'string' ? log.msg : '';
    if (!SAFE_LOG_PREFIX.test(message) || UNSAFE_LOG_CONTENT.test(message)) return null;
    const safeLog = { time: String(log.time ?? ''), msg: message.slice(0, 240) };
    if (Number.isInteger(log.seq)) safeLog.seq = log.seq;
    return safeLog;
}

function sanitizeLogs(logs) {
    return (Array.isArray(logs) ? logs : [])
        .map(sanitizeLogEntry)
        .filter(Boolean)
        .slice(0, 50);
}

async function fetchData() {
    try {
        const response = await fetch(
            '/stats?since=' + encodeURIComponent(lastStatsSeq),
            { headers: getAuthHeaders() },
        );
        if (!response.ok) {
            if (response.status === 401 || response.status === 503) {
                throw new Error("Authentication required");
            }
            throw new Error("Network response was not ok");
        }
        const data = await response.json();
        const incomingHistory = sanitizeRequestMetrics(data.request_history);
        const incomingLogs = sanitizeLogs(data.recent_logs);

        if (!receivedStatsSnapshot || data.full_snapshot) {
            historyData = incomingHistory.slice(0, 50);
            recentLogsData = incomingLogs;
            receivedStatsSnapshot = true;
        } else {
            historyData = mergeNewest(incomingHistory, historyData, 'id');
            recentLogsData = mergeNewest(incomingLogs, recentLogsData, 'seq');
        }

        if (Number.isInteger(data.stats_seq)) lastStatsSeq = data.stats_seq;
        updateDashboard({
            ...data,
            request_history: historyData,
            recent_logs: recentLogsData,
        });
    } catch (error) {
        console.error("Error fetching stats:", error);
        document.getElementById('server-status').textContent = "Offline (" + error.message + ")";
        document.getElementById('server-status').classList.remove('text-success');
        document.getElementById('server-status').classList.add('text-danger');
    }
}

function mergeNewest(incoming, existing, key) {
    const merged = [...incoming, ...existing];
    const seen = new Set();
    return merged.filter(item => {
        const value = item[key];
        if (value === undefined || seen.has(value)) return false;
        seen.add(value);
        return true;
    }).slice(0, 50);
}

let requestsChartCtx = document.getElementById('requestsChart').getContext('2d');
let modelChartCtx = document.getElementById('modelChart').getContext('2d');
let requestsChart, modelChart;
let previousHourlySignature = null;
let previousModelSignature = null;
let previousLogSignature = null;
let previousHistorySignature = null;

function appendTextCell(row, value, className = '') {
    const cell = document.createElement('td');
    if (className) cell.className = className;
    cell.textContent = value === undefined || value === null ? '' : String(value);
    row.appendChild(cell);
    return cell;
}

function renderModelUsage(distribution, tokens) {
    const body = document.getElementById('model-usage-body');
    const fragment = document.createDocumentFragment();
    const sortedModels = Object.keys(distribution).sort((a, b) => distribution[b] - distribution[a]);

    sortedModels.forEach(model => {
        const reqs = distribution[model];
        const modelTokens = tokens[model] || 0;
        const avg = reqs > 0 ? Math.round(modelTokens / reqs) : 0;
        const row = document.createElement('tr');
        appendTextCell(row, model, 'font-monospace');
        appendTextCell(row, reqs);
        appendTextCell(row, modelTokens.toLocaleString());
        appendTextCell(row, avg.toLocaleString());
        fragment.appendChild(row);
    });
    body.replaceChildren(fragment);
}

function renderLogs(logs) {
    const container = document.getElementById('logs-container');
    const fragment = document.createDocumentFragment();
    logs.forEach(log => {
        const line = document.createElement('div');
        const timestamp = document.createElement('span');
        timestamp.className = 'text-muted';
        timestamp.textContent = `[${log.time}]`;
        line.append(timestamp, document.createTextNode(` ${log.msg ?? ''}`));
        fragment.appendChild(line);
    });
    container.replaceChildren(fragment);
}

function renderHistory(history) {
    const body = document.getElementById('request-history-body');
    if (!history.length) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 5;
        cell.className = 'text-center text-muted';
        cell.textContent = 'No requests yet';
        row.appendChild(cell);
        body.replaceChildren(row);
        return;
    }

    const fragment = document.createDocumentFragment();
    history.forEach(req => {
        const row = document.createElement('tr');
        appendTextCell(row, req.timestamp);
        appendTextCell(row, req.model);
        appendTextCell(row, `${req.input_tokens} / ${req.output_tokens}`);
        appendTextCell(row, req.ttft || '-');
        appendTextCell(row, req.duration || '-');

        fragment.appendChild(row);
    });
    body.replaceChildren(fragment);
}

function initCharts() {
    requestsChart = new Chart(requestsChartCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Requests', data: [], borderColor: '#0d6efd', tension: 0.4 }] },
        options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { grid: { color: '#333' } }, y: { grid: { color: '#333' } } } }
    });

    modelChart = new Chart(modelChartCtx, {
        type: 'doughnut',
        data: { labels: [], datasets: [{ data: [], backgroundColor: ['#0d6efd', '#6610f2', '#6f42c1', '#d63384', '#dc3545'] }] },
        options: { responsive: true, plugins: { legend: { position: 'bottom' } }, cutout: '70%', elements: { arc: { borderWidth: 0 } } }
    });
}

function updateDashboard(data) {
    document.getElementById('server-status').textContent = "Online";
    document.getElementById('server-status').classList.remove('text-danger');
    document.getElementById('server-status').classList.add('text-success');

    document.getElementById('daily-requests').textContent = data.daily_requests || 0;
    document.getElementById('daily-tokens').textContent = (data.daily_tokens || 0).toLocaleString();
    document.getElementById('last-model').textContent = data.last_model || "-";
    document.getElementById('uptime').textContent = formatUptime(data.uptime);

    // Update charts if data exists
    const hourlySignature = JSON.stringify(data.hourly_stats || null);
    if (data.hourly_stats && hourlySignature !== previousHourlySignature) {
        requestsChart.data.labels = data.hourly_stats.labels;
        requestsChart.data.datasets[0].data = data.hourly_stats.values;
        requestsChart.update();
        previousHourlySignature = hourlySignature;
    }

    const modelSignature = JSON.stringify([data.model_distribution || {}, data.model_tokens || {}]);
    if (data.model_distribution && modelSignature !== previousModelSignature) {
        modelChart.data.labels = Object.keys(data.model_distribution);
        modelChart.data.datasets[0].data = Object.values(data.model_distribution);
        modelChart.update();
        previousModelSignature = modelSignature;

        // Update Model Usage Table
        renderModelUsage(data.model_distribution, data.model_tokens || {});
    }

    // Logs
    const logContainer = document.getElementById('logs-container');
    const logSignature = (data.recent_logs || []).map(log => log.seq ?? `${log.time}:${log.msg}`).join('|');
    if (logSignature !== previousLogSignature) {
        renderLogs(data.recent_logs || []);
        logContainer.scrollTop = logContainer.scrollHeight;
        previousLogSignature = logSignature;
    }

    // Request History
    const historySignature = (data.request_history || []).map(req => req.seq ?? req.id).join('|');
    if (historySignature !== previousHistorySignature) {
        renderHistory(data.request_history || []);
        previousHistorySignature = historySignature;
    }
}

function formatUptime(seconds) {
    if (!seconds) return "-";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
}

initCharts();
async function pollStats() {
    await fetchData();
    setTimeout(pollStats, 5000);
}
pollStats();

connectButton?.addEventListener('click', () => {
    resetStatsCursor();
    fetchData();
});

authKeyInput?.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
        resetStatsCursor();
        fetchData();
    }
});
