/**
 * ERR-EVAL REFERENCE IMPLEMENTATION
 * High-fidelity interaction logic & Chart.js configuration
 */

// STATE
let state = {
    data: null,
    filtered: [],
    masterChart: null,
    providers: {},
    currentTab: 'accuracy',
    selectedModels: new Set()
};

// AESTHETIC CONFIG (Matches CSS)
const THEME = {
    void: '#050505',
    text: '#e0e0e0',
    muted: '#666666',
    signal: '#70FF8B',
    signalDim: 'rgba(112, 255, 139, 0.2)',
    error: '#FF5C5C',
    grid: '#1f1f1f',
    font: 'Chivo Mono'
};

const DEFAULT_PROVIDER = { color: '#666666', label: 'Other', icon: '?' };

// INIT
document.addEventListener('DOMContentLoaded', async () => {
    Chart.defaults.font.family = THEME.font;
    Chart.defaults.color = THEME.muted;

    const loader = document.getElementById('loading-overlay');
    loader.classList.add('active');

    await new Promise(r => setTimeout(r, 800));

    await loadData();
    setupInteractions();
    populateModelFilter();
    renderAll();

    // Update timestamp
    if (state.data && state.data.generated_at) {
        const el = document.getElementById('last-run-display');
        if (el) el.textContent = `Last Updated: ${new Date(state.data.generated_at).toLocaleString()}`;
    }

    loader.classList.remove('active');
});

function getProvider(modelId) {
    const prefix = modelId.split('/')[0];
    if (state.providers && state.providers[prefix]) {
        return state.providers[prefix];
    }
    return DEFAULT_PROVIDER;
}

// Data source - switch to GitHub raw URL when deployed
const DATA_URL = 'data/results.json';
// For production: 'https://raw.githubusercontent.com/GustyCube/ERR-EVAL/main/frontend/data/results.json'

async function loadData() {
    try {
        // Add cache-busting query param to always get fresh data
        const cacheBuster = `?t=${Date.now()}`;
        const res = await fetch(DATA_URL + cacheBuster);
        if (!res.ok) throw new Error('Data Access Failure');
        const json = await res.json();

        state.data = json;
        state.providers = json.providers || {};
        state.filtered = [...(json.entries || [])];

        // Default all models selected
        state.filtered.forEach(entry => state.selectedModels.add(entry.model_id));

        state.filtered.sort((a, b) => b.overall_score - a.overall_score);
        state.filtered.forEach((item, index) => item.rank = index + 1);

        const total = state.filtered.length;
        document.getElementById('total-models').textContent = total.toString().padStart(2, '0');

    } catch (e) {
        console.warn('[ERR-EVAL] System Status: Awaiting Data', e);
        state.data = { entries: [] };
        state.filtered = [];
    }
}

function setupInteractions() {
    // Tab clicks
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.currentTab = e.target.dataset.tab;
            renderChart();
        });
    });

    // Search only (sort dropdown removed)
    const search = document.getElementById('search-input');
    if (search) {
        search.addEventListener('input', () => applyFilters());
    }
}

function populateModelFilter() {
    const container = document.getElementById('model-filter-dropdown');
    if (!container || !state.data.entries) return;

    container.innerHTML = '';

    // "Select All" Option
    const allDiv = document.createElement('div');
    allDiv.className = 'multi-select-option';
    allDiv.innerHTML = `<input type="checkbox" value="all" checked> <span>Select All</span>`;
    allDiv.querySelector('input').addEventListener('change', (e) => {
        const checkboxes = container.querySelectorAll('input[type="checkbox"]:not([value="all"])');
        checkboxes.forEach(cb => {
            cb.checked = e.target.checked;
            if (e.target.checked) state.selectedModels.add(cb.value);
            else state.selectedModels.delete(cb.value);
        });
        applyFilters();
    });
    container.appendChild(allDiv);

    // Individual Models
    state.data.entries.forEach(entry => {
        const div = document.createElement('div');
        div.className = 'multi-select-option';
        div.innerHTML = `<input type="checkbox" value="${entry.model_id}" checked> <span>${entry.model_name}</span>`;

        div.querySelector('input').addEventListener('change', (e) => {
            if (e.target.checked) state.selectedModels.add(entry.model_id);
            else state.selectedModels.delete(entry.model_id);

            const allCb = container.querySelector('input[value="all"]');
            allCb.checked = state.selectedModels.size === state.data.entries.length;

            applyFilters();
        });

        container.appendChild(div);
    });

    // Toggle dropdown visibility
    const btn = document.querySelector('.multi-select-btn');
    if (btn) {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            container.classList.toggle('show');
        });
    }

    // Close on click outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.multi-select-container')) {
            container.classList.remove('show');
        }
    });
}

function applyFilters() {
    const search = document.getElementById('search-input');
    const term = search.value.toLowerCase();

    let result = state.data.entries.filter(entry => {
        if (!state.selectedModels.has(entry.model_id)) return false;
        if (term && !entry.model_name.toLowerCase().includes(term) &&
            !entry.model_id.toLowerCase().includes(term)) return false;
        return true;
    });

    // Always sort by overall score
    result.sort((a, b) => b.overall_score - a.overall_score);

    result.forEach((item, index) => item.rank = index + 1);

    state.filtered = result;
    renderLeaderboard();
    renderChart();

    // Update filter button text
    const btn = document.querySelector('.multi-select-btn');
    if (btn) {
        const total = state.data.entries.length;
        const selected = state.selectedModels.size;
        btn.textContent = selected === total ? 'Filter Models (All)' : `Filter Models (${selected}/${total})`;
    }
}

function renderAll() {
    renderLeaderboard();
    renderChart();
}

function renderLeaderboard() {
    const tbody = document.getElementById('leaderboard-body');
    const stub = document.getElementById('no-data-message');

    if (!state.filtered.length) {
        tbody.innerHTML = '';
        stub.style.display = 'block';
        return;
    }

    stub.style.display = 'none';
    tbody.innerHTML = state.filtered.map(entry => {
        const provider = getProvider(entry.model_id);
        return `
        <tr>
            <td class="col-rank">${String(entry.rank).padStart(2, '0')}</td>
            <td class="col-model" style="border-left: 3px solid ${provider.color}; padding-left: 1rem;">
                ${entry.model_name}
                <span class="model-sub">${entry.model_id}</span>
            </td>
            <td class="col-score">${entry.overall_score.toFixed(2)}</td>
            ${['A', 'B', 'C', 'D', 'E'].map(t => `
                <td class="col-track" style="color: ${getScoreColor(entry.track_scores[t])}">
                    ${(entry.track_scores[t] || 0).toFixed(1)}
                </td>
            `).join('')}
        </tr>
    `}).join('');
}

function getScoreColor(val) {
    if (val >= 7) return THEME.signal;
    if (val >= 4) return '#ffffff';
    return THEME.muted;
}

function renderChart() {
    const ctx = document.getElementById('master-chart');
    if (!ctx) return;

    if (state.masterChart) state.masterChart.destroy();

    const chartType = state.currentTab;

    if (chartType === 'accuracy') renderAccuracyChart(ctx);
    else if (chartType === 'cost') renderCostChart(ctx);
    else if (chartType === 'speed') renderSpeedChart(ctx);
    else if (chartType === 'combined') renderCombinedChart(ctx);
}

// 1. Accuracy Chart (Original bar chart style)
function renderAccuracyChart(ctx) {
    const labels = state.filtered.map(m => m.model_name);
    const dataPoints = state.filtered.map(m => m.overall_score);
    const backgroundColors = state.filtered.map(m => getProvider(m.model_id).color);
    const borderColors = state.filtered.map(m => getProvider(m.model_id).color);

    state.masterChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Overall Score',
                data: dataPoints,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1,
                barPercentage: 0.6,
                hoverBackgroundColor: THEME.signal
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        color: THEME.muted,
                        font: { family: THEME.font, size: 10 },
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    grid: { color: THEME.grid },
                    beginAtZero: true,
                    max: 10,
                    title: {
                        display: true,
                        text: 'Reliability Score (0-10)',
                        color: THEME.muted,
                        font: { family: THEME.font }
                    },
                    ticks: { color: THEME.muted }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: THEME.void,
                    displayColors: false,
                    titleFont: { family: THEME.font },
                    bodyFont: { family: THEME.font },
                    borderColor: THEME.grid,
                    borderWidth: 1,
                    callbacks: {
                        label: (context) => ` Score: ${context.raw}`
                    }
                }
            },
            animation: { duration: 1500, easing: 'easeOutQuart' },
            layout: { padding: { top: 30 } }
        },
        plugins: [{
            id: 'chartTitle',
            beforeDraw(chart) {
                const ctx = chart.ctx;
                ctx.save();
                ctx.fillStyle = THEME.signal;
                ctx.font = 'bold 12px ' + THEME.font;
                ctx.textAlign = 'right';
                ctx.fillText('↑ HIGHER IS BETTER', chart.width - 20, 20);
                ctx.restore();
            }
        }]
    });
}

// 2. Cost Chart (Vertical bar, sorted low to high, cost on Y axis)
function renderCostChart(ctx) {
    const sorted = [...state.filtered].sort((a, b) => (a.avg_cost || 0) - (b.avg_cost || 0));
    const labels = sorted.map(m => m.model_name);
    const data = sorted.map(m => m.avg_cost || 0);
    const colors = sorted.map(m => getProvider(m.model_id).color);

    state.masterChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                barPercentage: 0.5,
                categoryPercentage: 0.8,
                maxBarThickness: 50
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: THEME.grid },
                    title: { display: true, text: 'Avg Cost per Item ($)', color: THEME.muted },
                    ticks: { color: THEME.muted, callback: (v) => `$${v.toFixed(4)}` }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        color: THEME.muted,
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: THEME.void,
                    borderColor: THEME.grid,
                    borderWidth: 1,
                    callbacks: { label: (c) => ` $${c.raw.toFixed(6)}` }
                }
            },
            layout: { padding: { top: 30 } }
        },
        plugins: [{
            id: 'chartTitle',
            beforeDraw(chart) {
                const ctx = chart.ctx;
                ctx.save();
                ctx.fillStyle = '#FF5C5C';
                ctx.font = 'bold 12px ' + THEME.font;
                ctx.textAlign = 'right';
                ctx.fillText('↓ LOWER IS BETTER', chart.width - 20, 20);
                ctx.restore();
            }
        }]
    });
}

// 3. Speed Chart (Vertical bar, latency)
function renderSpeedChart(ctx) {
    const labels = state.filtered.map(m => m.model_name);
    const data = state.filtered.map(m => m.avg_latency || 0);
    const colors = state.filtered.map(m => getProvider(m.model_id).color);

    state.masterChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{ data: data, backgroundColor: colors, barPercentage: 0.6, maxBarThickness: 40 }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: THEME.grid },
                    title: { display: true, text: 'Avg Latency (ms)', color: THEME.muted },
                    ticks: { color: THEME.muted }
                },
                x: { grid: { display: false }, ticks: { color: THEME.muted, maxRotation: 45, minRotation: 45 } }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: THEME.void,
                    borderColor: THEME.grid,
                    borderWidth: 1,
                    callbacks: { label: (c) => ` ${c.raw.toFixed(0)} ms` }
                }
            },
            layout: { padding: { top: 30 } }
        },
        plugins: [{
            id: 'chartTitle',
            beforeDraw(chart) {
                const ctx = chart.ctx;
                ctx.save();
                ctx.fillStyle = '#FF5C5C';
                ctx.font = 'bold 12px ' + THEME.font;
                ctx.textAlign = 'right';
                ctx.fillText('↓ LOWER IS BETTER', chart.width - 20, 20);
                ctx.restore();
            }
        }]
    });
}

// 4. Combined Chart (Scatter: Cost vs Accuracy)
function renderCombinedChart(ctx) {
    const points = state.filtered.map(m => ({
        x: m.avg_cost || 0,
        y: m.overall_score,
        label: m.model_name
    }));
    const colors = state.filtered.map(m => getProvider(m.model_id).color);

    state.masterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                data: points,
                backgroundColor: colors,
                borderColor: colors,
                pointRadius: 8,
                pointHoverRadius: 12
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'Avg Cost ($)', color: THEME.muted },
                    grid: { color: THEME.grid },
                    ticks: { color: THEME.muted }
                },
                y: {
                    beginAtZero: true,
                    max: 10,
                    title: { display: true, text: 'Reliability Score (0-10)', color: THEME.muted },
                    grid: { color: THEME.grid },
                    ticks: { color: THEME.muted }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: THEME.void,
                    borderColor: THEME.grid,
                    borderWidth: 1,
                    callbacks: {
                        label: (ctx) => {
                            const p = ctx.raw;
                            return `${p.label}: Score ${p.y.toFixed(2)} | $${p.x.toFixed(4)}`;
                        }
                    }
                }
            },
            layout: { padding: { top: 30 } }
        },
        plugins: [{
            id: 'chartTitle',
            beforeDraw(chart) {
                const ctx = chart.ctx;
                ctx.save();
                ctx.fillStyle = THEME.signal;
                ctx.font = 'bold 12px ' + THEME.font;
                ctx.textAlign = 'right';
                ctx.fillText('↖ TOP-LEFT = BEST (High Score, Low Cost)', chart.width - 20, 20);
                ctx.restore();
            }
        }]
    });
}
