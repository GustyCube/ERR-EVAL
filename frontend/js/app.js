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
    selectedModels: new Set(),
    iconCache: {} // Cache for loaded icon images
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

const LIGHT_THEME = {
    void: '#ffffff',
    text: '#1a1a1a',
    muted: '#666666',
    signal: '#00c853',
    signalDim: 'rgba(0, 200, 83, 0.2)',
    error: '#d32f2f',
    grid: '#e0e0e0',
    font: 'Chivo Mono'
};

// Get current theme colors
function getCurrentTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    return isDark ? THEME : LIGHT_THEME;
}

const DEFAULT_PROVIDER = { color: '#666666', label: 'Other', icon: '?' };

// INIT
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize theme from localStorage
    initTheme();
    
    // Initialize Chart.js defaults only if available
    if (typeof Chart !== 'undefined') {
        const currentTheme = getCurrentTheme();
        Chart.defaults.font.family = currentTheme.font;
        Chart.defaults.color = currentTheme.muted;
    }

    const loader = document.getElementById('loading-overlay');
    loader.classList.add('active');

    await new Promise(r => setTimeout(r, 800));

    await loadData();
    await preloadIcons(); // Load provider icons
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

// Theme Management
function initTheme() {
    // Check localStorage or default to dark
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme, false);
    
    // Setup toggle button
    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleTheme);
        updateThemeIcon();
    }
}

function setTheme(theme, save = true) {
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }
    
    if (save) {
        localStorage.setItem('theme', theme);
    }
    
    updateThemeIcon();
    
    // Update Chart.js defaults only if Chart is available
    if (typeof Chart !== 'undefined') {
        const currentTheme = getCurrentTheme();
        Chart.defaults.color = currentTheme.muted;
        Chart.defaults.font.family = currentTheme.font;
        
        // Re-render chart if it exists
        if (state.masterChart) {
            renderChart();
        }
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
}

function updateThemeIcon() {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const sunIcon = document.querySelector('.sun-icon');
    const moonIcon = document.querySelector('.moon-icon');
    
    if (sunIcon && moonIcon) {
        if (isDark) {
            sunIcon.style.display = 'none';
            moonIcon.style.display = 'block';
        } else {
            sunIcon.style.display = 'block';
            moonIcon.style.display = 'none';
        }
    }
}

function getProvider(modelId) {
    const prefix = modelId.split('/')[0];
    if (state.providers && state.providers[prefix]) {
        return state.providers[prefix];
    }
    return DEFAULT_PROVIDER;
}

// Preload provider icons as Image objects
async function preloadIcons() {
    const providers = state.providers || {};
    const loadPromises = [];

    for (const [key, provider] of Object.entries(providers)) {
        if (provider.icon && !state.iconCache[key]) {
            const img = new Image();
            const promise = new Promise((resolve) => {
                img.onload = () => {
                    state.iconCache[key] = img;
                    resolve();
                };
                img.onerror = () => resolve(); // Skip on error
            });
            img.src = `assets/${provider.icon}`;
            loadPromises.push(promise);
        }
    }

    await Promise.all(loadPromises);
}

// Icon plugin for Chart.js - draws provider icons above bars
// Takes sorted data array to match chart bar order
function createIconPlugin(sortedData) {
    return {
        id: 'iconPlugin',
        afterDatasetsDraw(chart) {
            const ctx = chart.ctx;
            const meta = chart.getDatasetMeta(0);
            const data = sortedData || state.filtered;

            meta.data.forEach((bar, index) => {
                const modelEntry = data[index];
                if (!modelEntry) return;

                const providerKey = modelEntry.model_id.split('/')[0];
                const icon = state.iconCache[providerKey];
                const provider = state.providers[providerKey] || {};

                if (icon) {
                    const iconSize = 20;
                    const x = bar.x - iconSize / 2;
                    const y = bar.y - iconSize - 5;

                    ctx.save();

                    // Draw background circle if icon_background is set
                    if (provider.icon_background) {
                        const centerX = bar.x;
                        const centerY = y + iconSize / 2;
                        const radius = iconSize / 2 + 3;

                        ctx.beginPath();
                        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
                        ctx.fillStyle = provider.icon_background;
                        ctx.fill();
                    }

                    ctx.drawImage(icon, x, y, iconSize, iconSize);
                    ctx.restore();
                }
            });
        }
    };
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
    const currentTheme = getCurrentTheme();
    if (val >= 7) return currentTheme.signal;
    if (val >= 4) return currentTheme.void === '#ffffff' ? '#ffffff' : '#1a1a1a';
    return currentTheme.muted;
}

function renderChart() {
    const ctx = document.getElementById('master-chart');
    if (!ctx || typeof Chart === 'undefined') return;

    if (state.masterChart) state.masterChart.destroy();

    const chartType = state.currentTab;

    if (chartType === 'accuracy') renderAccuracyChart(ctx);
    else if (chartType === 'cost') renderCostChart(ctx);
    else if (chartType === 'speed') renderSpeedChart(ctx);
    else if (chartType === 'combined') renderCombinedChart(ctx);
}

// 1. Accuracy Chart (Bar chart, sorted high to low)
function renderAccuracyChart(ctx) {
    const currentTheme = getCurrentTheme();
    const sorted = [...state.filtered].sort((a, b) => b.overall_score - a.overall_score);
    const labels = sorted.map(m => m.model_name);
    const dataPoints = sorted.map(m => m.overall_score);
    const backgroundColors = sorted.map(m => getProvider(m.model_id).color);
    const borderColors = sorted.map(m => getProvider(m.model_id).color);

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
                hoverBackgroundColor: currentTheme.signal
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        color: currentTheme.muted,
                        font: { family: currentTheme.font, size: 10 },
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    grid: { color: currentTheme.grid },
                    beginAtZero: true,
                    max: 10,
                    title: {
                        display: true,
                        text: 'Reliability Score (0-10)',
                        color: currentTheme.muted,
                        font: { family: currentTheme.font }
                    },
                    ticks: { color: currentTheme.muted }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: currentTheme.void,
                    titleColor: currentTheme.text,
                    bodyColor: currentTheme.text,
                    displayColors: false,
                    titleFont: { family: currentTheme.font },
                    bodyFont: { family: currentTheme.font },
                    borderColor: currentTheme.grid,
                    borderWidth: 1,
                    callbacks: {
                        label: (context) => ` Score: ${context.raw}`
                    }
                }
            },
            animation: { duration: 1500, easing: 'easeOutQuart' },
            layout: { padding: { top: 50 } } // Extra padding for icons
        },
        plugins: [
            createIconPlugin(sorted),
            {
                id: 'chartTitle',
                beforeDraw(chart) {
                    const ctx = chart.ctx;
                    ctx.save();
                    ctx.fillStyle = currentTheme.signal;
                    ctx.font = 'bold 12px ' + currentTheme.font;
                    ctx.textAlign = 'right';
                    ctx.fillText('↑ HIGHER IS BETTER', chart.width - 20, 20);
                    ctx.restore();
                }
            }
        ]
    });
}

// 2. Cost Chart (Vertical bar, sorted low to high, cost on Y axis)
function renderCostChart(ctx) {
    const currentTheme = getCurrentTheme();
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
                    grid: { color: currentTheme.grid },
                    title: { display: true, text: 'Avg Cost per Item ($)', color: currentTheme.muted },
                    ticks: { color: currentTheme.muted, callback: (v) => `$${v.toFixed(4)}` }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        color: currentTheme.muted,
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: currentTheme.void,
                    titleColor: currentTheme.text,
                    bodyColor: currentTheme.text,
                    borderColor: currentTheme.grid,
                    borderWidth: 1,
                    callbacks: { label: (c) => ` $${c.raw.toFixed(6)}` }
                }
            },
            layout: { padding: { top: 50 } }
        },
        plugins: [
            createIconPlugin(sorted),
            {
                id: 'chartTitle',
                beforeDraw(chart) {
                    const ctx = chart.ctx;
                    ctx.save();
                    ctx.fillStyle = currentTheme.error;
                    ctx.font = 'bold 12px ' + currentTheme.font;
                    ctx.textAlign = 'right';
                    ctx.fillText('↓ LOWER IS BETTER', chart.width - 20, 20);
                    ctx.restore();
                }
            }
        ]
    });
}

// 3. Speed Chart (Vertical bar, latency, sorted low to high)
function renderSpeedChart(ctx) {
    const currentTheme = getCurrentTheme();
    const sorted = [...state.filtered].sort((a, b) => (a.avg_latency || 0) - (b.avg_latency || 0));
    const labels = sorted.map(m => m.model_name);
    const data = sorted.map(m => m.avg_latency || 0);
    const colors = sorted.map(m => getProvider(m.model_id).color);

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
                    grid: { color: currentTheme.grid },
                    title: { display: true, text: 'Avg Latency (ms)', color: currentTheme.muted },
                    ticks: { color: currentTheme.muted }
                },
                x: { grid: { display: false }, ticks: { color: currentTheme.muted, maxRotation: 45, minRotation: 45 } }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: currentTheme.void,
                    titleColor: currentTheme.text,
                    bodyColor: currentTheme.text,
                    borderColor: currentTheme.grid,
                    borderWidth: 1,
                    callbacks: { label: (c) => ` ${c.raw.toFixed(0)} ms` }
                }
            },
            layout: { padding: { top: 50 } }
        },
        plugins: [
            createIconPlugin(sorted),
            {
                id: 'chartTitle',
                beforeDraw(chart) {
                    const ctx = chart.ctx;
                    ctx.save();
                    ctx.fillStyle = currentTheme.error;
                    ctx.font = 'bold 12px ' + currentTheme.font;
                    ctx.textAlign = 'right';
                    ctx.fillText('↓ LOWER IS BETTER', chart.width - 20, 20);
                    ctx.restore();
                }
            }
        ]
    });
}

// 4. Combined Chart (Scatter: Cost vs Accuracy)
function renderCombinedChart(ctx) {
    const currentTheme = getCurrentTheme();
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
                    title: { display: true, text: 'Avg Cost ($)', color: currentTheme.muted },
                    grid: { color: currentTheme.grid },
                    ticks: { color: currentTheme.muted }
                },
                y: {
                    beginAtZero: true,
                    max: 10,
                    title: { display: true, text: 'Reliability Score (0-10)', color: currentTheme.muted },
                    grid: { color: currentTheme.grid },
                    ticks: { color: currentTheme.muted }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: currentTheme.void,
                    titleColor: currentTheme.text,
                    bodyColor: currentTheme.text,
                    borderColor: currentTheme.grid,
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
                ctx.fillStyle = currentTheme.signal;
                ctx.font = 'bold 12px ' + currentTheme.font;
                ctx.textAlign = 'right';
                ctx.fillText('↖ TOP-LEFT = BEST (High Score, Low Cost)', chart.width - 20, 20);
                ctx.restore();
            }
        }]
    });
}
