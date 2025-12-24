/**
 * MIRAGE Benchmark - Frontend Application
 * Handles leaderboard display, search/sort, and chart rendering
 */

// Application state
let leaderboardData = null;
let filteredData = [];

// Chart instances
let radarChart = null;
let barChart = null;

// Chart.js configuration
const chartColors = {
    primary: 'rgba(99, 102, 241, 1)',
    primaryLight: 'rgba(99, 102, 241, 0.3)',
    secondary: 'rgba(139, 92, 246, 1)',
    secondaryLight: 'rgba(139, 92, 246, 0.3)',
    tertiary: 'rgba(6, 182, 212, 1)',
    tertiaryLight: 'rgba(6, 182, 212, 0.3)',
    success: 'rgba(16, 185, 129, 1)',
    successLight: 'rgba(16, 185, 129, 0.3)',
    warning: 'rgba(245, 158, 11, 1)',
    warningLight: 'rgba(245, 158, 11, 0.3)',
    text: 'rgba(240, 240, 245, 0.8)',
    textMuted: 'rgba(160, 160, 176, 0.6)',
    grid: 'rgba(255, 255, 255, 0.08)',
};

const modelColors = [
    { bg: chartColors.primaryLight, border: chartColors.primary },
    { bg: chartColors.secondaryLight, border: chartColors.secondary },
    { bg: chartColors.tertiaryLight, border: chartColors.tertiary },
    { bg: chartColors.successLight, border: chartColors.success },
    { bg: chartColors.warningLight, border: chartColors.warning },
];

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', async () => {
    await loadData();
    setupEventListeners();
    renderLeaderboard();
    renderCharts();
});

/**
 * Load leaderboard data from JSON
 */
async function loadData() {
    try {
        const response = await fetch('data/results.json');
        if (!response.ok) {
            throw new Error('No data available');
        }
        leaderboardData = await response.json();
        filteredData = [...(leaderboardData.entries || [])];

        // Update total models count
        document.getElementById('total-models').textContent = filteredData.length;
    } catch (error) {
        console.log('No results data yet:', error.message);
        leaderboardData = { entries: [] };
        filteredData = [];
    }
}

/**
 * Setup event listeners for search and sort
 */
function setupEventListeners() {
    const searchInput = document.getElementById('search-input');
    const sortSelect = document.getElementById('sort-select');

    searchInput.addEventListener('input', (e) => {
        filterAndSort(e.target.value, sortSelect.value);
    });

    sortSelect.addEventListener('change', (e) => {
        filterAndSort(searchInput.value, e.target.value);
    });
}

/**
 * Filter and sort leaderboard data
 */
function filterAndSort(searchTerm, sortBy) {
    let data = [...(leaderboardData.entries || [])];

    // Filter by search term
    if (searchTerm) {
        const term = searchTerm.toLowerCase();
        data = data.filter(entry =>
            entry.model_name.toLowerCase().includes(term) ||
            entry.model_id.toLowerCase().includes(term)
        );
    }

    // Sort
    if (sortBy === 'overall') {
        data.sort((a, b) => b.overall_score - a.overall_score);
    } else if (sortBy in (data[0]?.axis_scores || {})) {
        data.sort((a, b) => (b.axis_scores[sortBy] || 0) - (a.axis_scores[sortBy] || 0));
    }

    // Update ranks
    data.forEach((entry, idx) => entry.rank = idx + 1);

    filteredData = data;
    renderLeaderboard();
}

/**
 * Render the leaderboard table
 */
function renderLeaderboard() {
    const tbody = document.getElementById('leaderboard-body');
    const noDataMessage = document.getElementById('no-data-message');

    if (filteredData.length === 0) {
        tbody.innerHTML = '';
        noDataMessage.style.display = 'block';
        return;
    }

    noDataMessage.style.display = 'none';

    tbody.innerHTML = filteredData.map(entry => {
        const rankClass = entry.rank <= 3 ? `rank-${entry.rank}` : 'rank-other';

        return `
            <tr>
                <td class="rank-col">
                    <span class="rank-badge ${rankClass}">${entry.rank}</span>
                </td>
                <td class="model-col">
                    <div class="model-info">
                        <span class="model-name">${escapeHtml(entry.model_name)}</span>
                        <span class="model-id">${escapeHtml(entry.model_id)}</span>
                    </div>
                </td>
                <td class="score-col">
                    <div class="score-display">
                        <span class="score-value">${entry.overall_score.toFixed(2)}</span>
                        <span class="score-max">/ 10</span>
                    </div>
                </td>
                ${renderTrackScores(entry.track_scores)}
            </tr>
        `;
    }).join('');
}

/**
 * Render track score cells
 */
function renderTrackScores(trackScores) {
    const tracks = ['A', 'B', 'C', 'D', 'E'];

    return tracks.map(track => {
        const score = trackScores?.[track] ?? '-';
        const scoreClass = getScoreClass(score);
        const displayScore = typeof score === 'number' ? score.toFixed(1) : score;

        return `<td class="track-col"><span class="track-score ${scoreClass}">${displayScore}</span></td>`;
    }).join('');
}

/**
 * Get CSS class based on score value
 */
function getScoreClass(score) {
    if (typeof score !== 'number') return '';
    if (score >= 7) return 'high';
    if (score >= 4) return 'medium';
    return 'low';
}

/**
 * Render charts
 */
function renderCharts() {
    renderRadarChart();
    renderBarChart();
}

/**
 * Render radar chart for axis comparison
 */
function renderRadarChart() {
    const ctx = document.getElementById('radar-chart');
    if (!ctx) return;

    const top5 = filteredData.slice(0, 5);

    if (top5.length === 0) {
        // Show placeholder
        return;
    }

    const labels = [
        'Ambiguity Detection',
        'Hallucination Avoidance',
        'Localization',
        'Response Strategy',
        'Epistemic Tone'
    ];

    const datasets = top5.map((entry, idx) => ({
        label: entry.model_name,
        data: [
            entry.axis_scores?.ambiguity_detection || 0,
            entry.axis_scores?.hallucination_avoidance || 0,
            entry.axis_scores?.localization_of_uncertainty || 0,
            entry.axis_scores?.response_strategy || 0,
            entry.axis_scores?.epistemic_tone || 0,
        ],
        backgroundColor: modelColors[idx].bg,
        borderColor: modelColors[idx].border,
        borderWidth: 2,
        pointBackgroundColor: modelColors[idx].border,
        pointBorderColor: '#fff',
        pointBorderWidth: 1,
        pointRadius: 4,
    }));

    if (radarChart) {
        radarChart.destroy();
    }

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    min: 0,
                    max: 2,
                    ticks: {
                        stepSize: 0.5,
                        color: chartColors.textMuted,
                        backdropColor: 'transparent',
                    },
                    grid: {
                        color: chartColors.grid,
                    },
                    angleLines: {
                        color: chartColors.grid,
                    },
                    pointLabels: {
                        color: chartColors.text,
                        font: {
                            size: 11,
                        },
                    },
                },
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: chartColors.text,
                        padding: 16,
                        usePointStyle: true,
                        pointStyle: 'circle',
                    },
                },
            },
        },
    });
}

/**
 * Render bar chart for track performance
 */
function renderBarChart() {
    const ctx = document.getElementById('bar-chart');
    if (!ctx) return;

    const top5 = filteredData.slice(0, 5);

    if (top5.length === 0) {
        return;
    }

    const tracks = ['A', 'B', 'C', 'D', 'E'];
    const trackNames = {
        'A': 'Noisy Perception',
        'B': 'Ambiguous Semantics',
        'C': 'False Premise',
        'D': 'Underspecified',
        'E': 'Conflicts',
    };

    const datasets = top5.map((entry, idx) => ({
        label: entry.model_name,
        data: tracks.map(t => entry.track_scores?.[t] || 0),
        backgroundColor: modelColors[idx].bg,
        borderColor: modelColors[idx].border,
        borderWidth: 2,
        borderRadius: 6,
    }));

    if (barChart) {
        barChart.destroy();
    }

    barChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: tracks.map(t => trackNames[t]),
            datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: {
                        color: chartColors.text,
                    },
                    grid: {
                        color: chartColors.grid,
                    },
                },
                y: {
                    min: 0,
                    max: 10,
                    ticks: {
                        color: chartColors.textMuted,
                    },
                    grid: {
                        color: chartColors.grid,
                    },
                },
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: chartColors.text,
                        padding: 16,
                        usePointStyle: true,
                        pointStyle: 'rect',
                    },
                },
            },
        },
    });
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
