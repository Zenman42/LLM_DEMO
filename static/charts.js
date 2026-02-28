let chartInstances = {};  // { google: Chart, yandex: Chart }

const COLORS = [
    '#4299e1', '#48bb78', '#ed8936', '#9f7aea',
    '#f56565', '#38b2ac', '#ecc94b', '#667eea',
    '#fc8181', '#68d391', '#f6ad55', '#b794f4',
];

function getCurrentEngine() {
    var tab = document.querySelector('.tab-content.active');
    if (tab && tab.id === 'tabYandex') return 'yandex';
    return 'google';
}

function getChartParams() {
    var engine = getCurrentEngine();
    var cap = engine.charAt(0).toUpperCase() + engine.slice(1);
    var modeEl = document.querySelector('input[name="chartMode' + cap + '"]:checked');
    var mode = modeEl ? modeEl.value : 'keyword';
    var params = { mode: mode, engine: engine };

    if (mode === 'keyword') {
        var sel = document.getElementById('chartKeywordSelect' + cap);
        if (sel) params.keyword_id = sel.value;
    } else {
        // Collect checked keyword IDs from checkboxes
        var checks = document.querySelectorAll('#urlKeywordsList' + cap + ' input[type="checkbox"]:checked');
        var ids = Array.from(checks).map(function(c) { return c.value; });
        params.keyword_ids = ids.join(',');
    }

    var daysEl = document.getElementById('daysSelect' + cap);
    params.days = daysEl ? daysEl.value : '30';

    return params;
}

async function onUrlSelectChange() {
    var engine = getCurrentEngine();
    var cap = engine.charAt(0).toUpperCase() + engine.slice(1);
    var urlSel = document.getElementById('chartUrlSelect' + cap);
    if (!urlSel || !urlSel.value) return;

    var container = document.getElementById('urlKeywordsList' + cap);
    container.innerHTML = '<span class="muted">Loading...</span>';
    container.style.display = '';

    try {
        var r = await fetch('/api/url-keywords/' + PROJECT_ID +
            '?page_url=' + encodeURIComponent(urlSel.value) +
            '&engine=' + engine);
        var data = await r.json();
        renderUrlKeywords(data.keywords, cap);
    } catch (e) {
        container.innerHTML = '<span class="muted" style="color:var(--danger);">Failed to load</span>';
    }
}

function renderUrlKeywords(keywords, cap) {
    var container = document.getElementById('urlKeywordsList' + cap);
    if (!keywords || !keywords.length) {
        container.innerHTML = '<span class="muted">No keywords found for this URL</span>';
        container.style.display = '';
        clearChart();
        return;
    }
    container.style.display = '';
    var html = keywords.map(function(kw) {
        return '<label class="url-kw-item">' +
            '<input type="checkbox" value="' + kw.id + '" onchange="loadChart()">' +
            '<code>' + kw.keyword + '</code>' +
            '<span class="url-kw-imp">(' + kw.impressions.toLocaleString() + ')</span>' +
            '</label>';
    }).join('');
    container.innerHTML = html;
    clearChart();
}

function clearChart() {
    var engine = getCurrentEngine();
    if (chartInstances[engine]) {
        chartInstances[engine].destroy();
        chartInstances[engine] = null;
    }
}

async function loadChart() {
    if (typeof PROJECT_ID === 'undefined') return;

    var cp = getChartParams();

    // If URL mode and no keywords checked — clear chart
    if (cp.mode === 'url' && !cp.keyword_ids) {
        clearChart();
        return;
    }

    var params = new URLSearchParams({
        days: cp.days,
        engine: cp.engine,
        mode: cp.mode,
    });
    if (cp.keyword_id) params.set('keyword_id', cp.keyword_id);
    if (cp.keyword_ids) params.set('keyword_ids', cp.keyword_ids);

    try {
        var response = await fetch('/api/chart/' + PROJECT_ID + '?' + params);
        var data = await response.json();

        var engine = cp.engine;
        var cap = engine.charAt(0).toUpperCase() + engine.slice(1);

        if (chartInstances[engine]) {
            chartInstances[engine].destroy();
        }

        var ctx = document.getElementById('positionChart' + cap);
        if (!ctx) return;

        var colorIndex = 0;
        var datasets = data.datasets.map(function(ds) {
            var isGsc = ds.type === 'gsc';
            var color = isGsc ? COLORS[(colorIndex - 1 + COLORS.length) % COLORS.length]
                              : COLORS[colorIndex % COLORS.length];
            if (!isGsc) colorIndex++;
            return {
                label: ds.label,
                data: ds.data,
                borderColor: color,
                backgroundColor: color + '20',
                fill: false,
                tension: 0.3,
                pointRadius: isGsc ? 2 : 3,
                pointHoverRadius: isGsc ? 4 : 6,
                borderWidth: 2,
                borderDash: isGsc ? [6, 3] : [],
                spanGaps: true,
            };
        });

        // Compute Y-axis bounds from actual data
        var allValues = [];
        datasets.forEach(function(ds) {
            ds.data.forEach(function(v) { if (v !== null) allValues.push(v); });
        });
        var yMin = 1, yMax = 10;
        if (allValues.length > 0) {
            var dataMin = Math.min.apply(null, allValues);
            var dataMax = Math.max.apply(null, allValues);
            yMin = Math.max(0, Math.floor(dataMin) - 2);
            yMax = Math.ceil(dataMax) + 2;
            if (yMax - yMin < 5) yMax = yMin + 5;
        }
        var yStep = (yMax - yMin) <= 15 ? 1 : (yMax - yMin) <= 40 ? 5 : 10;

        chartInstances[engine] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: datasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        reverse: true,
                        min: yMin,
                        max: yMax,
                        title: { display: true, text: 'Position' },
                        ticks: { stepSize: yStep },
                    },
                    x: {
                        title: { display: true, text: 'Date' },
                        ticks: { maxRotation: 45, maxTicksLimit: 15 },
                    },
                },
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(ctx) {
                                var pos = ctx.parsed.y;
                                if (pos === null) return ctx.dataset.label + ': no data';
                                return ctx.dataset.label + ': #' + pos;
                            },
                        },
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 20,
                            boxHeight: 2,
                            usePointStyle: false,
                            padding: 15,
                            font: { size: 12 },
                        },
                    },
                },
                interaction: { mode: 'index', intersect: false },
            },
        });
    } catch (e) {
        console.error('Chart load error:', e);
    }
}
