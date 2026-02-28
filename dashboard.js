// ── Dashboard JavaScript ─────────────────────────────────────

// ── Auth check ───────────────────────────────────────────────
async function checkAuth() {
    try {
        const res  = await fetch('/api/me');
        const data = await res.json();
        if (!data.authenticated) {
            window.location.href = '/login';
            return null;
        }
        // Populate user info
        document.querySelectorAll('.user-name').forEach(el => el.textContent = data.name);
        document.querySelectorAll('.user-email').forEach(el => el.textContent = data.email);
        document.querySelectorAll('.user-avatar').forEach(el => el.textContent = data.initials);
        return data;
    } catch {
        window.location.href = '/login';
        return null;
    }
}

async function logout() {
    if (!confirm('Are you sure you want to logout?')) return;
    await fetch('/api/logout', { method: 'POST' });
    window.location.href = '/login';
}

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    const user = await checkAuth();
    if (!user) return;

    await loadStats();
    await loadSegments();
    initCharts();
});

// ── Load dashboard stats from API ────────────────────────────
let STATS = {};
async function loadStats() {
    try {
        const res  = await fetch('/api/dashboard/stats');
        STATS      = await res.json();
        animateMetrics();
    } catch (err) {
        console.error('Failed to load stats', err);
    }
}

// ── Load segments ─────────────────────────────────────────────
async function loadSegments() {
    try {
        const res      = await fetch('/api/dashboard/segments');
        const segments = await res.json();
        const emojis   = { 0:'🏆', 1:'💎', 2:'⚠️', 3:'💔' };
        const classes  = { 0:'champions', 1:'loyal', 2:'at-risk', 3:'lost' };

        const grid = document.querySelector('.segments-grid');
        if (!grid) return;
        grid.innerHTML = segments.map(s => `
            <div class="segment-card ${classes[s.id]}">
                <div class="segment-header">
                    <span class="segment-emoji">${emojis[s.id]}</span>
                    <h4>${s.name}</h4>
                </div>
                <div class="segment-count">~${Math.round(250 * (s.spent / 500))} customers</div>
                <div class="segment-stats">
                    <div class="stat-item">
                        <span class="stat-label">Avg Recency</span>
                        <span class="stat-value">${s.recency} days</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg Spent</span>
                        <span class="stat-value">$${s.spent}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg Transactions</span>
                        <span class="stat-value">${s.transactions}</span>
                    </div>
                </div>
            </div>`).join('');
    } catch (err) {
        console.error('Failed to load segments', err);
    }
}

// ── Animate metric counters ───────────────────────────────────
function animateMetrics() {
    setTimeout(() => {
        animateValue('totalCustomers', 0, STATS.total_customers || 1000, 1500, '', '');
        animateValue('totalRevenue',   0, STATS.total_revenue   || 500000, 1500, '$', '');
        animateValue('avgTransaction', 0, STATS.avg_transaction || 100, 1500, '$', '');
        animateValue('churnRate',      0, STATS.churn_rate      || 24.3, 1500, '', '%');
    }, 300);
}

function animateValue(id, start, end, duration, prefix = '', suffix = '') {
    const el = document.getElementById(id);
    if (!el) return;
    const range     = end - start;
    const increment = range / (duration / 16);
    let current     = start;

    const timer = setInterval(() => {
        current += increment;
        if (increment > 0 ? current >= end : current <= end) {
            current = end;
            clearInterval(timer);
        }
        let display;
        if (id === 'churnRate')      display = current.toFixed(1);
        else if (id === 'totalRevenue') display = '$' + (current / 1000).toFixed(0) + 'K';
        else                         display = Math.floor(current).toLocaleString();

        el.textContent = id === 'totalRevenue' ? display : prefix + display + suffix;
    }, 16);
}

// ── Charts ────────────────────────────────────────────────────
function initCharts() {
    Chart.defaults.color       = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(51,65,85,0.3)';
    Chart.defaults.font.family = "'Space Mono', monospace";

    // Amount Distribution
    const amountCtx = document.getElementById('amountChart');
    if (amountCtx && STATS.bucket_counts) {
        const labels = Object.keys(STATS.bucket_counts);
        const values = Object.values(STATS.bucket_counts);
        new Chart(amountCtx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{ label: 'Transactions', data: values,
                    backgroundColor: 'rgba(102,126,234,0.7)',
                    borderColor: '#667eea', borderWidth: 2, borderRadius: 8 }]
            },
            options: { responsive: true, maintainAspectRatio: true,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true, grid: { color: 'rgba(51,65,85,0.2)' } },
                          x: { grid: { display: false } } } }
        });
    }

    // Category doughnut
    const catCtx = document.getElementById('categoryChart');
    if (catCtx && STATS.cat_sales) {
        const catLabels = Object.keys(STATS.cat_sales);
        const catValues = Object.values(STATS.cat_sales);
        new Chart(catCtx, {
            type: 'doughnut',
            data: {
                labels: catLabels,
                datasets: [{ data: catValues,
                    backgroundColor: [
                        'rgba(102,126,234,0.8)', 'rgba(240,147,251,0.8)',
                        'rgba(74,222,128,0.8)',  'rgba(251,191,36,0.8)',
                        'rgba(96,165,250,0.8)'
                    ],
                    borderColor: '#1e293b', borderWidth: 3 }]
            },
            options: { responsive: true, maintainAspectRatio: true,
                plugins: { legend: { position: 'bottom', labels: { padding: 15, font: { size: 11 } } } } }
        });
    }

    // Age distribution
    const ageCtx = document.getElementById('ageChart');
    if (ageCtx && STATS.age_counts) {
        new Chart(ageCtx, {
            type: 'line',
            data: {
                labels: Object.keys(STATS.age_counts),
                datasets: [{ label: 'Customers', data: Object.values(STATS.age_counts),
                    backgroundColor: 'rgba(74,222,128,0.2)',
                    borderColor: '#4ade80', borderWidth: 3, fill: true, tension: 0.4,
                    pointBackgroundColor: '#4ade80', pointRadius: 5, pointHoverRadius: 7 }]
            },
            options: { responsive: true, maintainAspectRatio: true,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true, grid: { color: 'rgba(51,65,85,0.2)' } },
                          x: { grid: { display: false } } } }
        });
    }

    // Revenue Trend
    const revCtx = document.getElementById('revenueChart');
    if (revCtx && STATS.monthly_labels) {
        new Chart(revCtx, {
            type: 'line',
            data: {
                labels: STATS.monthly_labels,
                datasets: [{ label: 'Revenue', data: STATS.monthly_values,
                    backgroundColor: 'rgba(102,126,234,0.1)',
                    borderColor: '#667eea', borderWidth: 3, fill: true, tension: 0.4,
                    pointBackgroundColor: '#667eea', pointRadius: 6, pointHoverRadius: 8 }]
            },
            options: { responsive: true, maintainAspectRatio: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true,
                         ticks: { callback: v => '$' + (v/1000).toFixed(0) + 'K' },
                         grid: { color: 'rgba(51,65,85,0.2)' } },
                    x: { grid: { display: false } } } }
        });
    }
}

// ── Export ────────────────────────────────────────────────────
function exportData() {
    showNotification('Exporting data…', 'info');
    window.location.href = '/api/export/csv';
    setTimeout(() => showNotification('Download started!', 'success'), 500);
}

// ── Notification ──────────────────────────────────────────────
function showNotification(message, type = 'info') {
    document.querySelector('.notification')?.remove();
    const n = document.createElement('div');
    n.className   = `notification notification-${type}`;
    n.textContent = message;
    document.body.appendChild(n);
    setTimeout(() => {
        n.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => n.remove(), 300);
    }, 3000);
}

// Mobile sidebar toggle
document.getElementById('sidebarToggle')?.addEventListener('click', () => {
    document.querySelector('.sidebar')?.classList.toggle('open');
});

console.log('✅ Dashboard module loaded');
