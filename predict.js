// ── Prediction JavaScript ────────────────────────────────────

async function checkAuth() {
    try {
        const res  = await fetch('/api/me');
        const data = await res.json();
        if (!data.authenticated) { window.location.href = '/login'; return null; }
        document.querySelectorAll('.user-name').forEach(el => el.textContent = data.name);
        document.querySelectorAll('.user-email').forEach(el => el.textContent = data.email);
        document.querySelectorAll('.user-avatar').forEach(el => el.textContent = data.initials);
        return data;
    } catch { window.location.href = '/login'; return null; }
}

async function logout() {
    if (!confirm('Are you sure you want to logout?')) return;
    await fetch('/api/logout', { method: 'POST' });
    window.location.href = '/login';
}

document.addEventListener('DOMContentLoaded', async () => {
    const user = await checkAuth();
    if (!user) return;
    setupFormHandlers();
    loadPredictionHistory();
});

// ── Form visibility ───────────────────────────────────────────
function showChurnForm() {
    document.getElementById('churnForm').style.display = 'block';
    document.getElementById('churnForm').scrollIntoView({ behavior: 'smooth' });
}
function hideChurnForm() {
    document.getElementById('churnForm').style.display = 'none';
    document.getElementById('churnPredictionForm').reset();
    document.getElementById('churnResult').style.display = 'none';
}
function showSalesForm() {
    document.getElementById('salesForm').style.display = 'block';
    document.getElementById('salesForm').scrollIntoView({ behavior: 'smooth' });
}
function hideSalesForm() {
    document.getElementById('salesForm').style.display = 'none';
    document.getElementById('salesPredictionForm').reset();
    document.getElementById('salesResult').style.display = 'none';
}
function showSegmentForm() {
    showNotification('Customer segment classification coming soon!', 'info');
}

// ── Form handlers ─────────────────────────────────────────────
function setupFormHandlers() {
    document.getElementById('churnPredictionForm')?.addEventListener('submit', handleChurnPrediction);
    document.getElementById('salesPredictionForm')?.addEventListener('submit', handleSalesPrediction);
}

// ── Churn prediction ──────────────────────────────────────────
async function handleChurnPrediction(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data     = Object.fromEntries(formData);

    const submitBtn = e.target.querySelector('button[type="submit"]');
    const origText  = submitBtn.textContent;
    submitBtn.textContent = '🔮 Analyzing…';
    submitBtn.disabled    = true;

    try {
        const res  = await fetch('/api/predict/churn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                customer_id:     data.customer_id,
                age:             parseFloat(data.age),
                num_transactions: parseFloat(data.num_transactions),
                total_spent:     parseFloat(data.total_spent),
                avg_transaction: parseFloat(data.avg_transaction),
                recency:         parseFloat(data.recency),
                lifetime:        parseFloat(data.lifetime),
                gender:          data.gender,
                city:            'New York'
            })
        });
        const result = await res.json();

        if (result.success) {
            displayChurnResult(result);
            e.target.style.display = 'none';
            document.getElementById('churnResult').style.display = 'block';
            loadPredictionHistory();
        } else {
            showNotification(result.message || 'Prediction failed', 'error');
        }
    } catch (err) {
        showNotification('Network error: ' + err.message, 'error');
    } finally {
        submitBtn.textContent = origText;
        submitBtn.disabled    = false;
    }
}

function displayChurnResult(result) {
    document.getElementById('churnProbability').textContent = result.probability.toFixed(1) + '%';
    const statusEl = document.getElementById('churnStatus');
    const statusMap = {
        'High Risk':     { text: '🔴 High Risk – Immediate Action Required', cls: 'text-danger' },
        'Moderate Risk': { text: '🟡 Moderate Risk – Monitor Closely',        cls: 'text-warning' },
        'Low Risk':      { text: '🟢 Low Risk – Stable Customer',              cls: 'text-success' }
    };
    const s = statusMap[result.risk] || { text: result.risk, cls: '' };
    statusEl.textContent = s.text;
    statusEl.className   = `result-status ${s.cls}`;
    document.getElementById('churnFactors').innerHTML =
        result.factors.map(f => `<li>${f}</li>`).join('');
}

// ── Sales prediction ──────────────────────────────────────────
async function handleSalesPrediction(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data     = Object.fromEntries(formData);

    const submitBtn = e.target.querySelector('button[type="submit"]');
    const origText  = submitBtn.textContent;
    submitBtn.textContent = '📈 Predicting…';
    submitBtn.disabled    = true;

    try {
        const res  = await fetch('/api/predict/sales', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                customer_id:     data.customer_id,
                age:             parseFloat(data.age),
                num_transactions: parseFloat(data.num_transactions),
                avg_transaction: parseFloat(data.avg_transaction),
                recency:         parseFloat(data.recency),
                lifetime:        parseFloat(data.lifetime),
                gender:          'M'
            })
        });
        const result = await res.json();

        if (result.success) {
            document.getElementById('salesAmount').textContent      = '$' + result.predicted_sales.toFixed(2);
            document.getElementById('salesConfidence').textContent  = 'Model Confidence: ' + result.confidence + '%';
            document.getElementById('modelConfidence').textContent  = result.confidence + '%';
            document.getElementById('predictionRange').textContent  = '±$' + result.range.toFixed(2);
            e.target.style.display = 'none';
            document.getElementById('salesResult').style.display = 'block';
            loadPredictionHistory();
        } else {
            showNotification(result.message || 'Prediction failed', 'error');
        }
    } catch (err) {
        showNotification('Network error: ' + err.message, 'error');
    } finally {
        submitBtn.textContent = origText;
        submitBtn.disabled    = false;
    }
}

// ── Reset forms ───────────────────────────────────────────────
function resetChurnForm() {
    document.getElementById('churnPredictionForm').reset();
    document.getElementById('churnPredictionForm').style.display = 'block';
    document.getElementById('churnResult').style.display = 'none';
}
function resetSalesForm() {
    document.getElementById('salesPredictionForm').reset();
    document.getElementById('salesPredictionForm').style.display = 'block';
    document.getElementById('salesResult').style.display = 'none';
}
function saveResult(type) {
    showNotification(`${type.charAt(0).toUpperCase() + type.slice(1)} prediction saved!`, 'success');
}

// ── Prediction history ────────────────────────────────────────
async function loadPredictionHistory() {
    try {
        const res     = await fetch('/api/predictions/history');
        const history = await res.json();
        updatePredictionsTable(history);
    } catch {}
}

function updatePredictionsTable(history) {
    const tbody = document.getElementById('predictionsTableBody');
    if (!tbody || history.length === 0) return;
    tbody.innerHTML = history.slice(0, 10).map(pred => {
        const badge  = pred.type === 'churn' ? 'badge-churn' : 'badge-sales';
        const result = pred.type === 'churn'
            ? (pred.result > 70 ? '🔴 High Risk' : pred.result > 40 ? '🟡 Moderate' : '🟢 Low Risk')
            : '$' + parseFloat(pred.result).toFixed(0);
        return `<tr>
            <td>${pred.date}</td>
            <td><span class="badge ${badge}">${pred.type.charAt(0).toUpperCase() + pred.type.slice(1)}</span></td>
            <td>${pred.customer_id || '—'}</td>
            <td>${result}</td>
            <td>${pred.confidence}%</td>
            <td><button class="btn-small">View</button></td>
        </tr>`;
    }).join('');
}

// ── Notification ──────────────────────────────────────────────
function showNotification(message, type = 'info') {
    document.querySelector('.notification')?.remove();
    const n = document.createElement('div');
    n.className   = `notification notification-${type}`;
    n.textContent = message;
    document.body.appendChild(n);
    setTimeout(() => { n.style.animation = 'slideOutRight 0.3s ease-out'; setTimeout(() => n.remove(), 300); }, 3000);
}

console.log('✅ Prediction module loaded');
