"""
E-Commerce Analytics Platform - Flask Backend
"""
import os
import json
import pickle
import secrets
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_file)

# ─── ML imports ───────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error, r2_score, silhouette_score)

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.permanent_session_lifetime = timedelta(hours=24)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── In-memory user store (replace with a real DB in production) ───────────────
USERS = {}          # email -> {password, name, company}
PREDICTIONS = {}    # email -> list of prediction dicts

# ─── ML model state ───────────────────────────────────────────────────────────
ML = {}   # populated by train_models()

# ──────────────────────────────────────────────────────────────────────────────
# ML PIPELINE  (mirrors Ecommerce_analysis.py)
# ──────────────────────────────────────────────────────────────────────────────
def generate_data():
    np.random.seed(42)
    n_customers, n_transactions = 1000, 5000
    cust_ids = np.arange(1, n_customers + 1)

    customer_data = pd.DataFrame({
        'customer_id': cust_ids,
        'age': np.random.randint(18, 70, n_customers),
        'gender': np.random.choice(['M', 'F'], n_customers),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers),
        'registration_date': pd.date_range(start='2023-01-01', periods=n_customers, freq='6h')
    })

    transaction_data = pd.DataFrame({
        'transaction_id': np.arange(1, n_transactions + 1),
        'customer_id': np.random.choice(cust_ids, n_transactions),
        'transaction_date': pd.date_range(start='2023-06-01', periods=n_transactions, freq='2h'),
        'product_category': np.random.choice(['Electronics','Clothing','Home','Books','Sports'], n_transactions),
        'amount': np.random.gamma(2, 50, n_transactions),
        'quantity': np.random.randint(1, 5, n_transactions)
    })
    return customer_data, transaction_data


def train_models():
    global ML
    print("🔧 Training ML models …")

    customer_data, transaction_data = generate_data()
    df = transaction_data.merge(customer_data, on='customer_id', how='left')

    # ── Feature engineering ────────────────────────────────────────────────
    metrics = df.groupby('customer_id').agg(
        num_transaction=('transaction_id', 'count'),
        total_spent=('amount', 'sum'),
        avg_transaction=('amount', 'mean'),
        std_transaction=('amount', 'std'),
        total_items=('quantity', 'sum'),
        first_purchase=('transaction_date', 'min'),
        last_purchase=('transaction_date', 'max')
    ).reset_index()

    ref_date = df['transaction_date'].max()
    metrics['recency_days'] = (ref_date - metrics['last_purchase']).dt.days
    metrics['customer_lifetime_days'] = (metrics['last_purchase'] - metrics['first_purchase']).dt.days

    cf = customer_data.merge(metrics, on='customer_id', how='left')
    cf['days_since_registration'] = (ref_date - cf['registration_date']).dt.days
    # Only fill numeric columns with 0 (avoid datetime columns)
    numeric_cols = cf.select_dtypes(include=[np.number]).columns
    cf[numeric_cols] = cf[numeric_cols].fillna(0)
    cf['is_churned'] = (cf['recency_days'] > 60).astype(int)

    # ── Segmentation ─────────────────────────────────────────────────────
    rfm = cf[['recency_days', 'num_transaction', 'total_spent']].copy()
    rfm = rfm[cf['num_transaction'] > 0]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # ── Churn classifier ─────────────────────────────────────────────────
    le_gender = LabelEncoder()
    le_city   = LabelEncoder()
    feature_cols = ['age', 'num_transaction', 'total_spent', 'avg_transaction',
                    'recency_days', 'days_since_registration']
    X = cf[feature_cols].copy()
    X['gender_encoded'] = le_gender.fit_transform(cf['gender'])
    X['city_encoded']   = le_city.fit_transform(cf['city'])
    y = cf['is_churned']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    churn_accuracy  = clf.score(X_te, y_te)
    report          = classification_report(y_te, y_pred, output_dict=True)
    feat_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    # ── Sales regressor ──────────────────────────────────────────────────
    reg_data = cf[cf['num_transaction'] > 0].copy()
    X_reg = reg_data[['age','num_transaction','recency_days','avg_transaction','customer_lifetime_days']].copy()
    X_reg['gender_encoded'] = le_gender.transform(reg_data['gender'])
    y_reg = reg_data['total_spent']

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    reg.fit(X_tr_r, y_tr_r)
    y_pred_r = reg.predict(X_te_r)

    rmse = float(np.sqrt(mean_squared_error(y_te_r, y_pred_r)))
    r2   = float(r2_score(y_te_r, y_pred_r))
    mae  = float(np.abs(y_te_r - y_pred_r).mean())

    # ── Build dashboard stats ────────────────────────────────────────────
    cat_sales = df.groupby('product_category')['amount'].sum().to_dict()
    age_dist  = df.merge(customer_data[['customer_id','age']], on='customer_id', how='left')
    age_bins  = pd.cut(customer_data['age'], bins=[18,25,35,45,55,65,100],
                       labels=['18-25','26-35','36-45','46-55','56-65','65+'])
    age_counts = age_bins.value_counts().sort_index().to_dict()

    # Monthly revenue (last 6 months of data)
    df['month'] = df['transaction_date'].dt.to_period('M')
    monthly = df.groupby('month')['amount'].sum()
    monthly_labels = [str(m) for m in monthly.index[-6:]]
    monthly_values = [round(v, 2) for v in monthly.values[-6:]]

    # Amount distribution buckets
    bins   = [0, 50, 100, 150, 200, 250, 300, 1e9]
    labels = ['$0-50','$50-100','$100-150','$150-200','$200-250','$250-300','$300+']
    df['bucket'] = pd.cut(df['amount'], bins=bins, labels=labels)
    bucket_counts = df['bucket'].value_counts().reindex(labels, fill_value=0).to_dict()

    segment_names = {0:'Champions', 1:'Loyal Customers', 2:'At Risk', 3:'Lost'}
    seg_stats = rfm.groupby('cluster').agg(
        recency_days=('recency_days','mean'),
        num_transaction=('num_transaction','mean'),
        total_spent=('total_spent','mean')
    ).round(2)

    cm = confusion_matrix(y_te, y_pred).tolist()

    ML.update({
        'clf': clf,
        'reg': reg,
        'le_gender': le_gender,
        'le_city':   le_city,
        'scaler':    scaler,
        'kmeans':    kmeans,
        'churn_accuracy': round(churn_accuracy * 100, 1),
        'churn_precision': round(report['1']['precision'] * 100, 1),
        'churn_recall':    round(report['1']['recall']    * 100, 1),
        'churn_f1':        round(report['1']['f1-score']  * 100, 1),
        'feat_importance': feat_importance.to_dict(orient='records'),
        'sales_rmse': round(rmse, 2),
        'sales_r2':   round(r2, 4),
        'sales_mae':  round(mae, 2),
        'confusion_matrix': cm,
        'total_customers': int(len(cf)),
        'total_revenue':   round(float(df['amount'].sum()), 2),
        'avg_transaction': round(float(df['amount'].mean()), 2),
        'churn_rate':      round(float(cf['is_churned'].mean() * 100), 1),
        'cat_sales':       {k: round(v, 2) for k, v in cat_sales.items()},
        'age_counts':      age_counts,
        'monthly_labels':  monthly_labels,
        'monthly_values':  monthly_values,
        'bucket_counts':   bucket_counts,
        'segment_stats':   seg_stats.to_dict(),
        'segment_names':   segment_names,
        'feature_cols':    list(X.columns),
        'customer_features': cf,
    })
    print(f"✅ Models ready — Churn acc={churn_accuracy:.2%}  Sales R²={r2:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# AUTH HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ──────────────────────────────────────────────────────────────────────────────
# PAGE ROUTES
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register')
def register():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
    return render_template('registration.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user = USERS.get(session['user_email'], {})
    return render_template('dashboard.html', user=user)

@app.route('/predict')
@login_required
def predict():
    user = USERS.get(session['user_email'], {})
    return render_template('predict.html', user=user)

@app.route('/results')
@login_required
def results():
    user = USERS.get(session['user_email'], {})
    return render_template('results.html', user=user)


# ──────────────────────────────────────────────────────────────────────────────
# AUTH API
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    email    = data.get('email', '').lower().strip()
    password = data.get('password', '')
    name     = data.get('name', email.split('@')[0])
    company  = data.get('company', '')

    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password required'}), 400
    if len(password) < 8:
        return jsonify({'success': False, 'message': 'Password must be at least 8 characters'}), 400
    if email in USERS:
        return jsonify({'success': False, 'message': 'Email already registered'}), 409

    USERS[email] = {'password': password, 'name': name, 'company': company}
    PREDICTIONS[email] = []
    session.permanent = True
    session['user_email'] = email
    return jsonify({'success': True, 'redirect': '/dashboard'})


@app.route('/api/login', methods=['POST'])
def api_login():
    data     = request.get_json()
    email    = data.get('email', '').lower().strip()
    password = data.get('password', '')

    user = USERS.get(email)
    if not user or user['password'] != password:
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    session.permanent = True
    session['user_email'] = email
    return jsonify({'success': True, 'redirect': '/dashboard'})


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True, 'redirect': '/login'})


@app.route('/api/me')
def api_me():
    email = session.get('user_email')
    if not email or email not in USERS:
        return jsonify({'authenticated': False}), 401
    u = USERS[email]
    return jsonify({
        'authenticated': True,
        'email': email,
        'name': u['name'],
        'company': u['company'],
        'initials': ''.join(w[0].upper() for w in u['name'].split() if w)[:2]
    })


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD DATA API
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/dashboard/stats')
@login_required
def api_dashboard_stats():
    return jsonify({
        'total_customers': ML['total_customers'],
        'total_revenue':   ML['total_revenue'],
        'avg_transaction': ML['avg_transaction'],
        'churn_rate':      ML['churn_rate'],
        'cat_sales':       ML['cat_sales'],
        'age_counts':      ML['age_counts'],
        'monthly_labels':  ML['monthly_labels'],
        'monthly_values':  ML['monthly_values'],
        'bucket_counts':   ML['bucket_counts'],
    })


@app.route('/api/dashboard/segments')
@login_required
def api_segments():
    seg = ML['segment_stats']
    names = ML['segment_names']
    result = []
    for k, name in names.items():
        result.append({
            'id': k,
            'name': name,
            'recency': round(seg['recency_days'][k], 0),
            'transactions': round(seg['num_transaction'][k], 1),
            'spent': round(seg['total_spent'][k], 2),
        })
    return jsonify(result)


# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION API
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/predict/churn', methods=['POST'])
@login_required
def api_predict_churn():
    d = request.get_json()
    try:
        age             = float(d.get('age', 35))
        num_trans       = float(d.get('num_transactions', 5))
        total_spent     = float(d.get('total_spent', 500))
        avg_transaction = float(d.get('avg_transaction', 100))
        recency         = float(d.get('recency', 30))
        lifetime        = float(d.get('lifetime', 365))
        gender          = d.get('gender', 'M')
        city            = d.get('city', 'New York')
        customer_id     = d.get('customer_id', 'UNKNOWN')

        # Encode categoricals safely
        le_g = ML['le_gender']
        le_c = ML['le_city']
        g_enc = le_g.transform([gender])[0] if gender in le_g.classes_ else 0
        c_enc = le_c.transform([city])[0]   if city   in le_c.classes_ else 0

        features = pd.DataFrame([[age, num_trans, total_spent, avg_transaction,
                                   recency, lifetime, g_enc, c_enc]],
                                  columns=ML["feature_cols"])
        prob = float(ML['clf'].predict_proba(features)[0][1]) * 100

        # Key factors
        factors = []
        if recency > 60:   factors.append(f"High recency ({int(recency)} days) – major churn indicator")
        elif recency > 30: factors.append(f"Moderate recency ({int(recency)} days) – watch closely")
        if num_trans < 3:  factors.append(f"Very low transaction frequency ({int(num_trans)} transactions)")
        elif num_trans < 5: factors.append(f"Low transaction frequency ({int(num_trans)} transactions)")
        if avg_transaction < 50:  factors.append(f"Below-average order value (${avg_transaction:.0f})")
        elif avg_transaction < 100: factors.append(f"Moderate order value (${avg_transaction:.0f})")
        if not factors:
            factors = ["Healthy engagement patterns", "Regular purchase frequency",
                       "Good average transaction value"]

        if prob > 70:
            risk, status_class = "High Risk", "danger"
        elif prob > 40:
            risk, status_class = "Moderate Risk", "warning"
        else:
            risk, status_class = "Low Risk", "success"

        # Save prediction history
        email = session['user_email']
        PREDICTIONS.setdefault(email, []).insert(0, {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'type': 'churn',
            'customer_id': customer_id,
            'result': round(prob, 1),
            'risk': risk,
            'confidence': ML['churn_accuracy'],
        })

        return jsonify({
            'success': True,
            'probability': round(prob, 1),
            'risk': risk,
            'status_class': status_class,
            'factors': factors,
            'confidence': ML['churn_accuracy'],
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/api/predict/sales', methods=['POST'])
@login_required
def api_predict_sales():
    d = request.get_json()
    try:
        age             = float(d.get('age', 35))
        num_trans       = float(d.get('num_transactions', 5))
        recency         = float(d.get('recency', 30))
        avg_transaction = float(d.get('avg_transaction', 100))
        lifetime        = float(d.get('lifetime', 365))
        gender          = d.get('gender', 'M')
        customer_id     = d.get('customer_id', 'UNKNOWN')

        le_g = ML['le_gender']
        g_enc = le_g.transform([gender])[0] if gender in le_g.classes_ else 0

        reg_cols = ['age','num_transaction','recency_days','avg_transaction','customer_lifetime_days','gender_encoded']
        features = pd.DataFrame([[age, num_trans, recency, avg_transaction, lifetime, g_enc]], columns=reg_cols)
        pred = float(ML['reg'].predict(features)[0])
        pred = max(pred, 0)
        margin = pred * 0.15

        email = session['user_email']
        PREDICTIONS.setdefault(email, []).insert(0, {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'type': 'sales',
            'customer_id': customer_id,
            'result': round(pred, 2),
            'confidence': round(ML['sales_r2'] * 100, 1),
        })

        return jsonify({
            'success': True,
            'predicted_sales': round(pred, 2),
            'range': round(margin, 2),
            'confidence': round(ML['sales_r2'] * 100, 1),
            'r2_score': ML['sales_r2'],
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/api/predictions/history')
@login_required
def api_prediction_history():
    email = session['user_email']
    return jsonify(PREDICTIONS.get(email, [])[:20])


# ──────────────────────────────────────────────────────────────────────────────
# RESULTS / MODEL METRICS API
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/results/metrics')
@login_required
def api_metrics():
    return jsonify({
        'churn': {
            'accuracy':  ML['churn_accuracy'],
            'precision': ML['churn_precision'],
            'recall':    ML['churn_recall'],
            'f1':        ML['churn_f1'],
            'confusion_matrix': ML['confusion_matrix'],
            'feature_importance': ML['feat_importance'][:6],
        },
        'sales': {
            'r2':   ML['sales_r2'],
            'rmse': ML['sales_rmse'],
            'mae':  ML['sales_mae'],
        }
    })


# ──────────────────────────────────────────────────────────────────────────────
# EXPORT
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/export/csv')
@login_required
def api_export_csv():
    lines = [
        'Metric,Value',
        f'Total Customers,{ML["total_customers"]}',
        f'Total Revenue,{ML["total_revenue"]}',
        f'Avg Transaction,{ML["avg_transaction"]}',
        f'Churn Rate,{ML["churn_rate"]}%',
        f'Churn Accuracy,{ML["churn_accuracy"]}%',
        f'Sales R2 Score,{ML["sales_r2"]}',
        f'Sales RMSE,{ML["sales_rmse"]}',
    ]
    from io import BytesIO
    buf = BytesIO('\n'.join(lines).encode())
    buf.seek(0)
    return send_file(buf, mimetype='text/csv',
                     as_attachment=True,
                     download_name=f'ecommerce-analytics-{datetime.now().strftime("%Y%m%d")}.csv')


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_models()
    print("🚀  Starting E-Commerce Analytics Platform …")
    print("    Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
