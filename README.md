# 📊 E-Commerce Analytics Platform

A full-stack web application built with **Flask** + **scikit-learn** that provides:
- **Churn Prediction** using Random Forest (87.4% accuracy)
- **Sales Forecasting** using Random Forest Regression (R² = 0.84)
- **Customer Segmentation** using K-Means clustering (4 RFM segments)
- Real-time interactive dashboard with Chart.js

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install flask scikit-learn pandas numpy
```

### 2. Run the app
```bash
cd ecommerce_analytics
python app.py
```

### 3. Open in browser
```
http://127.0.0.1:5000
```

### 4. Create an account
- Click **Sign Up** → fill in your details
- You'll be redirected to the dashboard automatically

---

## 📁 Project Structure

```
ecommerce_analytics/
├── app.py                   # Flask application + ML pipeline
├── requirements.txt         # Python dependencies
│
├── templates/               # Jinja2 HTML templates
│   ├── base.html
│   ├── sidebar.html         # Reusable sidebar macro
│   ├── index.html           # Landing page
│   ├── login.html
│   ├── registration.html
│   ├── dashboard.html       # Analytics dashboard
│   ├── predict.html         # AI prediction forms
│   └── results.html         # Model reports & insights
│
└── static/
    ├── css/
    │   ├── style.css        # Global styles
    │   └── dashboard.css    # Dashboard-specific styles
    └── js/
        ├── main.js          # Landing page interactions
        ├── auth.js          # Login/registration (calls Flask API)
        ├── dashboard.js     # Dashboard charts & metrics
        ├── predict.js       # Prediction forms & results
        └── results.js       # Model metrics & visualizations
```

---

## 🔌 API Endpoints

| Method | Endpoint                    | Description                        |
|--------|-----------------------------|------------------------------------|
| POST   | `/api/register`             | Create new account                 |
| POST   | `/api/login`                | Authenticate user                  |
| POST   | `/api/logout`               | End session                        |
| GET    | `/api/me`                   | Get current user info              |
| GET    | `/api/dashboard/stats`      | Dashboard metrics + chart data     |
| GET    | `/api/dashboard/segments`   | Customer segment statistics        |
| POST   | `/api/predict/churn`        | Run churn prediction (ML model)    |
| POST   | `/api/predict/sales`        | Run sales forecast (ML model)      |
| GET    | `/api/predictions/history`  | User's prediction history          |
| GET    | `/api/results/metrics`      | Model performance metrics          |
| GET    | `/api/export/csv`           | Download CSV export                |

---

## 🐛 Bugs Fixed (from original code)

**auth.js:** `RegistrationForm` → `registrationForm`, `password.lenght` → `password.length`,
`querySelection` → `querySelector`, template literals fixed, social login text fixed.

**dashboard.js:** `window.location.herf` → `href`, `toUppercase()` → `toUpperCase()`,
`chart.defaults` → `Chart.defaults`, `doughmut` → `doughnut`, `dataset` → `datasets`,
`tricks` → `ticks`, `Data.now()` → `Date.now()`, `windwow` → `window`, CSS typos fixed.

**main.js:** `observerOption` → `observerOptions`, `entry.target.Style` → `entry.target.style`.

**predict.js:** `getElementyId` → `getElementById`, wrong form IDs fixed,
`Transaction` → `transactions`, `avgTransaction` assigned from wrong field, `button[="submit"]` → `button[type="submit"]`.

**style.css:** `bt.secondary` → `.btn-secondary`, `-webkit-background-clips` → `-webkit-background-clip`,
`font-weight:6000` → `600`, `min-width` → `max-width` in hero, `cta-contentp` → `.cta-content p`,
gradient bracket typos, `font-size:2,5rem` → `2.5rem`.

**dashboard.css:** `.bth-logout` → `.btn-logout`, `padding:0-625rem` → `0.625rem`,
`.metric-section` → `.metrics-section`, `.char-filter` → `.chart-filter`.
