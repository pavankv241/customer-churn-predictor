# Technical Overview — Customer Churn Scoring Service

## 1. Problem statement

Telecom providers lose revenue when customers cancel or stop renewing (churn). This system scores individual customers for churn risk so retention teams can prioritize outreach.

**Dataset:** IBM Telco Customer Churn (~7,000 customers). Positive class rate ≈ 26.6%.

**Output:** churn probability, binary decision at a business-tuned threshold, and optional batch verification when ground-truth labels are present.

## 2. System architecture

```
Client (Streamlit)
        │  HTTPS JSON / multipart CSV
        ▼
FastAPI service
        │
        ├─ Pydantic request validation
        ├─ PredictionService (shared ML pipeline)
        └─ SQLite audit log (prediction history)
```

| Layer | Path | Responsibility |
|-------|------|----------------|
| HTTP API | `api/` | Routes, schemas, middleware, persistence |
| ML core | `src/` | Cleaning, feature engineering, scoring |
| UI client | `web/` | Forms, CSV upload, charts; calls API only for scoring |
| Artifacts | `models/`, `reports/` | Trained pipeline and evaluation outputs |
| Data | `data/` | Training CSV, sample upload, SQLite DB file |

Scoring never retrains the model at request time. The pipeline is loaded once at API startup from `models/churn_pipeline.joblib`.

## 3. Machine learning methodology

### 3.1 Cleaning and features

- Coerce `TotalCharges` to numeric; drop invalid rows  
- Encode target `Churn` as 0/1  
- Engineered features:
  - `tenure_bucket` — 0–12, 13–24, 25–48, 49+  
  - `avg_monthly_spend` — `TotalCharges / max(tenure, 1)`  
  - `service_count` — count of active add-on services  

### 3.2 Imbalance handling

Stratified train/test split plus `class_weight="balanced"` where the estimator supports it. Synthetic oversampling (e.g. SMOTE) is intentionally deferred to keep the pipeline simple and reduce leakage risk.

### 3.3 Model comparison

Candidates evaluated on a held-out stratified test set:

| Model | Role |
|-------|------|
| Logistic Regression | Linear, interpretable baseline |
| Random Forest | Non-linear bagging baseline |
| Gradient Boosting | Selected production model |

**Selection rule:** highest **ROC AUC** on the holdout set.

**Selected model:** Gradient Boosting (ROC AUC ≈ 0.839).

### 3.4 Decision threshold

Default 0.5 is not used for production decisions. The threshold minimizes a simple cost matrix:

- Cost(false negative) = 5 (missed churner)  
- Cost(false positive) = 1 (unnecessary outreach)  

**Tuned threshold ≈ 0.125**, which raises recall on the churn class (≈ 0.91 at that cutoff) at the expense of precision. That tradeoff is intentional for retention use cases where missing a churner is more expensive than contacting a stable customer.

## 4. API surface

Base URL (production): `https://customer-churn-predictor-sl34.onrender.com`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service metadata |
| GET | `/health` | Liveness and model-loaded status |
| GET | `/docs` | OpenAPI interactive documentation |
| POST | `/predict` | Score one customer (JSON body) |
| POST | `/predict/batch` | Score a JSON list of customers |
| POST | `/predict/csv` | Score an uploaded Telco-format CSV |
| GET | `/predictions` | Recent scores from the SQLite audit log |

### Example — single prediction

```bash
curl -s -X POST https://customer-churn-predictor-sl34.onrender.com/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
    "tenure":1,"PhoneService":"No","MultipleLines":"No phone service",
    "InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes",
    "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
    "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85
  }'
```

Invalid payloads return **422** (validation). If the model file is missing at runtime, scoring endpoints return **503**.

Each successful score is appended to SQLite (`data/predictions.db`) with timestamp, source, probability, and decision.

## 5. User interface

Live UI: https://customer-churn-predictor-lhlfqhgmrqvhawhpjxkx6d.streamlit.app/

| Tab | Behavior |
|-----|----------|
| Overview | Dataset metrics and training reports (local artifacts) |
| Predict | Form → `POST /predict` |
| Upload CSV | File → `POST /predict/csv`; optional verification if `Churn` column exists |
| History | `GET /predictions` |
| Model Lab | Holdout comparison table, ROC, confusion matrix |
| Insights | Domain findings from EDA |

The UI resolves `API_URL` from Streamlit secrets or environment variables.

## 6. Deployment

| Component | Host | Notes |
|-----------|------|--------|
| API | Render (free web service) | `requirements-api.txt`, Python 3.12, `uvicorn` |
| UI | Streamlit Community Cloud | Secret: `API_URL` pointing at Render |
| CI | GitHub Actions | `pytest` on push/PR to `main` |

Local stack:

```bash
uvicorn api.main:app --reload --port 8000
export API_URL=http://127.0.0.1:8000
streamlit run app.py
```

Or: `docker compose up --build`.

Free-tier APIs may cold-start after idle periods (first request can take tens of seconds).

## 7. Design decisions

1. **API owns scoring** — the UI is a client, so other consumers (scripts, mobile, partners) can call the same contract.  
2. **Pydantic at the edge** — bad input fails before the model runs.  
3. **SQLite audit log** — lightweight persistence without a managed database on free hosting.  
4. **Cost-aware threshold** — aligns the binary decision with retention economics.  
5. **Lean API requirements** — Render builds install `requirements-api.txt` (no Streamlit/matplotlib) for faster, more reliable deploys.  
6. **Shared `src/` package** — training, API, and tests reuse one cleaning/scoring path.

## 8. Limitations and next steps

- Observational snapshot; not a causal model of churn.  
- Cost weights are proxies, not customer LTV.  
- No authentication on the public API.  
- Free hosts sleep when idle.  

Reasonable extensions: API auth, request metrics/alerting, time-based validation splits, probability calibration, and uplift modeling for treatment effect.

## 9. Repository map

```
api/           FastAPI application
src/           ML utilities (data, features, models, evaluate, batch)
web/           Streamlit client
models/        Trained pipeline + meta
reports/       Comparison metrics, ROC, confusion matrix
tests/         Unit and API tests
notebooks/     EDA notebook
train_model.py Train/compare models and write artifacts
```

## 10. References

- Source: https://github.com/pavankv241/customer-churn-predictor  
- UI: https://customer-churn-predictor-lhlfqhgmrqvhawhpjxkx6d.streamlit.app/  
- API docs: https://customer-churn-predictor-sl34.onrender.com/docs  
