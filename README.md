# Customer Churn Predictor

Predict which telecom customers are likely to leave using service and billing data. Compare models, tune a decision threshold, and score customers in a Streamlit app (form or CSV upload).

**Live demo:** https://customer-churn-predictor-lhlfqhgmrqvhawhpjxkx6d.streamlit.app/

**Source code:** https://github.com/pavankv241/customer-churn-predictor

## Why this project (interview framing)

| Weak demo | This repo |
|-----------|-----------|
| One script, one model | Modular `src/` package |
| Accuracy only | ROC AUC + precision/recall/F1 |
| Default 0.5 cutoff | Cost-based threshold (missed churn > wasted outreach) |
| No EDA story | Notebook + Insights tab |

## Problem

IBM Telco Customer Churn (~7k customers). Positive class ≈ 26%. Goal: rank likely churners and choose a decision threshold aligned with retention economics.

## Approach

1. **Clean** — coerce `TotalCharges`, drop invalid rows, encode target  
2. **Engineer** — `tenure_bucket`, `avg_monthly_spend`, `service_count`  
3. **Compare** — Logistic Regression, Random Forest, Gradient Boosting  
4. **Select** — best ROC AUC on stratified holdout  
5. **Tune threshold** — minimize `5 * FN + 1 * FP`  
6. **Serve** — Streamlit app (Overview / Predict / Upload CSV / Model Lab / Insights)

### Upload CSV demo

1. Open the **Upload CSV** tab  
2. Download the sample file (or use any Telco-format CSV)  
3. Click **Score uploaded customers**  
4. If `Churn` is present, the app **verifies** predictions against those labels  

The model was trained on `data/Telco-Customer-Churn.csv` and stored in `models/`. Uploads are scored only — not used for retraining.

Imbalance strategy: stratified split + `class_weight="balanced"` where supported. SMOTE is deferred on purpose (harder to explain, easy to leak if misused).

## Project layout

```
app.py                      # Streamlit UI
train_model.py              # Train, compare, tune, save artifacts
src/
  config.py  data.py  features.py  models.py  evaluate.py  explain.py
notebooks/01_eda_and_insights.ipynb
tests/
models/                     # best pipeline + meta
reports/                    # comparison, ROC, confusion, metrics.json
data/Telco-Customer-Churn.csv
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
pytest -q
```

## Latest training snapshot

Re-run `python train_model.py` to refresh. Typical holdout results:

- Models compared on ROC AUC
- Best model and tuned threshold written to `reports/metrics.json`
- Confusion matrix uses the **tuned** threshold (not 0.5)

## Free hosting

1. Push to GitHub (already public)  
2. [Streamlit Community Cloud](https://share.streamlit.io) → New app → Main file `app.py`  
3. Backup: [Hugging Face Spaces](https://huggingface.co/spaces) (Streamlit SDK)

## Questions you should be ready to answer

- Why ROC AUC over accuracy?  
- Why not start with SMOTE?  
- What do `tenure_bucket` / `service_count` capture?  
- Why is FN cost higher than FP in the threshold objective?  
- What would you add next (calibration, time-based validation, uplift modeling)?

## Limitations

- Single observational snapshot — not causal  
- No customer lifetime value in the cost matrix (proxy weights only)  
- No production monitoring / drift detection yet
