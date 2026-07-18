# Customer Churn Predictor

Predict which telecom customers are likely to leave, using the IBM Telco Customer Churn dataset and a scikit-learn Random Forest model.

**Free hosting (recommended):**

| Platform | Role |
|----------|------|
| [Streamlit Community Cloud](https://share.streamlit.io) | Primary — deploy from GitHub in a few clicks |
| [Hugging Face Spaces](https://huggingface.co/spaces) (Streamlit) | Backup — same `app.py` + `requirements.txt` |
| GitHub | Source repo required by both hosts |

## Quick start (local)

```bash
cd "Customer Churn"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py       # only needed if you retrain
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Project layout

```
app.py                      # Streamlit UI + predictions
train_model.py              # Train and save the model
requirements.txt
data/Telco-Customer-Churn.csv
models/churn_pipeline.joblib
models/model_meta.joblib
```

## Deploy on Streamlit Community Cloud (primary)

1. Create a **public** GitHub repo and push this project (include `models/` so the host does not retrain).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set **Main file path** to `app.py`.
4. Deploy. Share the public URL when it finishes.

Tips:

- Keep `requirements.txt` lean (already done — no TensorFlow).
- If the app sleeps after idle time, the first visit may take a few seconds to wake up.

## Deploy on Hugging Face Spaces (backup)

1. Create a new Space → SDK: **Streamlit**.
2. Upload the project files (or connect the GitHub repo), including `app.py`, `requirements.txt`, `data/`, and `models/`.
3. HF installs deps from `requirements.txt` and runs `app.py` automatically.

Optional Space metadata (add as `README.md` front matter on HF only, or keep a separate Space README):

```yaml
---
title: Customer Churn Predictor
sdk: streamlit
app_file: app.py
---
```

## Retraining

```bash
python train_model.py
```

This overwrites `models/churn_pipeline.joblib` and `models/model_meta.joblib`. Commit the new files before redeploying.

## Model notes

- Features: demographics, services, contract, charges
- Preprocessing: one-hot encoding for categoricals
- Metrics printed at train time (accuracy, F1, ROC AUC)
