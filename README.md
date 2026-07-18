# Customer Churn Scoring Service

Software engineering project: a **FastAPI** prediction service with validation, SQLite audit history, Docker Compose, CI, and a Streamlit client UI.

**Live UI:** https://customer-churn-predictor-lhlfqhgmrqvhawhpjxkx6d.streamlit.app/  
**Source:** https://github.com/pavankv241/customer-churn-predictor

## Free hosting

| Piece | Platform | Cost |
|-------|----------|------|
| UI | [Streamlit Community Cloud](https://share.streamlit.io) | Free |
| API | [Render](https://render.com) free web service | Free (sleeps after idle) |

### 1) Deploy the API on Render

1. Sign up at https://render.com with GitHub  
2. **New** → **Blueprint** (uses [`render.yaml`](render.yaml))  
   — or **Web Service** → repo `pavankv241/customer-churn-predictor`  
3. Settings if creating manually:
   - **Runtime:** Python  
   - **Build:** `pip install -r requirements.txt`  
   - **Start:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`  
   - **Instance:** Free  
4. Deploy → copy the URL, e.g. `https://customer-churn-api.onrender.com`  
5. Check: open `/health` and `/docs` on that URL  

First request after idle can take ~30–60s (free tier cold start).

### 2) Point Streamlit Cloud at the API

1. Open your app on https://share.streamlit.io  
2. **⋮** → **Settings** → **Secrets**  
3. Paste:

```toml
API_URL = "https://YOUR-SERVICE.onrender.com"
```

4. Save → reboot the app  

Predict / Upload / History will call Render. Overview charts still use bundled data/reports.

Local secrets example: [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)

## Architecture

```
User → Streamlit (web/) → HTTP JSON → FastAPI (api/)
                              ↓
                     Pydantic validation
                              ↓
                     PredictionService (src/ ML pipeline)
                              ↓
                     SQLite audit log (data/predictions.db)
```

| Layer | Responsibility |
|-------|----------------|
| `api/` | HTTP API, schemas, middleware, persistence |
| `src/` | Shared ML clean/features/score logic |
| `web/` | UI client (no direct model load for scoring) |
| `models/` | Trained pipeline artifact |
| `tests/` | Unit + API (`TestClient`) tests |

## API

After starting the API, open interactive docs: http://127.0.0.1:8000/docs

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + model status |
| POST | `/predict` | Score one customer (JSON) |
| POST | `/predict/batch` | Score a JSON list |
| POST | `/predict/csv` | Upload CSV batch |
| GET | `/predictions` | Recent audit history |

### Example

```bash
curl -s http://127.0.0.1:8000/health | jq

curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
    "tenure":1,"PhoneService":"No","MultipleLines":"No phone service",
    "InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes",
    "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
    "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85
  }' | jq
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — UI (talks to API)
export API_URL=http://127.0.0.1:8000
streamlit run app.py

# Tests
pytest -q
```

Retrain model (optional): `python train_model.py`

## Docker Compose

```bash
docker compose up --build
```

- API: http://localhost:8000/docs  
- UI: http://localhost:8501  

## CI

GitHub Actions (`.github/workflows/ci.yml`) installs deps and runs `pytest` on every push/PR to `main`.

## Design decisions

- **API owns scoring** — UI is a client; easier to swap frontends or call from other services  
- **Pydantic validation** — bad payloads fail with `422` before touching the model  
- **SQLite history** — lightweight audit trail without hosting Postgres  
- **Cost-tuned threshold** baked into saved model meta — API applies it consistently  
- **SMOTE deferred** — stratified split + class weights first (simpler, less leak risk)

## Resume line

Designed a FastAPI churn-scoring service with input validation, SQLite audit history, Docker Compose, and a Streamlit client; covered by API tests and GitHub Actions CI.
