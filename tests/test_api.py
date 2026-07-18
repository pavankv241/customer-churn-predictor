"""API tests using FastAPI TestClient."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import DATA_PATH

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_upload.csv"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"
    assert body["threshold"] is not None


def test_predict_one(client: TestClient):
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "customerID": "TEST-001",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_churn"] in {"Yes", "No"}
    assert "decision" in body


def test_predict_validation_error(client: TestClient):
    r = client.post("/predict", json={"gender": "Female"})
    assert r.status_code == 422


def test_predict_csv(client: TestClient):
    assert SAMPLE.exists()
    with SAMPLE.open("rb") as f:
        r = client.post(
            "/predict/csv",
            files={"file": ("sample_upload.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 20
    assert body["verification"] is not None
    assert "results" in body


def test_history(client: TestClient):
    r = client.get("/predictions?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert body["count"] >= 1


def test_dataset_present():
    assert DATA_PATH.exists()
