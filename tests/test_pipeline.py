"""Tests for training artifacts and prediction pipeline."""

from __future__ import annotations

import json

import joblib
import pandas as pd

from src.config import (
    COMPARISON_PATH,
    FEATURE_COLS,
    META_PATH,
    METRICS_PATH,
    PIPELINE_PATH,
)
from src.data import load_modeling_frame
from src.evaluate import tune_threshold
from src.features import engineer_features


def test_pipeline_artifact_exists_and_predicts():
    assert PIPELINE_PATH.exists()
    pipeline = joblib.load(PIPELINE_PATH)
    df = load_modeling_frame().head(5)
    X = df[FEATURE_COLS]
    proba = pipeline.predict_proba(X)
    assert proba.shape == (5, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_single_row_engineering_and_predict():
    pipeline = joblib.load(PIPELINE_PATH)
    row = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.0,
                "TotalCharges": 900.0,
                "Churn": 0,
            }
        ]
    )
    row = engineer_features(row)
    proba = float(pipeline.predict_proba(row[FEATURE_COLS])[0, 1])
    assert 0.0 <= proba <= 1.0


def test_metrics_threshold_in_unit_interval():
    assert METRICS_PATH.exists()
    metrics = json.loads(METRICS_PATH.read_text())
    threshold = float(metrics["threshold"])
    assert 0.0 < threshold < 1.0
    assert "best_model" in metrics
    assert COMPARISON_PATH.exists()


def test_meta_matches_metrics():
    meta = joblib.load(META_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
    assert meta["best_model"] == metrics["best_model"]
    assert abs(meta["threshold"] - metrics["threshold"]) < 1e-9


def test_tune_threshold_returns_valid_value():
    y_true = [0, 0, 1, 1, 1, 0]
    y_proba = [0.1, 0.2, 0.6, 0.8, 0.9, 0.3]
    t, cost = tune_threshold(y_true, y_proba)
    assert 0.05 <= t <= 0.95
    assert cost >= 0
