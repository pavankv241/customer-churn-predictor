"""Batch scoring for uploaded customer CSVs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS
from src.data import clean_data
from src.features import engineer_features

# Columns required in an upload (Telco-style). Churn is optional (for verification).
REQUIRED_INPUT_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def validate_upload(df: pd.DataFrame) -> list[str]:
    """Return list of missing required column names (empty if OK)."""
    return [c for c in REQUIRED_INPUT_COLS if c not in df.columns]


def score_customers(pipeline, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Clean → engineer → predict.
    Adds churn_probability, predicted_churn, risk_band.
    If a Churn column exists, also adds actual_churn and correct.
    """
    working = df.copy()
    had_churn = "Churn" in working.columns
    if not had_churn:
        working["Churn"] = "No"

    cleaned = clean_data(working)
    featured = engineer_features(cleaned)
    X = featured[FEATURE_COLS]
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = cleaned.copy()
    out["churn_probability"] = np.round(proba, 4)
    out["predicted_churn"] = np.where(pred == 1, "Yes", "No")
    out["risk_band"] = [
        "High" if p >= max(threshold + 0.15, 0.65) else ("Medium" if p >= threshold else "Low")
        for p in proba
    ]

    if had_churn:
        out["actual_churn"] = np.where(cleaned["Churn"] == 1, "Yes", "No")
        out["correct"] = out["predicted_churn"] == out["actual_churn"]

    return out


def verification_summary(scored: pd.DataFrame) -> dict[str, Any] | None:
    """If actual labels exist, summarize how well predictions match."""
    if "correct" not in scored.columns:
        return None
    n = len(scored)
    correct = int(scored["correct"].sum())
    return {
        "rows": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "predicted_churn_rate": float((scored["predicted_churn"] == "Yes").mean()),
        "actual_churn_rate": float((scored["actual_churn"] == "Yes").mean()),
    }
