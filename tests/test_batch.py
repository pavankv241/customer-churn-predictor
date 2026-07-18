"""Tests for CSV upload batch scoring."""

from __future__ import annotations

import joblib
import pandas as pd

from src.batch import score_customers, validate_upload, verification_summary
from src.config import DATA_PATH, META_PATH, PIPELINE_PATH


def test_validate_upload_detects_missing_cols():
    df = pd.DataFrame({"gender": ["Female"], "tenure": [1]})
    missing = validate_upload(df)
    assert "Contract" in missing
    assert "MonthlyCharges" in missing


def test_score_and_verify_sample():
    pipeline = joblib.load(PIPELINE_PATH)
    meta = joblib.load(META_PATH)
    raw = pd.read_csv(DATA_PATH).head(30)
    assert validate_upload(raw) == []
    scored = score_customers(pipeline, raw, float(meta["threshold"]))
    assert "churn_probability" in scored.columns
    assert "predicted_churn" in scored.columns
    summary = verification_summary(scored)
    assert summary is not None
    assert summary["rows"] == len(scored)
    assert 0.0 <= summary["accuracy"] <= 1.0
