"""Tests for data loading and cleaning."""

from __future__ import annotations

import pandas as pd

from src.config import FEATURE_COLS
from src.data import clean_data, load_modeling_frame, load_raw
from src.features import engineer_features


def test_load_raw_has_expected_columns():
    df = load_raw()
    assert "customerID" in df.columns
    assert "Churn" in df.columns
    assert len(df) > 1000


def test_clean_data_coerces_total_charges_and_churn():
    raw = load_raw()
    cleaned = clean_data(raw)
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    assert len(cleaned) <= len(raw)


def test_engineer_features_adds_columns():
    df = clean_data(load_raw()).head(20)
    out = engineer_features(df)
    for col in ("avg_monthly_spend", "tenure_bucket", "service_count"):
        assert col in out.columns
    assert out["service_count"].between(0, 6).all()


def test_modeling_frame_has_feature_cols():
    df = load_modeling_frame()
    missing = set(FEATURE_COLS) - set(df.columns)
    assert not missing, f"Missing features: {missing}"
    assert "Churn" in df.columns
