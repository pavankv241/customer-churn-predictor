"""Lightweight, explainable feature engineering."""

from __future__ import annotations

import pandas as pd

from src.config import SERVICE_FLAG_COLS


def _tenure_bucket(tenure: int) -> str:
    if tenure <= 12:
        return "0-12"
    if tenure <= 24:
        return "13-24"
    if tenure <= 48:
        return "25-48"
    return "49+"


def _is_active_service(value: object) -> int:
    return 1 if value == "Yes" else 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interview-friendly features:
    - avg_monthly_spend: TotalCharges / max(tenure, 1)
    - tenure_bucket: categorical tenure bands
    - service_count: number of optional add-on services
    """
    out = df.copy()
    tenure = out["tenure"].clip(lower=0)
    out["avg_monthly_spend"] = out["TotalCharges"] / tenure.clip(lower=1)
    out["tenure_bucket"] = tenure.apply(_tenure_bucket)
    out["service_count"] = out[SERVICE_FLAG_COLS].apply(
        lambda row: sum(_is_active_service(v) for v in row), axis=1
    )
    return out
