"""Load and clean the Telco churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_PATH
from src.features import engineer_features


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, drop bad TotalCharges rows, encode Churn as 0/1."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out = out.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    if out["Churn"].dtype == object:
        out["Churn"] = (out["Churn"] == "Yes").astype(int)
    return out


def load_modeling_frame(path: Path = DATA_PATH) -> pd.DataFrame:
    """Raw -> clean -> engineered features ready for modeling."""
    df = clean_data(load_raw(path))
    return engineer_features(df)
