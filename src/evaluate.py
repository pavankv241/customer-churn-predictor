"""Metrics, threshold tuning, and report helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "threshold": float(threshold),
    }


def expected_cost(y_true, y_proba, threshold: float) -> float:
    """Total cost under a simple FN/FP business matrix."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE)


def tune_threshold(
    y_true,
    y_proba,
    fn_cost: float = COST_FALSE_NEGATIVE,
    fp_cost: float = COST_FALSE_POSITIVE,
) -> tuple[float, float]:
    """
    Pick the probability threshold that minimizes expected business cost.
    FN (missed churn) is weighted heavier than FP (unnecessary outreach).
    """
    thresholds = np.linspace(0.05, 0.95, 37)
    best_t = 0.5
    best_cost = float("inf")
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fn * fn_cost + fp * fp_cost
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
    return best_t, float(best_cost)


def roc_payload(y_true, y_proba) -> dict[str, Any]:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(roc_auc_score(y_true, y_proba)),
    }


def confusion_payload(y_true, y_proba, threshold: float) -> dict[str, Any]:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "labels": ["Stay", "Churn"],
    }


def comparison_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
