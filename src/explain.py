"""Model explanation helpers for the Streamlit app."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline


def clean_feature_name(name: str) -> str:
    return name.replace("cat__", "").replace("num__", "")


def top_feature_importances(pipeline: Pipeline, top_k: int = 15) -> list[tuple[str, float]]:
    """
    Prefer tree feature_importances_; fall back to absolute logistic coefficients.
    """
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = [clean_feature_name(n) for n in preprocess.get_feature_names_out()]

    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        scores = np.abs(np.asarray(model.coef_).ravel())
    else:
        return []

    pairs = sorted(zip(names, scores.tolist()), key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def risk_band(proba: float, threshold: float) -> tuple[str, str]:
    """Map probability to a qualitative band relative to the tuned threshold."""
    if proba >= max(threshold + 0.15, 0.65):
        return "High risk", "red"
    if proba >= threshold:
        return "Medium risk", "orange"
    return "Low risk", "green"


def insight_bullets() -> list[str]:
    """Reusable interview talking points surfaced in the app."""
    return [
        "Month-to-month contracts churn far more than one- or two-year contracts — switching cost is low.",
        "New customers (tenure under 12 months) are the riskiest cohort; onboarding quality matters.",
        "Fiber optic + high monthly charges often coincide with churn — value perception issues.",
        "Electronic check payers churn more than automatic payment methods — payment friction / intent signal.",
        "Optional security/support add-ons correlate with lower churn — stickiness from product depth.",
        "We optimize a business cost threshold (missed churn costlier than wasted outreach), not default 0.5.",
    ]
