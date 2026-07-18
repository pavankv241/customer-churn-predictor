"""Candidate model pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BASE_CATEGORICAL, BASE_NUMERIC, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                BASE_CATEGORICAL,
            ),
            (
                "num",
                StandardScaler(),
                BASE_NUMERIC,
            ),
        ]
    )


def get_candidate_models() -> dict[str, object]:
    """Three classic baselines interviewers expect you to compare."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(model: object) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
