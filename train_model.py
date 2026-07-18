"""CLI: compare models, tune threshold, save best pipeline + reports."""

from __future__ import annotations

import json

import joblib
from sklearn.model_selection import train_test_split

from src.config import (
    COMPARISON_PATH,
    CONFUSION_PATH,
    FEATURE_COLS,
    META_PATH,
    METRICS_PATH,
    MODEL_DIR,
    PIPELINE_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    ROC_PATH,
    TEST_SIZE,
)
from src.data import load_modeling_frame
from src.evaluate import (
    classification_metrics,
    comparison_frame,
    confusion_payload,
    roc_payload,
    tune_threshold,
)
from src.explain import top_feature_importances
from src.models import build_pipeline, get_candidate_models


def main() -> None:
    df = load_modeling_frame()
    X = df[FEATURE_COLS]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rows = []
    fitted: dict[str, object] = {}
    proba_map: dict[str, object] = {}

    print("Training candidate models...\n")
    for name, estimator in get_candidate_models().items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, y_proba, threshold=0.5)
        metrics["model"] = name
        rows.append(metrics)
        fitted[name] = pipe
        proba_map[name] = y_proba
        print(
            f"{name:22s}  AUC={metrics['roc_auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  Recall={metrics['recall']:.4f}"
        )

    comparison = comparison_frame(rows)
    best_name = comparison.iloc[0]["model"]
    best_pipe = fitted[best_name]
    best_proba = proba_map[best_name]

    threshold, total_cost = tune_threshold(y_test, best_proba)
    tuned = classification_metrics(y_test, best_proba, threshold=threshold)

    print(f"\nBest model by ROC AUC: {best_name}")
    print(f"Tuned threshold (cost-minimizing): {threshold:.3f}")
    print(f"Expected cost at threshold: {total_cost:.1f}")
    print(
        f"At tuned threshold -> "
        f"Precision={tuned['precision']:.4f}  Recall={tuned['recall']:.4f}  "
        f"F1={tuned['f1']:.4f}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipe, PIPELINE_PATH)
    comparison.to_csv(COMPARISON_PATH, index=False)

    importances = top_feature_importances(best_pipe, top_k=15)
    roc = roc_payload(y_test, best_proba)
    cm = confusion_payload(y_test, best_proba, threshold)

    meta = {
        "best_model": best_name,
        "feature_cols": FEATURE_COLS,
        "threshold": threshold,
        "total_cost": total_cost,
        "metrics_at_0_5": {
            k: float(comparison.iloc[0][k])
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        },
        "metrics_at_tuned": tuned,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": float(y.mean()),
        "feature_importances": importances,
    }
    joblib.dump(meta, META_PATH)

    metrics_json = {
        **meta,
        "feature_importances": [
            {"feature": f, "importance": float(v)} for f, v in importances
        ],
        "cost_matrix": {"false_negative": 5.0, "false_positive": 1.0},
        "selection_rule": "Highest ROC AUC on held-out test set; threshold tuned for business cost.",
        "imbalance_strategy": (
            "Stratified split + class_weight='balanced' where supported. "
            "SMOTE deferred — prefer simpler, more explainable rebalancing first."
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics_json, indent=2))
    ROC_PATH.write_text(json.dumps(roc, indent=2))
    CONFUSION_PATH.write_text(json.dumps(cm, indent=2))

    print(f"\nSaved pipeline  -> {PIPELINE_PATH}")
    print(f"Saved meta      -> {META_PATH}")
    print(f"Saved reports   -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
