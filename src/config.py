"""Project paths and modeling constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

PIPELINE_PATH = MODEL_DIR / "churn_pipeline.joblib"
META_PATH = MODEL_DIR / "model_meta.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
ROC_PATH = REPORT_DIR / "roc_curve.json"
CONFUSION_PATH = REPORT_DIR / "confusion_matrix.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Cost matrix for threshold tuning (business-aware).
# False negatives (missed churners) cost more than false positives (wasted outreach).
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0

BASE_CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
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
    "tenure_bucket",
]

BASE_NUMERIC = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "service_count",
]

FEATURE_COLS = BASE_CATEGORICAL + BASE_NUMERIC

SERVICE_FLAG_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
