"""Prediction service wrapping the shared ML package."""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd

from api import db
from api.schemas import CustomerIn, PredictOut, VerificationSummary
from src.batch import score_customers, verification_summary
from src.config import META_PATH, PIPELINE_PATH
from src.explain import risk_band as risk_band_label


class PredictorService:
    def __init__(self) -> None:
        self.pipeline = None
        self.meta: dict[str, Any] = {}
        self.threshold: float = 0.5

    def load(self) -> None:
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(f"Model not found: {PIPELINE_PATH}")
        self.pipeline = joblib.load(PIPELINE_PATH)
        self.meta = joblib.load(META_PATH) if META_PATH.exists() else {}
        self.threshold = float(self.meta.get("threshold", 0.5))

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def predict_one(self, customer: CustomerIn, *, source: str = "api") -> PredictOut:
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        payload = customer.model_dump(exclude_none=True)
        row = pd.DataFrame([payload])
        scored = score_customers(self.pipeline, row, self.threshold)
        r = scored.iloc[0]
        proba = float(r["churn_probability"])
        predicted = str(r["predicted_churn"])
        band = str(r["risk_band"])
        label, _ = risk_band_label(proba, self.threshold)
        decision = "Likely to churn" if predicted == "Yes" else "Likely to stay"

        db.insert_prediction(
            source=source,
            customer_id=customer.customerID,
            churn_probability=proba,
            predicted_churn=predicted,
            risk_band=band,
            threshold=self.threshold,
            payload_json=json.dumps(customer.model_dump()),
        )

        return PredictOut(
            churn_probability=proba,
            predicted_churn=predicted,  # type: ignore[arg-type]
            risk_band=label,
            threshold=self.threshold,
            decision=decision,
            customerID=customer.customerID,
        )

    def predict_batch(
        self, customers: list[CustomerIn], *, source: str = "api_batch"
    ) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        rows = [c.model_dump(exclude_none=True) for c in customers]
        df = pd.DataFrame(rows)
        return self.predict_dataframe(df, source=source)

    def predict_dataframe(self, df: pd.DataFrame, *, source: str = "api_csv") -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        scored = score_customers(self.pipeline, df, self.threshold)
        for _, r in scored.iterrows():
            cid = None
            if "customerID" in scored.columns and pd.notna(r.get("customerID")):
                cid = str(r["customerID"])
            db.insert_prediction(
                source=source,
                customer_id=cid,
                churn_probability=float(r["churn_probability"]),
                predicted_churn=str(r["predicted_churn"]),
                risk_band=str(r["risk_band"]),
                threshold=self.threshold,
            )
        summary = verification_summary(scored)
        verification = VerificationSummary(**summary) if summary else None
        pred_yes = int((scored["predicted_churn"] == "Yes").sum())
        results = json.loads(scored.to_json(orient="records"))
        return {
            "count": len(scored),
            "predicted_churn_count": pred_yes,
            "predicted_churn_rate": pred_yes / len(scored) if len(scored) else 0.0,
            "threshold": self.threshold,
            "verification": verification,
            "results": results,
        }


predictor = PredictorService()
