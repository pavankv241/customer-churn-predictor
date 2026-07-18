"""Prediction history endpoint."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query

from api.db import list_predictions
from api.schemas import HistoryOut, PredictionRecord

router = APIRouter(tags=["history"])


@router.get("/predictions", response_model=HistoryOut)
def get_predictions(limit: int = Query(50, ge=1, le=500)) -> HistoryOut:
    rows = list_predictions(limit=limit)
    items = [
        PredictionRecord(
            id=int(r["id"]),
            created_at=datetime.fromisoformat(r["created_at"]),
            source=r["source"],
            customer_id=r["customer_id"],
            churn_probability=float(r["churn_probability"]),
            predicted_churn=r["predicted_churn"],
            risk_band=r["risk_band"],
            threshold=float(r["threshold"]),
        )
        for r in rows
    ]
    return HistoryOut(count=len(items), items=items)
