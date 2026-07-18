"""Predict endpoints."""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import BatchCustomerIn, BatchOut, CustomerIn, PredictOut
from api.services.predictor import predictor
from src.batch import validate_upload

router = APIRouter(tags=["predict"])


def _ensure_model() -> None:
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")


@router.post("/predict", response_model=PredictOut)
def predict_one(customer: CustomerIn) -> PredictOut:
    _ensure_model()
    try:
        return predictor.predict_one(customer, source="api")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/predict/batch", response_model=BatchOut)
def predict_batch(body: BatchCustomerIn) -> BatchOut:
    _ensure_model()
    try:
        result = predictor.predict_batch(body.customers, source="api_batch")
        return BatchOut(**result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/predict/csv", response_model=BatchOut)
async def predict_csv(file: UploadFile = File(...)) -> BatchOut:
    _ensure_model()
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")
    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    missing = validate_upload(df)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {', '.join(missing)}",
        )
    try:
        result = predictor.predict_dataframe(df, source="api_csv")
        return BatchOut(**result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
