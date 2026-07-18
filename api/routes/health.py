"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthOut
from api.services.predictor import predictor

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        status="ok" if predictor.loaded else "degraded",
        model_loaded=predictor.loaded,
        best_model=predictor.meta.get("best_model") if predictor.loaded else None,
        threshold=predictor.threshold if predictor.loaded else None,
    )
