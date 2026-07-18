"""Request/response schemas for the churn API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CustomerIn(BaseModel):
    """Telco-style customer features used for scoring."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = 0
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No", "Yes", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float = Field(ge=0, le=100000)
    customerID: Optional[str] = None
    Churn: Optional[Literal["Yes", "No"]] = None


class PredictOut(BaseModel):
    churn_probability: float
    predicted_churn: Literal["Yes", "No"]
    risk_band: str
    threshold: float
    decision: str
    customerID: Optional[str] = None


class BatchCustomerIn(BaseModel):
    customers: list[CustomerIn] = Field(min_length=1, max_length=5000)


class VerificationSummary(BaseModel):
    rows: int
    correct: int
    accuracy: float
    predicted_churn_rate: float
    actual_churn_rate: float


class BatchOut(BaseModel):
    count: int
    predicted_churn_count: int
    predicted_churn_rate: float
    threshold: float
    verification: Optional[VerificationSummary] = None
    results: list[dict[str, Any]]


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    best_model: Optional[str] = None
    threshold: Optional[float] = None


class PredictionRecord(BaseModel):
    id: int
    created_at: datetime
    source: str
    customer_id: Optional[str]
    churn_probability: float
    predicted_churn: str
    risk_band: str
    threshold: float


class HistoryOut(BaseModel):
    count: int
    items: list[PredictionRecord]
