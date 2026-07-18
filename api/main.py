"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.db import init_db
from api.routes import health, history, predict
from api.services.predictor import predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("churn.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    predictor.load()
    logger.info(
        "Model loaded best_model=%s threshold=%.3f",
        predictor.meta.get("best_model"),
        predictor.threshold,
    )
    yield


app = FastAPI(
    title="Customer Churn API",
    description="Score telecom customers for churn risk. ML model is a dependency; this service owns validation, scoring, and audit history.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s -> %s (%.1fms) request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


app.include_router(health.router)
app.include_router(predict.router)
app.include_router(history.router)
