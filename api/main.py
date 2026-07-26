"""
VELOX API — FastAPI service exposing vehicle valuation as a REST endpoint.

This is a separate deployable service from the Streamlit app: the app uses
the `velox` package directly (fast, single-process, ideal for Streamlit
Cloud), while this API exposes the same underlying models over HTTP for any
other consumer — a mobile app, another backend, a batch job, etc. Both share
the same `velox` core, so there is exactly one place the modeling logic lives.

Run locally:
    uvicorn api.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from velox import config
from velox.data import get_or_build_clean_dataset
from velox.explain import explain_prediction
from velox.logging_config import setup_logging
from velox.model import load_models
from velox.schemas import FeatureContribution, HealthResponse, PredictionResponse, VehicleProfile

setup_logging()
logger = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading VELOX models...")
    point_model, quantile_models, metrics = load_models()
    df = get_or_build_clean_dataset()
    _state["model"] = point_model
    _state["quantile_models"] = quantile_models
    _state["metrics"] = metrics
    _state["df"] = df
    logger.info("Models loaded — %d training samples.", metrics.get("n_samples", 0))
    yield
    _state.clear()


app = FastAPI(
    title="VELOX Vehicle Valuation API",
    description="Predicts used-car resale value from vehicle attributes, "
                 "with model-based P10/P50/P90 uncertainty bands and "
                 "per-prediction SHAP explanations.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_feature_row(profile: VehicleProfile) -> pd.DataFrame:
    brand = profile.brand if profile.brand in _state["metrics"]["brands"] else "Other"
    car_age = max(config.CURRENT_YEAR - profile.year, 0)
    return pd.DataFrame([{
        "brand": brand,
        "car_age": car_age,
        "km_driven": profile.km_driven,
        "fuel": profile.fuel,
        "seller_type": profile.seller_type,
        "transmission": profile.transmission,
        "owner": profile.owner,
    }])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded="model" in _state,
        n_training_samples=_state.get("metrics", {}).get("n_samples", 0),
    )


@app.get("/brands", response_model=list[str])
def brands() -> list[str]:
    return _state["metrics"]["brands"]


@app.post("/predict", response_model=PredictionResponse)
def predict(profile: VehicleProfile) -> PredictionResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    row = _to_feature_row(profile)
    point_pred = max(float(_state["model"].predict(row)[0]), 0)
    p10 = max(float(_state["quantile_models"]["p10"].predict(row)[0]), 0)
    p50 = max(float(_state["quantile_models"]["p50"].predict(row)[0]), 0)
    p90 = max(float(_state["quantile_models"]["p90"].predict(row)[0]), 0)

    df = _state["df"]
    brand_df = df[df["brand"] == row["brand"].iloc[0]]
    brand_median = float(brand_df["price_inr"].median()) if len(brand_df) >= 5 else float(df["price_inr"].median())

    contributions = explain_prediction(_state["model"], row)
    top_contributions = [
        FeatureContribution(label=r.label, shap_value=float(r.shap_value))
        for r in contributions.itertuples()
    ]

    return PredictionResponse(
        predicted_price_inr=point_pred,
        price_low=min(p10, point_pred),
        price_high=max(p90, point_pred),
        p10=p10, p50=p50, p90=p90,
        brand_median_inr=brand_median,
        top_contributions=top_contributions,
    )
