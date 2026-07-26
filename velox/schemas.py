"""
velox.schemas
Pydantic request/response models for the prediction API. Also used to
validate inputs before they ever reach the model, so bad input fails with a
clear 422 instead of a confusing downstream error.
"""

from typing import Literal

from pydantic import BaseModel, Field

from velox import config

FuelType = Literal["Petrol", "Diesel", "CNG", "LPG", "Electric"]
SellerType = Literal["Individual", "Dealer", "Trustmark Dealer"]
Transmission = Literal["Manual", "Automatic"]
Owner = Literal["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]


class VehicleProfile(BaseModel):
    brand: str = Field(..., description="Vehicle brand, e.g. 'Maruti'. Unknown brands are treated as 'Other'.")
    year: int = Field(..., ge=1980, le=config.CURRENT_YEAR, description="Manufacturing year")
    km_driven: int = Field(..., ge=0, le=1_000_000, description="Odometer reading in kilometers")
    fuel: FuelType
    transmission: Transmission
    seller_type: SellerType
    owner: Owner

    model_config = {
        "json_schema_extra": {
            "example": {
                "brand": "Maruti",
                "year": 2021,
                "km_driven": 40000,
                "fuel": "Petrol",
                "transmission": "Manual",
                "seller_type": "Individual",
                "owner": "First Owner",
            }
        }
    }


class FeatureContribution(BaseModel):
    label: str
    shap_value: float


class PredictionResponse(BaseModel):
    predicted_price_inr: float
    price_low: float
    price_high: float
    p10: float
    p50: float
    p90: float
    brand_median_inr: float
    top_contributions: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_training_samples: int
