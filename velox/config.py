"""
velox.config
Centralized configuration for the VELOX package. Values are overridable via
environment variables so the same code runs unchanged across local dev,
CI, Docker, and Streamlit Cloud.
"""

import os
from pathlib import Path

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("VELOX_DATA_DIR", BASE_DIR / "data"))
MODEL_DIR = Path(os.getenv("VELOX_MODEL_DIR", BASE_DIR))

RAW_DATA_PATH = DATA_DIR / "car_details.csv"
CLEAN_DATA_PATH = MODEL_DIR / "car_data_clean.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"
QUANTILE_MODELS_PATH = MODEL_DIR / "quantile_models.pkl"
METRICS_PATH = MODEL_DIR / "metrics.pkl"

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("VELOX_LOG_LEVEL", "INFO")

# ---------------- Temporal reference ----------------
CURRENT_YEAR = int(os.getenv("VELOX_CURRENT_YEAR", "2026"))  # "today", for vehicle-age calc

# ---- Market adjustment: scale 2020-21 listing prices to CURRENT_YEAR-equivalent ----
# CarDekho's public listings reflect the India used-car market as of ~2021.
# Genuinely fresh listings for CURRENT_YEAR aren't available as an open dataset,
# so prices are scaled using a documented used-car price appreciation rate
# (~8%/year, per Cars24/Team-BHP and Mordor Intelligence India used-car market
# reports, 2024-2025) instead of being left stale. Disclosed to the user in the
# app's Methodology tab.
DATA_REFERENCE_YEAR = int(os.getenv("VELOX_DATA_REFERENCE_YEAR", "2021"))
ANNUAL_MARKET_APPRECIATION = float(os.getenv("VELOX_ANNUAL_APPRECIATION", "0.08"))
MARKET_ADJUSTMENT = (1 + ANNUAL_MARKET_APPRECIATION) ** (CURRENT_YEAR - DATA_REFERENCE_YEAR)

# ---------------- Category labels shown in the UI / API ----------------
FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]
TRANSMISSIONS = ["Manual", "Automatic"]
OWNER_TYPES = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]

TOP_N_BRANDS = 12

FEATURES = ["brand", "car_age", "km_driven", "fuel", "seller_type", "transmission", "owner"]
CATEGORICAL_FEATURES = ["brand", "fuel", "seller_type", "transmission", "owner"]
NUMERIC_FEATURES = ["car_age", "km_driven"]
TARGET = "price_inr"

# ---------------- Modeling ----------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}

# ---------------- API ----------------
API_HOST = os.getenv("VELOX_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("VELOX_API_PORT", "8000"))
