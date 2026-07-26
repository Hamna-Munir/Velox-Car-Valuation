"""
velox.data
Loads and cleans the raw CarDekho used-car dataset into the schema the model
and app expect. Includes basic data-quality validation so a malformed or
truncated source file fails loudly and early instead of silently training a
bad model.
"""

import logging

import pandas as pd

from velox import config

logger = logging.getLogger(__name__)

REQUIRED_RAW_COLUMNS = {"name", "year", "selling_price", "km_driven", "fuel",
                         "seller_type", "transmission", "owner"}


class DataValidationError(ValueError):
    """Raised when the raw or cleaned dataset fails a sanity check."""


def _validate_raw(raw: pd.DataFrame) -> None:
    missing = REQUIRED_RAW_COLUMNS - set(raw.columns)
    if missing:
        raise DataValidationError(f"Raw dataset is missing required columns: {sorted(missing)}")
    if len(raw) < 100:
        raise DataValidationError(
            f"Raw dataset has only {len(raw)} rows — expected several thousand. "
            "Check that data/car_details.csv wasn't truncated."
        )


def _validate_clean(df: pd.DataFrame) -> None:
    if df[config.TARGET].isna().any():
        raise DataValidationError("Cleaned dataset contains null prices.")
    if (df[config.TARGET] <= 0).any():
        raise DataValidationError("Cleaned dataset contains non-positive prices.")
    if df["car_age"].min() < 0:
        raise DataValidationError("Cleaned dataset contains negative vehicle ages.")


def load_clean_dataset() -> pd.DataFrame:
    """Load data/car_details.csv and return the cleaned, model-ready dataframe.

    Applies brand bucketing (top N + "Other"), derives vehicle age from
    manufacturing year, scales prices to the configured market year, and
    drops extreme price outliers (top/bottom 0.5%).
    """
    logger.info("Loading raw dataset from %s", config.RAW_DATA_PATH)
    raw = pd.read_csv(config.RAW_DATA_PATH)
    _validate_raw(raw)

    raw = raw.dropna(subset=["selling_price", "year", "km_driven"])

    df = pd.DataFrame()
    brand = raw["name"].str.split().str[0]
    top_brands = brand.value_counts().head(config.TOP_N_BRANDS).index
    df["brand"] = brand.where(brand.isin(top_brands), "Other")

    df["year"] = raw["year"]
    df["car_age"] = (config.CURRENT_YEAR - raw["year"]).clip(lower=0)
    df["km_driven"] = raw["km_driven"]
    df["fuel"] = raw["fuel"]
    df["seller_type"] = raw["seller_type"]
    df["transmission"] = raw["transmission"]
    df["owner"] = raw["owner"]
    df[config.TARGET] = raw["selling_price"].astype(float) * config.MARKET_ADJUSTMENT

    # Drop implausible outliers (top/bottom 0.5%) to keep the model well-behaved
    lo, hi = df[config.TARGET].quantile([0.005, 0.995])
    df = df[(df[config.TARGET] >= lo) & (df[config.TARGET] <= hi)].reset_index(drop=True)

    _validate_clean(df)
    logger.info("Cleaned dataset: %d rows, %d brands", len(df), df["brand"].nunique())
    return df


def get_or_build_clean_dataset() -> pd.DataFrame:
    """Load the cached cleaned CSV if present, else build and cache it."""
    if config.CLEAN_DATA_PATH.exists():
        return pd.read_csv(config.CLEAN_DATA_PATH)
    df = load_clean_dataset()
    df.to_csv(config.CLEAN_DATA_PATH, index=False)
    return df
