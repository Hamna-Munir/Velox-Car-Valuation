import pandas as pd
import pytest

from velox import config
from velox.data import DataValidationError, load_clean_dataset


def test_load_clean_dataset_returns_expected_columns():
    df = load_clean_dataset()
    expected = set(config.FEATURES) | {config.TARGET, "year"}
    assert expected.issubset(df.columns)


def test_load_clean_dataset_has_rows():
    df = load_clean_dataset()
    assert len(df) > 1000


def test_no_null_prices():
    df = load_clean_dataset()
    assert not df[config.TARGET].isna().any()


def test_prices_are_positive():
    df = load_clean_dataset()
    assert (df[config.TARGET] > 0).all()


def test_car_age_is_non_negative():
    df = load_clean_dataset()
    assert (df["car_age"] >= 0).all()


def test_brand_bucketing_respects_top_n():
    df = load_clean_dataset()
    # +1 for the "Other" bucket
    assert df["brand"].nunique() <= config.TOP_N_BRANDS + 1


def test_market_adjustment_scales_prices_up():
    # With CURRENT_YEAR > DATA_REFERENCE_YEAR and positive appreciation,
    # the adjustment factor must be greater than 1.
    assert config.MARKET_ADJUSTMENT > 1.0


def test_validate_raw_rejects_missing_columns():
    from velox.data import _validate_raw

    bad_df = pd.DataFrame({"name": ["Maruti 800"], "year": [2010]})
    with pytest.raises(DataValidationError):
        _validate_raw(bad_df)
