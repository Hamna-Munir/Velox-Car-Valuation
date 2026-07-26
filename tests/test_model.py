import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from velox import config
from velox.model import build_pipeline, evaluate_quantile_coverage, load_models


@pytest.fixture(scope="module")
def trained_models():
    """Reuses whatever model.pkl / quantile_models.pkl already exist on disk
    (trained once via `python train_model.py`) rather than retraining for
    every test — hyperparameter search is too slow to repeat per-test.
    """
    return load_models()


def test_build_pipeline_returns_pipeline():
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    assert "preprocessor" in pipe.named_steps
    assert "model" in pipe.named_steps


def test_quantile_pipeline_sets_correct_loss_and_alpha():
    pipe = build_pipeline(loss="quantile", alpha=0.1)
    assert pipe.named_steps["model"].loss == "quantile"
    assert pipe.named_steps["model"].alpha == 0.1


def test_load_models_returns_expected_types(trained_models):
    point_model, quantile_models, metrics = trained_models
    assert isinstance(point_model, Pipeline)
    assert set(quantile_models.keys()) == {"p10", "p50", "p90"}
    assert "r2" in metrics


def test_point_model_r2_is_reasonable(trained_models):
    _, _, metrics = trained_models
    # Real-world used-car pricing from these features alone should comfortably
    # clear 0.6 R² — this guards against a silently broken training run.
    assert metrics["r2"] > 0.6


def test_quantile_ordering_is_sane(trained_models):
    """P10 must be <= P50 must be <= P90 for any given input, or the
    uncertainty band is nonsensical."""
    _, quantile_models, _ = trained_models
    sample = pd.DataFrame([{
        "brand": "Maruti", "car_age": 5, "km_driven": 40000, "fuel": "Petrol",
        "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner",
    }])
    p10 = quantile_models["p10"].predict(sample)[0]
    p50 = quantile_models["p50"].predict(sample)[0]
    p90 = quantile_models["p90"].predict(sample)[0]
    assert p10 <= p50 <= p90


def test_quantile_band_coverage_near_target(trained_models):
    """The P10-P90 band should contain roughly 80% of held-out prices."""
    from sklearn.model_selection import train_test_split

    from velox.data import load_clean_dataset

    _, quantile_models, _ = trained_models
    df = load_clean_dataset()
    X = df[config.FEATURES]
    y = df[config.TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    coverage = evaluate_quantile_coverage(quantile_models, X_test, y_test)
    # Allow a reasonably wide tolerance band — this is a sanity check, not a
    # strict calibration assertion.
    assert 0.60 <= coverage["p10_p90_coverage"] <= 0.95


def test_unknown_brand_does_not_crash_prediction(trained_models):
    point_model, _, _ = trained_models
    sample = pd.DataFrame([{
        "brand": "TotallyUnknownBrandXYZ", "car_age": 5, "km_driven": 40000,
        "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual",
        "owner": "First Owner",
    }])
    pred = point_model.predict(sample)[0]
    assert pred > 0
