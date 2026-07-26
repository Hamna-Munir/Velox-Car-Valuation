"""
velox.model
Builds and trains the VELOX pricing models.

Two things distinguish this from a naive "fit one regressor" script:

1. Hyperparameter tuning via RandomizedSearchCV with k-fold cross-validation,
   rather than hand-picked hyperparameters — the reported metrics come from
   held-out data the search never touched.
2. Quantile regression (P10 / P50 / P90) instead of a single point estimate
   plus a heuristic +/- MAE band. Each quantile model is trained with its own
   pinball loss, so the resulting interval is a real model-based estimate of
   *this specific vehicle profile's* price uncertainty — not just the
   historical spread of its brand.
"""

import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from velox import config

logger = logging.getLogger(__name__)

PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [150, 200, 300, 350, 450],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__min_samples_leaf": [1, 5, 10],
}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])


def build_pipeline(loss: str = "squared_error", alpha: float = 0.9) -> Pipeline:
    """A fresh preprocessing + GradientBoostingRegressor pipeline.

    loss="quantile" with `alpha` set to 0.1/0.5/0.9 trains a model for that
    specific quantile of the price distribution instead of the conditional mean.
    """
    kwargs = dict(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=config.RANDOM_STATE)
    if loss == "quantile":
        kwargs["loss"] = "quantile"
        kwargs["alpha"] = alpha
    model = GradientBoostingRegressor(**kwargs)
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def tune_point_estimator(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 25) -> Pipeline:
    """Randomized hyperparameter search with 5-fold CV for the point-estimate model."""
    base_pipeline = build_pipeline(loss="squared_error")
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=5,
        scoring="r2",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    logger.info("Running RandomizedSearchCV (%d iterations, 5-fold CV)...", n_iter)
    t0 = time.time()
    search.fit(X_train, y_train)
    logger.info("Search complete in %.1fs — best CV R²: %.4f", time.time() - t0, search.best_score_)
    logger.info("Best params: %s", search.best_params_)
    return search.best_estimator_, search.best_params_, search.best_score_


def train_quantile_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train one GradientBoostingRegressor per quantile (P10/P50/P90)."""
    models = {}
    for name, q in config.QUANTILES.items():
        pipe = build_pipeline(loss="quantile", alpha=q)
        pipe.fit(X_train, y_train)
        models[name] = pipe
        logger.info("Trained quantile model %s (alpha=%.2f)", name, q)
    return models


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": r2_score(y_test, preds),
    }


def evaluate_quantile_coverage(quantile_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """What fraction of true prices actually fall within the predicted P10-P90 band?

    For a well-calibrated model this should land close to 80%.
    """
    p10 = quantile_models["p10"].predict(X_test)
    p90 = quantile_models["p90"].predict(X_test)
    within_band = np.mean((y_test.values >= p10) & (y_test.values <= p90))
    return {"p10_p90_coverage": float(within_band), "target_coverage": 0.80}


def train_and_save() -> dict:
    """Full training pipeline: tune the point model, train quantile models,
    evaluate both, and persist all artifacts. Returns the metrics dict.
    """
    from velox.data import load_clean_dataset  # local import avoids circularity at module load

    df = load_clean_dataset()
    df.to_csv(config.CLEAN_DATA_PATH, index=False)

    X = df[config.FEATURES]
    y = df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    point_model, best_params, cv_r2 = tune_point_estimator(X_train, y_train)
    point_metrics = evaluate(point_model, X_test, y_test)
    logger.info("Point model test metrics: %s", point_metrics)

    quantile_models = train_quantile_models(X_train, y_train)
    coverage = evaluate_quantile_coverage(quantile_models, X_test, y_test)
    logger.info("Quantile band coverage: %s", coverage)

    joblib.dump(point_model, config.MODEL_PATH)
    joblib.dump(quantile_models, config.QUANTILE_MODELS_PATH)

    metrics = {
        **point_metrics,
        "cv_r2": cv_r2,
        "best_params": best_params,
        "coverage": coverage,
        "n_samples": len(df),
        "brands": sorted(df["brand"].unique().tolist()),
        "market_adjustment": config.MARKET_ADJUSTMENT,
        "market_year": config.CURRENT_YEAR,
        "data_reference_year": config.DATA_REFERENCE_YEAR,
        "annual_appreciation": config.ANNUAL_MARKET_APPRECIATION,
    }
    joblib.dump(metrics, config.METRICS_PATH)
    logger.info("Saved model.pkl, quantile_models.pkl, metrics.pkl, car_data_clean.csv")
    return metrics


def load_models():
    """Load the point-estimate model, quantile models, and metrics from disk,
    training them first if any artifact is missing.
    """
    if not (config.MODEL_PATH.exists() and config.QUANTILE_MODELS_PATH.exists() and config.METRICS_PATH.exists()):
        train_and_save()
    point_model = joblib.load(config.MODEL_PATH)
    quantile_models = joblib.load(config.QUANTILE_MODELS_PATH)
    metrics = joblib.load(config.METRICS_PATH)
    return point_model, quantile_models, metrics
