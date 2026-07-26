"""
velox.explain
Per-prediction explainability using SHAP TreeExplainer.

Global feature importance (from model.feature_importances_) tells you what
the model relies on *on average*. It doesn't tell you why *this specific*
prediction came out the way it did — whether this vehicle's price was pulled
up by its brand and pulled down by its mileage, for instance. SHAP values
answer that per-row question, which is what a real pricing/underwriting tool
needs to be able to justify an individual estimate.
"""

import logging

import numpy as np
import pandas as pd
import shap

from velox import config

logger = logging.getLogger(__name__)

_FIELD_LABELS = {
    "brand": "Brand", "fuel": "Fuel Type", "seller_type": "Seller Type",
    "transmission": "Transmission", "owner": "Ownership",
    "car_age": "Vehicle Age", "km_driven": "Kilometers Driven",
}


def _grouped_feature_names(preprocessor) -> list[str]:
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(config.CATEGORICAL_FEATURES)
    return list(config.NUMERIC_FEATURES) + list(cat_names)


def _group_of(name: str) -> str:
    for field in config.CATEGORICAL_FEATURES:
        if name.startswith(field):
            return field
    return name


def explain_prediction(model, input_df: pd.DataFrame) -> pd.DataFrame:
    """Return a small dataframe of {field, contribution} for one input row,
    showing how much each original field pushed the prediction up or down
    relative to the model's baseline (expected value).
    """
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]

    X_transformed = preprocessor.transform(input_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_transformed)
    if shap_values.ndim > 1:
        shap_values = shap_values[0]

    feature_names = _grouped_feature_names(preprocessor)
    contrib = pd.DataFrame({"feature": feature_names, "shap_value": shap_values})
    contrib["group"] = contrib["feature"].map(_group_of)
    grouped = contrib.groupby("group")["shap_value"].sum().reset_index()
    grouped["label"] = grouped["group"].map(_FIELD_LABELS).fillna(grouped["group"])
    grouped = grouped.sort_values("shap_value", key=np.abs, ascending=False).reset_index(drop=True)

    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value.item() if base_value.size == 1 else float(base_value[0])
    grouped["base_value"] = float(base_value)
    return grouped[["label", "shap_value", "base_value"]]


def global_feature_importance(model) -> pd.DataFrame:
    """Aggregate, model-level feature importance — grouped by original field."""
    preprocessor = model.named_steps["preprocessor"]
    feature_names = _grouped_feature_names(preprocessor)
    importances = model.named_steps["model"].feature_importances_

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df["group"] = df["feature"].map(_group_of)
    grouped = df.groupby("group")["importance"].sum().sort_values(ascending=True).reset_index()
    grouped["label"] = grouped["group"].map(_FIELD_LABELS).fillna(grouped["group"])
    return grouped[["label", "importance"]]
