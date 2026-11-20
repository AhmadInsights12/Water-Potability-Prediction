"""Model loading and prediction logic for the water-potability app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).resolve().with_name("water_model.pkl")
DATASET_PATH = Path(__file__).resolve().with_name("water_potability.csv")

FEATURE_LABELS = {
    "ph": "pH",
    "Hardness": "Hardness (mg/L)",
    "Solids": "Total dissolved solids (ppm)",
    "Chloramines": "Chloramines (ppm)",
    "Sulfate": "Sulfate (mg/L)",
    "Conductivity": "Conductivity (µS/cm)",
    "Organic_carbon": "Organic carbon (ppm)",
    "Trihalomethanes": "Trihalomethanes (µg/L)",
    "Turbidity": "Turbidity (NTU)",
}


@dataclass(frozen=True)
class WaterModel:
    estimator: Any
    features: tuple[str, ...]
    defaults: dict[str, float]


def load_model(path: Path = MODEL_PATH) -> WaterModel:
    """Load and validate the model artifact created by the training notebook."""
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact was not found: {path}")

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("The model artifact must contain a dictionary.")

    estimator = artifact.get("model")
    raw_features = artifact.get("features")
    raw_defaults = artifact.get("defaults")
    if estimator is None or not callable(getattr(estimator, "predict", None)):
        raise ValueError("The model artifact does not contain a usable classifier.")
    if not isinstance(raw_features, (list, tuple)) or not raw_features:
        raise ValueError("The model artifact does not contain a feature list.")
    if any(not isinstance(feature, str) for feature in raw_features):
        raise ValueError("Model feature names must be strings.")

    features = tuple(raw_features)
    if len(set(features)) != len(features):
        raise ValueError("The model feature list contains duplicate names.")
    if not isinstance(raw_defaults, dict):
        raise ValueError("The model artifact does not contain missing-value defaults.")

    defaults: dict[str, float] = {}
    for feature in features:
        if feature not in raw_defaults:
            raise ValueError(f"The model artifact has no default for {feature}.")
        try:
            default = float(raw_defaults[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(f"The default for {feature} is not numeric.") from error
        if not np.isfinite(default):
            raise ValueError(f"The default for {feature} must be finite.")
        defaults[feature] = default

    expected_count = getattr(estimator, "n_features_in_", len(features))
    if expected_count != len(features):
        raise ValueError("The model feature count does not match its feature list.")
    expected_names = getattr(estimator, "feature_names_in_", None)
    if expected_names is not None and tuple(expected_names) != features:
        raise ValueError("The model feature names do not match its feature list.")

    return WaterModel(estimator=estimator, features=features, defaults=defaults)


def predict_samples(samples: pd.DataFrame, bundle: WaterModel) -> pd.DataFrame:
    """Predict potability for rows containing the model's input features."""
    if not isinstance(samples, pd.DataFrame):
        raise TypeError("Samples must be provided as a pandas DataFrame.")
    if samples.empty:
        raise ValueError("At least one water sample is required.")

    missing_columns = [feature for feature in bundle.features if feature not in samples]
    if missing_columns:
        raise ValueError(
            "Missing required water measurements: " + ", ".join(missing_columns)
        )

    source = samples.loc[:, list(bundle.features)]
    parsed = pd.DataFrame(index=source.index)
    for feature in bundle.features:
        values = pd.to_numeric(source[feature], errors="coerce")
        invalid_values = source[feature].notna() & values.isna()
        if invalid_values.any():
            row = invalid_values[invalid_values].index[0]
            raise ValueError(
                f"'{feature}' must contain numeric values (invalid value at row {row})."
            )
        parsed[feature] = values

    if np.isinf(parsed.to_numpy(dtype=float)).any():
        raise ValueError("Measurements must be finite numbers.")

    imputed_counts = parsed.isna().sum(axis=1)
    features = parsed.fillna(bundle.defaults)
    predictions = np.asarray(bundle.estimator.predict(features))
    if predictions.ndim != 1 or len(predictions) != len(features):
        raise RuntimeError("The classifier returned an invalid prediction shape.")
    if not np.isin(predictions, [0, 1]).all():
        raise RuntimeError("The classifier returned a class other than 0 or 1.")

    result = features.copy()
    result["predicted_potability"] = np.where(
        predictions == 1, "Potable", "Not potable"
    )
    result["potable_probability"] = np.nan

    predict_proba = getattr(bundle.estimator, "predict_proba", None)
    classes = getattr(bundle.estimator, "classes_", None)
    if callable(predict_proba) and classes is not None and 1 in classes:
        probabilities = np.asarray(predict_proba(features))
        if probabilities.ndim != 2 or len(probabilities) != len(features):
            raise RuntimeError("The classifier returned invalid probability values.")
        result["potable_probability"] = probabilities[:, list(classes).index(1)]

    result["missing_values_imputed"] = imputed_counts.to_numpy()
    return result
