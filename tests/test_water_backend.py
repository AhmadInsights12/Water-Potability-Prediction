import unittest

import numpy as np
import pandas as pd

from water_backend import WaterModel, predict_samples


class ExampleClassifier:
    classes_ = np.array([0, 1])
    n_features_in_ = 2
    feature_names_in_ = np.array(["ph", "Hardness"], dtype=object)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([1 if value >= 7 else 0 for value in features["ph"]])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array(
            [[0.2, 0.8] if value >= 7 else [0.7, 0.3]
             for value in features["ph"]]
        )


class PredictSamplesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = WaterModel(
            estimator=ExampleClassifier(),
            features=("ph", "Hardness"),
            defaults={"ph": 7.0, "Hardness": 100.0},
        )

    def test_predictions_include_probability_and_missing_value_count(self) -> None:
        samples = pd.DataFrame(
            {"ph": [8.0, 6.0], "Hardness": [120.0, np.nan]}
        )

        results = predict_samples(samples, self.model)

        self.assertEqual(
            results["predicted_potability"].tolist(),
            ["Potable", "Not potable"],
        )
        self.assertEqual(results["potable_probability"].tolist(), [0.8, 0.3])
        self.assertEqual(results["missing_values_imputed"].tolist(), [0, 1])
        self.assertEqual(results.loc[1, "Hardness"], 100.0)

    def test_missing_required_column_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Hardness"):
            predict_samples(pd.DataFrame({"ph": [7.0]}), self.model)

    def test_non_numeric_measurement_is_rejected(self) -> None:
        samples = pd.DataFrame({"ph": ["unknown"], "Hardness": [100.0]})

        with self.assertRaisesRegex(ValueError, "ph.*numeric"):
            predict_samples(samples, self.model)

    def test_infinite_measurement_is_rejected(self) -> None:
        samples = pd.DataFrame({"ph": [np.inf], "Hardness": [100.0]})

        with self.assertRaisesRegex(ValueError, "finite"):
            predict_samples(samples, self.model)

    def test_empty_batch_is_rejected(self) -> None:
        samples = pd.DataFrame(columns=["ph", "Hardness"])

        with self.assertRaisesRegex(ValueError, "(?i)at least one"):
            predict_samples(samples, self.model)


if __name__ == "__main__":
    unittest.main()
