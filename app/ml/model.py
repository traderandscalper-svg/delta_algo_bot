from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


logger = logging.getLogger("MLModel")


class MLModel:
    """
    Machine-learning model wrapper for the Delta trading bot.

    The model predicts:
        DOWN = -1
        FLAT = 0
        UP   = 1
    """

    LABEL_NAMES = {
        -1: "DOWN",
        0: "FLAT",
        1: "UP",
    }

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 10,
        min_samples_leaf: int = 10,
        random_state: int = 42,
    ) -> None:

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

        self.feature_names: list[str] = []
        self.label_names: dict[str, str] = {
            "-1": "DOWN",
            "0": "FLAT",
            "1": "UP",
        }

        self.is_fitted = False

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: list[str] | None = None,
    ) -> "MLModel":

        if X is None or y is None:
            raise ValueError("Training data cannot be None.")

        if len(X) == 0:
            raise ValueError("Training dataset is empty.")

        if len(X) != len(y):
            raise ValueError(
                f"X/y length mismatch: {len(X)} != {len(y)}"
            )

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else []
        )

        logger.info(
            "Training ML model | samples=%d | features=%d",
            len(X),
            len(self.feature_names),
        )

        self.model.fit(X, y)

        self.is_fitted = True

        logger.info("ML model training completed.")

        return self

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:

        self._check_fitted()

        return self.model.predict(X)

    # ------------------------------------------------------------------
    # PROBABILITIES
    # ------------------------------------------------------------------

    def predict_proba(self, X: Any) -> np.ndarray:

        self._check_fitted()

        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # SINGLE PREDICTION
    # ------------------------------------------------------------------

    def predict_single(
        self,
        features: dict[str, float],
    ) -> dict[str, Any]:

        self._check_fitted()

        if not self.feature_names:
            raise ValueError(
                "Feature names are not available in the trained model."
            )

        values = []

        for feature_name in self.feature_names:
            value = features.get(feature_name, 0.0)

            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0

            if not np.isfinite(value):
                value = 0.0

            values.append(value)

        X = np.asarray(
            [values],
            dtype=float,
        )

        prediction = int(
            self.model.predict(X)[0]
        )

        probabilities = self.model.predict_proba(X)[0]

        classes = self.model.classes_

        probability_map: dict[str, float] = {}

        for class_value, probability in zip(
            classes,
            probabilities,
        ):
            class_int = int(class_value)

            probability_map[
                self.LABEL_NAMES.get(
                    class_int,
                    str(class_int),
                )
            ] = float(probability)

        confidence = float(
            max(probability_map.values())
            if probability_map
            else 0.0
        )

        return {
            "prediction": prediction,
            "label": self.LABEL_NAMES.get(
                prediction,
                str(prediction),
            ),
            "confidence": confidence,
            "probabilities": probability_map,
        }

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------

    def feature_importance(self) -> dict[str, float]:

        self._check_fitted()

        if not self.feature_names:
            return {}

        importances = self.model.feature_importances_

        result = {}

        for name, importance in zip(
            self.feature_names,
            importances,
        ):
            result[name] = float(importance)

        return dict(
            sorted(
                result.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def save(
        self,
        model_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> None:

        self._check_fitted()

        model_path = Path(model_path)

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "parameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
            },
        }

        joblib.dump(
            payload,
            model_path,
        )

        logger.info(
            "ML model saved: %s",
            model_path,
        )

        if metadata_path is not None:

            metadata_path = Path(metadata_path)

            metadata_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            metadata = {
                "feature_names": self.feature_names,
                "label_names": self.label_names,
                "parameters": payload["parameters"],
                "feature_importance": self.feature_importance(),
            }

            metadata_path.write_text(
                json.dumps(
                    metadata,
                    indent=2,
                ),
                encoding="utf-8",
            )

            logger.info(
                "ML metadata saved: %s",
                metadata_path,
            )

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_path: str | Path,
    ) -> "MLModel":

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"ML model not found: {model_path}"
            )

        payload = joblib.load(model_path)

        instance = cls(
            **payload.get(
                "parameters",
                {},
            )
        )

        instance.model = payload["model"]

        instance.feature_names = list(
            payload.get(
                "feature_names",
                [],
            )
        )

        instance.label_names = dict(
            payload.get(
                "label_names",
                {
                    "-1": "DOWN",
                    "0": "FLAT",
                    "1": "UP",
                },
            )
        )

        instance.is_fitted = True

        logger.info(
            "ML model loaded: %s",
            model_path,
        )

        return instance

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:

        if not self.is_fitted:
            raise RuntimeError(
                "ML model has not been trained or loaded."
            )