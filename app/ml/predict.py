from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.ml.model import MLModel


logger = logging.getLogger("MLPredictor")


class MLPredictor:
    """
    Real-time prediction service.

    This class does NOT place orders.

    It only converts the current feature vector
    into a machine-learning prediction.
    """

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:

        self.model_path = Path(model_path)

        self.model: MLModel | None = None

        self.last_prediction: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------------

    def load(self) -> None:

        self.model = MLModel.load(
            self.model_path
        )

        logger.info(
            "ML predictor ready | model=%s | features=%d",
            self.model_path,
            len(self.model.feature_names),
        )

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict(
        self,
        features: dict[str, float],
    ) -> dict[str, Any]:

        if self.model is None:
            self.load()

        if self.model is None:
            raise RuntimeError(
                "ML model could not be loaded."
            )

        result = self.model.predict_single(
            features
        )

        self.last_prediction = result

        logger.info(
            "ML Prediction | label=%s | confidence=%.2f%%",
            result["label"],
            result["confidence"] * 100.0,
        )

        return result

    # ------------------------------------------------------------------
    # MODEL STATUS
    # ------------------------------------------------------------------

    def is_available(self) -> bool:

        return self.model_path.exists()

    # ------------------------------------------------------------------
    # REQUIRED FEATURES
    # ------------------------------------------------------------------

    def required_features(self) -> list[str]:

        if self.model is None:
            self.load()

        if self.model is None:
            return []

        return list(
            self.model.feature_names
        )

    # ------------------------------------------------------------------
    # LAST PREDICTION
    # ------------------------------------------------------------------

    def get_last_prediction(
        self,
    ) -> dict[str, Any] | None:

        return self.last_prediction