"""Train and persist the production ML model used by delta_algo_bot.

The dataset is built only from historical market data and is split
chronologically to avoid training on future observations.

The saved artifact is compatible with app.ml.model.MLModel and therefore
can be loaded by the real-time predictor.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app.ml.model import MLModel
from app.config.settings import Settings


logger = logging.getLogger("MLTrainer")

LABEL_NAMES = {
    -1: "DOWN",
    0: "FLAT",
    1: "UP",
}

# These are available from the historical feature pipeline and can be
# reconstructed from the live MarketFeatures object. Keeping this list
# stable prevents the live predictor from receiving an unrelated feature
# vector filled with missing values.
LIVE_FEATURES = [
    "mid_close",
    "mid_range",
    "mid_change",
    "mid_return",
    "one_second_return",
    "intrasecond_volatility",
    "bid_close",
    "ask_close",
    "spread_mean",
    "spread_close",
    "imbalance_mean",
    "trade_count",
    "trade_volume",
    "trade_price_change",
    "l1_update_count",
    "mid_change_count",
    "bid_change_count",
    "ask_change_count",
    "l1_gap_seconds",
]
TARGET_COLUMNS = {
    "label",
    "future_return",
    "label_threshold",
    "timestamp_us",
    "timestamp_seconds",
    "symbol",
}


class MLTrainer:
    def __init__(self, test_fraction: float = 0.20) -> None:
        if not 0.05 <= test_fraction <= 0.40:
            raise ValueError("test_fraction must be between 0.05 and 0.40.")
        self.test_fraction = test_fraction

    def train(
        self,
        dataset_path: Path,
        model_path: Path,
    ) -> dict:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        if df.empty:
            raise RuntimeError("Training dataset is empty.")

        if "label" not in df.columns:
            raise RuntimeError("Training dataset is missing 'label'.")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)

        if "timestamp_us" in df.columns:
            df = df.sort_values("timestamp_us").reset_index(drop=True)

        feature_columns = [
            name for name in LIVE_FEATURES
            if name in df.columns
        ]
        if len(feature_columns) < 5:
            raise RuntimeError(
                "The aligned dataset does not contain enough live-compatible ML features."
            )

        X = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = df["label"]

        if y.nunique() < 2:
            raise RuntimeError("ML training requires at least two label classes.")

        split = int(len(df) * (1.0 - self.test_fraction))
        if split < 10 or len(df) - split < 5:
            raise RuntimeError("Not enough samples for chronological train/test split.")

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = MLModel(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(
            X_train.to_numpy(dtype=float),
            y_train.to_numpy(dtype=int),
            feature_names=feature_columns,
        )

        predictions = model.predict(X_test.to_numpy(dtype=float))

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(
                precision_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_test, predictions, average="macro", zero_division=0)
            ),
        }

        model_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = model_path.with_name("model_metadata.json")
        model.save(model_path, metadata_path)

        result = {
            "model_type": "RandomForestClassifier",
            "samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features": feature_columns,
            "metrics": metrics,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "label_distribution": {
                LABEL_NAMES.get(int(k), str(k)): int(v)
                for k, v in y.value_counts().sort_index().items()
            },
        }

        logger.info(
            "ML training complete | samples=%d | accuracy=%.4f | f1=%.4f | model=%s",
            result["samples"],
            metrics["accuracy"],
            metrics["f1_macro"],
            model_path,
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Delta ML prediction model.")
    parser.add_argument("--dataset", default=str(data_root / "ml" / "final_training_dataset.csv"))
    parser.add_argument("--model", default=str(data_root / "models" / "market_direction_model.joblib"))
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    result = MLTrainer(args.test_fraction).train(
        Path(args.dataset),
        Path(args.model),
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
