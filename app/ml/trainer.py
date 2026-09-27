"""
Train the first ML model for the Delta algorithmic trading bot.

The model predicts:

    0 = DOWN
    1 = FLAT
    2 = UP
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


logger = logging.getLogger(
    "MLTrainer"
)


LABEL_NAMES = {
    0: "DOWN",
    1: "FLAT",
    2: "UP",
}


NON_FEATURE_COLUMNS = {
    "timestamp_us",
    "timestamp_seconds",
    "symbol",
    "future_return",
    "label",
}


class MLTrainer:

    def __init__(
        self,
        test_fraction: float = 0.20,
    ) -> None:

        self.test_fraction = (
            test_fraction
        )

    # ============================================================
    # TRAIN
    # ============================================================

    def train(
        self,
        dataset_path: Path,
        model_path: Path,
    ) -> None:

        logger.info(
            "Loading ML dataset | %s",
            dataset_path,
        )

        df = pd.read_csv(
            dataset_path
        )

        if df.empty:
            raise RuntimeError(
                "Training dataset is empty."
            )

        required = {
            "label",
            "future_return",
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise RuntimeError(
                "Dataset is missing required "
                f"columns: {sorted(missing)}"
            )

        # --------------------------------------------------------
        # Clean
        # --------------------------------------------------------

        df = df.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        df = df.dropna(
            subset=[
                "label"
            ]
        )

        df["label"] = (
            df["label"]
            .astype(int)
        )

        # Preserve chronological order.
        if "timestamp_us" in df.columns:
            df = df.sort_values(
                "timestamp_us"
            ).reset_index(
                drop=True
            )

        # --------------------------------------------------------
        # Feature columns
        # --------------------------------------------------------

        feature_columns = [
            column
            for column in df.columns
            if column
            not in NON_FEATURE_COLUMNS
        ]

        if not feature_columns:
            raise RuntimeError(
                "No feature columns available."
            )

        X = df[
            feature_columns
        ].copy()

        y = df[
            "label"
        ].copy()

        X = X.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        X = X.fillna(0.0)

        # --------------------------------------------------------
        # Chronological split
        # --------------------------------------------------------

        split_index = int(
            len(df)
            * (
                1.0
                - self.test_fraction
            )
        )

        if split_index <= 0:
            raise RuntimeError(
                "Not enough data for training."
            )

        if split_index >= len(df):
            split_index = (
                len(df) - 1
            )

        X_train = X.iloc[
            :split_index
        ]

        X_test = X.iloc[
            split_index:
        ]

        y_train = y.iloc[
            :split_index
        ]

        y_test = y.iloc[
            split_index:
        ]

        logger.info(
            "Training samples: %d",
            len(X_train),
        )

        logger.info(
            "Testing samples: %d",
            len(X_test),
        )

        logger.info(
            "Features: %d",
            len(feature_columns),
        )

        # --------------------------------------------------------
        # Model
        # --------------------------------------------------------

        model = (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=31,
                min_samples_leaf=50,
                l2_regularization=1.0,
                random_state=42,
            )
        )

        logger.info(
            "Training HistGradientBoostingClassifier..."
        )

        model.fit(
            X_train,
            y_train,
        )

        # --------------------------------------------------------
        # Predictions
        # --------------------------------------------------------

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[
                0,
                1,
                2,
            ],
        )

        # --------------------------------------------------------
        # Save model
        # --------------------------------------------------------

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            model_path,
        )

        metadata_path = (
            model_path.parent
            / "model_metadata.json"
        )

        metadata = {
            "model_type": (
                "HistGradientBoostingClassifier"
            ),
            "feature_columns": (
                feature_columns
            ),
            "label_names": (
                LABEL_NAMES
            ),
            "train_samples": (
                int(len(X_train))
            ),
            "test_samples": (
                int(len(X_test))
            ),
            "metrics": {
                "accuracy": float(
                    accuracy
                ),
                "precision_macro": float(
                    precision
                ),
                "recall_macro": float(
                    recall
                ),
                "f1_macro": float(
                    f1
                ),
            },
        }

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        # --------------------------------------------------------
        # Report
        # --------------------------------------------------------

        print()
        print(
            "============================================================"
        )
        print(
            "                 ML TRAINING COMPLETE"
        )
        print(
            "============================================================"
        )

        print(
            f"Training samples : {len(X_train):,}"
        )

        print(
            f"Testing samples  : {len(X_test):,}"
        )

        print(
            f"Features         : {len(feature_columns)}"
        )

        print(
            f"Accuracy         : {accuracy:.4f}"
        )

        print(
            f"Precision        : {precision:.4f}"
        )

        print(
            f"Recall           : {recall:.4f}"
        )

        print(
            f"F1               : {f1:.4f}"
        )

        print()
        print(
            "Confusion Matrix"
        )

        print(
            "Rows = actual, Columns = predicted"
        )

        print(
            "             DOWN   FLAT     UP"
        )

        print(
            f"DOWN       {cm[0,0]:6d} "
            f"{cm[0,1]:7d} "
            f"{cm[0,2]:7d}"
        )

        print(
            f"FLAT       {cm[1,0]:6d} "
            f"{cm[1,1]:7d} "
            f"{cm[1,2]:7d}"
        )

        print(
            f"UP         {cm[2,0]:6d} "
            f"{cm[2,1]:7d} "
            f"{cm[2,2]:7d}"
        )

        print()
        print(
            "Detailed classification report:"
        )

        print(
            classification_report(
                y_test,
                predictions,
                labels=[
                    0,
                    1,
                    2,
                ],
                target_names=[
                    "DOWN",
                    "FLAT",
                    "UP",
                ],
                zero_division=0,
            )
        )

        print(
            f"Model saved   : {model_path}"
        )

        print(
            f"Metadata saved: {metadata_path}"
        )

        print(
            "============================================================"
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train the Delta ML prediction model."
        )
    )

    parser.add_argument(
        "--dataset",
        default=(
            "data/ml/"
            "training_dataset.csv"
        ),
    )

    parser.add_argument(
        "--model",
        default=(
            "data/models/"
            "market_direction_model.joblib"
        ),
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    trainer = MLTrainer(
        test_fraction=(
            args.test_fraction
        )
    )

    trainer.train(
        dataset_path=Path(
            args.dataset
        ),
        model_path=Path(
            args.model
        ),
    )


if __name__ == "__main__":
    main()