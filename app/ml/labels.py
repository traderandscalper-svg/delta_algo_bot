"""
Label generation for ML market-direction training.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger("MLLabels")


DOWN = -1
FLAT = 0
UP = 1


def calculate_future_return(
    current_price: float,
    future_price: float,
) -> float:
    """
    Calculate percentage return between current and future price.

    Example:
        100 -> 101 = 0.01
        100 -> 99  = -0.01
    """

    try:
        current_price = float(current_price)
        future_price = float(future_price)
    except (TypeError, ValueError):
        return 0.0

    if current_price <= 0:
        return 0.0

    return (
        (future_price - current_price)
        / current_price
    )


def create_label(
    current_price: float,
    future_price: float,
    threshold: float = 0.0003,
) -> int:
    """
    Create a market-direction label.

    Returns:
        -1 = DOWN
         0 = FLAT
         1 = UP
    """

    future_return = calculate_future_return(
        current_price,
        future_price,
    )

    if future_return > threshold:
        return UP

    if future_return < -threshold:
        return DOWN

    return FLAT


def create_direction_labels(
    df: pd.DataFrame,
    horizon: int = 10,
    threshold: float = 0.0003,
    price_column: str = "mid_price",
) -> pd.DataFrame:

    if df.empty:
        raise ValueError(
            "Cannot create labels from an empty dataframe."
        )

    if price_column not in df.columns:
        raise ValueError(
            f"Required price column not found: {price_column}"
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be greater than zero."
        )

    result = df.copy()

    current_price = pd.to_numeric(
        result[price_column],
        errors="coerce",
    )

    future_price = current_price.shift(
        -horizon
    )

    future_return = (
        (future_price - current_price)
        / current_price
    )

    result["future_return"] = future_return

    result["label"] = FLAT

    result.loc[
        future_return > threshold,
        "label",
    ] = UP

    result.loc[
        future_return < -threshold,
        "label",
    ] = DOWN

    result.loc[
        future_price.isna(),
        "label",
    ] = pd.NA

    result = result.dropna(
        subset=[
            "future_return",
            "label",
        ]
    )

    result["label"] = result[
        "label"
    ].astype(int)

    return result


def label_distribution(
    df: pd.DataFrame,
) -> dict[str, int]:

    if "label" not in df.columns:
        raise ValueError(
            "DataFrame does not contain a 'label' column."
        )

    counts = df["label"].value_counts()

    return {
        "DOWN": int(counts.get(DOWN, 0)),
        "FLAT": int(counts.get(FLAT, 0)),
        "UP": int(counts.get(UP, 0)),
    }


def log_label_distribution(
    df: pd.DataFrame,
) -> None:

    distribution = label_distribution(df)

    total = sum(
        distribution.values()
    )

    logger.info(
        "Label distribution | "
        "DOWN=%d | FLAT=%d | UP=%d | total=%d",
        distribution["DOWN"],
        distribution["FLAT"],
        distribution["UP"],
        total,
    )

    if total == 0:
        return

    for label, count in distribution.items():

        percentage = (
            count / total * 100.0
        )

        logger.info(
            "%s: %d (%.2f%%)",
            label,
            count,
            percentage,
        )


def create_labels(
    prices: list[float],
    horizon: int = 10,
    threshold: float = 0.0003,
) -> list[Any]:

    labels: list[Any] = []

    if not prices:
        return labels

    if horizon <= 0:
        raise ValueError(
            "horizon must be greater than zero."
        )

    for index in range(len(prices)):

        future_index = index + horizon

        if future_index >= len(prices):
            labels.append(None)
            continue

        labels.append(
            create_label(
                current_price=prices[index],
                future_price=prices[future_index],
                threshold=threshold,
            )
        )

    return labels