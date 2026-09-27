
"""
Diagnose the generated Delta ML dataset.

This script does not modify the dataset.

It checks whether the recorded price series is actually
moving and whether future returns contain excessive zeros.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATASET = Path(
    "data/ml/training_dataset.csv"
)


def main() -> None:

    if not DATASET.exists():

        raise SystemExit(
            f"Dataset not found: {DATASET}"
        )

    df = pd.read_csv(
        DATASET
    )

    if df.empty:

        raise SystemExit(
            "Dataset is empty."
        )

    required_columns = {
        "timestamp_us",
        "price",
        "future_return",
        "label",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise SystemExit(
            "Missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    df = df.sort_values(
        "timestamp_us"
    ).reset_index(
        drop=True
    )

    price = (
        pd.to_numeric(
            df["price"],
            errors="coerce",
        )
    )

    timestamps = (
        pd.to_numeric(
            df["timestamp_us"],
            errors="coerce",
        )
    )

    valid = (
        price.notna()
        & timestamps.notna()
    )

    price = price[valid].reset_index(
        drop=True
    )

    timestamps = timestamps[valid].reset_index(
        drop=True
    )

    # ============================================================
    # BASIC PRICE STATISTICS
    # ============================================================

    unique_prices = (
        price.nunique()
    )

    unchanged = int(
        (
            price.diff()
            == 0
        ).sum()
    )

    changed = int(
        (
            price.diff()
            != 0
        ).sum()
    )

    total_intervals = max(
        len(price) - 1,
        0,
    )

    unchanged_pct = (
        unchanged
        / max(
            total_intervals,
            1,
        )
        * 100.0
    )

    changed_pct = (
        changed
        / max(
            total_intervals,
            1,
        )
        * 100.0
    )

    # ============================================================
    # RETURNS
    # ============================================================

    returns_1s = (
        price.pct_change()
        .dropna()
    )

    future_returns = (
        pd.to_numeric(
            df["future_return"],
            errors="coerce",
        )
        .dropna()
    )

    # ============================================================
    # OUTPUT
    # ============================================================

    print()

    print(
        "============================================================"
    )

    print(
        "              DELTA ML DATASET DIAGNOSTIC"
    )

    print(
        "============================================================"
    )

    print(
        f"Rows:                  {len(df):,}"
    )

    print(
        f"Unique prices:         {unique_prices:,}"
    )

    print(
        f"Unchanged intervals:   {unchanged:,} "
        f"({unchanged_pct:.2f}%)"
    )

    print(
        f"Changed intervals:     {changed:,} "
        f"({changed_pct:.2f}%)"
    )

    print()

    print(
        f"First price:            {price.iloc[0]:.12f}"
    )

    print(
        f"Last price:             {price.iloc[-1]:.12f}"
    )

    print(
        f"Minimum price:          {price.min():.12f}"
    )

    print(
        f"Maximum price:          {price.max():.12f}"
    )

    print()

    print(
        "First 20 prices:"
    )

    for index, value in enumerate(
        price.head(20)
    ):

        print(
            f"  {index:>3}: "
            f"{value:.12f}"
        )

    # ============================================================
    # 1-SECOND RETURNS
    # ============================================================

    print()

    print(
        "1-second return statistics:"
    )

    if not returns_1s.empty:

        print(
            f"  Mean:       {returns_1s.mean():.10f}"
        )

        print(
            f"  Median:     {returns_1s.median():.10f}"
        )

        print(
            f"  Std:        {returns_1s.std():.10f}"
        )

        print(
            f"  Min:        {returns_1s.min():.10f}"
        )

        print(
            f"  Max:        {returns_1s.max():.10f}"
        )

        print(
            f"  Non-zero:   "
            f"{(returns_1s != 0).sum():,}"
            f"/{len(returns_1s):,}"
        )

    # ============================================================
    # FUTURE RETURNS
    # ============================================================

    print()

    print(
        "10-second future-return statistics:"
    )

    if not future_returns.empty:

        print(
            f"  Mean:       {future_returns.mean():.10f}"
        )

        print(
            f"  Median:     {future_returns.median():.10f}"
        )

        print(
            f"  Std:        {future_returns.std():.10f}"
        )

        print(
            f"  Min:        {future_returns.min():.10f}"
        )

        print(
            f"  Max:        {future_returns.max():.10f}"
        )

        print(
            f"  Zero:       "
            f"{(future_returns == 0).sum():,}"
            f"/{len(future_returns):,}"
        )

        print()

        print(
            "Future-return quantiles:"
        )

        for q in (
            0.50,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            0.99,
        ):

            value = float(
                future_returns.quantile(
                    q
                )
            )

            print(
                f"  Q{int(q * 100):02d}: "
                f"{value:.10f} "
                f"({value * 100:.6f}%)"
            )

    # ============================================================
    # LABEL DISTRIBUTION
    # ============================================================

    print()

    print(
        "Labels:"
    )

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():

        label_name = {
            -1: "DOWN",
             0: "FLAT",
             1: "UP",
        }.get(
            int(label),
            str(label),
        )

        print(
            f"  {label_name:<6}: "
            f"{int(count):,}"
        )

    # ============================================================
    # TIMESTAMPS
    # ============================================================

    if len(timestamps) > 1:

        intervals = (
            timestamps.diff()
            .dropna()
            / 1_000_000.0
        )

        print()

        print(
            "Timestamp intervals:"
        )

        print(
            f"  Minimum:    {intervals.min():.6f}s"
        )

        print(
            f"  Median:     {intervals.median():.6f}s"
        )

        print(
            f"  Maximum:    {intervals.max():.6f}s"
        )

    print()

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()
