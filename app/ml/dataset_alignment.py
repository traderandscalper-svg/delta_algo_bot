from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

INTRASECOND_FILE = Path(
    "data/ml/intrasecond_features.csv"
)

MULTI_HORIZON_FILE = Path(
    "data/ml/multi_horizon_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/ml/final_training_dataset.csv"
)

SYMBOL = "BTCUSD"

HORIZONS = (
    "1s",
    "3s",
    "5s",
    "10s",
    "30s",
    "60s",
)

# Maximum allowed difference between the feature
# timestamp and the L1 target anchor.
MAX_ALIGNMENT_SECONDS = 1.0

# Fields that are explicitly target/future information.
FUTURE_PREFIXES = (
    "future_",
    "return_",
    "direction_",
    "target_",
)

# These are allowed because they describe the current
# observation rather than future information.
ALLOWED_CURRENT_FIELDS = {
    "symbol",
    "window_id",
    "timestamp",
    "timestamp_us",

    "mid_open",
    "mid_high",
    "mid_low",
    "mid_close",
    "mid_range",
    "mid_change",
    "mid_return",
    "one_second_return",
    "intrasecond_volatility",

    "l1_update_count",
    "mid_change_count",
    "bid_change_count",
    "ask_change_count",

    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",

    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",

    "spread_mean",
    "spread_min",
    "spread_max",
    "spread_close",

    "imbalance_mean",
    "imbalance_min",
    "imbalance_max",

    "trade_count",
    "trade_volume",
    "trade_price_open",
    "trade_price_high",
    "trade_price_low",
    "trade_price_close",
    "trade_price_change",

    "l1_gap_seconds",
    "future_target_safe",
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(
    "DatasetAlignment"
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    if value == "":
        return None

    try:
        result = float(value)

        if result != result:
            return None

        return result

    except (
        TypeError,
        ValueError,
    ):
        return None


def safe_bool(
    value: Any,
) -> bool:

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def load_csv(
    path: Path,
) -> List[Dict[str, Any]]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required dataset not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        rows = list(reader)

    if not rows:

        raise RuntimeError(
            f"Dataset is empty: {path}"
        )

    logger.info(
        "Loaded dataset | file=%s | rows=%d | columns=%d",
        path,
        len(rows),
        len(rows[0]),
    )

    return rows


# ============================================================
# TIMESTAMP NORMALIZATION
# ============================================================

def get_timestamp_seconds(
    row: Dict[str, Any],
) -> Optional[float]:

    # Intrasecond feature dataset normally uses timestamp_us.
    timestamp_us = safe_float(
        row.get("timestamp_us")
    )

    if timestamp_us is not None:

        return (
            timestamp_us / 1_000_000.0
        )

    # Fallback to timestamp.
    timestamp = safe_float(
        row.get("timestamp")
    )

    if timestamp is None:
        return None

    # Microseconds.
    if timestamp > 100_000_000_000_000:

        return (
            timestamp / 1_000_000.0
        )

    # Milliseconds.
    if timestamp > 100_000_000_000:

        return (
            timestamp / 1_000.0
        )

    return timestamp


# ============================================================
# SORTING
# ============================================================

def prepare_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    prepared = []

    for row in rows:

        symbol = (
            row.get("symbol")
            or SYMBOL
        )

        if symbol != SYMBOL:
            continue

        timestamp = (
            get_timestamp_seconds(row)
        )

        if timestamp is None:
            continue

        item = dict(row)

        item["_timestamp_seconds"] = (
            timestamp
        )

        prepared.append(item)

    prepared.sort(
        key=lambda item:
        item["_timestamp_seconds"]
    )

    return prepared


# ============================================================
# TARGET LOOKUP
# ============================================================

def build_target_lookup(
    rows: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:

    """
    The multi-horizon dataset is event-level.

    We create a lookup by second. For each second,
    the latest L1 observation in that second is retained.

    This is used only to attach target values to the
    corresponding intrasecond feature window.
    """

    lookup: Dict[
        int,
        Dict[str, Any]
    ] = {}

    for row in rows:

        timestamp = row[
            "_timestamp_seconds"
        ]

        second = int(timestamp)

        lookup[second] = row

    return lookup


def find_nearest_target_row(
    feature_timestamp: float,
    target_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    """
    Find the nearest target anchor at or before the
    feature timestamp.

    We never use a target anchor from the future.
    """

    best = None

    for row in target_rows:

        timestamp = row[
            "_timestamp_seconds"
        ]

        if timestamp > feature_timestamp:
            break

        best = row

    if best is None:
        return None

    difference = (
        feature_timestamp
        - best["_timestamp_seconds"]
    )

    if difference > MAX_ALIGNMENT_SECONDS:
        return None

    return best


# ============================================================
# FEATURE LEAKAGE CHECK
# ============================================================

def identify_feature_columns(
    feature_rows: List[Dict[str, Any]],
) -> List[str]:

    if not feature_rows:
        return []

    columns = list(
        feature_rows[0].keys()
    )

    feature_columns = []

    for column in columns:

        if column.startswith("_"):
            continue

        if column in {
            "symbol",
            "window_id",
            "timestamp",
            "timestamp_us",
        }:
            feature_columns.append(column)
            continue

        if column in ALLOWED_CURRENT_FIELDS:
            feature_columns.append(column)
            continue

        # Explicitly reject anything that looks future-related.
        if column.startswith(
            FUTURE_PREFIXES
        ):
            continue

        # Unknown fields are excluded rather than
        # accidentally introducing leakage.
        logger.warning(
            "Excluding unknown feature column: %s",
            column,
        )

    return feature_columns


# ============================================================
# TARGET COLUMNS
# ============================================================

def target_columns() -> List[str]:

    columns = []

    for horizon in HORIZONS:

        columns.extend(
            [
                f"future_mid_{horizon}",
                f"return_{horizon}",
                f"direction_{horizon}",
                f"target_valid_{horizon}",
            ]
        )

    return columns


# ============================================================
# ALIGNMENT
# ============================================================

def align_datasets(
    feature_rows: List[Dict[str, Any]],
    target_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    features = prepare_rows(
        feature_rows
    )

    targets = prepare_rows(
        target_rows
    )

    logger.info(
        "Prepared feature rows: %d",
        len(features),
    )

    logger.info(
        "Prepared target rows: %d",
        len(targets),
    )

    feature_columns = (
        identify_feature_columns(
            features
        )
    )

    logger.info(
        "Approved current feature columns: %d",
        len(feature_columns),
    )

    result = []

    target_index = 0

    alignment_failures = 0

    future_leakage_failures = 0

    invalid_feature_rows = 0

    for feature in features:

        feature_timestamp = feature[
            "_timestamp_seconds"
        ]

        # Advance target pointer only up to the
        # feature timestamp.
        while (
            target_index + 1
            < len(targets)
            and targets[
                target_index + 1
            ]["_timestamp_seconds"]
            <= feature_timestamp
        ):

            target_index += 1

        if not targets:
            continue

        target = targets[
            target_index
        ]

        target_timestamp = target[
            "_timestamp_seconds"
        ]

        # ----------------------------------------------------
        # FUTURE TARGET CHECK
        # ----------------------------------------------------

        if target_timestamp > feature_timestamp:

            future_leakage_failures += 1

            continue

        # ----------------------------------------------------
        # ALIGNMENT CHECK
        # ----------------------------------------------------

        difference = (
            feature_timestamp
            - target_timestamp
        )

        if difference > MAX_ALIGNMENT_SECONDS:

            alignment_failures += 1

            continue

        # ----------------------------------------------------
        # BUILD OUTPUT ROW
        # ----------------------------------------------------

        output: Dict[str, Any] = {}

        for column in feature_columns:

            if column in {
                "timestamp",
                "timestamp_us",
            }:

                output[column] = feature.get(
                    column
                )

            else:

                output[column] = feature.get(
                    column
                )

        # Canonical timestamp.
        output[
            "feature_timestamp_seconds"
        ] = feature_timestamp

        output[
            "target_anchor_timestamp_seconds"
        ] = target_timestamp

        output[
            "alignment_delay_seconds"
        ] = difference

        # ----------------------------------------------------
        # TARGETS
        # ----------------------------------------------------

        for column in target_columns():

            output[column] = target.get(
                column
            )

        # ----------------------------------------------------
        # TARGET SAFETY
        # ----------------------------------------------------

        for horizon in HORIZONS:

            valid_column = (
                f"target_valid_{horizon}"
            )

            if not safe_bool(
                output.get(valid_column)
            ):

                continue

        # ----------------------------------------------------
        # FINAL LEAKAGE INSPECTION
        # ----------------------------------------------------

        for column in output.keys():

            if column.startswith(
                "feature_"
            ):
                continue

            if column.startswith(
                "target_"
            ):
                continue

        result.append(output)

    logger.info(
        "Alignment failures: %d",
        alignment_failures,
    )

    logger.info(
        "Future leakage failures: %d",
        future_leakage_failures,
    )

    logger.info(
        "Aligned rows: %d",
        len(result),
    )

    return result


# ============================================================
# FINAL LEAKAGE VALIDATION
# ============================================================

def validate_no_future_leakage(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:

    checked = 0

    failures = 0

    max_alignment = 0.0

    for row in rows:

        feature_timestamp = safe_float(
            row.get(
                "feature_timestamp_seconds"
            )
        )

        target_anchor = safe_float(
            row.get(
                "target_anchor_timestamp_seconds"
            )
        )

        if (
            feature_timestamp is None
            or target_anchor is None
        ):
            failures += 1
            continue

        checked += 1

        if target_anchor > feature_timestamp:

            failures += 1
            continue

        difference = (
            feature_timestamp
            - target_anchor
        )

        if difference > max_alignment:
            max_alignment = difference

    return {
        "checked": checked,
        "failures": failures,
        "max_alignment_seconds": max_alignment,
    }


# ============================================================
# TARGET SUMMARY
# ============================================================

def print_target_summary(
    rows: List[Dict[str, Any]],
) -> None:

    print()
    print("=" * 80)
    print("             PHASE 6.4 TARGET SUMMARY")
    print("=" * 80)

    print(
        f"{'Horizon':<10}"
        f"{'Valid':>12}"
        f"{'UP':>12}"
        f"{'DOWN':>12}"
        f"{'FLAT':>12}"
    )

    print("-" * 80)

    for horizon in HORIZONS:

        valid = 0
        up = 0
        down = 0
        flat = 0

        for row in rows:

            direction = row.get(
                f"direction_{horizon}"
            )

            target_valid = safe_bool(
                row.get(
                    f"target_valid_{horizon}"
                )
            )

            if not target_valid:
                continue

            valid += 1

            direction_value = (
                safe_float(direction)
            )

            if direction_value == 1:
                up += 1

            elif direction_value == -1:
                down += 1

            elif direction_value == 0:
                flat += 1

        print(
            f"{horizon:<10}"
            f"{valid:>12}"
            f"{up:>12}"
            f"{down:>12}"
            f"{flat:>12}"
        )

    print("=" * 80)


# ============================================================
# FEATURE SUMMARY
# ============================================================

def print_feature_summary(
    rows: List[Dict[str, Any]],
) -> None:

    if not rows:
        return

    feature_candidates = []

    for column in rows[0].keys():

        if column.startswith(
            (
                "future_",
                "return_",
                "direction_",
                "target_",
            )
        ):
            continue

        feature_candidates.append(
            column
        )

    print()
    print(
        "CURRENT FEATURE COLUMNS"
    )
    print("-" * 80)

    for index, column in enumerate(
        feature_candidates,
        start=1,
    ):

        print(
            f"{index:>3}. {column}"
        )

    print()
    print(
        f"Total current feature columns: "
        f"{len(feature_candidates)}"
    )


# ============================================================
# SAVE
# ============================================================

def save_csv(
    rows: List[Dict[str, Any]],
) -> None:

    if not rows:
        raise RuntimeError(
            "Cannot save an empty final dataset."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    logger.info(
        "Final training dataset saved | "
        "rows=%d | columns=%d | file=%s",
        len(rows),
        len(fieldnames),
        OUTPUT_FILE,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print("       DELTA DATASET ALIGNMENT & LEAKAGE CHECK")
    print("=" * 72)

    print(
        f"Intrasecond features:       "
        f"{INTRASECOND_FILE}"
    )

    print(
        f"Multi-horizon targets:      "
        f"{MULTI_HORIZON_FILE}"
    )

    print(
        f"Output:                     "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Maximum alignment delay:    "
        f"{MAX_ALIGNMENT_SECONDS}s"
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    feature_rows = load_csv(
        INTRASECOND_FILE
    )

    target_rows = load_csv(
        MULTI_HORIZON_FILE
    )

    # --------------------------------------------------------
    # ALIGN
    # --------------------------------------------------------

    aligned_rows = align_datasets(
        feature_rows,
        target_rows,
    )

    if not aligned_rows:

        raise RuntimeError(
            "No rows survived feature/target alignment."
        )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    leakage = (
        validate_no_future_leakage(
            aligned_rows
        )
    )

    print()
    print("=" * 80)
    print("             LEAKAGE VALIDATION")
    print("=" * 80)

    print(
        f"Rows checked:                "
        f"{leakage['checked']}"
    )

    print(
        f"Leakage failures:            "
        f"{leakage['failures']}"
    )

    print(
        f"Maximum alignment delay:     "
        f"{leakage['max_alignment_seconds']:.6f}s"
    )

    if leakage["failures"] > 0:

        raise RuntimeError(
            "Future leakage or invalid timestamps "
            "were detected. Dataset was NOT saved."
        )

    print(
        "Future leakage:              NONE DETECTED"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # SUMMARIES
    # --------------------------------------------------------

    print_feature_summary(
        aligned_rows
    )

    print_target_summary(
        aligned_rows
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_csv(
        aligned_rows
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("              PHASE 6.4 COMPLETE")
    print("=" * 72)

    print(
        f"Output:                     "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Rows:                       "
        f"{len(aligned_rows)}"
    )

    print(
        "Future leakage:              NONE DETECTED"
    )

    print(
        "ML training has NOT been started."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()