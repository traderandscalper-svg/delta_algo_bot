from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/ml/final_training_dataset.csv"
)

REPORT_FILE = Path(
    "data/ml/dataset_quality_report.txt"
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

# Values with an absolute return larger than this are
# flagged for inspection. This does NOT automatically
# mean they are invalid.
RETURN_OUTLIER_THRESHOLD = 0.01

# Extremely large feature values are flagged.
EXTREME_ABS_VALUE = 1e12

# A feature is considered constant if its number of
# unique non-null values is <= 1.
CONSTANT_FEATURE_LIMIT = 1

# Near-constant threshold.
NEAR_CONSTANT_RATIO = 0.995


# ============================================================
# HELPERS
# ============================================================

def safe_float(value: Any) -> Optional[float]:

    if value is None:
        return None

    if value == "":
        return None

    try:
        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError):

        return None


def is_missing(value: Any) -> bool:

    if value is None:
        return True

    if str(value).strip() == "":
        return True

    return False


def format_pct(
    numerator: int,
    denominator: int,
) -> str:

    if denominator == 0:
        return "0.00%"

    return (
        f"{100.0 * numerator / denominator:.2f}%"
    )


def percentile(
    values: List[float],
    percentage: float,
) -> Optional[float]:

    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        percentage
        / 100.0
        * (len(ordered) - 1)
    )

    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


def read_csv() -> List[Dict[str, str]]:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames is None:

            raise RuntimeError(
                "CSV has no header."
            )

        rows = list(reader)

    if not rows:

        raise RuntimeError(
            "CSV contains no data rows."
        )

    return rows


# ============================================================
# TIMESTAMP DETECTION
# ============================================================

def get_timestamp(
    row: Dict[str, str],
) -> Optional[float]:

    candidates = (
        "feature_timestamp_seconds",
        "timestamp_seconds",
        "timestamp_us",
    )

    for field in candidates:

        value = safe_float(
            row.get(field)
        )

        if value is None:
            continue

        if field.endswith("_us"):

            return value / 1_000_000.0

        return value

    return None


# ============================================================
# BASIC DATASET AUDIT
# ============================================================

def audit_basic_structure(
    rows: List[Dict[str, str]],
    columns: List[str],
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    result["rows"] = len(rows)

    result["columns"] = len(columns)

    result["expected_symbol_rows"] = 0

    result["unexpected_symbol_rows"] = 0

    result["missing_values"] = {}

    result["duplicate_rows"] = 0

    result["duplicate_timestamps"] = 0

    result["timestamp_missing"] = 0

    result["timestamp_out_of_order"] = 0

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    for column in columns:

        missing = sum(
            1
            for row in rows
            if is_missing(row.get(column))
        )

        result[
            "missing_values"
        ][column] = missing

    # --------------------------------------------------------
    # Duplicate complete rows
    # --------------------------------------------------------

    seen_rows = set()

    duplicate_rows = 0

    for row in rows:

        signature = tuple(
            row.get(column, "")
            for column in columns
        )

        if signature in seen_rows:

            duplicate_rows += 1

        else:

            seen_rows.add(signature)

    result[
        "duplicate_rows"
    ] = duplicate_rows

    # --------------------------------------------------------
    # Symbols
    # --------------------------------------------------------

    for row in rows:

        symbol = row.get(
            "symbol",
            "",
        )

        if symbol == SYMBOL:

            result[
                "expected_symbol_rows"
            ] += 1

        else:

            result[
                "unexpected_symbol_rows"
            ] += 1

    # --------------------------------------------------------
    # Timestamps
    # --------------------------------------------------------

    timestamps = []

    for row in rows:

        timestamp = get_timestamp(row)

        if timestamp is None:

            result[
                "timestamp_missing"
            ] += 1

        else:

            timestamps.append(timestamp)

    previous = None

    for timestamp in timestamps:

        if (
            previous is not None
            and timestamp < previous
        ):

            result[
                "timestamp_out_of_order"
            ] += 1

        previous = timestamp

    result[
        "duplicate_timestamps"
    ] = (
        len(timestamps)
        - len(set(timestamps))
    )

    if timestamps:

        result[
            "first_timestamp"
        ] = min(timestamps)

        result[
            "last_timestamp"
        ] = max(timestamps)

        result[
            "duration_seconds"
        ] = (
            max(timestamps)
            - min(timestamps)
        )

    else:

        result[
            "first_timestamp"
        ] = None

        result[
            "last_timestamp"
        ] = None

        result[
            "duration_seconds"
        ] = None

    return result


# ============================================================
# NUMERIC FEATURE AUDIT
# ============================================================

def numeric_columns(
    rows: List[Dict[str, str]],
    columns: List[str],
) -> List[str]:

    result = []

    for column in columns:

        numeric_count = 0

        nonempty_count = 0

        for row in rows:

            value = row.get(column)

            if is_missing(value):
                continue

            nonempty_count += 1

            if safe_float(value) is not None:

                numeric_count += 1

        if (
            nonempty_count > 0
            and numeric_count == nonempty_count
        ):

            result.append(column)

    return result


def audit_numeric_features(
    rows: List[Dict[str, str]],
    columns: List[str],
) -> Dict[str, Any]:

    result = {}

    numeric = numeric_columns(
        rows,
        columns,
    )

    result[
        "numeric_columns"
    ] = numeric

    result[
        "constant_columns"
    ] = []

    result[
        "near_constant_columns"
    ] = []

    result[
        "extreme_values"
    ] = {}

    result[
        "statistics"
    ] = {}

    for column in numeric:

        values = []

        extreme_count = 0

        for row in rows:

            value = safe_float(
                row.get(column)
            )

            if value is None:
                continue

            values.append(value)

            if abs(value) > EXTREME_ABS_VALUE:

                extreme_count += 1

        if not values:
            continue

        unique_count = len(
            set(values)
        )

        if (
            unique_count
            <= CONSTANT_FEATURE_LIMIT
        ):

            result[
                "constant_columns"
            ].append(column)

        most_common_count = Counter(
            values
        ).most_common(1)[0][1]

        dominant_ratio = (
            most_common_count
            / len(values)
        )

        if (
            dominant_ratio
            >= NEAR_CONSTANT_RATIO
            and unique_count > 1
        ):

            result[
                "near_constant_columns"
            ].append(
                (
                    column,
                    dominant_ratio,
                )
            )

        if extreme_count > 0:

            result[
                "extreme_values"
            ][column] = extreme_count

        result[
            "statistics"
        ][column] = {
            "count": len(values),
            "unique": unique_count,
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "q01": percentile(values, 1),
            "q05": percentile(values, 5),
            "q25": percentile(values, 25),
            "q75": percentile(values, 75),
            "q95": percentile(values, 95),
            "q99": percentile(values, 99),
        }

    return result


# ============================================================
# MARKET MICROSTRUCTURE AUDIT
# ============================================================

def audit_market_structure(
    rows: List[Dict[str, str]],
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    spread_negative = 0
    spread_zero = 0

    bid_above_ask = 0

    invalid_imbalance = 0

    negative_volume = 0
    negative_trade_count = 0

    mid_outside_book = 0

    l1_gap_rows = 0
    large_gap_rows = 0

    for row in rows:

        bid = safe_float(
            row.get("bid_close")
        )

        ask = safe_float(
            row.get("ask_close")
        )

        mid = safe_float(
            row.get("mid_close")
        )

        spread = safe_float(
            row.get("spread_close")
        )

        imbalance = safe_float(
            row.get("imbalance_mean")
        )

        trade_volume = safe_float(
            row.get("trade_volume")
        )

        trade_count = safe_float(
            row.get("trade_count")
        )

        gap = safe_float(
            row.get("l1_gap_seconds")
        )

        # ----------------------------------------------------
        # Bid/ask
        # ----------------------------------------------------

        if (
            bid is not None
            and ask is not None
        ):

            if bid > ask:

                bid_above_ask += 1

        # ----------------------------------------------------
        # Spread
        # ----------------------------------------------------

        if spread is not None:

            if spread < 0:

                spread_negative += 1

            elif spread == 0:

                spread_zero += 1

        # ----------------------------------------------------
        # Mid
        # ----------------------------------------------------

        if (
            bid is not None
            and ask is not None
            and mid is not None
        ):

            if (
                mid < bid
                or mid > ask
            ):

                mid_outside_book += 1

        # ----------------------------------------------------
        # Imbalance
        # ----------------------------------------------------

        if imbalance is not None:

            if (
                imbalance < -1.000001
                or imbalance > 1.000001
            ):

                invalid_imbalance += 1

        # ----------------------------------------------------
        # Trade volume
        # ----------------------------------------------------

        if (
            trade_volume is not None
            and trade_volume < 0
        ):

            negative_volume += 1

        # ----------------------------------------------------
        # Trade count
        # ----------------------------------------------------

        if (
            trade_count is not None
            and trade_count < 0
        ):

            negative_trade_count += 1

        # ----------------------------------------------------
        # Data gaps
        # ----------------------------------------------------

        if (
            gap is not None
            and gap > 0
        ):

            l1_gap_rows += 1

            if gap > 5.0:

                large_gap_rows += 1

    result[
        "spread_negative"
    ] = spread_negative

    result[
        "spread_zero"
    ] = spread_zero

    result[
        "bid_above_ask"
    ] = bid_above_ask

    result[
        "mid_outside_book"
    ] = mid_outside_book

    result[
        "invalid_imbalance"
    ] = invalid_imbalance

    result[
        "negative_trade_volume"
    ] = negative_volume

    result[
        "negative_trade_count"
    ] = negative_trade_count

    result[
        "rows_with_l1_gap"
    ] = l1_gap_rows

    result[
        "rows_with_gap_over_5s"
    ] = large_gap_rows

    return result


# ============================================================
# RETURN OUTLIER AUDIT
# ============================================================

def audit_returns(
    rows: List[Dict[str, str]],
) -> Dict[str, Any]:

    result = {}

    for horizon in HORIZONS:

        column = f"return_{horizon}"

        values = []

        outliers = []

        for index, row in enumerate(rows):

            value = safe_float(
                row.get(column)
            )

            if value is None:
                continue

            values.append(value)

            if abs(value) > RETURN_OUTLIER_THRESHOLD:

                outliers.append(
                    {
                        "row": index + 2,
                        "value": value,
                    }
                )

        if not values:

            result[horizon] = {
                "count": 0,
                "outliers": [],
            }

            continue

        result[horizon] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "q01": percentile(values, 1),
            "q05": percentile(values, 5),
            "q95": percentile(values, 95),
            "q99": percentile(values, 99),
            "outliers": outliers,
        }

    return result


# ============================================================
# TARGET VALIDITY AUDIT
# ============================================================

def audit_targets(
    rows: List[Dict[str, str]],
) -> Dict[str, Any]:

    result = {}

    for horizon in HORIZONS:

        valid_column = (
            f"target_valid_{horizon}"
        )

        direction_column = (
            f"direction_{horizon}"
        )

        valid = 0
        invalid = 0

        up = 0
        down = 0
        flat = 0
        unknown_direction = 0

        for row in rows:

            valid_value = (
                str(
                    row.get(
                        valid_column,
                        "",
                    )
                )
                .strip()
                .lower()
            )

            is_valid = (
                valid_value
                in {
                    "true",
                    "1",
                    "yes",
                }
            )

            if not is_valid:

                invalid += 1
                continue

            valid += 1

            direction = safe_float(
                row.get(
                    direction_column
                )
            )

            if direction == 1:

                up += 1

            elif direction == -1:

                down += 1

            elif direction == 0:

                flat += 1

            else:

                unknown_direction += 1

        result[horizon] = {
            "valid": valid,
            "invalid": invalid,
            "up": up,
            "down": down,
            "flat": flat,
            "unknown_direction": (
                unknown_direction
            ),
        }

    return result


# ============================================================
# FEATURE / TARGET SEPARATION
# ============================================================

def audit_column_roles(
    columns: List[str],
) -> Dict[str, Any]:

    feature_columns = []
    target_columns = []
    metadata_columns = []
    suspicious_columns = []

    for column in columns:

        lower = column.lower()

        if (
            lower.startswith("future_")
            or lower.startswith("return_")
            or lower.startswith("direction_")
            or lower.startswith("target_")
        ):

            target_columns.append(
                column
            )

        elif (
            lower in {
                "symbol",
                "window_id",
                "timestamp",
                "timestamp_us",
                "feature_timestamp_seconds",
                "alignment_delay_seconds",
                "target_anchor_timestamp_seconds",
            }
        ):

            metadata_columns.append(
                column
            )

        else:

            feature_columns.append(
                column
            )

        # Anything with obvious future terminology
        # that is not in the target group is flagged.
        if (
            "future" in lower
            and column not in target_columns
        ):

            suspicious_columns.append(
                column
            )

    return {
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "metadata_columns": metadata_columns,
        "suspicious_columns": suspicious_columns,
    }


# ============================================================
# TIME-SPLIT READINESS
# ============================================================

def audit_time_split(
    rows: List[Dict[str, str]],
) -> Dict[str, Any]:

    timestamps = []

    for row in rows:

        timestamp = get_timestamp(row)

        if timestamp is not None:

            timestamps.append(timestamp)

    timestamps.sort()

    if not timestamps:

        return {
            "available": False
        }

    count = len(timestamps)

    train_end = int(
        count * 0.70
    )

    validation_end = int(
        count * 0.85
    )

    train_end = min(
        max(train_end, 1),
        count,
    )

    validation_end = min(
        max(validation_end, train_end),
        count,
    )

    return {
        "available": True,
        "rows": count,
        "train_rows": train_end,
        "validation_rows": (
            validation_end - train_end
        ),
        "test_rows": (
            count - validation_end
        ),
        "train_start": timestamps[0],
        "train_end": (
            timestamps[train_end - 1]
        ),
        "validation_start": (
            timestamps[train_end]
            if train_end < count
            else None
        ),
        "validation_end": (
            timestamps[validation_end - 1]
            if validation_end > 0
            else None
        ),
        "test_start": (
            timestamps[validation_end]
            if validation_end < count
            else None
        ),
        "test_end": timestamps[-1],
    }


# ============================================================
# WRITE REPORT
# ============================================================

def build_report(
    rows: List[Dict[str, str]],
    columns: List[str],
) -> str:

    basic = audit_basic_structure(
        rows,
        columns,
    )

    numeric = audit_numeric_features(
        rows,
        columns,
    )

    market = audit_market_structure(
        rows
    )

    returns = audit_returns(
        rows
    )

    targets = audit_targets(
        rows
    )

    roles = audit_column_roles(
        columns
    )

    split = audit_time_split(
        rows
    )

    lines: List[str] = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "DELTA ALGORITHMIC TRADING BOT"
    )

    lines.append(
        "PHASE 6.5 DATASET QUALITY AUDIT"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    # --------------------------------------------------------
    # Basic
    # --------------------------------------------------------

    lines.append(
        "1. BASIC DATASET STRUCTURE"
    )

    lines.append("-" * 80)

    lines.append(
        f"Input file:                 {INPUT_FILE}"
    )

    lines.append(
        f"Rows:                       {basic['rows']}"
    )

    lines.append(
        f"Columns:                    {basic['columns']}"
    )

    lines.append(
        f"Expected symbol rows:       "
        f"{basic['expected_symbol_rows']}"
    )

    lines.append(
        f"Unexpected symbol rows:     "
        f"{basic['unexpected_symbol_rows']}"
    )

    lines.append(
        f"Duplicate complete rows:    "
        f"{basic['duplicate_rows']}"
    )

    lines.append(
        f"Duplicate timestamps:       "
        f"{basic['duplicate_timestamps']}"
    )

    lines.append(
        f"Missing timestamps:          "
        f"{basic['timestamp_missing']}"
    )

    lines.append(
        f"Out-of-order timestamps:     "
        f"{basic['timestamp_out_of_order']}"
    )

    if basic["duration_seconds"] is not None:

        lines.append(
            f"Time span:                  "
            f"{basic['duration_seconds']:.3f} sec"
        )

        lines.append(
            f"Time span:                  "
            f"{basic['duration_seconds'] / 60:.2f} min"
        )

    lines.append("")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    lines.append(
        "2. MISSING VALUES"
    )

    lines.append("-" * 80)

    missing_items = sorted(
        basic["missing_values"].items()
    )

    missing_found = False

    for column, count in missing_items:

        if count > 0:

            missing_found = True

            lines.append(
                f"{column:<40}"
                f"{count:>8} "
                f"({format_pct(count, len(rows))})"
            )

    if not missing_found:

        lines.append(
            "No missing values detected."
        )

    lines.append("")

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    lines.append(
        "3. NUMERIC FEATURE AUDIT"
    )

    lines.append("-" * 80)

    lines.append(
        f"Numeric columns:            "
        f"{len(numeric['numeric_columns'])}"
    )

    lines.append(
        f"Constant columns:           "
        f"{len(numeric['constant_columns'])}"
    )

    for column in numeric[
        "constant_columns"
    ]:

        lines.append(
            f"  CONSTANT: {column}"
        )

    lines.append(
        f"Near-constant columns:      "
        f"{len(numeric['near_constant_columns'])}"
    )

    for (
        column,
        ratio,
    ) in numeric[
        "near_constant_columns"
    ]:

        lines.append(
            f"  NEAR CONSTANT: {column} "
            f"({ratio:.2%})"
        )

    if numeric["extreme_values"]:

        lines.append(
            "Extreme-value fields:"
        )

        for (
            column,
            count,
        ) in numeric[
            "extreme_values"
        ].items():

            lines.append(
                f"  {column}: {count}"
            )

    else:

        lines.append(
            "No extreme numeric values detected."
        )

    lines.append("")

    # --------------------------------------------------------
    # Market structure
    # --------------------------------------------------------

    lines.append(
        "4. MARKET MICROSTRUCTURE AUDIT"
    )

    lines.append("-" * 80)

    lines.append(
        f"Negative spreads:           "
        f"{market['spread_negative']}"
    )

    lines.append(
        f"Zero spreads:               "
        f"{market['spread_zero']}"
    )

    lines.append(
        f"Bid above ask:              "
        f"{market['bid_above_ask']}"
    )

    lines.append(
        f"Mid outside bid/ask:        "
        f"{market['mid_outside_book']}"
    )

    lines.append(
        f"Invalid imbalance:          "
        f"{market['invalid_imbalance']}"
    )

    lines.append(
        f"Negative trade volume:      "
        f"{market['negative_trade_volume']}"
    )

    lines.append(
        f"Negative trade count:       "
        f"{market['negative_trade_count']}"
    )

    lines.append(
        f"Rows with L1 gap:            "
        f"{market['rows_with_l1_gap']}"
    )

    lines.append(
        f"Rows with gap >5s:          "
        f"{market['rows_with_gap_over_5s']}"
    )

    lines.append("")

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    lines.append(
        "5. RETURN DISTRIBUTION"
    )

    lines.append("-" * 80)

    for horizon in HORIZONS:

        data = returns[horizon]

        if data["count"] == 0:

            lines.append(
                f"{horizon}: no return data"
            )

            continue

        lines.append(
            f"{horizon}: "
            f"count={data['count']} | "
            f"min={data['min']:.8f} | "
            f"max={data['max']:.8f} | "
            f"mean={data['mean']:.8f} | "
            f"median={data['median']:.8f}"
        )

        lines.append(
            f"      Q01={data['q01']:.8f} | "
            f"Q05={data['q05']:.8f} | "
            f"Q95={data['q95']:.8f} | "
            f"Q99={data['q99']:.8f}"
        )

        if data["outliers"]:

            lines.append(
                f"      Flagged absolute returns "
                f">{RETURN_OUTLIER_THRESHOLD:.4f}: "
                f"{len(data['outliers'])}"
            )

    lines.append("")

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    lines.append(
        "6. TARGET VALIDITY"
    )

    lines.append("-" * 80)

    lines.append(
        f"{'Horizon':<10}"
        f"{'Valid':>10}"
        f"{'Invalid':>10}"
        f"{'UP':>10}"
        f"{'DOWN':>10}"
        f"{'FLAT':>10}"
        f"{'Unknown':>10}"
    )

    for horizon in HORIZONS:

        data = targets[horizon]

        lines.append(
            f"{horizon:<10}"
            f"{data['valid']:>10}"
            f"{data['invalid']:>10}"
            f"{data['up']:>10}"
            f"{data['down']:>10}"
            f"{data['flat']:>10}"
            f"{data['unknown_direction']:>10}"
        )

    lines.append("")

    # --------------------------------------------------------
    # Column roles
    # --------------------------------------------------------

    lines.append(
        "7. FEATURE / TARGET SEPARATION"
    )

    lines.append("-" * 80)

    lines.append(
        f"Feature columns:            "
        f"{len(roles['feature_columns'])}"
    )

    lines.append(
        f"Target columns:             "
        f"{len(roles['target_columns'])}"
    )

    lines.append(
        f"Metadata columns:           "
        f"{len(roles['metadata_columns'])}"
    )

    if roles["suspicious_columns"]:

        lines.append(
            "Suspicious future-related columns:"
        )

        for column in roles[
            "suspicious_columns"
        ]:

            lines.append(
                f"  {column}"
            )

    else:

        lines.append(
            "Suspicious future-related "
            "feature columns: NONE"
        )

    lines.append("")

    # --------------------------------------------------------
    # Time split
    # --------------------------------------------------------

    lines.append(
        "8. CHRONOLOGICAL SPLIT READINESS"
    )

    lines.append("-" * 80)

    if split["available"]:

        lines.append(
            f"Rows available:             "
            f"{split['rows']}"
        )

        lines.append(
            f"Training rows (70%):        "
            f"{split['train_rows']}"
        )

        lines.append(
            f"Validation rows (15%):      "
            f"{split['validation_rows']}"
        )

        lines.append(
            f"Test rows (15%):            "
            f"{split['test_rows']}"
        )

        lines.append(
            "Split method: chronological"
        )

    else:

        lines.append(
            "Unable to establish chronological split."
        )

    lines.append("")

    # --------------------------------------------------------
    # Overall verdict
    # --------------------------------------------------------

    serious_problems = []

    if basic[
        "unexpected_symbol_rows"
    ] > 0:

        serious_problems.append(
            "unexpected symbols"
        )

    if basic[
        "duplicate_timestamps"
    ] > 0:

        serious_problems.append(
            "duplicate timestamps"
        )

    if basic[
        "timestamp_out_of_order"
    ] > 0:

        serious_problems.append(
            "out-of-order timestamps"
        )

    if market[
        "spread_negative"
    ] > 0:

        serious_problems.append(
            "negative spreads"
        )

    if market[
        "bid_above_ask"
    ] > 0:

        serious_problems.append(
            "bid above ask"
        )

    if market[
        "mid_outside_book"
    ] > 0:

        serious_problems.append(
            "mid outside bid/ask"
        )

    if market[
        "invalid_imbalance"
    ] > 0:

        serious_problems.append(
            "invalid order-book imbalance"
        )

    if roles[
        "suspicious_columns"
    ]:

        serious_problems.append(
            "suspicious future-related feature"
        )

    lines.append(
        "=" * 80
    )

    lines.append(
        "9. OVERALL DATASET STATUS"
    )

    lines.append(
        "=" * 80
    )

    if serious_problems:

        lines.append(
            "STATUS: REVIEW REQUIRED"
        )

        lines.append(
            "Potential issues:"
        )

        for problem in serious_problems:

            lines.append(
                f"  - {problem}"
            )

    else:

        lines.append(
            "STATUS: STRUCTURALLY CLEAN"
        )

        lines.append(
            "No major structural market-data "
            "violations detected."
        )

    lines.append("")

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "This audit does NOT determine whether a "
        "trading strategy is profitable."
    )

    lines.append(
        "It only evaluates dataset integrity, "
        "feature quality, target validity, "
        "and leakage risk."
    )

    lines.append("")

    lines.append(
        "ML TRAINING: NOT STARTED"
    )

    lines.append(
        "LIVE TRADING: NOT ENABLED"
    )

    lines.append(
        "=" * 80
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print(
        "       DELTA PHASE 6.5 DATASET QUALITY AUDIT"
    )
    print("=" * 72)

    print(
        f"Input:  {INPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )

    print()

    rows = read_csv()

    if not rows:

        raise RuntimeError(
            "No rows found."
        )

    columns = list(
        rows[0].keys()
    )

    print(
        f"Rows:    {len(rows)}"
    )

    print(
        f"Columns: {len(columns)}"
    )

    print()

    report = build_report(
        rows,
        columns,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    basic = audit_basic_structure(
        rows,
        columns,
    )

    market = audit_market_structure(
        rows
    )

    roles = audit_column_roles(
        columns
    )

    print(
        "=" * 80
    )

    print(
        "PHASE 6.5 AUDIT SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Rows:                       "
        f"{len(rows)}"
    )

    print(
        f"Columns:                    "
        f"{len(columns)}"
    )

    print(
        f"Duplicate rows:             "
        f"{basic['duplicate_rows']}"
    )

    print(
        f"Duplicate timestamps:       "
        f"{basic['duplicate_timestamps']}"
    )

    print(
        f"Missing timestamps:         "
        f"{basic['timestamp_missing']}"
    )

    print(
        f"Out-of-order timestamps:     "
        f"{basic['timestamp_out_of_order']}"
    )

    print(
        f"Unexpected symbols:         "
        f"{basic['unexpected_symbol_rows']}"
    )

    print(
        f"Negative spreads:           "
        f"{market['spread_negative']}"
    )

    print(
        f"Bid above ask:              "
        f"{market['bid_above_ask']}"
    )

    print(
        f"Mid outside book:           "
        f"{market['mid_outside_book']}"
    )

    print(
        f"Invalid imbalance:          "
        f"{market['invalid_imbalance']}"
    )

    print(
        f"Suspicious future features: "
        f"{len(roles['suspicious_columns'])}"
    )

    print()

    print(
        f"Report saved: {REPORT_FILE}"
    )

    print()

    print(
        "=" * 72
    )

    print(
        "                 PHASE 6.5 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "ML training: NOT STARTED"
    )

    print(
        "Live trading: NOT ENABLED"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()