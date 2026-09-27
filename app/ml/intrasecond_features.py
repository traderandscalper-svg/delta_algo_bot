"""
Phase 6.2 - Intrasecond Market Feature Aggregation

Converts raw Delta Exchange L1/trade events into rich 1-second
feature windows without discarding intrasecond information.

This module does NOT train ML and does NOT modify raw market data.

Primary target:
    L1 mid-price

Important:
    The target remains a future L1-mid return.
    Intrasecond movement is preserved as FEATURES.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger("IntrasecondFeatures")


DEFAULT_INPUT_DIR = Path("data/market_data")

# One-second feature windows.
WINDOW_SECONDS = 1.0

# Future target horizon.
HORIZON_SECONDS = 10.0

# Any gap larger than this makes a future target unsafe.
MAX_ALLOWED_GAP_SECONDS = 5.0


# ================================================================
# BASIC HELPERS
# ================================================================

def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def to_microseconds(value: Any) -> Optional[int]:

    try:
        value = int(float(value))
    except (TypeError, ValueError):
        return None

    if value <= 0:
        return None

    if value < 100_000_000_000:
        return value * 1_000_000

    if value < 100_000_000_000_000:
        return value * 1_000

    if value < 100_000_000_000_000_000:
        return value

    return value // 1_000


def first_numeric(*values: Any) -> Optional[float]:

    for value in values:

        result = safe_float(value)

        if result is not None:
            return result

    return None


# ================================================================
# DELTA PAYLOAD
# ================================================================

def extract_payload(message: Dict[str, Any]) -> Dict[str, Any]:

    if not isinstance(message, dict):
        return {}

    data = message.get("d")

    if isinstance(data, list):

        if data and isinstance(data[0], dict):
            return data[0]

        return {}

    if isinstance(data, dict):
        return data

    return message


def extract_symbol(
    record: Dict[str, Any],
    payload: Dict[str, Any],
) -> Optional[str]:

    candidates = [
        record.get("symbol"),
        payload.get("sy"),
        payload.get("symbol"),
    ]

    for value in candidates:

        if value is None:
            continue

        text = str(value).strip()

        if text:
            return text

    return None


def extract_timestamp(
    record: Dict[str, Any],
    payload: Dict[str, Any],
) -> Optional[int]:

    candidates = [
        record.get("exchange_timestamp"),
        payload.get("ts"),
        payload.get("t"),
        record.get("received_at_us"),
    ]

    for value in candidates:

        timestamp = to_microseconds(value)

        if timestamp is not None:
            return timestamp

    return None


# ================================================================
# NORMALIZATION
# ================================================================

def normalize_event(
    record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    if not isinstance(record, dict):
        return None

    event_type = record.get("event_type")

    if event_type not in {
        "ob_l1",
        "trades",
        "ticker",
    }:
        return None

    normalized = record.get("normalized")

    if not isinstance(normalized, dict):
        normalized = {}

    message = record.get("message")

    if not isinstance(message, dict):
        message = {}

    payload = extract_payload(message)

    symbol = extract_symbol(
        record,
        payload,
    )

    timestamp_us = extract_timestamp(
        record,
        payload,
    )

    if symbol is None or timestamp_us is None:
        return None

    event = {
        "timestamp_us": timestamp_us,
        "symbol": symbol,
        "event_type": event_type,

        "bid": None,
        "ask": None,
        "bid_size": None,
        "ask_size": None,

        "trade_price": None,
        "trade_size": None,

        "ticker_price": None,
        "mark_price": None,
    }

    if event_type == "ob_l1":

        event["bid"] = first_numeric(
            normalized.get("bid"),
            payload.get("bp"),
            payload.get("bid"),
            payload.get("bid_price"),
        )

        event["ask"] = first_numeric(
            normalized.get("ask"),
            payload.get("ap"),
            payload.get("ask"),
            payload.get("ask_price"),
        )

        event["bid_size"] = first_numeric(
            normalized.get("bid_size"),
            payload.get("bs"),
            payload.get("bid_size"),
            payload.get("bid_quantity"),
        )

        event["ask_size"] = first_numeric(
            normalized.get("ask_size"),
            payload.get("as"),
            payload.get("ask_size"),
            payload.get("ask_quantity"),
        )

    elif event_type == "trades":

        event["trade_price"] = first_numeric(
            normalized.get("price"),
            payload.get("p"),
            payload.get("price"),
            payload.get("trade_price"),
        )

        event["trade_size"] = first_numeric(
            normalized.get("size"),
            payload.get("s"),
            payload.get("size"),
            payload.get("quantity"),
            payload.get("trade_size"),
            payload.get("qty"),
            payload.get("q"),
            payload.get("volume"),
        )

    elif event_type == "ticker":

        event["ticker_price"] = first_numeric(
            normalized.get("last_price"),
            normalized.get("price"),
            payload.get("last_price"),
            payload.get("last"),
            payload.get("lp"),
        )

        event["mark_price"] = first_numeric(
            normalized.get("mark_price"),
            payload.get("mark_price"),
            payload.get("mark"),
            payload.get("m"),
        )

    if (
        event["bid"] is not None
        and event["ask"] is not None
        and event["bid"] > 0
        and event["ask"] > 0
        and event["ask"] >= event["bid"]
    ):
        event["mid_price"] = (
            event["bid"] + event["ask"]
        ) / 2.0

    else:
        event["mid_price"] = None

    return event


# ================================================================
# RAW DATA LOADING
# ================================================================

def load_events(
    input_dir: Path,
) -> List[Dict[str, Any]]:

    files = sorted(
        input_dir.glob(
            "market_data_*.jsonl"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No market_data_*.jsonl files found in {input_dir}"
        )

    events: List[Dict[str, Any]] = []

    for file_path in files:

        logger.info(
            "Reading market data | file=%s",
            file_path,
        )

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError:
                    continue

                event = normalize_event(
                    record
                )

                if event is not None:

                    event["source_file"] = (
                        str(file_path)
                    )

                    event["line_number"] = (
                        line_number
                    )

                    events.append(event)

    events.sort(
        key=lambda x: (
            x["symbol"],
            x["timestamp_us"],
            x["line_number"],
        )
    )

    logger.info(
        "Usable events loaded: %d",
        len(events),
    )

    return events


# ================================================================
# L1 GAP DETECTION
# ================================================================

def calculate_l1_gaps(
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:

    l1_by_symbol = defaultdict(list)

    for event in events:

        if (
            event["event_type"] == "ob_l1"
            and event["mid_price"] is not None
        ):
            l1_by_symbol[
                event["symbol"]
            ].append(event)

    gap_by_timestamp: Dict[
        tuple[str, int],
        float,
    ] = {}

    symbol_gap_summary = {}

    for symbol, symbol_events in l1_by_symbol.items():

        symbol_events.sort(
            key=lambda x: x["timestamp_us"]
        )

        maximum_gap = 0.0
        gap_count = 0

        for index in range(
            1,
            len(symbol_events),
        ):

            previous = (
                symbol_events[index - 1]
                ["timestamp_us"]
            )

            current = (
                symbol_events[index]
                ["timestamp_us"]
            )

            gap = (
                current - previous
            ) / 1_000_000.0

            if gap > MAX_ALLOWED_GAP_SECONDS:

                gap_count += 1
                maximum_gap = max(
                    maximum_gap,
                    gap,
                )

            gap_by_timestamp[
                (
                    symbol,
                    current,
                )
            ] = gap

        symbol_gap_summary[
            symbol
        ] = {
            "large_gap_count": gap_count,
            "maximum_gap_seconds": maximum_gap,
        }

    return {
        "gaps": gap_by_timestamp,
        "summary": symbol_gap_summary,
    }


# ================================================================
# ONE-SECOND WINDOW AGGREGATION
# ================================================================

def aggregate_symbol(
    events: List[Dict[str, Any]],
    symbol: str,
    gap_info: Dict[str, Any],
) -> pd.DataFrame:

    symbol_events = [
        event
        for event in events
        if event["symbol"] == symbol
    ]

    symbol_events.sort(
        key=lambda x: (
            x["timestamp_us"],
            x["line_number"],
        )
    )

    if not symbol_events:
        return pd.DataFrame()

    start_us = (
        symbol_events[0]["timestamp_us"]
    )

    # ------------------------------------------------------------
    # Group into one-second windows.
    # ------------------------------------------------------------

    windows: Dict[
        int,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for event in symbol_events:

        window_id = int(
            (
                event["timestamp_us"]
                - start_us
            )
            // 1_000_000
        )

        windows[
            window_id
        ].append(event)

    rows: List[Dict[str, Any]] = []

    previous_close_mid = None

    for window_id in sorted(
        windows.keys()
    ):

        window_events = windows[
            window_id
        ]

        l1 = [
            event
            for event in window_events
            if (
                event["event_type"] == "ob_l1"
                and event["mid_price"] is not None
            )
        ]

        trades = [
            event
            for event in window_events
            if (
                event["event_type"] == "trades"
                and event["trade_price"] is not None
            )
        ]

        if not l1:
            continue

        l1.sort(
            key=lambda x: x["timestamp_us"]
        )

        mids = np.asarray(
            [
                event["mid_price"]
                for event in l1
            ],
            dtype=float,
        )

        bids = np.asarray(
            [
                event["bid"]
                for event in l1
                if event["bid"] is not None
            ],
            dtype=float,
        )

        asks = np.asarray(
            [
                event["ask"]
                for event in l1
                if event["ask"] is not None
            ],
            dtype=float,
        )

        spreads = (
            asks[:min(len(bids), len(asks))]
            - bids[:min(len(bids), len(asks))]
        )

        # --------------------------------------------------------
        # L1 changes inside this second.
        # --------------------------------------------------------

        mid_changes = np.diff(
            mids
        )

        mid_change_count = int(
            np.count_nonzero(
                mid_changes
            )
        )

        bid_values = [
            event["bid"]
            for event in l1
            if event["bid"] is not None
        ]

        ask_values = [
            event["ask"]
            for event in l1
            if event["ask"] is not None
        ]

        bid_change_count = 0

        for index in range(
            1,
            len(bid_values),
        ):

            if (
                bid_values[index]
                != bid_values[index - 1]
            ):
                bid_change_count += 1

        ask_change_count = 0

        for index in range(
            1,
            len(ask_values),
        ):

            if (
                ask_values[index]
                != ask_values[index - 1]
            ):
                ask_change_count += 1

        # --------------------------------------------------------
        # Order-book imbalance.
        # --------------------------------------------------------

        imbalance_values = []

        for event in l1:

            bid_size = event[
                "bid_size"
            ]

            ask_size = event[
                "ask_size"
            ]

            if (
                bid_size is not None
                and ask_size is not None
                and (
                    bid_size
                    + ask_size
                ) > 0
            ):

                imbalance_values.append(
                    (
                        bid_size
                        - ask_size
                    )
                    / (
                        bid_size
                        + ask_size
                    )
                )

        # --------------------------------------------------------
        # Trade features.
        # --------------------------------------------------------

        trade_prices = np.asarray(
            [
                trade["trade_price"]
                for trade in trades
            ],
            dtype=float,
        )

        trade_sizes = np.asarray(
            [
                trade["trade_size"]
                for trade in trades
                if trade["trade_size"] is not None
            ],
            dtype=float,
        )

        # --------------------------------------------------------
        # Mid return.
        # --------------------------------------------------------

        first_mid = float(
            mids[0]
        )

        last_mid = float(
            mids[-1]
        )

        if first_mid > 0:

            window_return = (
                last_mid
                / first_mid
                - 1.0
            )

        else:

            window_return = 0.0

        if (
            previous_close_mid is not None
            and previous_close_mid > 0
        ):

            one_second_return = (
                last_mid
                / previous_close_mid
                - 1.0
            )

        else:

            one_second_return = 0.0

        previous_close_mid = last_mid

        # --------------------------------------------------------
        # Intrasecond volatility.
        # --------------------------------------------------------

        if len(mids) >= 2:

            log_returns = np.diff(
                np.log(mids)
            )

            intrasecond_volatility = float(
                np.std(
                    log_returns
                )
            )

        else:

            intrasecond_volatility = 0.0

        # --------------------------------------------------------
        # Data gap.
        # --------------------------------------------------------

        first_timestamp = (
            l1[0]["timestamp_us"]
        )

        current_gap = gap_info.get(
            (
                symbol,
                first_timestamp,
            ),
            0.0,
        )

        # --------------------------------------------------------
        # Construct row.
        # --------------------------------------------------------

        row = {
            "symbol": symbol,

            "window_id": window_id,

            "timestamp_us":
                first_timestamp,

            # -------------------------
            # Primary L1 target features
            # -------------------------

            "mid_open":
                first_mid,

            "mid_high":
                float(np.max(mids)),

            "mid_low":
                float(np.min(mids)),

            "mid_close":
                last_mid,

            "mid_range":
                float(
                    np.max(mids)
                    - np.min(mids)
                ),

            "mid_change":
                last_mid - first_mid,

            "mid_return":
                window_return,

            "one_second_return":
                one_second_return,

            "intrasecond_volatility":
                intrasecond_volatility,

            # -------------------------
            # L1 activity
            # -------------------------

            "l1_update_count":
                len(l1),

            "mid_change_count":
                mid_change_count,

            "bid_change_count":
                bid_change_count,

            "ask_change_count":
                ask_change_count,

            # -------------------------
            # Bid
            # -------------------------

            "bid_open":
                bid_values[0]
                if bid_values
                else np.nan,

            "bid_high":
                max(bid_values)
                if bid_values
                else np.nan,

            "bid_low":
                min(bid_values)
                if bid_values
                else np.nan,

            "bid_close":
                bid_values[-1]
                if bid_values
                else np.nan,

            # -------------------------
            # Ask
            # -------------------------

            "ask_open":
                ask_values[0]
                if ask_values
                else np.nan,

            "ask_high":
                max(ask_values)
                if ask_values
                else np.nan,

            "ask_low":
                min(ask_values)
                if ask_values
                else np.nan,

            "ask_close":
                ask_values[-1]
                if ask_values
                else np.nan,

            # -------------------------
            # Spread
            # -------------------------

            "spread_mean":
                float(np.mean(spreads))
                if len(spreads)
                else np.nan,

            "spread_min":
                float(np.min(spreads))
                if len(spreads)
                else np.nan,

            "spread_max":
                float(np.max(spreads))
                if len(spreads)
                else np.nan,

            "spread_close":
                float(spreads[-1])
                if len(spreads)
                else np.nan,

            # -------------------------
            # Order-book imbalance
            # -------------------------

            "imbalance_mean":
                float(
                    np.mean(
                        imbalance_values
                    )
                )
                if imbalance_values
                else np.nan,

            "imbalance_min":
                float(
                    np.min(
                        imbalance_values
                    )
                )
                if imbalance_values
                else np.nan,

            "imbalance_max":
                float(
                    np.max(
                        imbalance_values
                    )
                )
                if imbalance_values
                else np.nan,

            # -------------------------
            # Trades
            # -------------------------

            "trade_count":
                len(trades),

            "trade_volume":
                float(
                    np.sum(
                        trade_sizes
                    )
                )
                if len(trade_sizes)
                else 0.0,

            "trade_price_open":
                float(trade_prices[0])
                if len(trade_prices)
                else np.nan,

            "trade_price_high":
                float(np.max(trade_prices))
                if len(trade_prices)
                else np.nan,

            "trade_price_low":
                float(np.min(trade_prices))
                if len(trade_prices)
                else np.nan,

            "trade_price_close":
                float(trade_prices[-1])
                if len(trade_prices)
                else np.nan,

            "trade_price_change":
                (
                    float(trade_prices[-1])
                    - float(trade_prices[0])
                )
                if len(trade_prices)
                else 0.0,

            # -------------------------
            # Data integrity
            # -------------------------

            "l1_gap_seconds":
                current_gap,

            "future_target_safe":
                current_gap
                <= MAX_ALLOWED_GAP_SECONDS,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ================================================================
# MULTI-SYMBOL AGGREGATION
# ================================================================

def build_features(
    events: List[Dict[str, Any]],
) -> pd.DataFrame:

    gap_data = calculate_l1_gaps(
        events
    )

    gap_info = gap_data[
        "gaps"
    ]

    symbols = sorted(
        {
            event["symbol"]
            for event in events
        }
    )

    frames = []

    for symbol in symbols:

        logger.info(
            "Aggregating 1-second features | "
            "symbol=%s",
            symbol,
        )

        frame = aggregate_symbol(
            events,
            symbol,
            gap_info,
        )

        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result.sort_values(
        [
            "symbol",
            "timestamp_us",
        ],
        inplace=True,
    )

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return result


# ================================================================
# FUTURE TARGET
# ================================================================

def add_future_target(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    if dataframe.empty:
        return dataframe

    output_frames = []

    horizon_us = int(
        HORIZON_SECONDS
        * 1_000_000
    )

    for symbol, group in dataframe.groupby(
        "symbol",
        sort=False,
    ):

        group = group.sort_values(
            "timestamp_us"
        ).copy()

        timestamps = (
            group[
                "timestamp_us"
            ]
            .astype(np.int64)
            .to_numpy()
        )

        prices = (
            group[
                "mid_close"
            ]
            .astype(float)
            .to_numpy()
        )

        future_returns = np.full(
            len(group),
            np.nan,
            dtype=float,
        )

        future_safe = np.zeros(
            len(group),
            dtype=bool,
        )

        future_index = 1

        for index in range(
            len(group)
        ):

            target_time = (
                timestamps[index]
                + horizon_us
            )

            if future_index <= index:
                future_index = index + 1

            while (
                future_index < len(group)
                and timestamps[
                    future_index
                ] < target_time
            ):

                future_index += 1

            if future_index >= len(group):
                continue

            current_price = prices[
                index
            ]

            future_price = prices[
                future_index
            ]

            if (
                not math.isfinite(
                    current_price
                )
                or not math.isfinite(
                    future_price
                )
                or current_price <= 0
            ):
                continue

            # ----------------------------------------------------
            # Check every one-second row crossed by the horizon.
            # If any row contains an unsafe gap, reject target.
            # ----------------------------------------------------

            crossed = group.iloc[
                index:future_index + 1
            ]

            if not bool(
                crossed[
                    "future_target_safe"
                ].all()
            ):
                continue

            future_returns[index] = (
                future_price
                / current_price
                - 1.0
            )

            future_safe[index] = True

        group[
            "future_mid_return_10s"
        ] = future_returns

        group[
            "future_target_valid"
        ] = future_safe

        output_frames.append(
            group
        )

    return pd.concat(
        output_frames,
        ignore_index=True,
    )


# ================================================================
# DIAGNOSTICS
# ================================================================

def print_diagnostics(
    dataframe: pd.DataFrame,
) -> None:

    print()
    print("=" * 72)
    print(
        "          PHASE 6.2 INTRASECOND FEATURE DIAGNOSTICS"
    )
    print("=" * 72)

    if dataframe.empty:
        print(
            "No feature rows were created."
        )
        return

    print(
        f"Feature rows:                 "
        f"{len(dataframe):,}"
    )

    print(
        f"Columns:                      "
        f"{len(dataframe.columns):,}"
    )

    print()

    print(
        "L1 update-count statistics:"
    )

    print(
        f"  Mean:                        "
        f"{dataframe['l1_update_count'].mean():.2f}"
    )

    print(
        f"  Median:                      "
        f"{dataframe['l1_update_count'].median():.2f}"
    )

    print(
        f"  Minimum:                     "
        f"{dataframe['l1_update_count'].min():.0f}"
    )

    print(
        f"  Maximum:                     "
        f"{dataframe['l1_update_count'].max():.0f}"
    )

    print()

    print(
        "Intrasecond movement:"
    )

    print(
        f"  Rows with mid change:        "
        f"{int((dataframe['mid_change'] != 0).sum()):,}"
    )

    print(
        f"  Rows without mid change:     "
        f"{int((dataframe['mid_change'] == 0).sum()):,}"
    )

    print(
        f"  Mean intrasecond volatility: "
        f"{dataframe['intrasecond_volatility'].mean():.10f}"
    )

    print()

    valid_returns = dataframe[
        "future_mid_return_10s"
    ].dropna()

    print(
        "10-second future target:"
    )

    print(
        f"  Valid targets:               "
        f"{len(valid_returns):,}"
    )

    print(
        f"  Invalid/unsafe targets:      "
        f"{int(dataframe['future_mid_return_10s'].isna().sum()):,}"
    )

    if len(valid_returns):

        nonzero = valid_returns[
            valid_returns != 0
        ]

        print(
            f"  Positive:                    "
            f"{int((valid_returns > 0).sum()):,}"
        )

        print(
            f"  Negative:                    "
            f"{int((valid_returns < 0).sum()):,}"
        )

        print(
            f"  Zero:                        "
            f"{int((valid_returns == 0).sum()):,}"
        )

        print(
            f"  Non-zero:                    "
            f"{len(nonzero):,}"
        )

        print(
            f"  Non-zero percentage:         "
            f"{len(nonzero) / len(valid_returns) * 100:.2f}%"
        )

        if len(nonzero):

            print()

            print(
                "  Non-zero absolute-return quantiles:"
            )

            absolute = np.abs(
                nonzero.to_numpy()
            )

            for q in (
                0.50,
                0.70,
                0.80,
                0.90,
                0.95,
                0.99,
            ):

                value = float(
                    np.quantile(
                        absolute,
                        q,
                    )
                )

                print(
                    f"    Q{int(q * 100):02d}: "
                    f"{value:.8f} "
                    f"({value * 100:.4f}%)"
                )

    unsafe = (
        ~dataframe[
            "future_target_safe"
        ]
    )

    print()

    print(
        f"Rows affected by data gaps:  "
        f"{int(unsafe.sum()):,}"
    )

    print(
        f"Gap-affected percentage:      "
        f"{unsafe.mean() * 100:.2f}%"
    )

    print()
    print(
        "Important:"
    )

    print(
        "The future target is calculated from "
        "future L1 mid-price, while intrasecond "
        "movement is preserved as model features."
    )

    print(
        "=" * 72
    )


# ================================================================
# SAVE
# ================================================================

def save_features(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Intrasecond feature dataset saved | "
        "rows=%d | columns=%d | file=%s",
        len(dataframe),
        len(dataframe.columns),
        output_path,
    )


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    input_dir = DEFAULT_INPUT_DIR

    output_path = Path(
        "data/ml/"
        "intrasecond_features.csv"
    )

    print()
    print("=" * 72)
    print(
        "       DELTA INTRASECOND FEATURE BUILDER"
    )
    print("=" * 72)

    print(
        f"Input:                       "
        f"{input_dir}"
    )

    print(
        f"Window:                      "
        f"{WINDOW_SECONDS:.1f} second"
    )

    print(
        f"Future horizon:              "
        f"{HORIZON_SECONDS:.1f} seconds"
    )

    print(
        f"Maximum allowed gap:         "
        f"{MAX_ALLOWED_GAP_SECONDS:.1f} seconds"
    )

    print()

    events = load_events(
        input_dir
    )

    features = build_features(
        events
    )

    if features.empty:
        raise SystemExit(
            "No feature rows created."
        )

    features = add_future_target(
        features
    )

    print_diagnostics(
        features
    )

    save_features(
        features,
        output_path,
    )

    print()
    print("=" * 72)
    print(
        "              PHASE 6.2 COMPLETE"
    )
    print("=" * 72)

    print(
        f"Output:                      "
        f"{output_path}"
    )

    print(
        "Raw market data was not modified."
    )

    print(
        "Existing training_dataset.csv was not modified."
    )

    print(
        "ML training has NOT been started."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()