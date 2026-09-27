"""
Phase 6 - Delta Exchange Market Data Quality Verification

Checks the quality of the raw historical market-data JSONL files.

This diagnostic does NOT modify the dataset and does NOT train ML.

It verifies:

- Total records
- Event-type distribution
- Exchange timestamps
- Timestamp gaps
- Events per second
- Trade price movement
- L1 bid/ask movement
- L1 mid-price movement
- L1 update frequency
- Trade update frequency
- Bid/ask spread
- 1-second resampling behavior
- 10-second L1 mid-price returns
- Stale L1 periods
- Missing/invalid records
- Whether the apparent low volatility is present in raw data

Primary target price:

    L1 mid-price = (bid + ask) / 2

Trade price remains a separate market-data field.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger("MarketDataQuality")


# ================================================================
# CONSTANTS
# ================================================================

EVENT_TYPES = {
    "ticker",
    "ob_l1",
    "trades",
}

DEFAULT_INPUT = Path("data/market_data")

HORIZON_SECONDS = 10.0


# ================================================================
# NUMERIC HELPERS
# ================================================================

def safe_float(value: Any) -> Optional[float]:
    """
    Convert a value to a finite float.
    """

    try:
        if value is None:
            return None

        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def to_microseconds(value: Any) -> Optional[int]:
    """
    Normalize Delta timestamps into microseconds.

    Supports:
        seconds
        milliseconds
        microseconds
        nanoseconds
    """

    try:
        timestamp = int(float(value))
    except (TypeError, ValueError):
        return None

    if timestamp <= 0:
        return None

    # Seconds
    if timestamp < 100_000_000_000:
        return timestamp * 1_000_000

    # Milliseconds
    if timestamp < 100_000_000_000_000:
        return timestamp * 1_000

    # Microseconds
    if timestamp < 100_000_000_000_000_000:
        return timestamp

    # Nanoseconds
    return timestamp // 1_000


# ================================================================
# PAYLOAD EXTRACTION
# ================================================================

def extract_payload(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Delta websocket payload.

    Handles:

        {"d": {...}}

    and

        {"d": [{...}]}

    and direct payloads.
    """

    if not isinstance(message, dict):
        return {}

    data = message.get("d")

    if isinstance(data, list):

        if not data:
            return {}

        first = data[0]

        if isinstance(first, dict):
            return first

        return {}

    if isinstance(data, dict):
        return data

    return message


# ================================================================
# SYMBOL
# ================================================================

def extract_symbol(
    record: Dict[str, Any],
    normalized: Dict[str, Any],
    payload: Dict[str, Any],
) -> Optional[str]:

    candidates = [
        normalized.get("symbol"),
        record.get("symbol"),
        payload.get("sy"),
        payload.get("symbol"),
    ]

    for candidate in candidates:

        if candidate is None:
            continue

        text = str(candidate).strip()

        if text:
            return text

    return None


# ================================================================
# TIMESTAMP
# ================================================================

def extract_timestamp(
    record: Dict[str, Any],
    normalized: Dict[str, Any],
    payload: Dict[str, Any],
    message: Dict[str, Any],
) -> Optional[int]:

    candidates = [
        record.get("exchange_timestamp"),
        normalized.get("exchange_timestamp"),
        payload.get("ts"),
        payload.get("t"),
        message.get("ts"),
        message.get("t"),
        record.get("received_at_us"),
    ]

    for candidate in candidates:

        timestamp = to_microseconds(candidate)

        if timestamp is not None:
            return timestamp

    return None


# ================================================================
# EVENT NORMALIZATION
# ================================================================

def normalize_record(
    record: Dict[str, Any],
    line_number: int,
) -> Optional[Dict[str, Any]]:

    if not isinstance(record, dict):
        return None

    event_type = record.get("event_type")

    if event_type not in EVENT_TYPES:
        return None

    normalized = record.get("normalized")

    if not isinstance(normalized, dict):
        normalized = {}

    message = record.get("message")

    if not isinstance(message, dict):
        message = {}

    payload = extract_payload(message)

    timestamp = extract_timestamp(
        record,
        normalized,
        payload,
        message,
    )

    if timestamp is None:
        return None

    symbol = extract_symbol(
        record,
        normalized,
        payload,
    )

    if symbol is None:
        return None

    result: Dict[str, Any] = {
        "timestamp_us": timestamp,
        "symbol": symbol,
        "event_type": event_type,
        "line_number": line_number,
        "trade_price": None,
        "trade_size": None,
        "bid": None,
        "ask": None,
        "bid_size": None,
        "ask_size": None,
        "ticker_price": None,
        "mark_price": None,
    }

    # ------------------------------------------------------------
    # TRADE
    # ------------------------------------------------------------

    if event_type == "trades":

        result["trade_price"] = first_numeric(
            normalized.get("price"),
            payload.get("p"),
            payload.get("price"),
            payload.get("trade_price"),
        )

        result["trade_size"] = first_numeric(
            normalized.get("size"),
            payload.get("s"),
            payload.get("size"),
            payload.get("quantity"),
            payload.get("trade_size"),
            payload.get("qty"),
            payload.get("q"),
            payload.get("volume"),
        )

    # ------------------------------------------------------------
    # L1
    # ------------------------------------------------------------

    elif event_type == "ob_l1":

        result["bid"] = first_numeric(
            normalized.get("bid"),
            payload.get("bp"),
            payload.get("bid"),
            payload.get("bid_price"),
        )

        result["ask"] = first_numeric(
            normalized.get("ask"),
            payload.get("ap"),
            payload.get("ask"),
            payload.get("ask_price"),
        )

        result["bid_size"] = first_numeric(
            normalized.get("bid_size"),
            payload.get("bs"),
            payload.get("bid_size"),
            payload.get("bid_quantity"),
        )

        result["ask_size"] = first_numeric(
            normalized.get("ask_size"),
            payload.get("as"),
            payload.get("ask_size"),
            payload.get("ask_quantity"),
        )

    # ------------------------------------------------------------
    # TICKER
    # ------------------------------------------------------------

    elif event_type == "ticker":

        result["ticker_price"] = first_numeric(
            normalized.get("last_price"),
            normalized.get("price"),
            payload.get("last_price"),
            payload.get("last"),
            payload.get("lp"),
        )

        result["mark_price"] = first_numeric(
            normalized.get("mark_price"),
            payload.get("mark_price"),
            payload.get("mark"),
            payload.get("m"),
        )

        result["bid"] = first_numeric(
            normalized.get("bid"),
            payload.get("bp"),
            payload.get("bid"),
        )

        result["ask"] = first_numeric(
            normalized.get("ask"),
            payload.get("ap"),
            payload.get("ask"),
        )

    # ------------------------------------------------------------
    # Mid price
    # ------------------------------------------------------------

    bid = result["bid"]
    ask = result["ask"]

    if (
        bid is not None
        and ask is not None
        and bid > 0
        and ask > 0
        and ask >= bid
    ):
        result["mid_price"] = (
            bid + ask
        ) / 2.0

    else:
        result["mid_price"] = None

    return result


def first_numeric(*values: Any) -> Optional[float]:

    for value in values:

        result = safe_float(value)

        if result is not None:
            return result

    return None


# ================================================================
# FILE DISCOVERY
# ================================================================

def find_input_files(
    input_path: Path,
) -> List[Path]:

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        return []

    return sorted(
        input_path.glob(
            "market_data_*.jsonl"
        )
    )


# ================================================================
# READ FILES
# ================================================================

def read_files(
    input_files: Iterable[Path],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:

    records: List[Dict[str, Any]] = []

    counters = Counter()

    for path in input_files:

        print()
        print(
            f"Reading: {path}"
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                counters["lines"] += 1

                line = line.strip()

                if not line:
                    counters["empty_lines"] += 1
                    continue

                try:
                    raw = json.loads(line)

                except json.JSONDecodeError:
                    counters["invalid_json"] += 1
                    continue

                event_type = raw.get(
                    "event_type"
                )

                if event_type in EVENT_TYPES:
                    counters[
                        f"raw_{event_type}"
                    ] += 1

                normalized = normalize_record(
                    raw,
                    line_number,
                )

                if normalized is None:
                    counters[
                        "unusable_records"
                    ] += 1
                    continue

                records.append(
                    normalized
                )

                counters[
                    "usable_records"
                ] += 1

    return records, dict(counters)


# ================================================================
# BASIC SUMMARY
# ================================================================

def print_basic_summary(
    records: List[Dict[str, Any]],
    counters: Dict[str, int],
) -> None:

    print()
    print("=" * 70)
    print("                 RAW DATA QUALITY SUMMARY")
    print("=" * 70)

    print(
        f"Total JSON lines:             "
        f"{counters.get('lines', 0):,}"
    )

    print(
        f"Invalid JSON lines:           "
        f"{counters.get('invalid_json', 0):,}"
    )

    print(
        f"Empty lines:                  "
        f"{counters.get('empty_lines', 0):,}"
    )

    print(
        f"Usable market records:        "
        f"{len(records):,}"
    )

    print(
        f"Unusable market records:      "
        f"{counters.get('unusable_records', 0):,}"
    )

    print()

    event_counter = Counter(
        record["event_type"]
        for record in records
    )

    print("Event distribution:")
    print(
        f"  Ticker:                    "
        f"{event_counter['ticker']:,}"
    )

    print(
        f"  L1 order book:             "
        f"{event_counter['ob_l1']:,}"
    )

    print(
        f"  Trades:                    "
        f"{event_counter['trades']:,}"
    )

    symbols = sorted(
        {
            record["symbol"]
            for record in records
        }
    )

    print()
    print(
        "Symbols: "
        + (
            ", ".join(symbols)
            if symbols
            else "NONE"
        )
    )


# ================================================================
# TIMESTAMP ANALYSIS
# ================================================================

def print_timestamp_analysis(
    records: List[Dict[str, Any]],
) -> None:

    if not records:
        return

    print()
    print("=" * 70)
    print("                    TIMESTAMP ANALYSIS")
    print("=" * 70)

    for symbol in sorted(
        {
            record["symbol"]
            for record in records
        }
    ):

        symbol_records = [
            record
            for record in records
            if record["symbol"] == symbol
        ]

        symbol_records.sort(
            key=lambda x: x["timestamp_us"]
        )

        timestamps = np.array(
            [
                record["timestamp_us"]
                for record in symbol_records
            ],
            dtype=np.int64,
        )

        if len(timestamps) < 2:
            continue

        gaps = np.diff(
            timestamps
        ) / 1_000_000.0

        positive_gaps = gaps[
            gaps > 0
        ]

        duplicate_count = int(
            np.sum(gaps == 0)
        )

        print()
        print(
            f"Symbol: {symbol}"
        )

        print(
            f"  First timestamp:           "
            f"{timestamps[0] / 1_000_000:.6f}"
        )

        print(
            f"  Last timestamp:            "
            f"{timestamps[-1] / 1_000_000:.6f}"
        )

        print(
            f"  Duration:                  "
            f"{(timestamps[-1] - timestamps[0]) / 1_000_000:.2f} sec"
        )

        print(
            f"  Duplicate timestamps:      "
            f"{duplicate_count:,}"
        )

        if len(positive_gaps):

            print(
                f"  Median event gap:          "
                f"{np.median(positive_gaps):.6f} sec"
            )

            print(
                f"  Mean event gap:            "
                f"{np.mean(positive_gaps):.6f} sec"
            )

            print(
                f"  Maximum event gap:         "
                f"{np.max(positive_gaps):.6f} sec"
            )

            print(
                f"  Gaps > 1 sec:              "
                f"{np.sum(positive_gaps > 1.0):,}"
            )

            print(
                f"  Gaps > 5 sec:              "
                f"{np.sum(positive_gaps > 5.0):,}"
            )

            print(
                f"  Gaps > 10 sec:             "
                f"{np.sum(positive_gaps > 10.0):,}"
            )


# ================================================================
# L1 ANALYSIS
# ================================================================

def print_l1_analysis(
    records: List[Dict[str, Any]],
) -> None:

    l1_records = [
        record
        for record in records
        if record["event_type"] == "ob_l1"
    ]

    if not l1_records:
        print()
        print(
            "NO L1 ORDER-BOOK RECORDS FOUND."
        )
        return

    print()
    print("=" * 70)
    print("                 L1 ORDER BOOK ANALYSIS")
    print("=" * 70)

    valid_l1 = []

    for record in l1_records:

        bid = record["bid"]
        ask = record["ask"]

        if (
            bid is not None
            and ask is not None
            and bid > 0
            and ask > 0
            and ask >= bid
        ):

            mid = (
                bid + ask
            ) / 2.0

            spread = (
                ask - bid
            )

            valid_l1.append(
                {
                    "timestamp_us":
                        record["timestamp_us"],
                    "symbol":
                        record["symbol"],
                    "bid":
                        bid,
                    "ask":
                        ask,
                    "mid":
                        mid,
                    "spread":
                        spread,
                }
            )

    print(
        f"Raw L1 records:              "
        f"{len(l1_records):,}"
    )

    print(
        f"Valid bid/ask L1 records:     "
        f"{len(valid_l1):,}"
    )

    if not valid_l1:
        return

    df = pd.DataFrame(
        valid_l1
    )

    for symbol, group in df.groupby(
        "symbol",
        sort=True,
    ):

        group = group.sort_values(
            "timestamp_us"
        )

        mids = group[
            "mid"
        ].astype(float)

        bids = group[
            "bid"
        ].astype(float)

        asks = group[
            "ask"
        ].astype(float)

        spreads = group[
            "spread"
        ].astype(float)

        mid_changes = (
            mids.diff()
            .fillna(0.0)
        )

        bid_changes = (
            bids.diff()
            .fillna(0.0)
        )

        ask_changes = (
            asks.diff()
            .fillna(0.0)
        )

        print()
        print(
            f"Symbol: {symbol}"
        )

        print(
            f"  L1 observations:           "
            f"{len(group):,}"
        )

        print(
            f"  Unique bid prices:         "
            f"{bids.nunique():,}"
        )

        print(
            f"  Unique ask prices:         "
            f"{asks.nunique():,}"
        )

        print(
            f"  Unique mid prices:         "
            f"{mids.nunique():,}"
        )

        print(
            f"  Bid changes:               "
            f"{int((bid_changes != 0).sum()):,}"
        )

        print(
            f"  Ask changes:               "
            f"{int((ask_changes != 0).sum()):,}"
        )

        print(
            f"  Mid-price changes:         "
            f"{int((mid_changes != 0).sum()):,}"
        )

        print(
            f"  Mid unchanged:             "
            f"{int((mid_changes == 0).sum()):,}"
        )

        print(
            f"  First mid:                 "
            f"{mids.iloc[0]:.8f}"
        )

        print(
            f"  Last mid:                  "
            f"{mids.iloc[-1]:.8f}"
        )

        total_return = (
            mids.iloc[-1]
            / mids.iloc[0]
            - 1.0
        )

        print(
            f"  Total mid return:          "
            f"{total_return:.8f} "
            f"({total_return * 100:.4f}%)"
        )

        print(
            f"  Average spread:            "
            f"{spreads.mean():.8f}"
        )

        print(
            f"  Minimum spread:            "
            f"{spreads.min():.8f}"
        )

        print(
            f"  Maximum spread:            "
            f"{spreads.max():.8f}"
        )


# ================================================================
# TRADE ANALYSIS
# ================================================================

def print_trade_analysis(
    records: List[Dict[str, Any]],
) -> None:

    trade_records = [
        record
        for record in records
        if record["event_type"] == "trades"
    ]

    if not trade_records:
        print()
        print(
            "NO TRADE RECORDS FOUND."
        )
        return

    print()
    print("=" * 70)
    print("                     TRADE ANALYSIS")
    print("=" * 70)

    print(
        f"Trade records:               "
        f"{len(trade_records):,}"
    )

    for symbol in sorted(
        {
            record["symbol"]
            for record in trade_records
        }
    ):

        group = [
            record
            for record in trade_records
            if record["symbol"] == symbol
        ]

        prices = [
            record["trade_price"]
            for record in group
            if record["trade_price"] is not None
        ]

        sizes = [
            record["trade_size"]
            for record in group
            if record["trade_size"] is not None
        ]

        if not prices:
            continue

        unique_prices = len(
            set(prices)
        )

        price_changes = sum(
            1
            for index in range(
                1,
                len(prices),
            )
            if prices[index]
            != prices[index - 1]
        )

        print()
        print(
            f"Symbol: {symbol}"
        )

        print(
            f"  Trades:                   "
            f"{len(group):,}"
        )

        print(
            f"  Valid trade prices:       "
            f"{len(prices):,}"
        )

        print(
            f"  Unique trade prices:      "
            f"{unique_prices:,}"
        )

        print(
            f"  Trade-price changes:      "
            f"{price_changes:,}"
        )

        print(
            f"  First trade price:        "
            f"{prices[0]:.8f}"
        )

        print(
            f"  Last trade price:         "
            f"{prices[-1]:.8f}"
        )

        if sizes:

            print(
                f"  Average trade size:       "
                f"{np.mean(sizes):.8f}"
            )

            print(
                f"  Maximum trade size:       "
                f"{np.max(sizes):.8f}"
            )


# ================================================================
# RAW L1 TIMING
# ================================================================

def print_l1_update_frequency(
    records: List[Dict[str, Any]],
) -> None:

    print()
    print("=" * 70)
    print("                  L1 UPDATE FREQUENCY")
    print("=" * 70)

    for symbol in sorted(
        {
            record["symbol"]
            for record in records
        }
    ):

        l1 = [
            record
            for record in records
            if (
                record["symbol"] == symbol
                and record["event_type"] == "ob_l1"
                and record["bid"] is not None
                and record["ask"] is not None
            )
        ]

        if len(l1) < 2:
            continue

        l1.sort(
            key=lambda x: x["timestamp_us"]
        )

        timestamps = np.array(
            [
                record["timestamp_us"]
                for record in l1
            ],
            dtype=np.int64,
        )

        gaps = (
            np.diff(timestamps)
            / 1_000_000.0
        )

        print()
        print(
            f"Symbol: {symbol}"
        )

        print(
            f"  L1 updates:                "
            f"{len(l1):,}"
        )

        print(
            f"  Median L1 gap:             "
            f"{np.median(gaps):.6f} sec"
        )

        print(
            f"  Mean L1 gap:               "
            f"{np.mean(gaps):.6f} sec"
        )

        print(
            f"  Maximum L1 gap:            "
            f"{np.max(gaps):.6f} sec"
        )

        print(
            f"  L1 gaps <= 1 sec:         "
            f"{np.sum(gaps <= 1.0):,}"
        )

        print(
            f"  L1 gaps > 1 sec:          "
            f"{np.sum(gaps > 1.0):,}"
        )

        print(
            f"  L1 gaps > 5 sec:          "
            f"{np.sum(gaps > 5.0):,}"
        )


# ================================================================
# 10-SECOND RAW L1 RETURN ANALYSIS
# ================================================================

def analyze_10_second_returns(
    records: List[Dict[str, Any]],
) -> None:

    print()
    print("=" * 70)
    print("           RAW L1 10-SECOND RETURN ANALYSIS")
    print("=" * 70)

    horizon_us = int(
        HORIZON_SECONDS
        * 1_000_000
    )

    for symbol in sorted(
        {
            record["symbol"]
            for record in records
        }
    ):

        l1 = [
            record
            for record in records
            if (
                record["symbol"] == symbol
                and record["event_type"] == "ob_l1"
                and record["bid"] is not None
                and record["ask"] is not None
            )
        ]

        if not l1:
            continue

        l1.sort(
            key=lambda x: (
                x["timestamp_us"],
                x["line_number"],
            )
        )

        # --------------------------------------------------------
        # Deduplicate same timestamp.
        # Last observation at a timestamp wins.
        # --------------------------------------------------------

        state_by_timestamp: Dict[
            int,
            float,
        ] = {}

        for record in l1:

            timestamp = int(
                record["timestamp_us"]
            )

            mid = (
                record["bid"]
                + record["ask"]
            ) / 2.0

            state_by_timestamp[
                timestamp
            ] = float(mid)

        timestamps = np.array(
            sorted(
                state_by_timestamp.keys()
            ),
            dtype=np.int64,
        )

        mids = np.array(
            [
                state_by_timestamp[
                    timestamp
                ]
                for timestamp in timestamps
            ],
            dtype=float,
        )

        returns: List[
            float
        ] = []

        future_index = 1

        for index in range(
            len(timestamps)
        ):

            target = (
                timestamps[index]
                + horizon_us
            )

            if future_index <= index:
                future_index = index + 1

            while (
                future_index
                < len(timestamps)
                and timestamps[
                    future_index
                ] < target
            ):
                future_index += 1

            if future_index >= len(
                timestamps
            ):
                break

            current = mids[index]
            future = mids[
                future_index
            ]

            if current <= 0:
                continue

            returns.append(
                (
                    future
                    - current
                )
                / current
            )

        if not returns:
            continue

        values = np.asarray(
            returns,
            dtype=float,
        )

        nonzero = values[
            values != 0
        ]

        print()
        print(
            f"Symbol: {symbol}"
        )

        print(
            f"  Raw L1 samples:            "
            f"{len(timestamps):,}"
        )

        print(
            f"  10-second returns:         "
            f"{len(values):,}"
        )

        print(
            f"  Positive:                  "
            f"{int((values > 0).sum()):,}"
        )

        print(
            f"  Negative:                  "
            f"{int((values < 0).sum()):,}"
        )

        print(
            f"  Zero:                      "
            f"{int((values == 0).sum()):,}"
        )

        print(
            f"  Non-zero:                  "
            f"{len(nonzero):,}"
        )

        print(
            f"  Non-zero percentage:       "
            f"{len(nonzero) / len(values) * 100:.2f}%"
        )

        print(
            f"  Min return:                "
            f"{values.min():.8f} "
            f"({values.min() * 100:.4f}%)"
        )

        print(
            f"  Max return:                "
            f"{values.max():.8f} "
            f"({values.max() * 100:.4f}%)"
        )

        print(
            f"  Mean return:               "
            f"{values.mean():.8f} "
            f"({values.mean() * 100:.4f}%)"
        )

        print(
            f"  Median return:             "
            f"{np.median(values):.8f} "
            f"({np.median(values) * 100:.4f}%)"
        )

        if len(nonzero):

            print()
            print(
                "  Non-zero return quantiles:"
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
                        np.abs(nonzero),
                        q,
                    )
                )

                print(
                    f"    Q{int(q * 100):02d}: "
                    f"{value:.8f} "
                    f"({value * 100:.4f}%)"
                )


# ================================================================
# 1-SECOND RESAMPLING VERIFICATION
# ================================================================

def analyze_resampling(
    records: List[Dict[str, Any]],
) -> None:

    print()
    print("=" * 70)
    print("               1-SECOND RESAMPLING CHECK")
    print("=" * 70)

    for symbol in sorted(
        {
            record["symbol"]
            for record in records
        }
    ):

        l1 = [
            record
            for record in records
            if (
                record["symbol"] == symbol
                and record["event_type"] == "ob_l1"
                and record["bid"] is not None
                and record["ask"] is not None
            )
        ]

        if not l1:
            continue

        l1.sort(
            key=lambda x: (
                x["timestamp_us"],
                x["line_number"],
            )
        )

        state_by_timestamp: Dict[
            int,
            float,
        ] = {}

        for record in l1:

            timestamp = int(
                record["timestamp_us"]
            )

            mid = (
                record["bid"]
                + record["ask"]
            ) / 2.0

            state_by_timestamp[
                timestamp
            ] = float(mid)

        timestamps = sorted(
            state_by_timestamp.keys()
        )

        if len(timestamps) < 2:
            continue

        first_timestamp = timestamps[0]
        last_timestamp = timestamps[-1]

        interval_us = 1_000_000

        sample_timestamp = (
            first_timestamp
        )

        index = 0
        latest_mid = None

        samples: List[
            float
        ] = []

        while (
            sample_timestamp
            <= last_timestamp
        ):

            while (
                index < len(timestamps)
                and timestamps[index]
                <= sample_timestamp
            ):

                latest_mid = (
                    state_by_timestamp[
                        timestamps[index]
                    ]
                )

                index += 1

            if latest_mid is not None:
                samples.append(
                    latest_mid
                )

            sample_timestamp += (
                interval_us
            )

        if not samples:
            continue

        values = np.asarray(
            samples,
            dtype=float,
        )

        changes = np.diff(
            values
        )

        print()
        print(
            f"Symbol: {symbol}"
        )

        print(
            f"  1-second samples:         "
            f"{len(values):,}"
        )

        print(
            f"  Unique mid prices:        "
            f"{len(np.unique(values)):,}"
        )

        print(
            f"  Changed samples:          "
            f"{int((changes != 0).sum()):,}"
        )

        print(
            f"  Unchanged samples:        "
            f"{int((changes == 0).sum()):,}"
        )

        print(
            f"  Unchanged percentage:     "
            f"{(changes == 0).sum() / max(len(changes), 1) * 100:.2f}%"
        )


# ================================================================
# FINAL INTERPRETATION
# ================================================================

def print_interpretation(
    records: List[Dict[str, Any]],
) -> None:

    print()
    print("=" * 70)
    print("                    DATA QUALITY VERDICT")
    print("=" * 70)

    l1 = [
        record
        for record in records
        if (
            record["event_type"] == "ob_l1"
            and record["bid"] is not None
            and record["ask"] is not None
        )
    ]

    trades = [
        record
        for record in records
        if (
            record["event_type"] == "trades"
            and record["trade_price"] is not None
        )
    ]

    if not l1:
        print()
        print(
            "CRITICAL: No valid L1 bid/ask data was found."
        )
        print(
            "Do NOT proceed to ML training."
        )
        return

    if not trades:
        print()
        print(
            "WARNING: No valid trade-price data was found."
        )

    mids = np.asarray(
        [
            (
                record["bid"]
                + record["ask"]
            ) / 2.0
            for record in l1
        ],
        dtype=float,
    )

    unique_mids = len(
        np.unique(mids)
    )

    if len(mids) > 1:

        changes = np.diff(
            mids
        )

        changed = int(
            (changes != 0).sum()
        )

        change_percentage = (
            changed
            / len(changes)
            * 100.0
        )

    else:

        changed = 0
        change_percentage = 0.0

    print()

    print(
        f"Valid raw L1 observations:    "
        f"{len(l1):,}"
    )

    print(
        f"Unique raw L1 mid prices:      "
        f"{unique_mids:,}"
    )

    print(
        f"Raw L1 mid changes:            "
        f"{changed:,}"
    )

    print(
        f"Raw L1 change percentage:      "
        f"{change_percentage:.2f}%"
    )

    print()

    if change_percentage < 1.0:

        print(
            "RESULT: L1 movement is very sparse."
        )

        print(
            "The low 10-second target movement may "
            "be caused by the actual captured L1 stream."
        )

        print(
            "Collecting more data is recommended "
            "before ML training."
        )

    elif change_percentage < 10.0:

        print(
            "RESULT: L1 movement exists but is relatively sparse."
        )

        print(
            "More historical data is recommended "
            "before training a high-frequency model."
        )

    else:

        print(
            "RESULT: L1 is changing frequently."
        )

        print(
            "If the ML dataset still shows mostly-zero "
            "returns, investigate the resampling/target "
            "construction rather than the exchange feed."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This diagnostic does not determine whether a "
        "strategy is profitable."
    )

    print(
        "It only determines whether the market data "
        "is sufficiently dynamic and correctly captured "
        "for the next ML pipeline stage."
    )

    print(
        "=" * 70
    )


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Verify the quality of historical "
            "Delta Exchange market data."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "JSONL file or directory containing "
            "market data."
        ),
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

    input_path = Path(
        args.input
    )

    input_files = find_input_files(
        input_path
    )

    if not input_files:

        raise SystemExit(
            "No market-data JSONL files found."
        )

    print()
    print("=" * 70)
    print("          DELTA MARKET DATA QUALITY VERIFICATION")
    print("=" * 70)

    print(
        f"Input: {input_path}"
    )

    print(
        f"Files: {len(input_files)}"
    )

    for path in input_files:
        print(
            f"  - {path}"
        )

    records, counters = read_files(
        input_files
    )

    if not records:

        raise SystemExit(
            "No usable market records found."
        )

    print_basic_summary(
        records,
        counters,
    )

    print_timestamp_analysis(
        records
    )

    print_l1_analysis(
        records
    )

    print_trade_analysis(
        records
    )

    print_l1_update_frequency(
        records
    )

    analyze_10_second_returns(
        records
    )

    analyze_resampling(
        records
    )

    print_interpretation(
        records
    )


if __name__ == "__main__":
    main()