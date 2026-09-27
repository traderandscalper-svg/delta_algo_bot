from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/market_data")
OUTPUT_FILE = Path("data/ml/multi_horizon_dataset.csv")

SYMBOL = "BTCUSD"

HORIZONS = {
    "1s": 1.0,
    "3s": 3.0,
    "5s": 5.0,
    "10s": 10.0,
    "30s": 30.0,
    "60s": 60.0,
}

MAX_ALLOWED_GAP_SECONDS = 5.0

# 0.001%
DIRECTION_THRESHOLD = 0.00001


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger("MultiHorizonDataset")


# ============================================================
# HELPERS
# ============================================================

def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        result = float(value)

        if result != result:
            return None

        return result

    except (TypeError, ValueError):
        return None


def extract_timestamp(record: Dict[str, Any]) -> Optional[float]:
    """
    Extract exchange timestamp.

    The actual validated Delta structure is:

        record["exchange_timestamp"]

    and also:

        record["message"]["ts"]
    """

    value = record.get("exchange_timestamp")

    if value is None:
        message = record.get("message")

        if isinstance(message, dict):
            value = message.get("ts")

    if value is None:
        value = record.get("ts")

    timestamp = safe_float(value)

    if timestamp is None:
        return None

    # Delta timestamps in the current dataset are microseconds.
    if timestamp > 100_000_000_000_000:
        return timestamp / 1_000_000.0

    # Milliseconds fallback.
    if timestamp > 100_000_000_000:
        return timestamp / 1_000.0

    return timestamp


def extract_l1_event(
    record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Parse the exact validated structure used by the
    current Delta market-data collector.

    Example:

    {
        "event_type": "ob_l1",
        "symbol": "BTCUSD",
        "exchange_timestamp": ...,
        "message": {
            "ap": "...",
            "as": "...",
            "bp": "...",
            "bs": "...",
            "sy": "BTCUSD",
            "ts": ...
        }
    }
    """

    event_type = (
        record.get("event_type")
        or record.get("type")
    )

    if event_type != "ob_l1":
        return None

    # --------------------------------------------------------
    # MESSAGE
    # --------------------------------------------------------

    message = record.get("message")

    if not isinstance(message, dict):
        return None

    # --------------------------------------------------------
    # SYMBOL
    # --------------------------------------------------------

    symbol = (
        record.get("symbol")
        or message.get("sy")
    )

    if symbol != SYMBOL:
        return None

    # --------------------------------------------------------
    # PRICES
    # --------------------------------------------------------

    bid = safe_float(
        message.get("bp")
    )

    ask = safe_float(
        message.get("ap")
    )

    if bid is None or ask is None:
        return None

    if bid <= 0 or ask <= 0:
        return None

    if ask < bid:
        return None

    # --------------------------------------------------------
    # SIZES
    # --------------------------------------------------------

    bid_size = safe_float(
        message.get("bs")
    )

    ask_size = safe_float(
        message.get("as")
    )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp = extract_timestamp(record)

    if timestamp is None:
        return None

    # --------------------------------------------------------
    # MID
    # --------------------------------------------------------

    mid = (bid + ask) / 2.0

    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid": mid,
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_l1_events() -> List[Dict[str, Any]]:

    events: List[Dict[str, Any]] = []

    files = sorted(
        INPUT_DIR.glob("market_data_*.jsonl")
    )

    if not files:
        raise FileNotFoundError(
            f"No market_data_*.jsonl files found in {INPUT_DIR}"
        )

    total_lines = 0
    raw_l1 = 0
    valid_l1 = 0

    for file_path in files:

        logger.info(
            "Reading L1 data | file=%s",
            file_path,
        )

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            for line in handle:

                total_lines += 1

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(record, dict):
                    continue

                event_type = (
                    record.get("event_type")
                    or record.get("type")
                )

                if event_type != "ob_l1":
                    continue

                raw_l1 += 1

                event = extract_l1_event(record)

                if event is None:
                    continue

                valid_l1 += 1

                events.append(event)

    logger.info(
        "Total JSON lines inspected: %d",
        total_lines,
    )

    logger.info(
        "Raw L1 records found: %d",
        raw_l1,
    )

    logger.info(
        "Valid BTCUSD L1 records: %d",
        valid_l1,
    )

    events.sort(
        key=lambda item: item["timestamp"]
    )

    return events


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_events(
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not events:
        return []

    result = []

    previous_timestamp = None

    for event in events:

        timestamp = event["timestamp"]

        if (
            previous_timestamp is not None
            and timestamp == previous_timestamp
        ):
            continue

        result.append(event)

        previous_timestamp = timestamp

    return result


# ============================================================
# FUTURE EVENT
# ============================================================

def find_future_event(
    events: List[Dict[str, Any]],
    start_index: int,
    target_timestamp: float,
) -> Optional[Dict[str, Any]]:

    for index in range(
        start_index + 1,
        len(events),
    ):

        event = events[index]

        if event["timestamp"] >= target_timestamp:
            return event

    return None


# ============================================================
# GAP VALIDATION
# ============================================================

def target_interval_is_safe(
    events: List[Dict[str, Any]],
    start_index: int,
    target_timestamp: float,
) -> bool:

    previous_timestamp = events[start_index]["timestamp"]

    for index in range(
        start_index + 1,
        len(events),
    ):

        current_timestamp = events[index]["timestamp"]

        if current_timestamp > target_timestamp:
            break

        gap = (
            current_timestamp
            - previous_timestamp
        )

        if gap > MAX_ALLOWED_GAP_SECONDS:
            return False

        previous_timestamp = current_timestamp

    return True


# ============================================================
# RETURN
# ============================================================

def calculate_return(
    current_mid: float,
    future_mid: Optional[float],
) -> Optional[float]:

    if future_mid is None:
        return None

    if current_mid <= 0:
        return None

    return (
        future_mid - current_mid
    ) / current_mid


def calculate_direction(
    future_return: Optional[float],
) -> Optional[int]:

    if future_return is None:
        return None

    if future_return > DIRECTION_THRESHOLD:
        return 1

    if future_return < -DIRECTION_THRESHOLD:
        return -1

    return 0


# ============================================================
# DATASET
# ============================================================

def build_dataset(
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    logger.info(
        "Building multi-horizon dataset | events=%d",
        len(events),
    )

    for index, event in enumerate(events):

        timestamp = event["timestamp"]

        current_mid = event["mid"]

        # ----------------------------------------------------
        # BASE FEATURES
        # ----------------------------------------------------

        row: Dict[str, Any] = {
            "symbol": event["symbol"],
            "timestamp": timestamp,

            "mid_price": current_mid,

            "bid": event["bid"],
            "ask": event["ask"],

            "spread": (
                event["ask"]
                - event["bid"]
            ),

            "bid_size": event["bid_size"],
            "ask_size": event["ask_size"],
        }

        # ----------------------------------------------------
        # ORDER BOOK IMBALANCE
        # ----------------------------------------------------

        bid_size = event["bid_size"]
        ask_size = event["ask_size"]

        if (
            bid_size is not None
            and ask_size is not None
        ):

            denominator = (
                bid_size
                + ask_size
            )

            if denominator > 0:

                row["orderbook_imbalance"] = (
                    bid_size
                    - ask_size
                ) / denominator

            else:

                row["orderbook_imbalance"] = None

        else:

            row["orderbook_imbalance"] = None

        # ----------------------------------------------------
        # HORIZONS
        # ----------------------------------------------------

        for horizon_name, horizon_seconds in HORIZONS.items():

            target_timestamp = (
                timestamp
                + horizon_seconds
            )

            safe = target_interval_is_safe(
                events,
                index,
                target_timestamp,
            )

            future_event = None

            if safe:

                future_event = find_future_event(
                    events,
                    index,
                    target_timestamp,
                )

            future_mid = (
                future_event["mid"]
                if future_event is not None
                else None
            )

            future_return = calculate_return(
                current_mid,
                future_mid,
            )

            direction = calculate_direction(
                future_return
            )

            row[
                f"future_mid_{horizon_name}"
            ] = future_mid

            row[
                f"return_{horizon_name}"
            ] = future_return

            row[
                f"direction_{horizon_name}"
            ] = direction

            row[
                f"target_valid_{horizon_name}"
            ] = (
                future_return is not None
            )

        rows.append(row)

    return rows


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_horizon_statistics(
    rows: List[Dict[str, Any]],
) -> None:

    print()
    print("=" * 80)
    print("          PHASE 6.3 MULTI-HORIZON DIAGNOSTICS")
    print("=" * 80)

    print(
        f"Raw feature rows:             {len(rows)}"
    )

    print()

    print(
        f"{'Horizon':<10}"
        f"{'Valid':>10}"
        f"{'Non-zero':>12}"
        f"{'Non-zero %':>14}"
        f"{'Positive':>12}"
        f"{'Negative':>12}"
        f"{'Zero':>10}"
    )

    print("-" * 80)

    for horizon_name in HORIZONS:

        returns = [
            row[f"return_{horizon_name}"]
            for row in rows
            if row[f"return_{horizon_name}"] is not None
        ]

        positive = sum(
            1
            for value in returns
            if value > 0
        )

        negative = sum(
            1
            for value in returns
            if value < 0
        )

        zero = sum(
            1
            for value in returns
            if value == 0
        )

        non_zero = (
            positive
            + negative
        )

        non_zero_percentage = (
            non_zero
            / len(returns)
            * 100
            if returns
            else 0.0
        )

        print(
            f"{horizon_name:<10}"
            f"{len(returns):>10}"
            f"{non_zero:>12}"
            f"{non_zero_percentage:>13.2f}%"
            f"{positive:>12}"
            f"{negative:>12}"
            f"{zero:>10}"
        )

    print("=" * 80)


def print_return_ranges(
    rows: List[Dict[str, Any]],
) -> None:

    print()
    print("RETURN RANGE DIAGNOSTICS")
    print("-" * 80)

    for horizon_name in HORIZONS:

        values = [
            row[f"return_{horizon_name}"]
            for row in rows
            if row[f"return_{horizon_name}"] is not None
        ]

        if not values:

            print(
                f"{horizon_name:<6}"
                " No valid returns"
            )

            continue

        minimum = min(values)
        maximum = max(values)

        mean = (
            sum(values)
            / len(values)
        )

        print(
            f"{horizon_name:<6}"
            f" min={minimum:.8f}"
            f" ({minimum * 100:.4f}%)"
            f" max={maximum:.8f}"
            f" ({maximum * 100:.4f}%)"
            f" mean={mean:.8f}"
        )


# ============================================================
# SAVE
# ============================================================

def save_csv(
    rows: List[Dict[str, Any]],
) -> None:

    if not rows:
        raise RuntimeError(
            "Cannot save empty dataset."
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
        "Multi-horizon dataset saved | "
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
    print("       DELTA MULTI-HORIZON DATASET BUILDER")
    print("=" * 72)

    print(
        f"Input:                       {INPUT_DIR}"
    )

    print(
        f"Output:                      {OUTPUT_FILE}"
    )

    print(
        "Horizons:                    "
        + ", ".join(HORIZONS.keys())
    )

    print(
        f"Maximum allowed gap:         "
        f"{MAX_ALLOWED_GAP_SECONDS} seconds"
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    events = load_l1_events()

    logger.info(
        "Usable L1 events loaded: %d",
        len(events),
    )

    if len(events) < 100:

        raise RuntimeError(
            "Too few L1 events to build the dataset."
        )

    # --------------------------------------------------------
    # DEDUPLICATE
    # --------------------------------------------------------

    events = deduplicate_events(
        events
    )

    logger.info(
        "L1 events after timestamp "
        "deduplication: %d",
        len(events),
    )

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    rows = build_dataset(
        events
    )

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    print_horizon_statistics(
        rows
    )

    print_return_ranges(
        rows
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_csv(rows)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("              PHASE 6.3 COMPLETE")
    print("=" * 72)

    print(
        f"Output:                      {OUTPUT_FILE}"
    )

    print(
        f"Rows:                        {len(rows)}"
    )

    print(
        "Raw market data was not modified."
    )

    print(
        "Intrasecond feature dataset "
        "was not modified."
    )

    print(
        "ML training has NOT been started."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()