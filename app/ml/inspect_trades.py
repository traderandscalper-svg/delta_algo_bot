"""
Inspect historical Delta Exchange trade messages.

This script is READ-ONLY.

Important:
- Historical data is not subject to the live 10-second freshness rule.
- Raw trade messages are structurally validated directly.
- Stored `valid` / `validation_reason` values are NOT trusted.
- This allows previously collected trades to be inspected and reused
  for ML dataset construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


DATA_DIR = Path("data/market_data")

MAX_VALID_TRADES = 10
MAX_INVALID_TRADES = 20


def find_latest_file() -> Path:
    files = sorted(DATA_DIR.glob("market_data_*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No market-data files found in: {DATA_DIR}"
        )

    return files[-1]


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(record, dict):
                continue

            record["_jsonl_line"] = line_number
            records.append(record)

    return records


def get_raw_message(
    record: dict[str, Any],
) -> Optional[dict[str, Any]]:

    message = record.get("message")

    if isinstance(message, dict):
        return message

    return None


def is_trade_record(
    record: dict[str, Any],
) -> bool:

    if record.get("event_type") == "trades":
        return True

    message = get_raw_message(record)

    if isinstance(message, dict):
        return message.get("type") == "trades"

    return False


def to_float(
    value: Any,
) -> Optional[float]:

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if result != result:
        return None

    return result


def validate_historical_trade(
    record: dict[str, Any],
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:

    message = get_raw_message(record)

    if message is None:
        return False, None, "Missing raw message"

    if message.get("type") != "trades":
        return False, None, "Message type is not trades"

    price = to_float(message.get("p"))
    size = to_float(message.get("s"))
    symbol = message.get("sy")
    timestamp = message.get("ts")

    if price is None:
        return False, None, "Invalid or missing trade price"

    if price <= 0:
        return False, None, "Trade price must be greater than zero"

    if size is None:
        return False, None, "Invalid or missing trade size"

    if size <= 0:
        return False, None, "Trade size must be greater than zero"

    if not symbol:
        return False, None, "Missing trade symbol"

    try:
        timestamp_us = int(timestamp)
    except (TypeError, ValueError):
        return False, None, "Invalid or missing exchange timestamp"

    if timestamp_us <= 0:
        return False, None, "Exchange timestamp must be positive"

    normalized = {
        "event_type": "trades",
        "symbol": str(symbol),
        "price": price,
        "size": size,
        "side": None,
        "trade_id": None,
        "exchange_timestamp": timestamp_us,
    }

    return True, normalized, None


def print_trade(
    record: dict[str, Any],
    number: int,
    title: str,
    valid: bool,
    normalized: Optional[dict[str, Any]],
    error: Optional[str],
) -> None:

    print("-" * 70)
    print(f"{title} #{number}")
    print(f"JSONL line: {record.get('_jsonl_line')}")
    print(f"Historical validation: {valid}")
    print(f"Validation reason: {error}")

    print("\nRAW MESSAGE:")
    print(
        json.dumps(
            record.get("message"),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nHISTORICAL NORMALIZED:")
    print(
        json.dumps(
            normalized,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nSTORED COLLECTOR METADATA:")
    print(
        f"Stored valid: "
        f"{record.get('valid')}"
    )
    print(
        f"Stored validation reason: "
        f"{record.get('validation_reason')}"
    )


def main() -> None:

    path = find_latest_file()
    records = read_records(path)

    trades = [
        record
        for record in records
        if is_trade_record(record)
    ]

    valid_trades = []
    invalid_trades = []

    for record in trades:

        valid, normalized, error = (
            validate_historical_trade(record)
        )

        if valid:
            valid_trades.append(
                (
                    record,
                    normalized,
                    error,
                )
            )
        else:
            invalid_trades.append(
                (
                    record,
                    normalized,
                    error,
                )
            )

    print("=" * 70)
    print("DELTA HISTORICAL TRADE INSPECTOR")
    print("=" * 70)
    print(f"File: {path}")
    print("Mode: Historical structural validation")
    print("Source file: READ-ONLY")
    print()

    print("=" * 70)
    print("VALID HISTORICAL TRADE EXAMPLES")
    print("=" * 70)

    if not valid_trades:
        print("No structurally valid historical trades found.")

    for number, (
        record,
        normalized,
        error,
    ) in enumerate(
        valid_trades[:MAX_VALID_TRADES],
        start=1,
    ):

        print_trade(
            record=record,
            number=number,
            title="VALID HISTORICAL TRADE",
            valid=True,
            normalized=normalized,
            error=error,
        )

    print()
    print("=" * 70)
    print("INVALID HISTORICAL TRADE RECORDS")
    print("=" * 70)

    if not invalid_trades:
        print(
            "No structurally invalid historical trades found."
        )

    for number, (
        record,
        normalized,
        error,
    ) in enumerate(
        invalid_trades[:MAX_INVALID_TRADES],
        start=1,
    ):

        print_trade(
            record=record,
            number=number,
            title="INVALID HISTORICAL TRADE",
            valid=False,
            normalized=normalized,
            error=error,
        )

    print()
    print("=" * 70)
    print("TRADE SUMMARY")
    print("=" * 70)

    print(
        f"Total trade records:       {len(trades)}"
    )

    print(
        f"Structurally valid:        {len(valid_trades)}"
    )

    print(
        f"Structurally invalid:      {len(invalid_trades)}"
    )

    print()

    if trades:
        usable_percentage = (
            len(valid_trades)
            / len(trades)
            * 100.0
        )
    else:
        usable_percentage = 0.0

    print(
        f"Historically usable:       "
        f"{usable_percentage:.2f}%"
    )

    print()
    print(
        "Note: Historical trades can be old and still be valid "
        "for dataset construction."
    )

    print(
        "The live validator's stale-message protection is not "
        "used here."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()