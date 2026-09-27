
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SignalPerformanceTracker:
    """
    Phase 7.5:
    Tracks the performance of generated strategy signals over multiple
    future 1-second candle horizons.

    Horizons:
        1s, 3s, 5s, 10s, 30s, 60s

    Important:
        - Signals are never evaluated against candles that existed before
          the signal.
        - Pending signals are processed BEFORE the current signal is added.
        - No orders are placed.
    """

    HORIZONS = (1, 3, 5, 10, 30, 60)

    def __init__(
        self,
        output_directory: str = "data/strategy",
    ) -> None:
        self.logger = logging.getLogger("SignalPerformanceTracker")

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.pending_signals: list[dict[str, Any]] = []
        self.completed_signal_keys: set[tuple[str, int, int]] = set()

        self.stats: dict[int, dict[str, Any]] = {
            horizon: self._empty_stats()
            for horizon in self.HORIZONS
        }

        self.completed_signal_records = 0

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "flat_count": 0,
            "positive_rate": 0.0,
            "average_return": 0.0,
            "average_directional_return": 0.0,
        }

    @staticmethod
    def _action(signal: Any) -> str:
        action = getattr(signal, "action", "HOLD")

        if hasattr(action, "value"):
            action = action.value

        return str(action).upper()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _iso_timestamp(timestamp_us: int) -> str:
        if timestamp_us <= 0:
            return ""

        return datetime.fromtimestamp(
            timestamp_us / 1_000_000,
            tz=timezone.utc,
        ).isoformat()

    def _output_file(self) -> Path:
        date_string = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.output_directory / (
            f"signal_performance_{date_string}.jsonl"
        )

    def _write_record(self, record: dict[str, Any]) -> None:
        path = self._output_file()

        with path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    record,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )

    def record_signal(
        self,
        signal: Any,
        candle: Any,
    ) -> None:
        """
        Register a newly generated signal.

        This should be called AFTER process_candle().
        """

        symbol = str(
            getattr(signal, "symbol", None)
            or getattr(candle, "symbol", "UNKNOWN")
        )

        timestamp = self._safe_int(
            getattr(
                signal,
                "timestamp",
                getattr(candle, "end_timestamp", 0),
            )
        )

        entry_price = self._safe_float(
            getattr(
                signal,
                "price",
                getattr(candle, "close", 0.0),
            )
        )

        if entry_price <= 0:
            entry_price = self._safe_float(
                getattr(candle, "close", 0.0)
            )

        if timestamp <= 0 or entry_price <= 0:
            return

        action = self._action(signal)

        key_base = (symbol, timestamp)

        for horizon in self.HORIZONS:
            key = (symbol, timestamp, horizon)

            if key in self.completed_signal_keys:
                continue

        pending = {
            "symbol": symbol,
            "signal_timestamp": timestamp,
            "signal_time": self._iso_timestamp(timestamp),
            "entry_price": entry_price,
            "action": action,
            "confidence": self._safe_float(
                getattr(signal, "confidence", 0.0)
            ),
            "score": self._safe_float(
                getattr(signal, "score", 0.0)
            ),
            "quality": self._safe_float(
                getattr(signal, "quality", 0.0)
            ),
            "regime": str(
                getattr(signal, "regime", "")
            ),
            "votes": getattr(
                signal,
                "votes",
                getattr(signal, "strategy_votes", {}),
            ),
            "reason": str(
                getattr(signal, "reason", "")
            ),
            "candle_end_timestamp": self._safe_int(
                getattr(candle, "end_timestamp", timestamp)
            ),
        }

        # Avoid duplicate registration of the same signal.
        for existing in self.pending_signals:
            if (
                existing["symbol"] == symbol
                and existing["signal_timestamp"] == timestamp
            ):
                return

        self.pending_signals.append(pending)

    def process_candle(
        self,
        candle: Any,
    ) -> None:
        """
        Process newly completed 1-second candle.

        Must be called BEFORE record_signal() for the current candle.
        """

        symbol = str(
            getattr(candle, "symbol", "UNKNOWN")
        )

        timestamp = self._safe_int(
            getattr(candle, "end_timestamp", 0)
        )

        close_price = self._safe_float(
            getattr(candle, "close", 0.0)
        )

        if timestamp <= 0 or close_price <= 0:
            return

        remaining: list[dict[str, Any]] = []

        for pending in self.pending_signals:
            if pending["symbol"] != symbol:
                remaining.append(pending)
                continue

            signal_timestamp = pending["signal_timestamp"]

            elapsed_seconds = (
                timestamp - signal_timestamp
            ) / 1_000_000

            if elapsed_seconds <= 0:
                remaining.append(pending)
                continue

            for horizon in self.HORIZONS:
                if elapsed_seconds < horizon:
                    continue

                key = (
                    pending["symbol"],
                    pending["signal_timestamp"],
                    horizon,
                )

                if key in self.completed_signal_keys:
                    continue

                entry_price = pending["entry_price"]

                raw_return = (
                    close_price - entry_price
                ) / entry_price

                action = pending["action"]

                if action == "BUY":
                    directional_return = raw_return
                elif action == "SELL":
                    directional_return = -raw_return
                else:
                    # HOLD is measured as market movement.
                    directional_return = raw_return

                if directional_return > 0:
                    outcome = "POSITIVE"
                elif directional_return < 0:
                    outcome = "NEGATIVE"
                else:
                    outcome = "FLAT"

                stats = self.stats[horizon]

                stats["count"] += 1

                if action == "BUY":
                    stats["buy_count"] += 1
                elif action == "SELL":
                    stats["sell_count"] += 1
                else:
                    stats["hold_count"] += 1

                if outcome == "POSITIVE":
                    stats["positive_count"] += 1
                elif outcome == "NEGATIVE":
                    stats["negative_count"] += 1
                else:
                    stats["flat_count"] += 1

                count = stats["count"]

                stats["average_return"] += (
                    raw_return - stats["average_return"]
                ) / count

                stats["average_directional_return"] += (
                    directional_return
                    - stats["average_directional_return"]
                ) / count

                non_flat = (
                    stats["positive_count"]
                    + stats["negative_count"]
                )

                if non_flat:
                    stats["positive_rate"] = (
                        stats["positive_count"]
                        / non_flat
                    )

                record = {
                    "record_type": "signal_performance",
                    "version": 1,
                    "symbol": pending["symbol"],
                    "signal_timestamp": pending[
                        "signal_timestamp"
                    ],
                    "signal_time": pending[
                        "signal_time"
                    ],
                    "entry_price": entry_price,
                    "action": action,
                    "confidence": pending[
                        "confidence"
                    ],
                    "score": pending["score"],
                    "quality": pending["quality"],
                    "regime": pending["regime"],
                    "votes": pending["votes"],
                    "reason": pending["reason"],
                    "horizon_seconds": horizon,
                    "exit_timestamp": timestamp,
                    "exit_time": self._iso_timestamp(
                        timestamp
                    ),
                    "exit_price": close_price,
                    "raw_return": raw_return,
                    "directional_return": directional_return,
                    "outcome": outcome,
                }

                self._write_record(record)

                self.completed_signal_keys.add(key)
                self.completed_signal_records += 1

            # Keep pending signal until its largest horizon is completed.
            if elapsed_seconds < max(self.HORIZONS):
                remaining.append(pending)

        self.pending_signals = remaining

    def get_stats(self) -> dict[str, Any]:
        return {
            "horizons": {
                str(horizon): dict(values)
                for horizon, values in self.stats.items()
            },
            "pending_signals": len(
                self.pending_signals
            ),
            "completed_signal_records": (
                self.completed_signal_records
            ),
        }

