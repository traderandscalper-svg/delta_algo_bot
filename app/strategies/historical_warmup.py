
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class HistoricalCandleWarmup:
    """
    Loads real historical 1-second candles into the strategy engine.

    Historical candles are used ONLY to initialize strategy history.
    They do not generate live signals, learning samples, or trades.
    """

    def __init__(
        self,
        candle_file: str = "data/market_data/candles.jsonl",
        max_candles: int = 200,
    ) -> None:
        self.logger = logging.getLogger(
            "HistoricalCandleWarmup"
        )

        self.candle_file = Path(candle_file)
        self.max_candles = max_candles

    def load(self) -> list[Any]:
        if not self.candle_file.exists():
            self.logger.warning(
                "Historical candle file not found: %s",
                self.candle_file,
            )
            return []

        candles: list[Any] = []
        seen: set[tuple[str, int, int]] = set()

        with self.candle_file.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if int(data.get("interval_seconds", 0)) != 1:
                    continue

                symbol = str(
                    data.get("symbol", "")
                ).strip()

                start_timestamp = int(
                    data.get(
                        "start_timestamp",
                        0,
                    )
                )

                end_timestamp = int(
                    data.get(
                        "end_timestamp",
                        0,
                    )
                )

                close = float(
                    data.get("close", 0.0)
                )

                if (
                    not symbol
                    or start_timestamp <= 0
                    or end_timestamp <= 0
                    or close <= 0
                ):
                    continue

                key = (
                    symbol,
                    start_timestamp,
                    end_timestamp,
                )

                if key in seen:
                    continue

                seen.add(key)

                candles.append(
                    SimpleNamespace(
                        symbol=symbol,
                        interval_seconds=1,
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        open=float(
                            data.get("open", close)
                        ),
                        high=float(
                            data.get("high", close)
                        ),
                        low=float(
                            data.get("low", close)
                        ),
                        close=close,
                        volume=float(
                            data.get("volume", 0.0)
                        ),
                        trade_count=int(
                            data.get(
                                "trade_count",
                                0,
                            )
                        ),
                        first_trade_timestamp=int(
                            data.get(
                                "first_trade_timestamp",
                                0,
                            )
                        ),
                        last_trade_timestamp=int(
                            data.get(
                                "last_trade_timestamp",
                                0,
                            )
                        ),
                    )
                )

        candles.sort(
            key=lambda candle: (
                candle.symbol,
                candle.start_timestamp,
            )
        )

        # Keep only the latest candles per symbol.
        by_symbol: dict[str, list[Any]] = {}

        for candle in candles:
            by_symbol.setdefault(
                candle.symbol,
                [],
            ).append(candle)

        result: list[Any] = []

        for symbol, symbol_candles in by_symbol.items():
            result.extend(
                symbol_candles[-self.max_candles :]
            )

            self.logger.info(
                "Historical warm-up | symbol=%s | "
                "1s_candles=%d",
                symbol,
                min(
                    len(symbol_candles),
                    self.max_candles,
                ),
            )

        return sorted(
            result,
            key=lambda candle: (
                candle.symbol,
                candle.start_timestamp,
            ),
        )

    def load_into_engine(
        self,
        strategy_engine: Any,
        strategy_history: dict[str, deque],
        max_history: int,
    ) -> int:
        candles = self.load()

        if not candles:
            return 0

        loaded = 0

        for candle in candles:
            symbol = candle.symbol

            history = strategy_history.setdefault(
                symbol,
                deque(maxlen=max_history),
            )

            history.append(candle)

            # Seed the Phase strategy engine directly.
            key = (
                symbol,
                int(candle.interval_seconds),
            )

            if hasattr(strategy_engine, "_candles"):
                buffer = strategy_engine._candles.setdefault(
                    key,
                    deque(maxlen=strategy_engine.max_candles),
                )

                buffer.append(candle)

            loaded += 1

        self.logger.info(
            "Historical strategy warm-up completed | "
            "candles_loaded=%d",
            loaded,
        )

        return loaded
