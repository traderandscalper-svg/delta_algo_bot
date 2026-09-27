
"""
Local OHLCV candle aggregation from validated trade events.

This module does not submit orders and does not use exchange-generated
candles. It builds candles from validated local trade events.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.market_data.models import MarketDataEvent


CandleHandler = Callable[["Candle"], None]


@dataclass
class Candle:
    symbol: str
    interval_seconds: int
    start_timestamp: int
    end_timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    first_trade_timestamp: int
    last_trade_timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _CandleBuilder:
    symbol: str
    interval_seconds: int
    start_timestamp: int
    end_timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    first_trade_timestamp: int
    last_trade_timestamp: int

    @classmethod
    def create(
        cls,
        symbol: str,
        interval_seconds: int,
        start_timestamp: int,
        price: float,
        quantity: float,
        trade_timestamp: int,
    ) -> "_CandleBuilder":
        end_timestamp = (
            start_timestamp
            + interval_seconds * 1_000_000
        )

        return cls(
            symbol=symbol,
            interval_seconds=interval_seconds,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=quantity,
            trade_count=1,
            first_trade_timestamp=trade_timestamp,
            last_trade_timestamp=trade_timestamp,
        )

    def add_trade(
        self,
        price: float,
        quantity: float,
        trade_timestamp: int,
    ) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += quantity
        self.trade_count += 1
        self.last_trade_timestamp = trade_timestamp

    def to_candle(self) -> Candle:
        """
        Convert the current builder into a completed candle.

        Named to_candle() instead of close() to avoid conflicting
        with the dataclass field named close.
        """
        return Candle(
            symbol=self.symbol,
            interval_seconds=self.interval_seconds,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            trade_count=self.trade_count,
            first_trade_timestamp=self.first_trade_timestamp,
            last_trade_timestamp=self.last_trade_timestamp,
        )


class CandleAggregator:
    """
    Thread-safe local candle aggregator.

    Timestamps are represented in integer microseconds.

    A trade can be assigned to multiple configured intervals,
    such as 1-second, 5-second, 15-second, and 60-second candles.
    """

    def __init__(
        self,
        intervals_seconds: Optional[List[int]] = None,
        output_directory: str = "data/market_data",
        candle_handler: Optional[CandleHandler] = None,
    ) -> None:
        self.logger = logging.getLogger(
            "CandleAggregator"
        )

        self.intervals_seconds = sorted(
            set(intervals_seconds or [1, 5, 15, 60])
        )

        if not self.intervals_seconds:
            raise ValueError(
                "At least one candle interval is required."
            )

        if any(
            interval <= 0
            for interval in self.intervals_seconds
        ):
            raise ValueError(
                "Candle intervals must be positive."
            )

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_file = (
            self.output_directory
            / "candles.jsonl"
        )

        self.candle_handler = candle_handler

        self._builders: Dict[
            tuple[str, int],
            _CandleBuilder,
        ] = {}

        self._lock = threading.RLock()

        self._closed_candles = 0
        self._ignored_events = 0

    @staticmethod
    def _get_value(
        payload: Dict[str, Any],
        *keys: str,
    ) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]

        return None

    @classmethod
    def _extract_trade(
        cls,
        event: MarketDataEvent,
    ) -> Optional[tuple[str, float, float, int]]:
        if event.event_type != "trades":
            return None

        payload = event.payload

        symbol = cls._get_value(
            payload,
            "sy",
            "symbol",
            "product_symbol",
        )

        price_value = cls._get_value(
            payload,
            "p",
            "price",
            "trade_price",
        )

        quantity_value = cls._get_value(
            payload,
            "s",
            "size",
            "q",
            "quantity",
            "trade_size",
        )

        timestamp = (
            event.exchange_timestamp
            or event.received_timestamp
        )

        if symbol is None:
            return None

        try:
            price = float(price_value)
            quantity = float(quantity_value)
            timestamp = int(timestamp)

        except (TypeError, ValueError):
            return None

        if not symbol:
            return None

        if price <= 0:
            return None

        if quantity <= 0:
            return None

        if timestamp <= 0:
            return None

        return (
            str(symbol),
            price,
            quantity,
            timestamp,
        )

    @staticmethod
    def _bucket_start(
        timestamp: int,
        interval_seconds: int,
    ) -> int:
        interval_microseconds = (
            interval_seconds * 1_000_000
        )

        return (
            timestamp // interval_microseconds
        ) * interval_microseconds

    def _write_candle(
        self,
        candle: Candle,
    ) -> None:
        record = candle.to_dict()

        try:
            with self.output_file.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        except OSError:
            self.logger.exception(
                "Unable to save completed candle."
            )

    def _emit_candle(
        self,
        candle: Candle,
    ) -> None:
        self._write_candle(candle)

        self._closed_candles += 1

        self.logger.debug(
            "Candle closed | symbol=%s | interval=%ss | "
            "start=%s | open=%s | high=%s | low=%s | "
            "close=%s | volume=%s | trades=%s",
            candle.symbol,
            candle.interval_seconds,
            candle.start_timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.trade_count,
        )

        if self.candle_handler is not None:
            try:
                self.candle_handler(candle)

            except Exception:
                self.logger.exception(
                    "Candle handler failed."
                )

    def _process_trade_for_interval(
        self,
        symbol: str,
        price: float,
        quantity: float,
        timestamp: int,
        interval_seconds: int,
    ) -> None:
        key = (
            symbol,
            interval_seconds,
        )

        bucket_start = self._bucket_start(
            timestamp,
            interval_seconds,
        )

        current_builder = self._builders.get(key)

        if current_builder is None:
            self._builders[key] = (
                _CandleBuilder.create(
                    symbol=symbol,
                    interval_seconds=interval_seconds,
                    start_timestamp=bucket_start,
                    price=price,
                    quantity=quantity,
                    trade_timestamp=timestamp,
                )
            )

            return

        if bucket_start < current_builder.start_timestamp:
            self._ignored_events += 1

            self.logger.warning(
                "Out-of-order trade ignored | symbol=%s | "
                "interval=%s | timestamp=%s | current_start=%s",
                symbol,
                interval_seconds,
                timestamp,
                current_builder.start_timestamp,
            )

            return

        if bucket_start == current_builder.start_timestamp:
            current_builder.add_trade(
                price=price,
                quantity=quantity,
                trade_timestamp=timestamp,
            )

            return

        completed_candle = (
            current_builder.to_candle()
        )

        self._emit_candle(
            completed_candle
        )

        self._builders[key] = (
            _CandleBuilder.create(
                symbol=symbol,
                interval_seconds=interval_seconds,
                start_timestamp=bucket_start,
                price=price,
                quantity=quantity,
                trade_timestamp=timestamp,
            )
        )

    def process_event(
        self,
        event: MarketDataEvent,
    ) -> None:
        extracted_trade = (
            self._extract_trade(event)
        )

        if extracted_trade is None:
            if event.event_type == "trades":
                with self._lock:
                    self._ignored_events += 1

            return

        symbol, price, quantity, timestamp = (
            extracted_trade
        )

        with self._lock:
            for interval_seconds in (
                self.intervals_seconds
            ):
                self._process_trade_for_interval(
                    symbol=symbol,
                    price=price,
                    quantity=quantity,
                    timestamp=timestamp,
                    interval_seconds=interval_seconds,
                )

    def flush(self) -> None:
        """
        Close all currently open candles.

        Use this during controlled shutdown or testing.
        """
        with self._lock:
            builders = list(
                self._builders.values()
            )

            self._builders.clear()

            for builder in builders:
                self._emit_candle(
                    builder.to_candle()
                )

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "active_builders": len(
                    self._builders
                ),
                "closed_candles": (
                    self._closed_candles
                ),
                "ignored_events": (
                    self._ignored_events
                ),
            }

    def clear(self) -> None:
        with self._lock:
            self._builders.clear()