
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MarketDataEvent:
    """
    Normalized market-data event.

    The raw Delta message is preserved in `payload`.

    `normalized` contains exchange-independent fields that
    downstream strategy, indicator, risk, and ML components
    can use without knowing Delta's compact field format.

    All timestamps are represented as integer microseconds.
    """

    event_type: str
    symbol: Optional[str]
    exchange_timestamp: Optional[int]
    received_timestamp: int

    payload: Dict[str, Any] = field(
        default_factory=dict
    )

    normalized: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def latency_ms(self) -> Optional[float]:
        """
        Calculate approximate event latency.

        Returns None if the exchange timestamp is unavailable.
        """

        if self.exchange_timestamp is None:
            return None

        difference = (
            self.received_timestamp
            - self.exchange_timestamp
        )

        return difference / 1000.0

    @property
    def is_market_data(self) -> bool:
        """
        Return True for actual market-data events.
        """

        return self.event_type in {
            "ticker",
            "ob_l1",
            "ob_l2",
            "ob_updates",
            "trades",
        }

    @property
    def last_price(self) -> Optional[float]:
        """
        Return normalized last traded/last ticker price.
        """

        value = self.normalized.get(
            "last_price"
        )

        if value is None:
            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @property
    def mark_price(self) -> Optional[float]:
        """
        Return normalized mark price.
        """

        value = self.normalized.get(
            "mark_price"
        )

        if value is None:
            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @property
    def bid(self) -> Optional[float]:
        """
        Return normalized best bid.
        """

        value = self.normalized.get(
            "bid"
        )

        if value is None:
            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @property
    def ask(self) -> Optional[float]:
        """
        Return normalized best ask.
        """

        value = self.normalized.get(
            "ask"
        )

        if value is None:
            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None


@dataclass
class MarketDataStats:
    """
    Runtime statistics for the market-data collector.
    """

    total_messages: int = 0
    valid_messages: int = 0
    invalid_messages: int = 0

    ticker_messages: int = 0
    orderbook_messages: int = 0
    trade_messages: int = 0
    system_messages: int = 0

    unknown_messages: int = 0

    last_received_timestamp: Optional[int] = None
    last_exchange_timestamp: Optional[int] = None

    # ---------------------------------------------------------
    # Additional diagnostics
    # ---------------------------------------------------------

    ticker_valid_messages: int = 0
    ticker_invalid_messages: int = 0

    orderbook_valid_messages: int = 0
    orderbook_invalid_messages: int = 0

    trade_valid_messages: int = 0
    trade_invalid_messages: int = 0

    def record_message(
        self,
        event_type: str,
        valid: bool,
        received_timestamp: int,
        exchange_timestamp: Optional[int],
    ) -> None:

        self.total_messages += 1

        if valid:
            self.valid_messages += 1
        else:
            self.invalid_messages += 1

        if event_type == "ticker":

            self.ticker_messages += 1

            if valid:
                self.ticker_valid_messages += 1
            else:
                self.ticker_invalid_messages += 1

        elif event_type in {
            "ob_l1",
            "ob_l2",
            "ob_updates",
        }:

            self.orderbook_messages += 1

            if valid:
                self.orderbook_valid_messages += 1
            else:
                self.orderbook_invalid_messages += 1

        elif event_type == "trades":

            self.trade_messages += 1

            if valid:
                self.trade_valid_messages += 1
            else:
                self.trade_invalid_messages += 1

        elif event_type == "system_status":

            self.system_messages += 1

        else:

            self.unknown_messages += 1

        self.last_received_timestamp = (
            received_timestamp
        )

        self.last_exchange_timestamp = (
            exchange_timestamp
        )
