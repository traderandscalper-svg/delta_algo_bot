
"""
Thread-safe market-data state management.

This module stores the latest validated market-data events.
It does not submit orders or make trading decisions.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.market_data.models import MarketDataEvent


@dataclass
class SymbolMarketState:
    """Latest known market state for one symbol."""

    symbol: str

    latest_ticker: Optional[Dict[str, Any]] = None
    latest_orderbook: Optional[Dict[str, Any]] = None
    latest_trade: Optional[Dict[str, Any]] = None

    last_event_type: Optional[str] = None
    last_exchange_timestamp: Optional[int] = None
    last_received_timestamp: Optional[int] = None

    valid_event_count: int = 0
    invalid_event_count: int = 0

    @property
    def best_bid(self) -> Optional[float]:
        if not self.latest_orderbook:
            return None

        value = self.latest_orderbook.get("bp")

        if value is None:
            value = self.latest_orderbook.get("bid_price")

        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def best_ask(self) -> Optional[float]:
        if not self.latest_orderbook:
            return None

        value = self.latest_orderbook.get("ap")

        if value is None:
            value = self.latest_orderbook.get("ask_price")

        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def mid_price(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask

        if bid is None or ask is None:
            return None

        return (bid + ask) / 2.0

    @property
    def spread(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask

        if bid is None or ask is None:
            return None

        return ask - bid

    @property
    def spread_bps(self) -> Optional[float]:
        mid = self.mid_price
        spread = self.spread

        if mid is None or spread is None or mid <= 0:
            return None

        return (spread / mid) * 10_000


class MarketDataStateManager:
    """
    Maintain the latest validated market-data state.

    Thread-safe:
        WebSocket callbacks update the state.
        The trading engine reads the state.
    """

    def __init__(
        self,
        stale_timeout_seconds: float = 5.0,
    ) -> None:
        self.stale_timeout_seconds = stale_timeout_seconds

        self._states: Dict[str, SymbolMarketState] = {}
        self._lock = threading.RLock()

        self._last_event_monotonic: Optional[float] = None
        self._total_events = 0
        self._invalid_events = 0

    def update(self, event: MarketDataEvent) -> None:
        """Update state using a validated market-data event."""

        if not event.symbol:
            return

        with self._lock:
            state = self._states.setdefault(
                event.symbol,
                SymbolMarketState(symbol=event.symbol),
            )

            event_type = event.event_type
            payload = event.payload

            if event_type == "ticker":
                state.latest_ticker = dict(payload)

            elif event_type in {
                "ob_l1",
                "ob_l2",
                "ob_updates",
            }:
                state.latest_orderbook = dict(payload)

            elif event_type == "trades":
                state.latest_trade = dict(payload)

            state.last_event_type = event_type
            state.last_exchange_timestamp = (
                event.exchange_timestamp
            )
            state.last_received_timestamp = (
                event.received_timestamp
            )

            state.valid_event_count += 1

            self._total_events += 1
            self._last_event_monotonic = time.monotonic()

    def record_invalid_event(self) -> None:
        """Record an invalid event without updating market state."""

        with self._lock:
            self._invalid_events += 1

    def get_state(
        self,
        symbol: str,
    ) -> Optional[SymbolMarketState]:
        """Return a snapshot of the latest state for a symbol."""

        with self._lock:
            state = self._states.get(symbol)

            if state is None:
                return None

            return SymbolMarketState(
                symbol=state.symbol,
                latest_ticker=(
                    dict(state.latest_ticker)
                    if state.latest_ticker is not None
                    else None
                ),
                latest_orderbook=(
                    dict(state.latest_orderbook)
                    if state.latest_orderbook is not None
                    else None
                ),
                latest_trade=(
                    dict(state.latest_trade)
                    if state.latest_trade is not None
                    else None
                ),
                last_event_type=state.last_event_type,
                last_exchange_timestamp=(
                    state.last_exchange_timestamp
                ),
                last_received_timestamp=(
                    state.last_received_timestamp
                ),
                valid_event_count=state.valid_event_count,
                invalid_event_count=state.invalid_event_count,
            )

    def get_all_states(self) -> Dict[str, SymbolMarketState]:
        """Return snapshots for all known symbols."""

        with self._lock:
            symbols = list(self._states.keys())

        result: Dict[str, SymbolMarketState] = {}

        for symbol in symbols:
            state = self.get_state(symbol)

            if state is not None:
                result[symbol] = state

        return result

    def is_stale(self, symbol: str) -> bool:
        """Check whether the latest event for a symbol is stale."""

        state = self.get_state(symbol)

        if state is None:
            return True

        if state.last_received_timestamp is None:
            return True

        now_us = time.time_ns() // 1_000
        age_seconds = (
            now_us - state.last_received_timestamp
        ) / 1_000_000

        return age_seconds > self.stale_timeout_seconds

    def is_healthy(self, symbol: str) -> bool:
        """Check basic market-data health for a symbol."""

        state = self.get_state(symbol)

        if state is None:
            return False

        if self.is_stale(symbol):
            return False

        if state.best_bid is not None:
            if state.best_bid <= 0:
                return False

        if state.best_ask is not None:
            if state.best_ask <= 0:
                return False

        bid = state.best_bid
        ask = state.best_ask

        if bid is not None and ask is not None:
            if ask < bid:
                return False

        return True

    def get_health_summary(self) -> Dict[str, Any]:
        """Return a compact health summary."""

        with self._lock:
            symbols = list(self._states.keys())

            summary = {
                "symbols": {},
                "total_events": self._total_events,
                "invalid_events": self._invalid_events,
            }

        for symbol in symbols:
            state = self.get_state(symbol)

            if state is None:
                continue

            summary["symbols"][symbol] = {
                "healthy": self.is_healthy(symbol),
                "stale": self.is_stale(symbol),
                "last_event_type": state.last_event_type,
                "best_bid": state.best_bid,
                "best_ask": state.best_ask,
                "mid_price": state.mid_price,
                "spread": state.spread,
                "spread_bps": state.spread_bps,
                "valid_event_count": state.valid_event_count,
            }

        return summary

    def clear(self) -> None:
        """Clear all stored market state."""

        with self._lock:
            self._states.clear()
            self._last_event_monotonic = None
            self._total_events = 0
            self._invalid_events = 0