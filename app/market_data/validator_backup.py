
"""
Market-data validation and normalization for Delta Exchange.

This module:

    - Validates incoming WebSocket messages.
    - Supports Delta compact fields.
    - Correctly handles Delta compact ticker payloads.
    - Extracts symbols from top-level and nested payloads.
    - Normalizes ticker data.
    - Normalizes L1 order-book data.
    - Normalizes trade data.
    - Applies freshness checks to market-data messages.
    - Ignores stale timestamp checks for control messages.
    - Tracks market-data validation statistics.

Important Delta ticker example:

{
    "d": [{
        "m": "83968.58289658",
        "m24hc": "-0.3353",
        "ohlc": [
            84276.5,
            85250.0,
            83143.0,
            83967.5
        ],
        "oi": [
            "843041",
            "0.0000"
        ],
        "q": [
            "83967.5",
            "245",
            "83963.5",
            "942",
            null
        ],
        "s": "BTCUSD"
    }],
    "sp": "83983.3",
    "sy": "BTCUSD",
    "ts": 1790398040587142,
    "type": "ticker"
}

Ticker compact interpretation:

    m       -> mark price
    m24hc   -> 24h change
    ohlc[0] -> open
    ohlc[1] -> high
    ohlc[2] -> low
    ohlc[3] -> ticker close / last price
    oi[0]   -> open interest
    oi[1]   -> open-interest change
    q[0]    -> ask price
    q[1]    -> ask size
    q[2]    -> bid price
    q[3]    -> bid size
    sp      -> special/reference price

Important:

    Trade payload fields are NOT assumed to use ticker
    compact-field semantics.

    In particular, "s" is NOT automatically interpreted
    as trade size because "s" is the ticker symbol field
    in the compact ticker payload.

The raw exchange message is NEVER discarded.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from app.market_data.models import (
    MarketDataEvent,
    MarketDataStats,
)


class MarketDataValidator:
    """
    Validate and normalize Delta Exchange WebSocket messages.
    """

    MARKET_EVENT_TYPES = {
        "ticker",
        "ob_l1",
        "ob_l2",
        "ob_updates",
        "trades",
    }

    CONTROL_EVENT_TYPES = {
        "system_status",
        "subscriptions",
        "heartbeat",
    }

    SUPPORTED_EVENT_TYPES = (
        MARKET_EVENT_TYPES
        | CONTROL_EVENT_TYPES
    )

    SYMBOL_KEYS = {
        "sy",
        "symbol",
        "product_symbol",
        "instrument_symbol",
        "contract_symbol",
    }

    def __init__(
        self,
        stale_timeout_seconds: float = 10.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:

        self.stale_timeout_seconds = (
            stale_timeout_seconds
        )

        self.logger = (
            logger
            or logging.getLogger(
                self.__class__.__name__
            )
        )

        self.stats = MarketDataStats()

    # =============================================================
    # MAIN VALIDATION
    # =============================================================

    def validate(
        self,
        message: Any,
    ) -> tuple[
        bool,
        Optional[MarketDataEvent],
        Optional[str],
    ]:

        received_timestamp = (
            self._current_timestamp_us()
        )

        if not isinstance(message, dict):

            self._record_invalid(
                event_type="unknown",
                received_timestamp=received_timestamp,
                exchange_timestamp=None,
            )

            return (
                False,
                None,
                "Message is not a dictionary",
            )

        event_type = (
            self._extract_event_type(
                message
            )
        )

        if not event_type:

            self._record_invalid(
                event_type="unknown",
                received_timestamp=received_timestamp,
                exchange_timestamp=None,
            )

            return (
                False,
                None,
                "Missing event type",
            )

        event_type = str(
            event_type
        ).lower()

        if (
            event_type
            not in self.SUPPORTED_EVENT_TYPES
        ):

            self.stats.unknown_messages += 1

            self.logger.debug(
                "Unknown market-data event type | "
                "type=%s",
                event_type,
            )

            return (
                False,
                None,
                f"Unsupported event type: {event_type}",
            )

        symbol = (
            self._extract_symbol(
                message
            )
        )

        if (
            event_type in self.MARKET_EVENT_TYPES
            and not symbol
        ):

            self._record_invalid(
                event_type=event_type,
                received_timestamp=received_timestamp,
                exchange_timestamp=None,
            )

            return (
                False,
                None,
                "Missing symbol for market-data event",
            )

        exchange_timestamp = (
            self._extract_exchange_timestamp(
                message
            )
        )

        if (
            event_type in self.MARKET_EVENT_TYPES
            and exchange_timestamp is not None
        ):

            if exchange_timestamp <= 0:

                self._record_invalid(
                    event_type=event_type,
                    received_timestamp=received_timestamp,
                    exchange_timestamp=exchange_timestamp,
                )

                return (
                    False,
                    None,
                    "Invalid exchange timestamp",
                )

            age_seconds = (
                received_timestamp
                - exchange_timestamp
            ) / 1_000_000.0

            if age_seconds < -5:

                self._record_invalid(
                    event_type=event_type,
                    received_timestamp=received_timestamp,
                    exchange_timestamp=exchange_timestamp,
                )

                return (
                    False,
                    None,
                    "Exchange timestamp is in the future",
                )

            if (
                age_seconds
                > self.stale_timeout_seconds
            ):

                self._record_invalid(
                    event_type=event_type,
                    received_timestamp=received_timestamp,
                    exchange_timestamp=exchange_timestamp,
                )

                return (
                    False,
                    None,
                    (
                        "Stale message: "
                        f"{age_seconds:.2f} "
                        "seconds old"
                    ),
                )

        validation_error = (
            self._validate_event_payload(
                event_type=event_type,
                message=message,
            )
        )

        if validation_error is not None:

            self._record_invalid(
                event_type=event_type,
                received_timestamp=received_timestamp,
                exchange_timestamp=exchange_timestamp,
            )

            return (
                False,
                None,
                validation_error,
            )

        normalized = (
            self._normalize_event(
                event_type=event_type,
                message=message,
            )
        )

        normalized_event = MarketDataEvent(
            event_type=event_type,
            symbol=symbol,
            exchange_timestamp=exchange_timestamp,
            received_timestamp=received_timestamp,
            payload=message,
            normalized=normalized,
        )

        self.stats.record_message(
            event_type=event_type,
            valid=True,
            received_timestamp=received_timestamp,
            exchange_timestamp=exchange_timestamp,
        )

        return (
            True,
            normalized_event,
            None,
        )

    # =============================================================
    # EVENT VALIDATION
    # =============================================================

    def _validate_event_payload(
        self,
        event_type: str,
        message: dict[str, Any],
    ) -> Optional[str]:

        if event_type == "ob_l1":

            return self._validate_orderbook_l1(
                message
            )

        if event_type in {
            "ob_l2",
            "ob_updates",
        }:

            return self._validate_orderbook_data(
                message
            )

        if event_type == "trades":

            return self._validate_trade_data(
                message
            )

        if event_type == "ticker":

            return self._validate_ticker_data(
                message
            )

        if event_type in self.CONTROL_EVENT_TYPES:

            return None

        return None

    # =============================================================
    # DELTA PAYLOAD EXTRACTION
    # =============================================================

    @staticmethod
    def _get_delta_payload(
        message: dict[str, Any],
    ) -> dict[str, Any]:

        data = message.get("d")

        if isinstance(data, list):

            for item in data:

                if isinstance(item, dict):

                    return item

        if isinstance(data, dict):

            return data

        payload = message.get(
            "payload"
        )

        if isinstance(payload, dict):

            return payload

        data = message.get(
            "data"
        )

        if isinstance(data, dict):

            return data

        return message

    # =============================================================
    # ORDER BOOK VALIDATION
    # =============================================================

    def _validate_orderbook_l1(
        self,
        message: dict[str, Any],
    ) -> Optional[str]:

        payload = (
            self._get_delta_payload(
                message
            )
        )

        bid_price = (
            self._find_first_value(
                payload,
                "bp",
                "bid_price",
                "best_bid",
            )
        )

        bid_size = (
            self._find_first_value(
                payload,
                "bs",
                "bid_size",
                "best_bid_size",
            )
        )

        ask_price = (
            self._find_first_value(
                payload,
                "ap",
                "ask_price",
                "best_ask",
            )
        )

        ask_size = (
            self._find_first_value(
                payload,
                "as",
                "ask_size",
                "best_ask_size",
            )
        )

        if bid_price is None:
            return "Missing bid price"

        if bid_size is None:
            return "Missing bid size"

        if ask_price is None:
            return "Missing ask price"

        if ask_size is None:
            return "Missing ask size"

        if not self._is_number(bid_price):
            return "Invalid bid price"

        if not self._is_number(bid_size):
            return "Invalid bid size"

        if not self._is_number(ask_price):
            return "Invalid ask price"

        if not self._is_number(ask_size):
            return "Invalid ask size"

        if float(bid_price) <= 0:
            return "Bid price must be positive"

        if float(ask_price) <= 0:
            return "Ask price must be positive"

        if float(bid_size) < 0:
            return "Bid size cannot be negative"

        if float(ask_size) < 0:
            return "Ask size cannot be negative"

        if float(ask_price) < float(bid_price):
            return "Ask price is below bid price"

        return None

    # =============================================================
    # ORDER BOOK L2 VALIDATION
    # =============================================================

    def _validate_orderbook_data(
        self,
        message: dict[str, Any],
    ) -> Optional[str]:

        payload = (
            self._get_delta_payload(
                message
            )
        )

        if not payload:

            return "Empty order-book payload"

        return None

    # =============================================================
    # TRADE VALIDATION
    # =============================================================

    def _validate_trade_data(
        self,
        message: dict[str, Any],
    ) -> Optional[str]:

        payload = (
            self._get_delta_payload(
                message
            )
        )

        price = (
            self._find_first_value(
                payload,
                "p",
                "price",
                "trade_price",
            )
        )

        size = (
            self._find_first_value(
                payload,
                "size",
                "quantity",
                "trade_size",
                "qty",
                "q",
                "volume",
            )
        )

        if price is None:

            return "Missing trade price"

        if size is None:

            return "Missing trade size"

        if not self._is_number(price):

            return "Invalid trade price"

        if not self._is_number(size):

            return "Invalid trade size"

        if float(price) <= 0:

            return "Trade price must be positive"

        if float(size) <= 0:

            return "Trade size must be positive"

        return None

    # =============================================================
    # TICKER VALIDATION
    # =============================================================

    def _validate_ticker_data(
        self,
        message: dict[str, Any],
    ) -> Optional[str]:

        payload = (
            self._get_delta_payload(
                message
            )
        )

        if not payload:

            return "Empty ticker payload"

        ohlc = payload.get(
            "ohlc"
        )

        if isinstance(
            ohlc,
            (list, tuple),
        ):

            if len(ohlc) >= 4:

                close = ohlc[3]

                if (
                    self._is_number(close)
                    and float(close) > 0
                ):

                    return None

        price = (
            self._find_first_value(
                payload,
                "last_price",
                "price",
                "close",
                "p",
            )
        )

        mark_price = (
            self._find_first_value(
                payload,
                "mark_price",
                "mark",
            )
        )

        if (
            price is not None
            and self._is_number(price)
            and float(price) > 0
        ):

            return None

        if (
            mark_price is not None
            and self._is_number(mark_price)
            and float(mark_price) > 0
        ):

            return None

        return (
            "Ticker contains no valid price "
            "or mark price"
        )

    # =============================================================
    # EVENT NORMALIZATION
    # =============================================================

    def _normalize_event(
        self,
        event_type: str,
        message: dict[str, Any],
    ) -> dict[str, Any]:

        if event_type == "ticker":

            return self._normalize_ticker(
                message
            )

        if event_type == "ob_l1":

            return self._normalize_orderbook_l1(
                message
            )

        if event_type == "trades":

            return self._normalize_trade(
                message
            )

        return {}

    # =============================================================
    # TICKER NORMALIZATION
    # =============================================================

    def _normalize_ticker(
        self,
        message: dict[str, Any],
    ) -> dict[str, Any]:

        payload = (
            self._get_delta_payload(
                message
            )
        )

        symbol = (
            payload.get("s")
            or message.get("sy")
            or message.get("symbol")
        )

        mark_price = self._safe_float(
            payload.get("m")
        )

        if mark_price is None:

            mark_price = self._safe_float(
                payload.get("mark_price")
            )

        ohlc = payload.get(
            "ohlc"
        )

        open_price = None
        high_price = None
        low_price = None
        close_price = None

        if isinstance(
            ohlc,
            (list, tuple),
        ):

            if len(ohlc) >= 1:
                open_price = self._safe_float(
                    ohlc[0]
                )

            if len(ohlc) >= 2:
                high_price = self._safe_float(
                    ohlc[1]
                )

            if len(ohlc) >= 3:
                low_price = self._safe_float(
                    ohlc[2]
                )

            if len(ohlc) >= 4:
                close_price = self._safe_float(
                    ohlc[3]
                )

        last_price = close_price

        if last_price is None:

            last_price = self._safe_float(
                payload.get("last_price")
            )

        if last_price is None:

            last_price = self._safe_float(
                payload.get("price")
            )

        quote = payload.get(
            "q"
        )

        ask = None
        ask_size = None
        bid = None
        bid_size = None

        if isinstance(
            quote,
            (list, tuple),
        ):

            if len(quote) >= 1:
                ask = self._safe_float(
                    quote[0]
                )

            if len(quote) >= 2:
                ask_size = self._safe_float(
                    quote[1]
                )

            if len(quote) >= 3:
                bid = self._safe_float(
                    quote[2]
                )

            if len(quote) >= 4:
                bid_size = self._safe_float(
                    quote[3]
                )

        oi = payload.get(
            "oi"
        )

        open_interest = None
        open_interest_change = None

        if isinstance(
            oi,
            (list, tuple),
        ):

            if len(oi) >= 1:
                open_interest = self._safe_float(
                    oi[0]
                )

            if len(oi) >= 2:
                open_interest_change = self._safe_float(
                    oi[1]
                )

        change_24h = self._safe_float(
            payload.get("m24hc")
        )

        special_price = self._safe_float(
            message.get("sp")
        )

        exchange_timestamp = (
            self._extract_exchange_timestamp(
                message
            )
        )

        return {
            "event_type": "ticker",
            "symbol": symbol,

            "last_price": last_price,
            "mark_price": mark_price,

            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,

            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,

            "open_interest": open_interest,
            "open_interest_change": (
                open_interest_change
            ),

            "change_24h": change_24h,

            "special_price": special_price,

            "exchange_timestamp": (
                exchange_timestamp
            ),
        }

    # =============================================================
    # ORDER BOOK NORMALIZATION
    # =============================================================

    def _normalize_orderbook_l1(
        self,
        message: dict[str, Any],
    ) -> dict[str, Any]:

        payload = (
            self._get_delta_payload(
                message
            )
        )

        bid = self._safe_float(
            self._find_first_value(
                payload,
                "bp",
                "bid_price",
                "best_bid",
            )
        )

        bid_size = self._safe_float(
            self._find_first_value(
                payload,
                "bs",
                "bid_size",
                "best_bid_size",
            )
        )

        ask = self._safe_float(
            self._find_first_value(
                payload,
                "ap",
                "ask_price",
                "best_ask",
            )
        )

        ask_size = self._safe_float(
            self._find_first_value(
                payload,
                "as",
                "ask_size",
                "best_ask_size",
            )
        )

        mid_price = None

        if (
            bid is not None
            and ask is not None
        ):

            mid_price = (
                bid + ask
            ) / 2.0

        spread = None

        if (
            bid is not None
            and ask is not None
        ):

            spread = (
                ask - bid
            )

        spread_bps = None

        if (
            spread is not None
            and mid_price
            and mid_price > 0
        ):

            spread_bps = (
                spread
                / mid_price
                * 10_000
            )

        return {
            "event_type": "ob_l1",

            "symbol": (
                payload.get("sy")
                or message.get("sy")
                or message.get("symbol")
            ),

            "bid": bid,
            "bid_size": bid_size,

            "ask": ask,
            "ask_size": ask_size,

            "mid_price": mid_price,
            "spread": spread,
            "spread_bps": spread_bps,
        }

    # =============================================================
    # TRADE NORMALIZATION
    # =============================================================

    def _normalize_trade(
        self,
        message: dict[str, Any],
    ) -> dict[str, Any]:

        payload = (
            self._get_delta_payload(
                message
            )
        )

        price = self._safe_float(
            self._find_first_value(
                payload,
                "p",
                "price",
                "trade_price",
            )
        )

        size = self._safe_float(
            self._find_first_value(
                payload,
                "size",
                "quantity",
                "trade_size",
                "qty",
                "q",
                "volume",
            )
        )

        symbol = (
            payload.get("sy")
            or payload.get("symbol")
            or message.get("sy")
            or message.get("symbol")
        )

        return {
            "event_type": "trades",
            "symbol": symbol,
            "price": price,
            "size": size,

            # Preserve useful trade-side fields
            # when Delta supplies them.
            "side": (
                payload.get("side")
                or payload.get("taker_side")
                or payload.get("direction")
            ),

            "trade_id": (
                payload.get("trade_id")
                or payload.get("id")
            ),

            "exchange_timestamp": (
                self._extract_exchange_timestamp(
                    message
                )
            ),
        }

    # =============================================================
    # EVENT TYPE
    # =============================================================

    @staticmethod
    def _extract_event_type(
        message: dict[str, Any],
    ) -> Optional[str]:

        event_type = (
            message.get("type")
            or message.get("event")
            or message.get("channel")
        )

        if event_type is None:

            return None

        return str(event_type)

    # =============================================================
    # SYMBOL
    # =============================================================

    @classmethod
    def _extract_symbol(
        cls,
        message: dict[str, Any],
    ) -> Optional[str]:

        def find_symbol(
            value: Any,
        ) -> Optional[str]:

            if isinstance(
                value,
                dict,
            ):

                for key in cls.SYMBOL_KEYS:

                    candidate = value.get(
                        key
                    )

                    if (
                        candidate is not None
                        and isinstance(
                            candidate,
                            str,
                        )
                        and candidate.strip()
                    ):

                        return candidate.strip()

                for nested_value in value.values():

                    result = find_symbol(
                        nested_value
                    )

                    if result is not None:

                        return result

            elif isinstance(
                value,
                list,
            ):

                for item in value:

                    result = find_symbol(
                        item
                    )

                    if result is not None:

                        return result

            return None

        return find_symbol(
            message
        )

    # =============================================================
    # TIMESTAMP
    # =============================================================

    @classmethod
    def _extract_exchange_timestamp(
        cls,
        message: dict[str, Any],
    ) -> Optional[int]:

        def find_timestamp(
            value: Any,
        ) -> Any:

            if isinstance(
                value,
                dict,
            ):

                if value.get("ts") is not None:

                    return value["ts"]

                for key in (
                    "timestamp",
                    "event_time",
                    "exchange_timestamp",
                ):

                    if value.get(key) is not None:

                        return value[key]

                for nested_value in value.values():

                    result = find_timestamp(
                        nested_value
                    )

                    if result is not None:

                        return result

            elif isinstance(
                value,
                list,
            ):

                for item in value:

                    result = find_timestamp(
                        item
                    )

                    if result is not None:

                        return result

            return None

        timestamp = find_timestamp(
            message
        )

        if timestamp is None:

            return None

        try:

            timestamp_value = float(
                timestamp
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        if timestamp_value < 10**11:

            return int(
                timestamp_value
                * 1_000_000
            )

        if timestamp_value < 10**14:

            return int(
                timestamp_value
                * 1_000
            )

        return int(
            timestamp_value
        )

    # =============================================================
    # RECURSIVE VALUE SEARCH
    # =============================================================

    @classmethod
    def _find_first_value(
        cls,
        value: Any,
        *keys: str,
    ) -> Any:

        if isinstance(
            value,
            dict,
        ):

            for key in keys:

                if key in value:

                    return value[key]

            for nested_value in value.values():

                result = (
                    cls._find_first_value(
                        nested_value,
                        *keys,
                    )
                )

                if result is not None:

                    return result

        elif isinstance(
            value,
            list,
        ):

            for item in value:

                result = (
                    cls._find_first_value(
                        item,
                        *keys,
                    )
                )

                if result is not None:

                    return result

        return None

    # =============================================================
    # NUMBER HELPERS
    # =============================================================

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> Optional[float]:

        if value is None:

            return None

        if isinstance(
            value,
            bool,
        ):

            return None

        try:

            result = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        if result != result:

            return None

        return result

    @classmethod
    def _is_number(
        cls,
        value: Any,
    ) -> bool:

        return (
            cls._safe_float(
                value
            )
            is not None
        )

    # =============================================================
    # STATISTICS
    # =============================================================

    def _record_invalid(
        self,
        event_type: str,
        received_timestamp: int,
        exchange_timestamp: Optional[int],
    ) -> None:

        self.stats.record_message(
            event_type=event_type,
            valid=False,
            received_timestamp=received_timestamp,
            exchange_timestamp=exchange_timestamp,
        )

    @staticmethod
    def _current_timestamp_us() -> int:

        return (
            time.time_ns()
            // 1_000
        )

    def get_stats(
        self,
    ) -> MarketDataStats:

        return self.stats

    def reset_stats(
        self,
    ) -> None:

        self.stats = (
            MarketDataStats()
        )
"""
Persistent Delta Exchange public market-data collector.

Collects:
    - Ticker
    - L1 order book
    - Trades
    - System status
    - Subscription events

Features:
    - Real Delta Exchange market data
    - Delta compact ticker parsing
    - Market-data validation
    - Normalized event handling
    - Raw message preservation
    - Persistent JSONL dataset
    - Daily file rotation
    - Buffered file writing
    - Automatic WebSocket reconnection
    - Graceful shutdown
    - Runtime health statistics

No private authentication.
No order placement.
No simulated market data.
"""



import json
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import websocket

from app.market_data.validator import MarketDataValidator


class MarketDataCollector:
    """
    Persistent Delta Exchange public market-data collector.
    """

    def __init__(
        self,
        websocket_url: str,
        symbols: Optional[list] = None,
        output_directory: str = "data/market_data",
        event_handler: Optional[
            Callable[[Any], None]
        ] = None,
        stale_timeout_seconds: float = 10.0,
        reconnect_delay_seconds: float = 5.0,
    ) -> None:

        self.logger = logging.getLogger(
            "MarketDataCollector"
        )

        self.websocket_url = websocket_url

        self.symbols = (
            symbols
            if symbols
            else ["BTCUSD"]
        )

        self.event_handler = event_handler

        self.stale_timeout_seconds = (
            stale_timeout_seconds
        )

        self.reconnect_delay_seconds = (
            reconnect_delay_seconds
        )

        # =========================================================
        # STORAGE
        # =========================================================

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_file = (
            self.output_directory
            / "market_data.jsonl"
        )

        self.current_date = (
            time.strftime(
                "%Y-%m-%d"
            )
        )

        self.file_handle = None

        # =========================================================
        # VALIDATOR
        # =========================================================

        self.validator = (
            MarketDataValidator(
                stale_timeout_seconds=(
                    self.stale_timeout_seconds
                )
            )
        )

        # =========================================================
        # THREAD / STATE
        # =========================================================

        self.stop_event = (
            threading.Event()
        )

        self.ws: Optional[
            websocket.WebSocketApp
        ] = None

        self.connected = False

        self.message_lock = (
            threading.Lock()
        )

        self.state_lock = (
            threading.Lock()
        )

        # =========================================================
        # DIAGNOSTICS
        # =========================================================

        self.trade_diagnostic_count = 0

        # =========================================================
        # STATISTICS
        # =========================================================

        self.last_status_log = (
            time.monotonic()
        )

        self.total_written = 0
        self.total_bytes = 0

        self.dataset_start_time = (
            time.time()
        )

        self.logger.info(
            "MarketDataCollector initialized | "
            "symbols=%s | websocket=%s | output=%s",
            self.symbols,
            self.websocket_url,
            self.output_file,
        )

    # ============================================================
    # SUBSCRIPTION
    # ============================================================

    def _build_subscription_message(
        self,
    ) -> Dict[str, Any]:

        return {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {
                        "name": "ticker",
                        "symbols": self.symbols,
                    },
                    {
                        "name": "ob_l1",
                        "symbols": self.symbols,
                    },
                    {
                        "name": "trades",
                        "symbols": self.symbols,
                    },
                    {
                        "name": "system_status",
                    },
                ]
            },
        }

    # ============================================================
    # STORAGE
    # ============================================================

    def _open_storage(
        self,
    ) -> None:

        today = time.strftime(
            "%Y-%m-%d"
        )

        if (
            self.file_handle is not None
            and today == self.current_date
        ):

            return

        self._close_storage()

        self.current_date = today

        daily_file = (
            self.output_directory
            / f"market_data_{today}.jsonl"
        )

        try:

            self.file_handle = (
                daily_file.open(
                    "a",
                    encoding="utf-8",
                    buffering=1024 * 1024,
                )
            )

            self.output_file = (
                daily_file
            )

            self.logger.info(
                "Dataset storage opened | "
                "file=%s",
                daily_file,
            )

        except OSError:

            self.logger.exception(
                "Unable to open dataset storage."
            )

    def _close_storage(
        self,
    ) -> None:

        if self.file_handle is None:

            return

        try:

            self.file_handle.flush()
            self.file_handle.close()

        except Exception:

            self.logger.exception(
                "Error closing dataset storage."
            )

        finally:

            self.file_handle = None

    # ============================================================
    # EVENT METADATA
    # ============================================================

    def _extract_event_metadata(
        self,
        event: Any,
        message: Dict[str, Any],
    ) -> Dict[str, Any]:

        event_type = getattr(
            event,
            "event_type",
            message.get("type"),
        )

        payload = getattr(
            event,
            "payload",
            message,
        )

        if not isinstance(
            payload,
            dict,
        ):

            payload = message

        symbol = getattr(
            event,
            "symbol",
            None,
        )

        if not symbol:

            symbol = (
                payload.get("sy")
                or payload.get("symbol")
                or message.get("sy")
                or message.get("symbol")
            )

        exchange_timestamp = getattr(
            event,
            "exchange_timestamp",
            None,
        )

        if exchange_timestamp is None:

            exchange_timestamp = (
                message.get(
                    "timestamp"
                )
                or message.get(
                    "ts"
                )
                or message.get(
                    "t"
                )
            )

        sequence = (
            payload.get("sequence")
            or payload.get("seq")
            or message.get("sequence")
            or message.get("seq")
        )

        return {
            "event_type": event_type,
            "symbol": symbol,
            "exchange_timestamp": (
                exchange_timestamp
            ),
            "sequence": sequence,
        }

    # ============================================================
    # WRITE EVENT
    # ============================================================

    def _write_event(
        self,
        message: Dict[str, Any],
        received_at: int,
        is_valid: bool,
        validation_reason: Optional[str],
        event: Any = None,
    ) -> None:

        with self.message_lock:

            self._open_storage()

            metadata = (
                self._extract_event_metadata(
                    event,
                    message,
                )
            )

            normalized = getattr(
                event,
                "normalized",
                None,
            )

            if not isinstance(
                normalized,
                dict,
            ):

                normalized = {}

            record = {
                "received_at_us": received_at,

                "received_at_ns": (
                    received_at * 1000
                ),

                "valid": is_valid,

                "validation_reason": (
                    validation_reason
                ),

                "event_type": metadata[
                    "event_type"
                ],

                "symbol": metadata[
                    "symbol"
                ],

                "exchange_timestamp": (
                    metadata[
                        "exchange_timestamp"
                    ]
                ),

                "sequence": metadata[
                    "sequence"
                ],

                "normalized": normalized,

                # Complete original Delta message.
                "message": message,
            }

            try:

                if self.file_handle is None:

                    self._open_storage()

                if self.file_handle is None:

                    return

                line = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                    )
                    + "\n"
                )

                self.file_handle.write(
                    line
                )

                self.total_written += 1

                self.total_bytes += len(
                    line.encode(
                        "utf-8"
                    )
                )

            except (
                OSError,
                TypeError,
                ValueError,
            ):

                self.logger.exception(
                    "Unable to write "
                    "market-data dataset record."
                )

    # ============================================================
    # FORWARD EVENT
    # ============================================================

    def _forward_event(
        self,
        event: Any,
    ) -> None:

        if self.event_handler is None:

            return

        try:

            self.event_handler(
                event
            )

        except Exception:

            self.logger.exception(
                "Market-data event "
                "handler failed."
            )

    # ============================================================
    # WEBSOCKET ON OPEN
    # ============================================================

    def on_open(
        self,
        ws: websocket.WebSocketApp,
    ) -> None:

        with self.state_lock:

            self.connected = True

        self.logger.info(
            "Public WebSocket connected | "
            "url=%s",
            self.websocket_url,
        )

        subscription = (
            self._build_subscription_message()
        )

        try:

            ws.send(
                json.dumps(
                    subscription
                )
            )

            self.logger.info(
                "Subscription request sent | "
                "symbols=%s",
                self.symbols,
            )

        except Exception:

            self.logger.exception(
                "Failed to send WebSocket "
                "subscription."
            )

    # ============================================================
    # WEBSOCKET MESSAGE
    # ============================================================

    def on_message(
        self,
        ws: websocket.WebSocketApp,
        raw_message: str,
    ) -> None:

        received_at = (
            time.time_ns()
            // 1_000
        )

        if not raw_message:

            self.logger.warning(
                "Received empty WebSocket message."
            )

            return

        # --------------------------------------------------------
        # Parse JSON
        # --------------------------------------------------------

        try:

            message = json.loads(
                raw_message
            )

        except json.JSONDecodeError:

            self.logger.warning(
                "Received invalid JSON message: %s",
                raw_message[:500],
            )

            return

        if not isinstance(
            message,
            dict,
        ):

            self.logger.warning(
                "Received non-object JSON message."
            )

            return

        message_type = (
            message.get(
                "type"
            )
        )

        # --------------------------------------------------------
        # Validate + normalize
        # --------------------------------------------------------

        try:

            (
                is_valid,
                event,
                reason,
            ) = self.validator.validate(
                message
            )

        except Exception:

            self.logger.exception(
                "Market-data validation failed."
            )

            self._write_event(
                message=message,
                received_at=received_at,
                is_valid=False,
                validation_reason=(
                    "validator_exception"
                ),
            )

            return

        # --------------------------------------------------------
        # Persist every observation
        # --------------------------------------------------------

        self._write_event(
            message=message,
            received_at=received_at,
            is_valid=is_valid,
            validation_reason=reason,
            event=event,
        )

        # --------------------------------------------------------
        # Invalid event
        # --------------------------------------------------------

        if not is_valid:

            self.logger.warning(
                "Invalid market-data message | "
                "reason=%s | type=%s",
                reason,
                message_type,
            )

            self._periodic_status_log()

            return

        if event is None:

            self._periodic_status_log()

            return

        # --------------------------------------------------------
        # Forward normalized event
        # --------------------------------------------------------

        self._forward_event(
            event
        )

        event_type = getattr(
            event,
            "event_type",
            message_type,
        )

        # --------------------------------------------------------
        # Event-specific logging
        # --------------------------------------------------------

        if event_type == "ob_l1":

            self._log_orderbook_update(
                event
            )

        elif event_type == "trades":

            self._log_trade_update(
                event
            )

        elif event_type == "ticker":

            self._log_ticker_update(
                event
            )

        elif event_type == "system_status":

            self.logger.info(
                "System status update: %s",
                event.payload,
            )

        elif event_type == "subscriptions":

            self.logger.info(
                "Subscription confirmation: %s",
                event.payload,
            )

        elif event_type == "heartbeat":

            self.logger.debug(
                "Heartbeat received."
            )

        self._periodic_status_log()

    # ============================================================
    # WEBSOCKET ERROR
    # ============================================================

    def on_error(
        self,
        ws: websocket.WebSocketApp,
        error: Any,
    ) -> None:

        self.logger.error(
            "Public WebSocket error: %s",
            error,
        )

    # ============================================================
    # WEBSOCKET CLOSE
    # ============================================================

    def on_close(
        self,
        ws: websocket.WebSocketApp,
        close_status_code: Any,
        close_msg: Any,
    ) -> None:

        with self.state_lock:

            self.connected = False

        self.logger.warning(
            "Public WebSocket closed | "
            "code=%s | message=%s",
            close_status_code,
            close_msg,
        )

    # ============================================================
    # L1 ORDER BOOK LOGGING
    # ============================================================

    def _log_orderbook_update(
        self,
        event: Any,
    ) -> None:

        normalized = getattr(
            event,
            "normalized",
            None,
        )

        if not isinstance(
            normalized,
            dict,
        ):

            normalized = {}

        self.logger.info(
            "L1 | symbol=%s | "
            "bid=%s (%s) | "
            "ask=%s (%s) | "
            "mid=%s | spread=%s | "
            "spread_bps=%s",
            normalized.get(
                "symbol"
            ),
            normalized.get(
                "bid"
            ),
            normalized.get(
                "bid_size"
            ),
            normalized.get(
                "ask"
            ),
            normalized.get(
                "ask_size"
            ),
            normalized.get(
                "mid_price"
            ),
            normalized.get(
                "spread"
            ),
            normalized.get(
                "spread_bps"
            ),
        )

    # ============================================================
    # TRADE LOGGING
    # ============================================================

    def _log_trade_update(
        self,
        event: Any,
    ) -> None:

        normalized = getattr(
            event,
            "normalized",
            None,
        )

        if not isinstance(
            normalized,
            dict,
        ):

            normalized = {}

        self.trade_diagnostic_count += 1

        self.logger.info(
            "Trade | symbol=%s | "
            "price=%s | size=%s | side=%s | id=%s",
            normalized.get(
                "symbol"
            ),
            normalized.get(
                "price"
            ),
            normalized.get(
                "size"
            ),
            normalized.get(
                "side"
            ),
            normalized.get(
                "trade_id"
            ),
        )

        # Print the first few raw trade payloads.
        # This is deliberately limited so the log does
        # not become enormous.
        if self.trade_diagnostic_count <= 10:

            try:

                self.logger.info(
                    "TRADE DIAGNOSTIC RAW #%s | %s",
                    self.trade_diagnostic_count,
                    json.dumps(
                        getattr(
                            event,
                            "payload",
                            {},
                        ),
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                    ),
                )

            except Exception:

                self.logger.exception(
                    "Unable to print "
                    "trade diagnostic payload."
                )

    # ============================================================
    # TICKER LOGGING
    # ============================================================

    def _log_ticker_update(
        self,
        event: Any,
    ) -> None:

        normalized = getattr(
            event,
            "normalized",
            None,
        )

        if not isinstance(
            normalized,
            dict,
        ):

            normalized = {}

        self.logger.info(
            "Ticker | symbol=%s | "
            "last=%s | mark=%s | "
            "bid=%s | ask=%s | "
            "bid_size=%s | ask_size=%s | "
            "OI=%s | 24h=%s",
            normalized.get(
                "symbol"
            ),
            normalized.get(
                "last_price"
            ),
            normalized.get(
                "mark_price"
            ),
            normalized.get(
                "bid"
            ),
            normalized.get(
                "ask"
            ),
            normalized.get(
                "bid_size"
            ),
            normalized.get(
                "ask_size"
            ),
            normalized.get(
                "open_interest"
            ),
            normalized.get(
                "change_24h"
            ),
        )

    # ============================================================
    # PERIODIC HEALTH LOG
    # ============================================================

    def _periodic_status_log(
        self,
    ) -> None:

        now = time.monotonic()

        if (
            now
            - self.last_status_log
            < 10
        ):

            return

        self.last_status_log = now

        try:

            stats = (
                self.validator.get_stats()
            )

            mb = (
                self.total_bytes
                / (
                    1024
                    * 1024
                )
            )

            elapsed = max(
                time.time()
                - self.dataset_start_time,
                1,
            )

            rate = (
                self.total_written
                / elapsed
            )

            self.logger.info(
                "Market-data health | "
                "connected=%s | "
                "total=%s | "
                "valid=%s | "
                "invalid=%s | "
                "ticker=%s | "
                "ticker_valid=%s | "
                "ticker_invalid=%s | "
                "orderbook=%s | "
                "trades=%s | "
                "stored=%s | "
                "size=%.2f MB | "
                "write_rate=%.1f records/s",
                self.connected,
                stats.total_messages,
                stats.valid_messages,
                stats.invalid_messages,
                stats.ticker_messages,
                stats.ticker_valid_messages,
                stats.ticker_invalid_messages,
                stats.orderbook_messages,
                stats.trade_messages,
                self.total_written,
                mb,
                rate,
            )

        except Exception:

            self.logger.exception(
                "Unable to log "
                "market-data health."
            )

    # ============================================================
    # WEBSOCKET CREATION
    # ============================================================

    def _create_websocket(
        self,
    ) -> websocket.WebSocketApp:

        return websocket.WebSocketApp(
            self.websocket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

    # ============================================================
    # MAIN RUN LOOP
    # ============================================================

    def run(
        self,
    ) -> None:

        self.logger.info(
            "Starting market-data collector."
        )

        with self.message_lock:

            self._open_storage()

        while not self.stop_event.is_set():

            try:

                self.ws = (
                    self._create_websocket()
                )

                self.ws.run_forever(
                    ping_interval=30,
                    ping_timeout=10,
                    suppress_origin=True,
                )

            except TypeError as exc:

                if (
                    "suppress_origin"
                    in str(exc)
                ):

                    self.logger.warning(
                        "WebSocket client does not "
                        "support suppress_origin. "
                        "Retrying without it."
                    )

                    try:

                        if self.ws is not None:

                            self.ws.run_forever(
                                ping_interval=30,
                                ping_timeout=10,
                            )

                    except Exception:

                        self.logger.exception(
                            "WebSocket retry failed."
                        )

                else:

                    self.logger.exception(
                        "WebSocket TypeError."
                    )

            except Exception:

                self.logger.exception(
                    "Market-data WebSocket "
                    "run failed."
                )

            finally:

                with self.state_lock:

                    self.connected = False

            if self.stop_event.is_set():

                break

            self.logger.warning(
                "WebSocket disconnected. "
                "Reconnecting in %.1f seconds.",
                self.reconnect_delay_seconds,
            )

            self.stop_event.wait(
                self.reconnect_delay_seconds
            )

        with self.message_lock:

            self._close_storage()

        self.logger.info(
            "Market-data collector stopped | "
            "records=%s | size=%.2f MB",
            self.total_written,
            self.total_bytes
            / (
                1024
                * 1024
            ),
        )

    # ============================================================
    # START
    # ============================================================

    def start(
        self,
    ) -> threading.Thread:

        thread = threading.Thread(
            target=self.run,
            name="MarketDataCollector",
            daemon=True,
        )

        thread.start()

        return thread

    # ============================================================
    # STOP
    # ============================================================

    def stop(
        self,
    ) -> None:

        self.logger.info(
            "Stopping market-data collector."
        )

        self.stop_event.set()

        with self.state_lock:

            self.connected = False

        if self.ws is not None:

            try:

                self.ws.close()

            except Exception:

                self.logger.exception(
                    "Error closing WebSocket."
                )

        with self.message_lock:

            self._close_storage()


# =================================================================
# STANDALONE EXECUTION
# =================================================================

def configure_logging() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


def main() -> None:

    configure_logging()

    collector = MarketDataCollector(
        websocket_url=(
            "wss://socket-ind-pub.testnet.deltaex.org"
        ),
        symbols=[
            "BTCUSD"
        ],
        output_directory=(
            "data/market_data"
        ),
    )

    def handle_shutdown(
        signum: int,
        frame: Any,
    ) -> None:

        collector.stop()

    signal.signal(
        signal.SIGINT,
        handle_shutdown,
    )

    if hasattr(
        signal,
        "SIGTERM",
    ):

        signal.signal(
            signal.SIGTERM,
            handle_shutdown,
        )

    try:

        collector.run()

    except KeyboardInterrupt:

        collector.stop()


if __name__ == "__main__":

    main()
