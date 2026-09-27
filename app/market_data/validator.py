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

    For Delta trade messages:

        p  -> trade price
        s  -> trade size
        sy -> symbol

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

        # Delta trade schema:
        # "s" = trade size
        # "sy" = symbol
        size = (
            self._find_first_value(
                payload,
                "s",
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

        # Delta trade schema:
        # "s" = trade size
        # "sy" = symbol
        size = self._safe_float(
            self._find_first_value(
                payload,
                "s",
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