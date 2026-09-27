
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

from __future__ import annotations

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
        """
        Build Delta public WebSocket subscription.
        """

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
                        "name": "ob_l2",
                        "symbols": self.symbols,
                    },
                    {
                        "name": "mark_price",
                        "symbols": [f"MARK:{symbol}" for symbol in self.symbols],
                    },
                    {
                        "name": "funding_rate",
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
        """
        Open today's persistent JSONL dataset.

        A new file is automatically created when
        the calendar date changes.
        """

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
    # DELTA PAYLOAD HELPERS
    # ============================================================

    @staticmethod
    def _get_delta_payload(
        message: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract Delta's actual payload.

        Delta ticker messages commonly use:

            message["d"][0]

        Example:

            {
                "type": "ticker",
                "d": [
                    {
                        "s": "BTCUSD",
                        "m": "83968.5",
                        "ohlc": [...]
                    }
                ]
            }
        """

        data = message.get(
            "d"
        )

        if isinstance(
            data,
            list,
        ):

            for item in data:

                if isinstance(
                    item,
                    dict,
                ):
                    return item

        if isinstance(
            data,
            dict,
        ):
            return data

        payload = message.get(
            "payload"
        )

        if isinstance(
            payload,
            dict,
        ):
            return payload

        data = message.get(
            "data"
        )

        if isinstance(
            data,
            dict,
        ):
            return data

        return message

    # ============================================================
    # EVENT METADATA
    # ============================================================

    def _extract_event_metadata(
        self,
        event: Any,
        message: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract normalized event metadata.
        """

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
        """
        Persist one complete market-data observation.

        Raw exchange message is always preserved.

        Normalized event data is also stored when available.
        """

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

                # Normalized stable schema.
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

        # --------------------------------------------------------
        # Determine event type early for diagnostics.
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Valid but no event
        # --------------------------------------------------------

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

            # IMPORTANT:
            # Pass the entire normalized event,
            # NOT event.payload.
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
        """
        Log normalized L1 order-book data.
        """

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
        """
        Log normalized trade data.
        """

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
            "Trade | symbol=%s | "
            "price=%s | size=%s",
            normalized.get(
                "symbol"
            ),
            normalized.get(
                "price"
            ),
            normalized.get(
                "size"
            ),
        )

    # ============================================================
    # TICKER LOGGING
    # ============================================================

    def _log_ticker_update(
        self,
        event: Any,
    ) -> None:
        """
        Log normalized Delta ticker data.

        Delta compact ticker:

            d[0].m
                -> mark_price

            d[0].ohlc[3]
                -> last_price / close

            d[0].q[0]
                -> ask

            d[0].q[1]
                -> ask_size

            d[0].q[2]
                -> bid

            d[0].q[3]
                -> bid_size

            d[0].oi[0]
                -> open_interest

            d[0].m24hc
                -> 24h change
        """

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

