import json
import logging
import threading
import time
from typing import Callable, Optional

import websocket


class DeltaPublicWebSocket:

    def __init__(
        self,
        url: str,
        on_message_callback: Optional[
            Callable[[dict], None]
        ] = None,
    ):

        self.url = url

        self.logger = logging.getLogger(
            "DeltaPublicWebSocket"
        )

        self.on_message_callback = (
            on_message_callback
        )

        self.ws = None

        self.thread = None

        self.running = False

        self.last_message_time = 0.0

    # --------------------------------------------------------
    # Subscribe
    # --------------------------------------------------------

    def subscribe(
        self,
        channels: list[dict],
    ):

        message = {
            "type": "subscribe",
            "payload": {
                "channels": channels
            },
        }

        self.send(message)

    def send(self, message: dict):

        if self.ws is None:

            raise RuntimeError(
                "WebSocket is not connected."
            )

        self.ws.send(
            json.dumps(message)
        )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    def _on_open(self, ws):

        self.logger.info(
            "Delta public WebSocket connected."
        )

        # Phase 2 test subscription.
        #
        # We intentionally begin with BTCUSD.
        #
        # Later the subscription manager will dynamically
        # manage the complete instrument universe.

        self.subscribe([
            {
                "name": "ticker",
                "symbols": [
                    "BTCUSD"
                ],
            },
            {
                "name": "ob_l1",
                "symbols": [
                    "BTCUSD"
                ],
            },
            {
                "name": "trades",
                "symbols": [
                    "BTCUSD"
                ],
            },
            {
                "name": "ob_l2",
                "symbols": [
                    "BTCUSD"
                ],
            },
            {
                "name": "mark_price",
                "symbols": [
                    "MARK:BTCUSD"
                ],
            },
            {
                "name": "funding_rate",
                "symbols": [
                    "BTCUSD"
                ],
            },
            {
                "name": "system_status",
            },
        ])

        self.logger.info(
            "Phase 2 market-data subscriptions sent."
        )

    def _on_message(self, ws, message):

        self.last_message_time = time.monotonic()

        try:

            data = json.loads(message)

        except json.JSONDecodeError:

            self.logger.warning(
                "Received non-JSON WebSocket message."
            )

            return

        message_type = data.get(
            "type",
            "unknown",
        )

        self.logger.debug(
            "WS message type: %s",
            message_type,
        )

        if self.on_message_callback:

            self.on_message_callback(
                data
            )

    def _on_error(self, ws, error):

        self.logger.error(
            "Delta WebSocket error: %s",
            error,
        )

    def _on_close(
        self,
        ws,
        close_status_code,
        close_msg,
    ):

        self.logger.warning(
            "Delta WebSocket closed. "
            "code=%s message=%s",
            close_status_code,
            close_msg,
        )

    # --------------------------------------------------------
    # Connection
    # --------------------------------------------------------

    def start(self):

        if self.running:

            return

        self.running = True

        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="delta-public-websocket",
        )

        self.thread.start()

    def _run(self):

        reconnect_delay = 5

        while self.running:

            try:

                self.logger.info(
                    "Connecting to Delta public WebSocket..."
                )

                self.ws.run_forever(
                    ping_interval=30,
                    ping_timeout=10,
                )

            except Exception as exc:

                self.logger.exception(
                    "WebSocket runtime error: %s",
                    exc,
                )

            if not self.running:

                break

            self.logger.warning(
                "WebSocket disconnected. "
                "Reconnecting in %s seconds.",
                reconnect_delay,
            )

            time.sleep(
                reconnect_delay
            )

    def stop(self):

        self.running = False

        if self.ws:

            try:
                self.ws.close()

            except Exception:
                pass

        if self.thread:

            self.thread.join(
                timeout=5
            )

        self.logger.info(
            "Delta public WebSocket stopped."
        )