import time
import threading
import logging


class Watchdog:
    """
    Basic engine watchdog.

    Phase 1 only monitors engine liveness.
    Later it will also monitor:
        - websocket heartbeat
        - market-data freshness
        - order acknowledgement
        - exchange connectivity
        - execution latency
        - state synchronization
    """

    def __init__(
        self,
        heartbeat_timeout: int,
        logger: logging.Logger,
    ):
        self.heartbeat_timeout = heartbeat_timeout
        self.logger = logger

        self._last_heartbeat = time.monotonic()
        self._running = False

        self._thread = None

    def heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    def start(self) -> None:
        if self._running:
            return

        self._running = True

        self._thread = threading.Thread(
            target=self._monitor,
            daemon=True,
            name="engine-watchdog",
        )

        self._thread.start()

        self.logger.info("Watchdog started.")

    def stop(self) -> None:
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2)

        self.logger.info("Watchdog stopped.")

    def _monitor(self) -> None:
        while self._running:
            elapsed = time.monotonic() - self._last_heartbeat

            if elapsed > self.heartbeat_timeout:
                self.logger.warning(
                    "Watchdog detected stale engine heartbeat: %.2fs",
                    elapsed,
                )

            time.sleep(1)