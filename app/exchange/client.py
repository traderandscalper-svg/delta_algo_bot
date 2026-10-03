import logging
from typing import Any, Optional

from app.config.settings import Settings
from app.exchange.instruments import InstrumentManager
from app.exchange.rest import DeltaRestClient
from app.exchange.websocket import DeltaPublicWebSocket


class DeltaClient:

    def __init__(
        self,
        settings: Settings,
    ):
        self.settings = settings

        self.logger = logging.getLogger(
            "DeltaClient"
        )

        # ----------------------------------------------------
        # REST
        # ----------------------------------------------------

        self.rest = DeltaRestClient(
            settings
        )

        # ----------------------------------------------------
        # Instruments
        # ----------------------------------------------------

        self.instruments = InstrumentManager(
            self.rest
        )

        # ----------------------------------------------------
        # Public WebSocket
        # ----------------------------------------------------

        self.public_ws = DeltaPublicWebSocket(
            url=settings.delta_public_ws_url,
            on_message_callback=(
                self._handle_market_message
            ),
        )

        # ----------------------------------------------------
        # Account state
        # ----------------------------------------------------

        self.account_equity: Optional[float] = None

        self.last_wallet_response: Optional[
            dict
        ] = None

    # ========================================================
    # REST CONNECTION
    # ========================================================

    def test_rest_connection(self):
        """
        Test public REST connectivity.
        """

        self.logger.info(
            "Testing Delta REST connection..."
        )

        products = self.rest.get_products()

        result = products.get(
            "result",
            [],
        )

        self.logger.info(
            "REST connection successful. "
            "Products received: %d",
            len(result),
        )

        return products

    # ========================================================
    # PRIVATE REST CONNECTION
    # ========================================================

    def test_private_rest_connection(self):
        """
        Test authenticated REST connectivity
        and load account equity.
        """

        self.logger.info(
            "Testing authenticated Delta REST..."
        )

        balances = (
            self.rest.get_wallet_balances()
        )

        if not isinstance(
            balances,
            dict,
        ):
            raise RuntimeError(
                "Delta wallet response is not a dictionary."
            )

        self.last_wallet_response = balances

        equity = self._extract_equity(
            balances
        )

        if equity is None:
            self.logger.error(
                "Unable to extract account equity "
                "from Delta wallet response."
            )

            self.logger.error(
                "Wallet response keys: %s",
                list(balances.keys()),
            )

            raise RuntimeError(
                "Delta account equity is unavailable."
            )

        self.account_equity = equity

        self.logger.info(
            "Authenticated REST connection successful | "
            "account_equity=%.8f",
            self.account_equity,
        )

        return balances

    # ========================================================
    # ACCOUNT EQUITY
    # ========================================================

    def get_account_equity(
        self,
    ) -> float:
        """
        Refresh account equity from Delta.
        """

        balances = (
            self.rest.get_wallet_balances()
        )

        if not isinstance(
            balances,
            dict,
        ):
            raise RuntimeError(
                "Invalid Delta wallet response."
            )

        self.last_wallet_response = balances

        equity = self._extract_equity(
            balances
        )

        if equity is None:
            raise RuntimeError(
                "Unable to extract account equity "
                "from Delta wallet response."
            )

        self.account_equity = equity

        self.logger.info(
            "Account equity refreshed | "
            "equity=%.8f",
            equity,
        )

        return equity

    @staticmethod
    def _extract_equity(
        balances: Any,
    ) -> Optional[float]:
        """
        Extract net equity from the Delta wallet response.

        Expected structure:

        {
            "meta": {
                "net_equity": "..."
            },
            "result": [...]
        }
        """

        if not isinstance(
            balances,
            dict,
        ):
            return None

        meta = balances.get(
            "meta"
        )

        if isinstance(
            meta,
            dict,
        ):
            value = meta.get(
                "net_equity"
            )

            if value is not None:
                try:
                    equity = float(
                        value
                    )

                    if equity >= 0:
                        return equity

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        # ----------------------------------------------------
        # Fallbacks for different wallet response structures
        # ----------------------------------------------------

        for key in (
            "net_equity",
            "equity",
            "total_equity",
        ):
            value = balances.get(
                key
            )

            if value is not None:
                try:
                    equity = float(
                        value
                    )

                    if equity >= 0:
                        return equity

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        return None

    # ========================================================
    # INSTRUMENTS
    # ========================================================

    def load_instruments(self, required_symbols=None):

        self.logger.info(
            "Loading Delta instruments..."
        )

        instruments = (
            self.instruments.load(required_symbols=required_symbols)
        )

        self.logger.info(
            "Delta instruments loaded."
        )

        return instruments

    # ========================================================
    # PUBLIC WEBSOCKET
    # ========================================================

    def start_public_market_data(self):

        self.logger.info(
            "Starting Delta public market data..."
        )

        self.public_ws.start()

    def stop(self):

        self.logger.info(
            "Stopping Delta connections..."
        )

        try:
            self.public_ws.stop()

        except Exception:
            self.logger.exception(
                "Error stopping Delta public WebSocket."
            )

    # ========================================================
    # MARKET MESSAGE HANDLER
    # ========================================================

    def _handle_market_message(
        self,
        message: dict[str, Any],
    ):

        if not isinstance(
            message,
            dict,
        ):
            return

        message_type = message.get(
            "type"
        )

        if message_type == "ticker":

            self.logger.info(
                "TICKER: %s",
                message,
            )

        elif message_type == "ob_l1":

            self.logger.info(
                "ORDERBOOK L1: %s",
                message,
            )

        elif message_type == "trades":

            self.logger.info(
                "TRADE: %s",
                message,
            )

        elif message_type == "system_status":

            self.logger.warning(
                "SYSTEM STATUS: %s",
                message,
            )

        elif message_type == "subscriptions":

            self.logger.info(
                "Subscriptions confirmed: %s",
                message,
            )