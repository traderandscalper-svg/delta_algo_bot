
import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests

from app.config.settings import Settings
from app.exchange.authentication import create_rest_headers


class DeltaRestClient:
    """
    REST API client for Delta Exchange India.

    Supports public and authenticated requests.
    Supports authenticated account and trading-order requests.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        self.logger = logging.getLogger("DeltaRestClient")

        self.base_url = settings.delta_rest_url.rstrip("/")

        self.api_key = settings.delta_api_key
        self.api_secret = settings.delta_api_secret

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "delta-algo-bot/0.1",
            }
        )

        self.timeout = 15

        self.logger.info(
            "Delta REST client initialized | base_url=%s",
            self.base_url,
        )

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _build_query_string(
        params: Optional[Dict[str, Any]]
    ) -> str:
        """
        Convert query parameters into a URL-encoded string.

        The resulting string is used for signing and is also
        sent as the request's query parameters.
        """

        if not params:
            return ""

        clean_params = {
            key: value
            for key, value in params.items()
            if value is not None
        }

        return urlencode(
            clean_params,
            doseq=True,
        )

    @staticmethod
    def _serialize_body(
        json_body: Optional[Dict[str, Any]]
    ) -> str:
        """
        Serialize JSON body consistently.

        The same serialized body must be used for signing
        and sending the HTTP request.
        """

        if json_body is None:
            return ""

        return json.dumps(
            json_body,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _create_authenticated_headers(
        self,
        method: str,
        path: str,
        query_string: str = "",
        body: str = "",
    ) -> Dict[str, str]:
        """
        Generate authenticated request headers.
        """

        if not self.api_key:
            raise RuntimeError(
                "Delta API key is missing. "
                "Check DELTA_API_KEY in .env."
            )

        if not self.api_secret:
            raise RuntimeError(
                "Delta API secret is missing. "
                "Check DELTA_API_SECRET in .env."
            )

        return create_rest_headers(
            api_key=self.api_key,
            api_secret=self.api_secret,
            method=method,
            path=path,
            query_string=query_string,
            body=body,
        )

    def _log_api_error(
        self,
        response: requests.Response,
        method: str,
        path: str,
    ) -> None:
        """
        Log the API error response.

        Never log API secrets or authentication signatures.
        """

        try:
            response_body = response.json()

        except ValueError:
            response_body = response.text[:2000]

        self.logger.error(
            "Delta API request failed | "
            "status=%s | method=%s | path=%s | response=%s",
            response.status_code,
            method.upper(),
            path,
            response_body,
        )

    # ============================================================
    # CORE REQUEST METHOD
    # ============================================================

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        authenticated: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a Delta REST API request.

        Parameters:
            method:
                HTTP method, e.g. GET or POST.

            path:
                API path, e.g. /v2/products.

            params:
                Query parameters.

            json_body:
                JSON request body.

            authenticated:
                Whether the request requires authentication.
        """

        method = method.upper()

        if not path.startswith("/"):
            path = "/" + path

        query_string = self._build_query_string(params)

        body = self._serialize_body(json_body)

        url = f"{self.base_url}{path}"

        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "delta-algo-bot/0.1",
        }

        if json_body is not None:
            headers["Content-Type"] = "application/json"

        if authenticated:
            auth_headers = self._create_authenticated_headers(
                method=method,
                path=path,
                query_string=query_string,
                body=body,
            )

            headers.update(auth_headers)

        request_kwargs: Dict[str, Any] = {
            "method": method,
            "url": url,
            "headers": headers,
            "timeout": self.timeout,
        }

        if params:
            request_kwargs["params"] = params

        if json_body is not None:
            request_kwargs["data"] = body

        self.logger.debug(
            "REST request | method=%s | path=%s | authenticated=%s",
            method,
            path,
            authenticated,
        )

        try:
            response = self.session.request(
                **request_kwargs
            )

        except requests.Timeout as exc:
            self.logger.error(
                "REST request timed out | method=%s | path=%s",
                method,
                path,
            )

            raise RuntimeError(
                f"Delta REST request timed out: "
                f"{method} {path}"
            ) from exc

        except requests.RequestException as exc:
            self.logger.error(
                "REST connection error | "
                "method=%s | path=%s | error=%s",
                method,
                path,
                str(exc),
            )

            raise RuntimeError(
                f"Delta REST connection failed: "
                f"{method} {path}"
            ) from exc

        if not response.ok:
            self._log_api_error(
                response=response,
                method=method,
                path=path,
            )

            response.raise_for_status()

        try:
            response_data = response.json()

        except ValueError as exc:
            self.logger.error(
                "Delta API returned invalid JSON | "
                "method=%s | path=%s | response=%s",
                method,
                path,
                response.text[:1000],
            )

            raise RuntimeError(
                f"Delta API returned invalid JSON: "
                f"{method} {path}"
            ) from exc

        self.logger.debug(
            "REST request successful | "
            "method=%s | path=%s | status=%s",
            method,
            path,
            response.status_code,
        )

        return response_data

    # ============================================================
    # PUBLIC ENDPOINTS
    # ============================================================

    def get_products(self) -> Dict[str, Any]:
        """
        Retrieve available products/instruments.
        """

        return self.request(
            method="GET",
            path="/v2/products",
            params={
                "page_size": 100,
            },
            authenticated=False,
        )

    def get_tickers(self) -> Dict[str, Any]:
        """
        Retrieve ticker information.
        """

        return self.request(
            method="GET",
            path="/v2/tickers",
            authenticated=False,
        )

    # ============================================================
    # AUTHENTICATED ENDPOINTS
    # ============================================================

    def get_wallet_balances(self) -> Dict[str, Any]:
        """
        Retrieve wallet balances.
        """

        return self.request(
            method="GET",
            path="/v2/wallet/balances",
            authenticated=True,
        )

    def get_positions(self) -> Dict[str, Any]:
        """Retrieve real-time open positions from Delta."""
        return self.request(
            method="GET",
            path="/v2/positions",
            authenticated=True,
        )

    def get_open_orders(self) -> Dict[str, Any]:
        """
        Retrieve open orders.
        """

        return self.request(
            method="GET",
            path="/v2/orders",
            authenticated=True,
        )

    # ============================================================
    # LIVE TRADING ORDERS
    # ============================================================

    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place one authenticated Delta production/testnet order."""
        if not isinstance(order, dict):
            raise ValueError("order must be a dictionary")
        return self.request(
            method="POST",
            path="/v2/orders",
            json_body=order,
            authenticated=True,
        )

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self.request(
            method="DELETE",
            path=f"/v2/orders/{order_id}",
            authenticated=True,
        )

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self.request(
            method="GET",
            path=f"/v2/orders/{order_id}",
            authenticated=True,
        )

    def get_order_by_client_order_id(self, client_order_id: str) -> Dict[str, Any]:
        """Retrieve the latest exchange state for a bot-generated client order id."""
        return self.request(
            method="GET",
            path=f"/v2/orders/client_order_id/{client_order_id}",
            authenticated=True,
        )

    def get_fills(self, product_symbol: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve authenticated fills for post-order verification."""
        params = {"product_symbol": product_symbol} if product_symbol else None
        return self.request(
            method="GET",
            path="/v2/fills",
            params=params,
            authenticated=True,
        )

    def get_order_history(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        params = {"product_symbol": symbol} if symbol else None
        return self.request(
            method="GET",
            path="/v2/orders/history",
            params=params,
            authenticated=True,
        )

    # ============================================================
    # CONNECTION TESTS
    # ============================================================

    def test_public_connection(self) -> Dict[str, Any]:
        """
        Test public REST connectivity.
        """

        self.logger.info(
            "Testing public REST connection..."
        )

        result = self.get_products()

        self.logger.info(
            "Public REST connection successful."
        )

        return result

    def test_private_connection(self) -> Dict[str, Any]:
        """
        Test authenticated REST connectivity.
        """

        self.logger.info(
            "Testing authenticated REST connection..."
        )

        result = self.get_wallet_balances()

        self.logger.info(
            "Authenticated REST connection successful."
        )

        return result

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def close(self) -> None:
        """
        Close the HTTP session.
        """

        self.logger.info(
            "Closing Delta REST session."
        )

        self.session.close()