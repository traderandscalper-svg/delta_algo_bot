import logging
from typing import Any


class InstrumentManager:

    def __init__(self, rest_client):

        self.rest = rest_client

        self.logger = logging.getLogger(
            "InstrumentManager"
        )

        self.products: list[dict[str, Any]] = []

        self.by_symbol: dict[str, dict[str, Any]] = {}

    def load(self, required_symbols: list[str] | None = None) -> list[dict[str, Any]]:

        response = self.rest.get_products()

        products = response.get(
            "result",
            [],
        )

        if not isinstance(products, list):
            products = []

        # The products endpoint is cursor-paginated. The first page is
        # sufficient for discovery, but the trading universe can contain
        # products outside that page. Resolve required instruments directly
        # by symbol so execution never depends on product ordering.
        loaded_symbols = {
            str(product.get("symbol", "")).upper()
            for product in products
            if isinstance(product, dict) and product.get("symbol")
        }

        for symbol in required_symbols or []:
            normalized = str(symbol).strip().upper()
            if not normalized or normalized in loaded_symbols:
                continue

            try:
                exact_response = self.rest.get_product(normalized)
                exact_product = exact_response.get("result")
                if isinstance(exact_product, dict) and exact_product.get("symbol"):
                    products.append(exact_product)
                    loaded_symbols.add(str(exact_product["symbol"]).upper())
                    self.logger.info(
                        "Resolved required Delta product directly | symbol=%s | product_id=%s",
                        exact_product.get("symbol"),
                        exact_product.get("id"),
                    )
                else:
                    self.logger.error(
                        "Required Delta product lookup returned no product | symbol=%s",
                        normalized,
                    )
            except Exception:
                self.logger.exception(
                    "Required Delta product lookup failed | symbol=%s",
                    normalized,
                )

        self.products = products

        self.by_symbol = {
            str(product["symbol"]).upper(): product
            for product in products
            if isinstance(product, dict) and "symbol" in product
        }

        self.logger.info(
            "Loaded %d Delta products.",
            len(self.products),
        )

        return self.products

    def get(self, symbol: str):

        return self.by_symbol.get(symbol)

    def symbols(self) -> list[str]:

        return list(
            self.by_symbol.keys()
        )

    def futures_symbols(self) -> list[str]:

        result = []

        for product in self.products:

            contract_type = str(
                product.get(
                    "contract_type",
                    ""
                )
            ).lower()

            if contract_type in {
                "futures",
                "perpetual_futures",
            }:

                result.append(
                    product["symbol"]
                )

        return result