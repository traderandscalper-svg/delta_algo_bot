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

    def load(self) -> list[dict[str, Any]]:

        response = self.rest.get_products()

        products = response.get(
            "result",
            [],
        )

        self.products = products

        self.by_symbol = {
            product["symbol"]: product
            for product in products
            if "symbol" in product
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