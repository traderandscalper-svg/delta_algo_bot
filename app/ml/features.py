"""
Feature engineering for the Delta Exchange ML pipeline.

The features are calculated only from information available
at or before the current observation.

No future information is used while creating features.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Optional


class FeatureEngine:
    """
    Incremental feature calculator.

    Features include:

        - price
        - mark price
        - bid/ask
        - spread
        - spread in basis points
        - order-book imbalance
        - short-term returns
        - momentum
        - volatility
        - OHLC relationships
        - open interest
        - open-interest change
        - 24h change
        - trade flow
    """

    def __init__(
        self,
        max_history: int = 300,
    ) -> None:

        self.max_history = max_history

        self.prices: Deque[float] = deque(
            maxlen=max_history
        )

        self.returns: Deque[float] = deque(
            maxlen=max_history
        )

        self.trade_volume: Deque[float] = deque(
            maxlen=max_history
        )

        self.buy_volume: Deque[float] = deque(
            maxlen=max_history
        )

        self.sell_volume: Deque[float] = deque(
            maxlen=max_history
        )

    # ============================================================
    # PUBLIC
    # ============================================================

    def update(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Update state and return the feature vector.

        The observation should contain the latest combined
        ticker/order-book/trade information.
        """

        price = self._float(
            observation.get("price")
        )

        if price is None or price <= 0:
            return {}

        previous_price = (
            self.prices[-1]
            if self.prices
            else None
        )

        self.prices.append(price)

        current_return = 0.0

        if (
            previous_price is not None
            and previous_price > 0
        ):
            current_return = (
                price / previous_price
            ) - 1.0

        self.returns.append(
            current_return
        )

        bid = self._float(
            observation.get("bid")
        )

        ask = self._float(
            observation.get("ask")
        )

        bid_size = self._float(
            observation.get("bid_size")
        )

        ask_size = self._float(
            observation.get("ask_size")
        )

        mark_price = self._float(
            observation.get("mark_price")
        )

        open_price = self._float(
            observation.get("open")
        )

        high_price = self._float(
            observation.get("high")
        )

        low_price = self._float(
            observation.get("low")
        )

        open_interest = self._float(
            observation.get("open_interest")
        )

        open_interest_change = self._float(
            observation.get(
                "open_interest_change"
            )
        )

        change_24h = self._float(
            observation.get("change_24h")
        )

        trade_price = self._float(
            observation.get("trade_price")
        )

        trade_size = self._float(
            observation.get("trade_size")
        )

        trade_side = observation.get(
            "trade_side"
        )

        # --------------------------------------------------------
        # Spread
        # --------------------------------------------------------

        spread = 0.0

        if (
            bid is not None
            and ask is not None
            and ask >= bid
        ):
            spread = ask - bid

        mid_price = None

        if (
            bid is not None
            and ask is not None
        ):
            mid_price = (
                bid + ask
            ) / 2.0

        spread_bps = 0.0

        if (
            mid_price is not None
            and mid_price > 0
        ):
            spread_bps = (
                spread
                / mid_price
                * 10_000.0
            )

        # --------------------------------------------------------
        # Order-book imbalance
        # --------------------------------------------------------

        orderbook_imbalance = 0.0

        if (
            bid_size is not None
            and ask_size is not None
        ):

            total_depth = (
                bid_size
                + ask_size
            )

            if total_depth > 0:
                orderbook_imbalance = (
                    bid_size - ask_size
                ) / total_depth

        # --------------------------------------------------------
        # Trade flow
        # --------------------------------------------------------

        if trade_size is not None:
            self.trade_volume.append(
                max(trade_size, 0.0)
            )

            if trade_side == "buy":
                self.buy_volume.append(
                    max(trade_size, 0.0)
                )
                self.sell_volume.append(0.0)

            elif trade_side == "sell":
                self.sell_volume.append(
                    max(trade_size, 0.0)
                )
                self.buy_volume.append(0.0)

            else:
                self.buy_volume.append(0.0)
                self.sell_volume.append(0.0)

        total_buy = sum(
            self.buy_volume
        )

        total_sell = sum(
            self.sell_volume
        )

        total_trade_volume = (
            total_buy + total_sell
        )

        trade_imbalance = 0.0

        if total_trade_volume > 0:
            trade_imbalance = (
                total_buy - total_sell
            ) / total_trade_volume

        # --------------------------------------------------------
        # Returns
        # --------------------------------------------------------

        return_1 = self._return_over(
            1
        )

        return_3 = self._return_over(
            3
        )

        return_5 = self._return_over(
            5
        )

        return_10 = self._return_over(
            10
        )

        return_20 = self._return_over(
            20
        )

        # --------------------------------------------------------
        # Volatility
        # --------------------------------------------------------

        volatility_5 = self._volatility(
            5
        )

        volatility_10 = self._volatility(
            10
        )

        volatility_20 = self._volatility(
            20
        )

        # --------------------------------------------------------
        # Momentum
        # --------------------------------------------------------

        momentum_5 = (
            return_5
        )

        momentum_10 = (
            return_10
        )

        # --------------------------------------------------------
        # Price location inside OHLC range
        # --------------------------------------------------------

        range_position = 0.5

        if (
            high_price is not None
            and low_price is not None
            and high_price > low_price
        ):
            range_position = (
                price - low_price
            ) / (
                high_price - low_price
            )

            range_position = max(
                0.0,
                min(
                    1.0,
                    range_position,
                ),
            )

        # --------------------------------------------------------
        # Distance from moving averages
        # --------------------------------------------------------

        sma_5 = self._sma(5)
        sma_10 = self._sma(10)
        sma_20 = self._sma(20)

        distance_sma_5 = 0.0
        distance_sma_10 = 0.0
        distance_sma_20 = 0.0

        if sma_5:
            distance_sma_5 = (
                price / sma_5
            ) - 1.0

        if sma_10:
            distance_sma_10 = (
                price / sma_10
            ) - 1.0

        if sma_20:
            distance_sma_20 = (
                price / sma_20
            ) - 1.0

        # --------------------------------------------------------
        # Mark-price basis
        # --------------------------------------------------------

        mark_basis = 0.0

        if (
            mark_price is not None
            and mark_price > 0
        ):
            mark_basis = (
                price / mark_price
            ) - 1.0

        # --------------------------------------------------------
        # Build feature vector
        # --------------------------------------------------------

        features = {
            "price": price,
            "mark_price": (
                mark_price
                if mark_price is not None
                else price
            ),

            "bid": (
                bid
                if bid is not None
                else price
            ),

            "ask": (
                ask
                if ask is not None
                else price
            ),

            "bid_size": (
                bid_size
                if bid_size is not None
                else 0.0
            ),

            "ask_size": (
                ask_size
                if ask_size is not None
                else 0.0
            ),

            "spread": spread,
            "spread_bps": spread_bps,

            "orderbook_imbalance": (
                orderbook_imbalance
            ),

            "trade_imbalance": (
                trade_imbalance
            ),

            "trade_volume": (
                total_trade_volume
            ),

            "return_1": return_1,
            "return_3": return_3,
            "return_5": return_5,
            "return_10": return_10,
            "return_20": return_20,

            "volatility_5": volatility_5,
            "volatility_10": volatility_10,
            "volatility_20": volatility_20,

            "momentum_5": momentum_5,
            "momentum_10": momentum_10,

            "sma_5_distance": (
                distance_sma_5
            ),

            "sma_10_distance": (
                distance_sma_10
            ),

            "sma_20_distance": (
                distance_sma_20
            ),

            "range_position": (
                range_position
            ),

            "mark_basis": mark_basis,

            "open_interest": (
                open_interest
                if open_interest is not None
                else 0.0
            ),

            "open_interest_change": (
                open_interest_change
                if open_interest_change is not None
                else 0.0
            ),

            "change_24h": (
                change_24h
                if change_24h is not None
                else 0.0
            ),

            "open_price": (
                open_price
                if open_price is not None
                else price
            ),

            "high_price": (
                high_price
                if high_price is not None
                else price
            ),

            "low_price": (
                low_price
                if low_price is not None
                else price
            ),

            "trade_price": (
                trade_price
                if trade_price is not None
                else price
            ),
        }

        return {
            key: self._finite(
                value
            )
            for key, value in features.items()
        }

    # ============================================================
    # CALCULATIONS
    # ============================================================

    def _return_over(
        self,
        periods: int,
    ) -> float:

        if len(self.prices) <= periods:
            return 0.0

        old_price = list(
            self.prices
        )[-periods - 1]

        current_price = self.prices[-1]

        if old_price <= 0:
            return 0.0

        return (
            current_price
            / old_price
        ) - 1.0

    def _volatility(
        self,
        periods: int,
    ) -> float:

        values = list(
            self.returns
        )[-periods:]

        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        return math.sqrt(
            max(variance, 0.0)
        )

    def _sma(
        self,
        periods: int,
    ) -> Optional[float]:

        if len(self.prices) < periods:
            return None

        values = list(
            self.prices
        )[-periods:]

        return sum(values) / len(values)

    @staticmethod
    def _float(
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        try:
            number = float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

        if not math.isfinite(number):
            return None

        return number

    @staticmethod
    def _finite(
        value: float,
    ) -> float:

        if not math.isfinite(value):
            return 0.0

        return float(value)