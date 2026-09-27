
from __future__ import annotations

from typing import List

from app.market_data.candle_aggregator import Candle
from app.strategies.base import BaseStrategy
from app.strategies.signals import (
    SignalAction,
    StrategySignal,
)


class MeanReversionStrategy(BaseStrategy):

    name = "mean_reversion"

    def __init__(self) -> None:
        super().__init__(
            minimum_candles=25
        )

    @staticmethod
    def _mean(
        values: List[float],
    ) -> float:

        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _standard_deviation(
        values: List[float],
        mean: float,
    ) -> float:

        if len(values) < 2:
            return 0.0

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        return variance ** 0.5

    def evaluate(
        self,
        candles: List[Candle],
    ) -> StrategySignal:

        candle = candles[-1]

        if not self.can_evaluate(
            candles
        ):
            return self.hold(
                candle,
                "Insufficient mean-reversion data.",
            )

        lookback = 20

        recent = candles[-lookback:]

        prices = [
            float(c.close)
            for c in recent
            if c.close > 0
        ]

        if len(prices) < lookback:
            return self.hold(
                candle,
                "Invalid mean-reversion price history.",
            )

        mean_price = self._mean(
            prices
        )

        standard_deviation = (
            self._standard_deviation(
                prices,
                mean_price,
            )
        )

        if mean_price <= 0:
            return self.hold(
                candle,
                "Invalid mean price.",
            )

        deviation = (
            candle.close - mean_price
        ) / mean_price

        if standard_deviation > 0:
            z_score = (
                candle.close - mean_price
            ) / standard_deviation
        else:
            z_score = 0.0

        previous_close = (
            candles[-2].close
        )

        price_change = (
            candle.close - previous_close
        )

        # ---------------------------------------------
        # Threshold
        # ---------------------------------------------

        threshold = 0.0005

        indicators = {
            "mean_price": mean_price,
            "standard_deviation": standard_deviation,
            "deviation": deviation,
            "z_score": z_score,
            "previous_close": previous_close,
            "current_close": candle.close,
            "price_change": price_change,
        }

        # ---------------------------------------------
        # Downside extension -> BUY
        # ---------------------------------------------

        if deviation <= -threshold:

            # Prefer evidence that price has started
            # moving back toward the mean.
            recovering = (
                price_change > 0
            )

            if recovering:

                confidence = (
                    0.50
                    + min(
                        abs(deviation) * 100,
                        0.20,
                    )
                    + min(
                        abs(z_score) * 0.08,
                        0.20,
                    )
                )

                confidence = min(
                    0.95,
                    confidence,
                )

                score = min(
                    1.0,
                    abs(deviation) * 100,
                )

                return StrategySignal(
                    strategy_name=self.name,
                    symbol=candle.symbol,
                    action=SignalAction.BUY,
                    confidence=confidence,
                    timestamp=candle.end_timestamp,
                    score=score,
                    quality=confidence,
                    regime="MEAN_REVERSION_UP",
                    reason=(
                        "Price is below its recent mean "
                        "and the latest candle shows "
                        "recovery toward the mean."
                    ),
                    indicators=indicators,
                )

        # ---------------------------------------------
        # Upside extension -> SELL
        # ---------------------------------------------

        if deviation >= threshold:

            recovering = (
                price_change < 0
            )

            if recovering:

                confidence = (
                    0.50
                    + min(
                        abs(deviation) * 100,
                        0.20,
                    )
                    + min(
                        abs(z_score) * 0.08,
                        0.20,
                    )
                )

                confidence = min(
                    0.95,
                    confidence,
                )

                score = -min(
                    1.0,
                    abs(deviation) * 100,
                )

                return StrategySignal(
                    strategy_name=self.name,
                    symbol=candle.symbol,
                    action=SignalAction.SELL,
                    confidence=confidence,
                    timestamp=candle.end_timestamp,
                    score=score,
                    quality=confidence,
                    regime="MEAN_REVERSION_DOWN",
                    reason=(
                        "Price is above its recent mean "
                        "and the latest candle shows "
                        "reversal toward the mean."
                    ),
                    indicators=indicators,
                )

        return StrategySignal(
            strategy_name=self.name,
            symbol=candle.symbol,
            action=SignalAction.HOLD,
            confidence=0.20,
            timestamp=candle.end_timestamp,
            score=0.0,
            quality=0.20,
            regime="RANGE",
            reason=(
                "Price is not sufficiently extended "
                "from its recent mean or reversal "
                "confirmation is absent."
            ),
            indicators=indicators,
        )

