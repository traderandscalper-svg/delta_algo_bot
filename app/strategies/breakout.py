from __future__ import annotations

from typing import List

from app.market_data.candle_aggregator import Candle
from app.strategies.base import BaseStrategy
from app.strategies.indicators import (
    closes,
    highest,
    lowest,
    atr_percent,
    volume_ratio,
)
from app.strategies.signals import (
    SignalAction,
    StrategySignal,
)


class BreakoutStrategy(BaseStrategy):

    name = "breakout"

    def __init__(self) -> None:
        super().__init__(
            minimum_candles=30
        )

    def evaluate(
        self,
        candles: List[Candle],
    ) -> StrategySignal:

        candle = candles[-1]
        prices = closes(candles)

        previous_high = highest(
            prices[:-1],
            20,
        )

        previous_low = lowest(
            prices[:-1],
            20,
        )

        volatility_percent = atr_percent(
            candles,
            14,
        )

        vol_ratio = volume_ratio(
            candles,
            20,
        )

        price = prices[-1]

        if (
            previous_high is None
            or previous_low is None
            or volatility_percent is None
        ):
            return self.hold(
                candle,
                "Insufficient breakout data.",
            )

        breakout_size_up = (
            (price - previous_high)
            / previous_high
            if previous_high != 0
            else 0.0
        )

        breakout_size_down = (
            (previous_low - price)
            / previous_low
            if previous_low != 0
            else 0.0
        )

        volume_confirmation = (
            vol_ratio is not None
            and vol_ratio >= 1.05
        )

        # Avoid calling tiny fluctuations breakouts.
        minimum_breakout = max(
            0.0002,
            volatility_percent * 0.10,
        )

        if (
            breakout_size_up
            > minimum_breakout
        ):

            confidence = (
                0.55
                + min(
                    breakout_size_up * 20,
                    0.20,
                )
            )

            if volume_confirmation:
                confidence += 0.10

            confidence = min(
                0.95,
                confidence,
            )

            return StrategySignal(
                strategy_name=self.name,
                symbol=candle.symbol,
                action=SignalAction.BUY,
                confidence=confidence,
                timestamp=candle.end_timestamp,
                score=min(
                    1.0,
                    breakout_size_up * 100,
                ),
                quality=confidence,
                regime="BREAKOUT_UP",
                reason=(
                    "Price broke above the "
                    "previous range high."
                ),
                indicators={
                    "previous_high": previous_high,
                    "previous_low": previous_low,
                    "breakout_size": breakout_size_up,
                    "atr_percent": volatility_percent,
                    "volume_ratio": (
                        vol_ratio
                        if vol_ratio is not None
                        else 0.0
                    ),
                    "volume_confirmation": (
                        1.0
                        if volume_confirmation
                        else 0.0
                    ),
                },
            )

        if (
            breakout_size_down
            > minimum_breakout
        ):

            confidence = (
                0.55
                + min(
                    breakout_size_down * 20,
                    0.20,
                )
            )

            if volume_confirmation:
                confidence += 0.10

            confidence = min(
                0.95,
                confidence,
            )

            return StrategySignal(
                strategy_name=self.name,
                symbol=candle.symbol,
                action=SignalAction.SELL,
                confidence=confidence,
                timestamp=candle.end_timestamp,
                score=-min(
                    1.0,
                    breakout_size_down * 100,
                ),
                quality=confidence,
                regime="BREAKOUT_DOWN",
                reason=(
                    "Price broke below the "
                    "previous range low."
                ),
                indicators={
                    "previous_high": previous_high,
                    "previous_low": previous_low,
                    "breakout_size": breakout_size_down,
                    "atr_percent": volatility_percent,
                    "volume_ratio": (
                        vol_ratio
                        if vol_ratio is not None
                        else 0.0
                    ),
                    "volume_confirmation": (
                        1.0
                        if volume_confirmation
                        else 0.0
                    ),
                },
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
            reason="No confirmed range breakout.",
            indicators={
                "previous_high": previous_high,
                "previous_low": previous_low,
                "atr_percent": volatility_percent,
                "volume_ratio": (
                    vol_ratio
                    if vol_ratio is not None
                    else 0.0
                ),
            },
        )