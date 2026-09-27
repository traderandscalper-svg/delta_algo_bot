
from __future__ import annotations

from typing import Dict, List

from app.market_data.candle_aggregator import Candle
from app.strategies.base import BaseStrategy
from app.strategies.signals import SignalAction, StrategySignal


class MomentumStrategy(BaseStrategy):
    """
    Short-term price/volume momentum strategy.

    Uses only completed candles already present in the candle buffer.
    No future data is accessed.

    Signal logic:
        - Measure price return over a configurable lookback.
        - Measure the current candle body strength.
        - Compare current volume with recent average volume.
        - Require directional agreement before generating BUY/SELL.
    """

    name = "momentum"

    def __init__(
        self,
        minimum_candles: int = 20,
        lookback: int = 10,
        minimum_return: float = 0.00010,
        minimum_body_ratio: float = 0.45,
        volume_multiplier: float = 1.10,
    ) -> None:

        super().__init__(
            minimum_candles=minimum_candles
        )

        self.lookback = max(
            2,
            int(lookback),
        )

        self.minimum_return = max(
            0.0,
            float(minimum_return),
        )

        self.minimum_body_ratio = max(
            0.0,
            min(1.0, float(minimum_body_ratio)),
        )

        self.volume_multiplier = max(
            0.0,
            float(volume_multiplier),
        )

    @staticmethod
    def _safe_ratio(
        numerator: float,
        denominator: float,
    ) -> float:

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def evaluate(
        self,
        candles: List[Candle],
    ) -> StrategySignal:

        current = candles[-1]

        if len(candles) < self.minimum_candles:

            return self.hold(
                candle=current,
                reason=(
                    "Insufficient candles for momentum analysis."
                ),
                confidence=0.0,
            )

        lookback_start = candles[
            -self.lookback
        ]

        reference_close = float(
            lookback_start.close
        )

        current_close = float(
            current.close
        )

        if reference_close <= 0:

            return self.hold(
                candle=current,
                reason="Invalid reference price.",
                confidence=0.0,
            )

        price_return = (
            current_close
            - reference_close
        ) / reference_close

        candle_range = (
            float(current.high)
            - float(current.low)
        )

        candle_body = abs(
            float(current.close)
            - float(current.open)
        )

        body_ratio = self._safe_ratio(
            candle_body,
            candle_range,
        )

        recent_candles = candles[
            -self.lookback:
        ]

        volumes = [
            max(0.0, float(c.volume))
            for c in recent_candles[:-1]
        ]

        if volumes:

            average_volume = (
                sum(volumes)
                / len(volumes)
            )

        else:

            average_volume = 0.0

        current_volume = max(
            0.0,
            float(current.volume),
        )

        volume_ratio = self._safe_ratio(
            current_volume,
            average_volume,
        )

        bullish_candles = 0
        bearish_candles = 0

        for candle in recent_candles:

            if candle.close > candle.open:

                bullish_candles += 1

            elif candle.close < candle.open:

                bearish_candles += 1

        total_directional = (
            bullish_candles
            + bearish_candles
        )

        if total_directional > 0:

            directional_dominance = (
                bullish_candles
                - bearish_candles
            ) / total_directional

        else:

            directional_dominance = 0.0

        bullish = (
            price_return
            >= self.minimum_return
            and current.close > current.open
            and body_ratio
            >= self.minimum_body_ratio
            and directional_dominance > 0
            and (
                volume_ratio >= self.volume_multiplier
                or average_volume <= 0
            )
        )

        bearish = (
            price_return
            <= -self.minimum_return
            and current.close < current.open
            and body_ratio
            >= self.minimum_body_ratio
            and directional_dominance < 0
            and (
                volume_ratio >= self.volume_multiplier
                or average_volume <= 0
            )
        )

        normalized_return = (
            abs(price_return)
            / max(
                self.minimum_return,
                1e-12,
            )
        )

        return_strength = max(
            0.0,
            min(
                1.0,
                normalized_return / 3.0,
            ),
        )

        body_strength = max(
            0.0,
            min(
                1.0,
                body_ratio,
            ),
        )

        volume_strength = max(
            0.0,
            min(
                1.0,
                volume_ratio / 2.0,
            )
            if average_volume > 0
            else 0.5,
        )

        directional_strength = max(
            0.0,
            min(
                1.0,
                abs(directional_dominance),
            ),
        )

        confidence = (
            0.35 * return_strength
            + 0.25 * body_strength
            + 0.20 * volume_strength
            + 0.20 * directional_strength
        )

        confidence = self.safe_confidence(
            confidence
        )

        score_direction = (
            1.0
            if price_return > 0
            else -1.0
            if price_return < 0
            else 0.0
        )

        score = (
            score_direction
            * confidence
        )

        indicators: Dict[str, float] = {
            "price_return": price_return,
            "body_ratio": body_ratio,
            "volume_ratio": volume_ratio,
            "directional_dominance": (
                directional_dominance
            ),
            "bullish_candles": float(
                bullish_candles
            ),
            "bearish_candles": float(
                bearish_candles
            ),
            "lookback": float(
                self.lookback
            ),
            "current_open": float(
                current.open
            ),
            "current_high": float(
                current.high
            ),
            "current_low": float(
                current.low
            ),
            "current_close": float(
                current.close
            ),
            "current_volume": current_volume,
        }

        if bullish:

            return StrategySignal(
                strategy_name=self.name,
                symbol=current.symbol,
                action=SignalAction.BUY,
                confidence=confidence,
                timestamp=current.end_timestamp,
                reason=(
                    "Bullish short-term momentum detected."
                ),
                score=score,
                quality=confidence,
                regime="MOMENTUM_UP",
                indicators=indicators,
            )

        if bearish:

            return StrategySignal(
                strategy_name=self.name,
                symbol=current.symbol,
                action=SignalAction.SELL,
                confidence=confidence,
                timestamp=current.end_timestamp,
                reason=(
                    "Bearish short-term momentum detected."
                ),
                score=score,
                quality=confidence,
                regime="MOMENTUM_DOWN",
                indicators=indicators,
            )

        return StrategySignal(
            strategy_name=self.name,
            symbol=current.symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            timestamp=current.end_timestamp,
            reason=(
                "Momentum conditions are not strong enough."
            ),
            score=0.0,
            quality=0.20,
            regime="RANGE",
            indicators=indicators,
        )

