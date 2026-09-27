
from __future__ import annotations

from typing import Dict, List

from app.market_data.candle_aggregator import Candle
from app.strategies.base import BaseStrategy
from app.strategies.indicators import (
    closes,
    ema,
    returns,
    rsi,
    slope,
    trend_strength,
)
from app.strategies.signals import (
    SignalAction,
    StrategySignal,
)


class TrendStrategy(BaseStrategy):
    """
    Multi-factor trend-following strategy.

    Uses:
        - EMA alignment
        - price vs EMA
        - regression slope
        - directional efficiency
        - short/medium returns
        - RSI as an exhaustion filter

    RSI does not automatically block a strong trend.
    """

    name = "trend"

    def __init__(
        self,
        minimum_candles: int = 35,
    ) -> None:
        super().__init__(
            minimum_candles=minimum_candles
        )

    def evaluate(
        self,
        candles: List[Candle],
    ) -> StrategySignal:

        candle = candles[-1]

        if not self.can_evaluate(candles):
            return self.hold(
                candle,
                "Insufficient trend data.",
            )

        prices = closes(candles)

        if len(prices) < self.minimum_candles:
            return self.hold(
                candle,
                "Insufficient clean price data.",
            )

        fast = ema(prices, 9)
        medium = ema(prices, 21)
        slow = ema(prices, 50)

        current_rsi = rsi(
            prices,
            14,
        )

        price_slope = slope(
            prices,
            20,
        )

        strength = trend_strength(
            prices,
            20,
        )

        short_return = returns(
            prices,
            10,
        )

        medium_return = returns(
            prices,
            20,
        )

        if (
            fast is None
            or medium is None
            or current_rsi is None
            or price_slope is None
            or strength is None
            or short_return is None
            or medium_return is None
        ):
            return self.hold(
                candle,
                "Insufficient trend indicators.",
            )

        current_price = float(
            prices[-1]
        )

        if current_price <= 0:
            return self.hold(
                candle,
                "Invalid current price.",
            )

        # -------------------------------------------------
        # Directional evidence
        # -------------------------------------------------

        bullish_points = 0.0
        bearish_points = 0.0

        if fast > medium:
            bullish_points += 1.0
        elif fast < medium:
            bearish_points += 1.0

        if current_price > medium:
            bullish_points += 1.0
        elif current_price < medium:
            bearish_points += 1.0

        if slow is not None:

            if fast > slow:
                bullish_points += 1.0
            elif fast < slow:
                bearish_points += 1.0

        if price_slope > 0:
            bullish_points += 1.0
        elif price_slope < 0:
            bearish_points += 1.0

        if short_return > 0:
            bullish_points += 1.0
        elif short_return < 0:
            bearish_points += 1.0

        if medium_return > 0:
            bullish_points += 1.0
        elif medium_return < 0:
            bearish_points += 1.0

        total_points = (
            bullish_points
            + bearish_points
        )

        if total_points <= 0:
            directional_score = 0.0
        else:
            directional_score = (
                bullish_points
                - bearish_points
            ) / total_points

        # -------------------------------------------------
        # Trend strength
        # -------------------------------------------------

        strong_direction = strength >= 0.35
        very_strong_direction = strength >= 0.70

        # -------------------------------------------------
        # RSI handling
        #
        # In a normal trend:
        #   BUY  -> RSI >= 50 and < 90
        #   SELL -> RSI <= 50 and > 10
        #
        # In an exceptionally clean trend, RSI may reach
        # 100 or 0. Do not reject such a trend automatically.
        # -------------------------------------------------

        bullish_rsi_ok = (
            (
                current_rsi >= 50.0
                and current_rsi < 90.0
            )
            or (
                very_strong_direction
                and directional_score >= 0.50
                and current_rsi >= 70.0
            )
        )

        bearish_rsi_ok = (
            (
                current_rsi <= 50.0
                and current_rsi > 10.0
            )
            or (
                very_strong_direction
                and directional_score <= -0.50
                and current_rsi <= 30.0
            )
        )

        # -------------------------------------------------
        # Final trend conditions
        # -------------------------------------------------

        bullish_trend = (
            bullish_points >= 4.0
            and directional_score >= 0.50
            and strong_direction
            and bullish_rsi_ok
        )

        bearish_trend = (
            bearish_points >= 4.0
            and directional_score <= -0.50
            and strong_direction
            and bearish_rsi_ok
        )

        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        direction_strength = min(
            1.0,
            abs(directional_score),
        )

        efficiency_strength = min(
            1.0,
            max(0.0, strength),
        )

        return_strength = min(
            1.0,
            abs(medium_return) / 0.001,
        )

        confidence = (
            0.45 * direction_strength
            + 0.30 * efficiency_strength
            + 0.15 * return_strength
            + 0.10
        )

        if very_strong_direction:
            confidence += 0.05

        confidence = self.safe_confidence(
            min(confidence, 0.95)
        )

        indicators: Dict[str, float] = {
            "ema_9": float(fast),
            "ema_21": float(medium),
            "ema_50": (
                float(slow)
                if slow is not None
                else 0.0
            ),
            "rsi_14": float(current_rsi),
            "slope_20": float(price_slope),
            "trend_strength": float(strength),
            "return_10": float(short_return),
            "return_20": float(medium_return),
            "bullish_points": float(
                bullish_points
            ),
            "bearish_points": float(
                bearish_points
            ),
            "directional_score": float(
                directional_score
            ),
            "current_price": current_price,
        }

        # -------------------------------------------------
        # BUY
        # -------------------------------------------------

        if bullish_trend:

            return StrategySignal(
                strategy_name=self.name,
                symbol=candle.symbol,
                action=SignalAction.BUY,
                confidence=confidence,
                timestamp=candle.end_timestamp,
                score=directional_score,
                quality=confidence,
                regime="TREND_UP",
                reason=(
                    "Multiple directional factors "
                    "confirm an upward trend."
                ),
                indicators=indicators,
            )

        # -------------------------------------------------
        # SELL
        # -------------------------------------------------

        if bearish_trend:

            return StrategySignal(
                strategy_name=self.name,
                symbol=candle.symbol,
                action=SignalAction.SELL,
                confidence=confidence,
                timestamp=candle.end_timestamp,
                score=directional_score,
                quality=confidence,
                regime="TREND_DOWN",
                reason=(
                    "Multiple directional factors "
                    "confirm a downward trend."
                ),
                indicators=indicators,
            )

        # -------------------------------------------------
        # HOLD
        # -------------------------------------------------

        return StrategySignal(
            strategy_name=self.name,
            symbol=candle.symbol,
            action=SignalAction.HOLD,
            confidence=0.20,
            timestamp=candle.end_timestamp,
            score=directional_score,
            quality=0.20,
            regime="RANGE",
            reason=(
                "Trend evidence is insufficient "
                "or directional agreement is weak."
            ),
            indicators=indicators,
        )
def _aggregate(
    self,
    candle: Candle,
    signals: List[StrategySignal],
) -> AggregatedSignal:

    if not signals:

        return AggregatedSignal(
            symbol=candle.symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            timestamp=candle.end_timestamp,
            reason="No strategy has enough data.",
            strategy_votes={},
            contributing_strategies=[],
            score=0.0,
            quality=0.0,
            regime="WARMUP",
        )

    buy_weight = 0.0
    sell_weight = 0.0

    score_sum = 0.0
    quality_sum = 0.0
    quality_count = 0

    votes: Dict[
        str,
        SignalAction,
    ] = {}

    contributors: List[str] = []

    # Only directional signals should determine the
    # dominant market regime. HOLD/RANGE signals are
    # deliberately excluded from regime voting.
    directional_regimes: Dict[str, int] = {}

    for signal in signals:

        votes[
            signal.strategy_name
        ] = signal.action

        effective_weight = (
            signal.confidence
            * max(
                0.25,
                signal.quality
                if signal.quality > 0
                else 1.0,
            )
        )

        if signal.action == SignalAction.BUY:

            buy_weight += effective_weight

            if signal.regime:
                directional_regimes[
                    signal.regime
                ] = (
                    directional_regimes.get(
                        signal.regime,
                        0,
                    )
                    + 1
                )

        elif signal.action == SignalAction.SELL:

            sell_weight += effective_weight

            if signal.regime:
                directional_regimes[
                    signal.regime
                ] = (
                    directional_regimes.get(
                        signal.regime,
                        0,
                    )
                    + 1
                )

        score_sum += signal.score
        quality_sum += signal.quality
        quality_count += 1

    directional_weight = (
        buy_weight
        + sell_weight
    )

    average_score = (
        score_sum
        / max(len(signals), 1)
    )

    average_quality = (
        quality_sum
        / max(quality_count, 1)
    )

    # If at least one strategy has a real directional
    # signal, use its regime. HOLD/RANGE cannot override it.
    dominant_regime = (
        max(
            directional_regimes,
            key=directional_regimes.get,
        )
        if directional_regimes
        else "RANGE"
    )

    if directional_weight <= 0:

        return AggregatedSignal(
            symbol=candle.symbol,
            action=SignalAction.HOLD,
            confidence=0.20,
            timestamp=candle.end_timestamp,
            reason="All strategies are neutral.",
            strategy_votes=votes,
            contributing_strategies=[],
            score=average_score,
            quality=average_quality,
            regime=dominant_regime,
        )

    difference = (
        buy_weight
        - sell_weight
    )

    normalized_direction = (
        difference
        / directional_weight
    )

    # Require meaningful directional agreement.
    if abs(normalized_direction) < 0.15:

        return AggregatedSignal(
            symbol=candle.symbol,
            action=SignalAction.HOLD,
            confidence=0.20,
            timestamp=candle.end_timestamp,
            reason=(
                "Strategy disagreement is too high."
            ),
            strategy_votes=votes,
            contributing_strategies=[],
            score=average_score,
            quality=average_quality,
            regime=dominant_regime,
        )

    if buy_weight > sell_weight:

        action = SignalAction.BUY

        confidence = (
            buy_weight
            / directional_weight
        )

    else:

        action = SignalAction.SELL

        confidence = (
            sell_weight
            / directional_weight
        )

    confidence *= (
        0.75
        + (
            average_quality
            * 0.25
        )
    )

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    for signal in signals:

        if signal.action == action:
            contributors.append(
                signal.strategy_name
            )

    reason = (
        f"BUY weight={buy_weight:.3f}, "
        f"SELL weight={sell_weight:.3f}, "
        f"score={average_score:.3f}, "
        f"quality={average_quality:.3f}"
    )

    result = AggregatedSignal(
        symbol=candle.symbol,
        action=action,
        confidence=confidence,
        timestamp=candle.end_timestamp,
        reason=reason,
        strategy_votes=votes,
        contributing_strategies=contributors,
        score=normalized_direction,
        quality=average_quality,
        regime=dominant_regime,
    )

    self.logger.info(
        "Strategy signal | "
        "symbol=%s | interval=%ss | "
        "action=%s | confidence=%.3f | "
        "score=%.3f | quality=%.3f | "
        "regime=%s | votes=%s | "
        "contributors=%s",
        candle.symbol,
        candle.interval_seconds,
        result.action.value,
        result.confidence,
        result.score,
        result.quality,
        result.regime,
        votes,
        contributors,
    )

    return result
