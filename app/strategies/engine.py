
from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List

from app.market_data.candle_aggregator import Candle

from app.strategies.base import BaseStrategy
from app.strategies.breakout import BreakoutStrategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.signals import (
    AggregatedSignal,
    SignalAction,
    StrategySignal,
)
from app.strategies.trend import TrendStrategy


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class StrategyEngine:
    """
    Central strategy engine.

    Responsibilities:
        - Maintain candle history per symbol/timeframe.
        - Run all configured strategies.
        - Collect strategy signals.
        - Weight directional signals.
        - Resolve BUY/SELL/HOLD.
        - Select the most relevant market regime.
        - Persist structured signal diagnostics.

    The engine does NOT:
        - Place orders.
        - Manage positions.
        - Access private exchange APIs.
        - Make execution decisions.

    Phase 7.4:
        Every evaluated candle produces a structured diagnostic
        JSONL record under:

            data/strategy/

        This is for inspection, backtesting and later ML work.
    """

    def __init__(
        self,
        max_candles: int = 200,
        diagnostics_directory: str = "data/strategy",
    ) -> None:

        self.logger = logging.getLogger(
            "StrategyEngine"
        )

        self.max_candles = max(
            50,
            int(max_candles),
        )

        self._candles: Dict[
            tuple[str, int],
            Deque[Candle],
        ] = {}

        self.strategies: List[
            BaseStrategy
        ] = [
            MomentumStrategy(),
            TrendStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
        ]

        # Adaptive weights are applied at aggregation time.
        # They are updated by the Phase 12 orchestration layer
        # after enough completed trade outcomes exist.
        self.strategy_weights = {
            strategy.name: 1.0
            for strategy in self.strategies
        }

        # -------------------------------------------------
        # Phase 7.4 diagnostics
        # -------------------------------------------------

        self.diagnostics_directory = Path(
            diagnostics_directory
        )

        self.diagnostics_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.logger.info(
            "Strategy engine initialized | strategies=%s",
            [
                strategy.name
                for strategy in self.strategies
            ],
        )

        self.logger.info(
            "Strategy diagnostics enabled | directory=%s",
            self.diagnostics_directory,
        )

    # =====================================================
    # CANDLE INPUT
    # =====================================================

    def add_candle(
        self,
        candle: Candle,
    ) -> AggregatedSignal:

        key = (
            candle.symbol,
            candle.interval_seconds,
        )

        if key not in self._candles:

            self._candles[key] = deque(
                maxlen=self.max_candles
            )

        self._candles[key].append(
            candle
        )

        candles = list(
            self._candles[key]
        )

        signals: List[
            StrategySignal
        ] = []

        strategy_errors: List[str] = []

        for strategy in self.strategies:

            if not strategy.can_evaluate(
                candles
            ):
                continue

            try:

                signal = strategy.evaluate(
                    candles
                )

                signals.append(
                    signal
                )

            except Exception as exc:

                strategy_errors.append(
                    strategy.name
                )

                self.logger.exception(
                    "Strategy evaluation failed | "
                    "strategy=%s | symbol=%s",
                    strategy.name,
                    candle.symbol,
                )

        result = self._aggregate(
            candle=candle,
            signals=signals,
        )

        self._write_diagnostic(
            candle=candle,
            result=result,
            signals=signals,
            strategy_errors=strategy_errors,
            buffer_size=len(candles),
        )

        return result

    # =====================================================
    # SIGNAL AGGREGATION
    # =====================================================

    def _aggregate(
        self,
        candle: Candle,
        signals: List[StrategySignal],
    ) -> AggregatedSignal:

        # -------------------------------------------------
        # No strategy has enough data yet.
        # -------------------------------------------------

        if not signals:

            return AggregatedSignal(
                symbol=candle.symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                timestamp=candle.end_timestamp,
                reason=(
                    "No strategy has enough data."
                ),
                strategy_votes={},
                contributing_strategies=[],
                score=0.0,
                quality=0.0,
                regime="WARMUP",
            )

        # -------------------------------------------------
        # Directional weights.
        # -------------------------------------------------

        buy_weight = 0.0
        sell_weight = 0.0

        score_sum = 0.0
        quality_sum = 0.0
        quality_count = 0

        votes: Dict[
            str,
            SignalAction,
        ] = {}

        # -------------------------------------------------
        # Process strategy signals.
        # -------------------------------------------------

        for signal in signals:

            votes[
                signal.strategy_name
            ] = signal.action

            # Quality-adjusted confidence.
            #
            # A strategy with zero quality should still
            # retain a minimum weighting instead of being
            # completely discarded.
            strategy_weight = clamp(
                float(
                    self.strategy_weights.get(
                        signal.strategy_name,
                        1.0,
                    )
                ),
                0.50,
                1.50,
            )

            effective_weight = (
                signal.confidence
                * max(
                    0.25,
                    signal.quality
                    if signal.quality > 0
                    else 1.0,
                )
                * strategy_weight
            )

            if signal.action == SignalAction.BUY:

                buy_weight += effective_weight

            elif signal.action == SignalAction.SELL:

                sell_weight += effective_weight

            score_sum += signal.score
            quality_sum += signal.quality
            quality_count += 1

        # -------------------------------------------------
        # Total directional weight.
        # -------------------------------------------------

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

        # -------------------------------------------------
        # No directional signal.
        # -------------------------------------------------

        if directional_weight <= 0:

            return AggregatedSignal(
                symbol=candle.symbol,
                action=SignalAction.HOLD,
                confidence=0.20,
                timestamp=candle.end_timestamp,
                reason=(
                    "All strategies are neutral."
                ),
                strategy_votes=votes,
                contributing_strategies=[],
                score=average_score,
                quality=average_quality,
                regime="RANGE",
            )

        # -------------------------------------------------
        # Directional balance.
        #
        # +1.0 = completely BUY
        # -1.0 = completely SELL
        #  0.0 = perfectly balanced
        # -------------------------------------------------

        difference = (
            buy_weight
            - sell_weight
        )

        normalized_direction = (
            difference
            / directional_weight
        )

        # -------------------------------------------------
        # Prevent trading when strategies strongly
        # disagree.
        # -------------------------------------------------

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
                regime="MIXED",
            )

        # -------------------------------------------------
        # Determine final direction.
        # -------------------------------------------------

        if buy_weight > sell_weight:

            action = SignalAction.BUY

        else:

            action = SignalAction.SELL

        # -------------------------------------------------
        # Determine contributing strategies.
        # -------------------------------------------------

        directional_signals = [
            signal
            for signal in signals
            if signal.action
            in (
                SignalAction.BUY,
                SignalAction.SELL,
            )
        ]

        contributing_signals = [
            signal
            for signal in directional_signals
            if signal.action == action
        ]

        contributors = [
            signal.strategy_name
            for signal in contributing_signals
        ]

        # -------------------------------------------------
        # Determine market regime.
        # -------------------------------------------------

        regime_priority = {
            "BREAKOUT_UP": 100,
            "BREAKOUT_DOWN": 100,

            "MOMENTUM_UP": 90,
            "MOMENTUM_DOWN": 90,

            "TREND_UP": 80,
            "TREND_DOWN": 80,

            "MEAN_REVERSION_UP": 70,
            "MEAN_REVERSION_DOWN": 70,

            "RANGE": 0,
            "UNKNOWN": 0,
        }

        best_regime = "UNKNOWN"
        best_priority = -1

        for signal in contributing_signals:

            regime = (
                signal.regime
                or "UNKNOWN"
            )

            priority = regime_priority.get(
                regime,
                10,
            )

            if priority > best_priority:

                best_priority = priority
                best_regime = regime

        # -------------------------------------------------
        # Fallback regime.
        # -------------------------------------------------

        if best_regime == "UNKNOWN":

            if action == SignalAction.BUY:

                best_regime = "TREND_UP"

            else:

                best_regime = "TREND_DOWN"

        # -------------------------------------------------
        # Final confidence.
        # -------------------------------------------------

        if buy_weight > sell_weight:

            directional_confidence = (
                buy_weight
                / directional_weight
            )

        else:

            directional_confidence = (
                sell_weight
                / directional_weight
            )

        confidence = (
            directional_confidence
            * (
                0.75
                + (
                    average_quality
                    * 0.25
                )
            )
        )

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        # -------------------------------------------------
        # Explanation.
        # -------------------------------------------------

        reason = (
            f"BUY weight={buy_weight:.3f}, "
            f"SELL weight={sell_weight:.3f}, "
            f"score={average_score:.3f}, "
            f"quality={average_quality:.3f}"
        )

        # -------------------------------------------------
        # Final aggregated signal.
        # -------------------------------------------------

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
            regime=best_regime,
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
            {
                name: action.value
                for name, action in votes.items()
            },
            contributors,
        )

        return result

    # =====================================================
    # PHASE 7.4 DIAGNOSTICS
    # =====================================================

    @staticmethod
    def _timestamp_to_iso(
        timestamp: int,
    ) -> str:

        try:
            return datetime.fromtimestamp(
                timestamp / 1_000_000,
                tz=timezone.utc,
            ).isoformat()
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return ""

    @staticmethod
    def _signal_to_dict(
        signal: StrategySignal,
    ) -> Dict[str, Any]:

        return {
            "strategy_name": signal.strategy_name,
            "symbol": signal.symbol,
            "action": signal.action.value,
            "confidence": signal.confidence,
            "score": signal.score,
            "quality": signal.quality,
            "regime": signal.regime,
            "timestamp": signal.timestamp,
            "reason": signal.reason,
            "indicators": dict(
                signal.indicators
            ),
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "metadata": dict(
                signal.metadata
            ),
        }

    def _diagnostic_path(
        self,
        timestamp: int,
    ) -> Path:

        try:
            dt = datetime.fromtimestamp(
                timestamp / 1_000_000,
                tz=timezone.utc,
            )

            filename = (
                f"strategy_signals_"
                f"{dt.strftime('%Y-%m-%d')}.jsonl"
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):

            filename = (
                "strategy_signals_unknown.jsonl"
            )

        return (
            self.diagnostics_directory
            / filename
        )

    def _write_diagnostic(
        self,
        candle: Candle,
        result: AggregatedSignal,
        signals: List[StrategySignal],
        strategy_errors: List[str],
        buffer_size: int,
    ) -> None:
        """
        Persist one complete strategy evaluation.

        JSONL is intentionally used instead of CSV because
        strategy indicators and metadata can have variable
        fields.
        """

        record: Dict[str, Any] = {
            "record_type": "strategy_signal",
            "diagnostic_version": 1,

            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "symbol": candle.symbol,

            "interval_seconds": (
                candle.interval_seconds
            ),

            "candle": {
                "start_timestamp": (
                    candle.start_timestamp
                ),
                "end_timestamp": (
                    candle.end_timestamp
                ),
                "start_time": self._timestamp_to_iso(
                    candle.start_timestamp
                ),
                "end_time": self._timestamp_to_iso(
                    candle.end_timestamp
                ),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "trade_count": candle.trade_count,
                "first_trade_timestamp": (
                    candle.first_trade_timestamp
                ),
                "last_trade_timestamp": (
                    candle.last_trade_timestamp
                ),
            },

            "engine": {
                "buffer_size": buffer_size,
                "configured_strategies": [
                    strategy.name
                    for strategy in self.strategies
                ],
                "evaluated_strategy_count": len(
                    signals
                ),
                "strategy_errors": list(
                    strategy_errors
                ),
            },

            "final_signal": {
                "action": result.action.value,
                "confidence": result.confidence,
                "score": result.score,
                "quality": result.quality,
                "regime": result.regime,
                "timestamp": result.timestamp,
                "timestamp_iso": (
                    self._timestamp_to_iso(
                        result.timestamp
                    )
                ),
                "reason": result.reason,
                "strategy_votes": {
                    name: action.value
                    for name, action
                    in result.strategy_votes.items()
                },
                "contributing_strategies": list(
                    result.contributing_strategies
                ),
            },

            "strategy_signals": [
                self._signal_to_dict(signal)
                for signal in signals
            ],
        }

        path = self._diagnostic_path(
            candle.end_timestamp
        )

        try:

            with path.open(
                "a",
                encoding="utf-8",
            ) as handle:

                handle.write(
                    json.dumps(
                        record,
                        separators=(
                            ",",
                            ":",
                        ),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )

                handle.write("\n")

        except Exception:

            # Diagnostics must NEVER stop market-data
            # processing or strategy evaluation.
            self.logger.exception(
                "Strategy diagnostic write failed | "
                "path=%s | symbol=%s",
                path,
                candle.symbol,
            )

    # =====================================================
    # BUFFER INFORMATION
    # =====================================================

    def get_buffer_size(
        self,
        symbol: str,
        interval_seconds: int,
    ) -> int:

        key = (
            symbol,
            interval_seconds,
        )

        candles = self._candles.get(
            key
        )

        if candles is None:
            return 0

        return len(candles)

    # =====================================================
    # CLEAR ALL HISTORY
    # =====================================================

    def clear(self) -> None:

        self._candles.clear()

        self.logger.info(
            "Strategy candle history cleared."
        )

