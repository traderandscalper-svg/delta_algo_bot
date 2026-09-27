
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from app.market_data.candle_aggregator import Candle
from app.strategies.signals import StrategySignal


class BaseStrategy(ABC):
    """
    Common interface for every trading strategy.

    All strategies must:
        1. Have a unique name.
        2. Specify how many candles they need.
        3. Return a StrategySignal from evaluate().
        4. Never access future candles.
        5. Return HOLD when there is insufficient evidence.
    """

    name: str = "base"

    def __init__(
        self,
        minimum_candles: int = 20,
    ) -> None:
        self.minimum_candles = max(
            1,
            int(minimum_candles),
        )

    def can_evaluate(
        self,
        candles: List[Candle],
    ) -> bool:
        """
        Check whether enough historical candles are available.

        The strategy must only use candles supplied to evaluate().
        """
        return len(candles) >= self.minimum_candles

    @staticmethod
    def safe_confidence(
        value: float,
    ) -> float:
        """
        Clamp confidence to the safe range [0.0, 1.0].
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if value != value:  # NaN check
            return 0.0

        return max(
            0.0,
            min(1.0, value),
        )

    def hold(
        self,
        candle: Candle,
        reason: str,
        confidence: float = 0.0,
        indicators: Dict[str, float] | None = None,
    ) -> StrategySignal:
        """
        Create a standardized HOLD signal.
        """
        return StrategySignal(
            strategy_name=self.name,
            symbol=candle.symbol,
            action=self._hold_action(),
            confidence=self.safe_confidence(
                confidence
            ),
            timestamp=candle.end_timestamp,
            reason=reason,
            indicators=indicators or {},
        )

    @staticmethod
    def _hold_action():
        """
        Import lazily to avoid unnecessary circular imports.
        """
        from app.strategies.signals import SignalAction

        return SignalAction.HOLD

    @abstractmethod
    def evaluate(
        self,
        candles: List[Candle],
    ) -> StrategySignal:
        """
        Evaluate the strategy using historical candles.

        Implementations must not:
            - access future candles
            - modify the candle history
            - place live orders
            - depend on unavailable future information
        """
        raise NotImplementedError
