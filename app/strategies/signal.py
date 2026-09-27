
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class SignalAction(str, Enum):
    """
    Standard action used by the strategy engine.

    BUY/SELL are the canonical values because the existing
    StrategyEngine, TrendStrategy and BreakoutStrategy use them.

    LONG/SHORT are aliases so newer strategies can still use
    long/short terminology without creating a second action system.
    """

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    LONG = "BUY"
    SHORT = "SELL"


@dataclass(frozen=True)
class StrategySignal:
    """
    Signal produced by an individual strategy.

    Timestamps use integer microseconds, matching Candle and
    the rest of the market-data pipeline.
    """

    strategy_name: str
    symbol: str
    action: SignalAction
    confidence: float
    timestamp: int

    reason: str

    score: float = 0.0
    quality: float = 0.0
    regime: str = "UNKNOWN"

    indicators: Dict[str, float] = field(
        default_factory=dict
    )

    stop_loss: float | None = None
    take_profit: float | None = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        # ---------------------------------------------
        # Normalize action
        # ---------------------------------------------

        if not isinstance(
            self.action,
            SignalAction,
        ):
            object.__setattr__(
                self,
                "action",
                SignalAction(self.action),
            )

        # ---------------------------------------------
        # Normalize confidence
        # ---------------------------------------------

        try:
            confidence = float(
                self.confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        if confidence != confidence:
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        # ---------------------------------------------
        # Normalize score
        # ---------------------------------------------

        try:
            score = float(self.score)
        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        if score != score:
            score = 0.0

        score = max(
            -1.0,
            min(1.0, score),
        )

        object.__setattr__(
            self,
            "score",
            score,
        )

        # ---------------------------------------------
        # Normalize quality
        # ---------------------------------------------

        try:
            quality = float(
                self.quality
            )
        except (
            TypeError,
            ValueError,
        ):
            quality = 0.0

        if quality != quality:
            quality = 0.0

        quality = max(
            0.0,
            min(1.0, quality),
        )

        object.__setattr__(
            self,
            "quality",
            quality,
        )

        # ---------------------------------------------
        # Normalize timestamp
        # ---------------------------------------------

        try:
            timestamp = int(
                self.timestamp
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "StrategySignal timestamp must "
                "be an integer microsecond timestamp."
            ) from exc

        if timestamp <= 0:
            raise ValueError(
                "StrategySignal timestamp must be positive."
            )

        object.__setattr__(
            self,
            "timestamp",
            timestamp,
        )

        # ---------------------------------------------
        # Normalize regime
        # ---------------------------------------------

        regime = str(
            self.regime
        ).strip()

        if not regime:
            regime = "UNKNOWN"

        object.__setattr__(
            self,
            "regime",
            regime,
        )

    @property
    def is_entry(self) -> bool:
        """
        True when the strategy is requesting a directional entry.
        """

        return self.action in (
            SignalAction.BUY,
            SignalAction.SELL,
        )

    @property
    def is_buy(self) -> bool:
        return self.action == SignalAction.BUY

    @property
    def is_sell(self) -> bool:
        return self.action == SignalAction.SELL

    @property
    def is_hold(self) -> bool:
        return self.action == SignalAction.HOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "score": self.score,
            "quality": self.quality,
            "regime": self.regime,
            "indicators": dict(
                self.indicators
            ),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": dict(
                self.metadata
            ),
        }

    def __str__(self) -> str:
        return (
            "StrategySignal("
            f"strategy={self.strategy_name}, "
            f"symbol={self.symbol}, "
            f"action={self.action.value}, "
            f"confidence={self.confidence:.3f}, "
            f"score={self.score:.3f}, "
            f"quality={self.quality:.3f}, "
            f"regime={self.regime}, "
            f"timestamp={self.timestamp}, "
            f"reason={self.reason}"
            ")"
        )


@dataclass(frozen=True)
class AggregatedSignal:
    """
    Combined signal produced by StrategyEngine after evaluating
    all currently available strategies.
    """

    symbol: str
    action: SignalAction
    confidence: float
    timestamp: int

    reason: str

    strategy_votes: Dict[
        str,
        SignalAction,
    ] = field(
        default_factory=dict
    )

    contributing_strategies: List[str] = field(
        default_factory=list
    )

    score: float = 0.0
    quality: float = 0.0
    regime: str = "UNKNOWN"

    def __post_init__(self) -> None:
        # Normalize action.
        if not isinstance(
            self.action,
            SignalAction,
        ):
            object.__setattr__(
                self,
                "action",
                SignalAction(self.action),
            )

        # Confidence.
        try:
            confidence = float(
                self.confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        if confidence != confidence:
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        # Score.
        try:
            score = float(
                self.score
            )
        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        if score != score:
            score = 0.0

        score = max(
            -1.0,
            min(1.0, score),
        )

        object.__setattr__(
            self,
            "score",
            score,
        )

        # Quality.
        try:
            quality = float(
                self.quality
            )
        except (
            TypeError,
            ValueError,
        ):
            quality = 0.0

        if quality != quality:
            quality = 0.0

        quality = max(
            0.0,
            min(1.0, quality),
        )

        object.__setattr__(
            self,
            "quality",
            quality,
        )

        # Timestamp.
        try:
            timestamp = int(
                self.timestamp
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "AggregatedSignal timestamp must "
                "be an integer microsecond timestamp."
            ) from exc

        if timestamp <= 0:
            raise ValueError(
                "AggregatedSignal timestamp must be positive."
            )

        object.__setattr__(
            self,
            "timestamp",
            timestamp,
        )

        # Regime.
        regime = str(
            self.regime
        ).strip()

        if not regime:
            regime = "UNKNOWN"

        object.__setattr__(
            self,
            "regime",
            regime
        )

    @property
    def is_entry(self) -> bool:
        return self.action in (
            SignalAction.BUY,
            SignalAction.SELL,
        )

    @property
    def is_buy(self) -> bool:
        return self.action == SignalAction.BUY

    @property
    def is_sell(self) -> bool:
        return self.action == SignalAction.SELL

    @property
    def is_hold(self) -> bool:
        return self.action == SignalAction.HOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "strategy_votes": {
                name: action.value
                for name, action
                in self.strategy_votes.items()
            },
            "contributing_strategies": list(
                self.contributing_strategies
            ),
            "score": self.score,
            "quality": self.quality,
            "regime": self.regime,
        }

    def __str__(self) -> str:
        return (
            "AggregatedSignal("
            f"symbol={self.symbol}, "
            f"action={self.action.value}, "
            f"confidence={self.confidence:.3f}, "
            f"score={self.score:.3f}, "
            f"quality={self.quality:.3f}, "
            f"regime={self.regime}, "
            f"timestamp={self.timestamp}, "
            f"reason={self.reason}"
            ")"
        )

