import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any
import json
import os
import hashlib

from app.config.settings import Settings
from app.exchange.client import DeltaClient
from app.market_data.candle_aggregator import CandleAggregator
from app.market_data.collector import MarketDataCollector
from app.safety.watchdog import Watchdog
from app.strategies.engine import StrategyEngine as Phase9StrategyEngine
from app.ml.predict import MLPredictor
from app.strategies.signal_performance_tracker import SignalPerformanceTracker
from app.core.persistence import EnginePersistence


# ============================================================
# DELTA ALGORITHMIC TRADING ENGINE
# PHASE 1 -> PHASE 14
# ============================================================
#
# Real Delta market data
# Real account equity
# Real L1 order book
# Real trade feed
# Real trade-derived candles
# Multi-timeframe analysis
# Strategy ensemble
# Risk management
# Paper execution
# Dynamic SL / TP
# Trailing stop
# Portfolio analytics
# Adaptive strategy weighting
# ML-ready learning layer
# Autonomous monitoring
#
# IMPORTANT:
# Real exchange order execution is intentionally disabled.
# ============================================================


class EngineState(str, Enum):
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    CONNECTING = "CONNECTING"
    RUNNING = "RUNNING"
    SAFE_MODE = "SAFE_MODE"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    SHUTDOWN = "SHUTDOWN"
    ERROR = "ERROR"


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ============================================================
# GENERAL HELPERS
# ============================================================


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default

        value = float(value)

        if math.isfinite(value):
            return value

        return default

    except (TypeError, ValueError):
        return default


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, value))


def extract_payload(event: Any) -> dict[str, Any]:
    if event is None:
        return {}

    if isinstance(event, dict):
        return event

    payload = getattr(event, "payload", None)

    if isinstance(payload, dict):
        return payload

    data = getattr(event, "data", None)

    if isinstance(data, dict):
        return data

    return {}


def get_event_timestamp(event: Any) -> float:
    timestamp = (
        getattr(event, "exchange_timestamp", None)
        or getattr(event, "received_timestamp", None)
    )

    if timestamp is None:
        return time.time()

    timestamp = safe_float(timestamp, time.time())

    if timestamp > 1_000_000_000_000:
        return timestamp / 1_000_000

    if timestamp > 1_000_000_000:
        return timestamp

    return timestamp


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class StrategySignal:
    strategy: str
    action: SignalAction
    confidence: float
    reason: str
    timestamp: float


@dataclass
class AggregatedSignal:
    action: SignalAction
    confidence: float
    reason: str
    timestamp: float

    votes: list[str] = field(default_factory=list)
    strategy_samples: list[StrategySignal] = field(
        default_factory=list
    )

    score: float = 0.0
    regime: str = "UNKNOWN"
    timeframe_confirmation: float = 0.0
    quality_score: float = 0.0
    learning_id: str = ""


@dataclass
class MarketFeatures:
    symbol: str
    timestamp: float

    timeframe: int = 1

    price: float = 0.0
    previous_price: float = 0.0

    return_1: float = 0.0
    return_5: float = 0.0
    return_15: float = 0.0
    return_60: float = 0.0

    rsi: float = 50.0
    volatility: float = 0.0

    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0

    mark_price: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0
    open_interest_change: float = 0.0

    spread: float = 0.0
    spread_bps: float = 0.0
    orderbook_imbalance: float = 0.0

    buy_volume: float = 0.0
    sell_volume: float = 0.0

    trade_flow: float = 0.0
    trade_flow_acceleration: float = 0.0

    trend: float = 0.0
    momentum: float = 0.0
    mean_reversion: float = 0.0

    regime: str = "UNKNOWN"
    data_quality: str = "INVALID"

    multi_timeframe_score: float = 0.0
    signal_quality: float = 0.0

    warmup_complete: bool = False


@dataclass
class PaperPosition:
    symbol: str
    side: str
    quantity: float

    entry_price: float
    entry_timestamp: float

    stop_loss: float
    take_profit: float
    trailing_stop: float

    entry_wall_time: float = 0.0

    highest_price: float = 0.0
    lowest_price: float = 0.0

    strategy_names: list[str] = field(
        default_factory=list
    )

    entry_confidence: float = 0.0
    entry_notional: float = 0.0

    unrealized_pnl: float = 0.0
    max_unrealized_pnl: float = 0.0
    min_unrealized_pnl: float = 0.0

    initial_risk: float = 0.0

    trailing_activated: bool = False
    exit_reason: str = ""
    learning_id: str = ""
    entry_signal_action: str = "HOLD"


# ============================================================
# FEATURE ENGINE
# ============================================================


class FeatureEngine:

    TIMEFRAMES = (1, 5, 15, 60)

    def __init__(
        self,
        stale_data_timeout_seconds: int = 5,
    ):
        self.logger = logging.getLogger(
            "FeatureEngine"
        )

        self.stale_data_timeout_seconds = (
            stale_data_timeout_seconds
        )

        self.price_history = {
            timeframe: deque(maxlen=5000)
            for timeframe in self.TIMEFRAMES
        }

        self.volume_history = {
            timeframe: deque(maxlen=5000)
            for timeframe in self.TIMEFRAMES
        }

        self.latest_candles: dict[str, dict] = {}

        self.rsi_history = deque(maxlen=5000)
        self.flow_history = deque(maxlen=200)

        self.latest_live_microstructure = {}
        self.latest_market_context = {
            "mark_price": 0.0,
            "funding_rate": 0.0,
            "open_interest": 0.0,
            "open_interest_change": 0.0,
        }

        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.trade_count = 0

        self.last_trade_price = 0.0
        self.last_trade_timestamp = 0.0

        self.last_feature_timestamp = 0.0

    # --------------------------------------------------------

    def process_event(self, event: Any) -> None:
        payload = extract_payload(event)

        event_type = getattr(
            event,
            "event_type",
            None,
        )

        if event_type is None:
            event_type = payload.get("type")

        timestamp = get_event_timestamp(event)

        if event_type == "ob_l1":
            self._process_l1(
                payload,
                timestamp,
            )

        elif event_type == "trades":
            self._process_trade(
                payload,
                timestamp,
            )

        elif event_type == "ticker":
            self._process_ticker(
                payload,
                timestamp,
            )

        elif event_type == "mark_price":
            self._process_mark_price(payload)

        elif event_type == "funding_rate":
            self._process_funding_rate(payload)

    # --------------------------------------------------------

    def _process_l1(
        self,
        payload: dict[str, Any],
        timestamp: float,
    ) -> None:

        bid = safe_float(
            payload.get(
                "bp",
                payload.get(
                    "bid",
                    payload.get(
                        "best_bid",
                        payload.get(
                            "bid_price",
                            0.0,
                        ),
                    ),
                ),
            )
        )

        ask = safe_float(
            payload.get(
                "ap",
                payload.get(
                    "ask",
                    payload.get(
                        "best_ask",
                        payload.get(
                            "ask_price",
                            0.0,
                        ),
                    ),
                ),
            )
        )

        bid_size = safe_float(
            payload.get(
                "bs",
                payload.get(
                    "bid_size",
                    payload.get(
                        "best_bid_size",
                        0.0,
                    ),
                ),
            )
        )

        ask_size = safe_float(
            payload.get(
                "as",
                payload.get(
                    "ask_size",
                    payload.get(
                        "best_ask_size",
                        0.0,
                    ),
                ),
            )
        )

        if bid <= 0 or ask <= 0:
            return

        if ask < bid:
            return

        self.latest_live_microstructure = {
            "bid": bid,
            "ask": ask,
            "bid_size": max(
                bid_size,
                0.0,
            ),
            "ask_size": max(
                ask_size,
                0.0,
            ),
            "received_time": time.monotonic(),
            "timestamp": timestamp,
        }

    # --------------------------------------------------------

    def _process_mark_price(self, payload: dict[str, Any]) -> None:
        if isinstance(payload.get("d"), list) and payload["d"] and isinstance(payload["d"][0], dict):
            payload = payload["d"][0]
        value = safe_float(payload.get("p", payload.get("mark_price", payload.get("price"))))
        if value > 0:
            self.latest_market_context["mark_price"] = value

    def _process_funding_rate(self, payload: dict[str, Any]) -> None:
        if isinstance(payload.get("d"), list) and payload["d"] and isinstance(payload["d"][0], dict):
            payload = payload["d"][0]
        rate = safe_float(payload.get("fr", payload.get("funding_rate")))
        if math.isfinite(rate):
            self.latest_market_context["funding_rate"] = rate

    def _process_trade(
        self,
        payload: dict[str, Any],
        timestamp: float,
    ) -> None:

        price = safe_float(
            payload.get(
                "p",
                payload.get(
                    "price",
                    payload.get(
                        "trade_price",
                        0.0,
                    ),
                ),
            )
        )

        size = safe_float(
            payload.get(
                "s",
                payload.get(
                    "size",
                    payload.get(
                        "q",
                        payload.get(
                            "quantity",
                            payload.get(
                                "trade_size",
                                0.0,
                            ),
                        ),
                    ),
                ),
            )
        )

        if price <= 0:
            return

        size = max(
            size,
            0.0,
        )

        self.trade_count += 1

        self.last_trade_price = price
        self.last_trade_timestamp = timestamp

        book = self.latest_live_microstructure

        bid = safe_float(
            book.get("bid")
        )

        ask = safe_float(
            book.get("ask")
        )

        # Do not interpret Delta's maker/taker
        # role directly as buy/sell direction.
        explicit_side = str(
            payload.get(
                "side",
                payload.get(
                    "taker_side",
                    payload.get(
                        "aggressor_side",
                        "",
                    ),
                ),
            )
        ).lower()

        if explicit_side in (
            "buy",
            "b",
            "bid",
        ):

            self.buy_volume += size

        elif explicit_side in (
            "sell",
            "s",
            "ask",
        ):

            self.sell_volume += size

        elif bid > 0 and ask > 0:

            midpoint = (
                bid + ask
            ) / 2

            if price >= ask:

                self.buy_volume += size

            elif price <= bid:

                self.sell_volume += size

            elif price > midpoint:

                self.buy_volume += (
                    size * 0.75
                )

                self.sell_volume += (
                    size * 0.25
                )

            elif price < midpoint:

                self.buy_volume += (
                    size * 0.25
                )

                self.sell_volume += (
                    size * 0.75
                )

            else:

                self.buy_volume += (
                    size * 0.50
                )

                self.sell_volume += (
                    size * 0.50
                )

        else:

            self.buy_volume += (
                size * 0.50
            )

            self.sell_volume += (
                size * 0.50
            )

    # --------------------------------------------------------

    def _process_ticker(
        self,
        payload: dict[str, Any],
        timestamp: float,
    ) -> None:
        mark = safe_float(payload.get("m", payload.get("mark_price")))
        if mark > 0:
            self.latest_market_context["mark_price"] = mark
        oi = payload.get("oi")
        if isinstance(oi, (list, tuple)):
            if len(oi) > 0:
                self.latest_market_context["open_interest"] = safe_float(oi[0])
            if len(oi) > 1:
                self.latest_market_context["open_interest_change"] = safe_float(oi[1])
        else:
            self.latest_market_context["open_interest"] = safe_float(payload.get("open_interest"), self.latest_market_context["open_interest"])
            self.latest_market_context["open_interest_change"] = safe_float(payload.get("open_interest_change"), self.latest_market_context["open_interest_change"])


        data = payload.get("d")

        if not isinstance(
            data,
            list,
        ):
            return

        if not data:
            return

        item = data[0]

        if not isinstance(
            item,
            dict,
        ):
            return

        symbol = item.get("s")

        if not symbol:
            return

        ohlc = item.get(
            "ohlc",
            [],
        )

        self.latest_candles.setdefault(
            symbol,
            {},
        )

        ticker_mark = safe_float(
            item.get("m")
        )

        ticker_close = safe_float(
            ohlc[3]
            if len(ohlc) > 3
            else 0
        )

        ticker_open_interest = safe_float(
            (
                item.get("oi")
                or [0]
            )[0]
        )

        self.latest_candles[
            symbol
        ].update(
            {
                # Delta field "m" is mark price, not last trade.
                "mark_price": ticker_mark,
                "ticker_price": ticker_close,
                "ticker_close": ticker_close,
                "timestamp": timestamp,
                "open": safe_float(
                    ohlc[0]
                    if len(ohlc) > 0
                    else 0
                ),
                "high": safe_float(
                    ohlc[1]
                    if len(ohlc) > 1
                    else 0
                ),
                "low": safe_float(
                    ohlc[2]
                    if len(ohlc) > 2
                    else 0
                ),
                "close": ticker_close,
                "open_interest": ticker_open_interest,
                "change_24h": safe_float(
                    item.get("m24hc")
                ),
            }
        )

        self.logger.info(
            "Ticker | symbol=%s | last=%s | mark=%s | oi=%s",
            symbol,
            ticker_close,
            ticker_mark,
            ticker_open_interest,
        )

    # --------------------------------------------------------

    def process_candle(
        self,
        candle: Any,
    ) -> MarketFeatures:

        symbol = getattr(
            candle,
            "symbol",
            "UNKNOWN",
        )

        timeframe = int(
            safe_float(
                getattr(
                    candle,
                    "interval_seconds",
                    1,
                ),
                1,
            )
        )

        if timeframe not in self.TIMEFRAMES:
            timeframe = 1

        timestamp = safe_float(
            getattr(
                candle,
                "end_timestamp",
                time.time(),
            ),
            time.time(),
        )

        if timestamp > 1_000_000_000_000:
            timestamp /= 1_000_000

        close = safe_float(
            getattr(
                candle,
                "close",
                0.0,
            )
        )

        volume = safe_float(
            getattr(
                candle,
                "volume",
                0.0,
            )
        )

        if close <= 0:

            return MarketFeatures(
                symbol=symbol,
                timestamp=timestamp,
                timeframe=timeframe,
                data_quality="INVALID",
            )

        history = self.price_history[
            timeframe
        ]

        previous_price = (
            history[-1]
            if history
            else close
        )

        history.append(close)

        if volume > 0:

            self.volume_history[
                timeframe
            ].append(volume)

        self.latest_candles[
            symbol
        ] = {
            "timeframe": timeframe,
            "close": close,
            "timestamp": timestamp,
        }

        # Higher timeframes update confirmation
        # data only. They do not independently
        # generate entry signals.
        if timeframe != 1:

            return MarketFeatures(
                symbol=symbol,
                timestamp=timestamp,
                timeframe=timeframe,
                price=close,
                previous_price=previous_price,
                data_quality="GOOD",
                warmup_complete=(
                    self.warmup_complete()
                ),
            )

        return self._build_execution_features(
            symbol,
            timestamp,
            close,
            previous_price,
        )

    # --------------------------------------------------------

    def _build_execution_features(
        self,
        symbol,
        timestamp,
        close,
        previous_price,
    ) -> MarketFeatures:

        return_1 = self._timeframe_return(
            1,
            1,
        )

        return_5 = self._timeframe_return(
            5,
            1,
        )

        return_15 = self._timeframe_return(
            15,
            1,
        )

        return_60 = self._timeframe_return(
            60,
            1,
        )

        rsi = self._calculate_rsi(
            1,
            14,
        )

        volatility = (
            self._calculate_volatility(
                1,
                30,
            )
        )

        (
            bid,
            ask,
            bid_size,
            ask_size,
        ) = self._get_microstructure()

        spread = (
            max(
                ask - bid,
                0.0,
            )
            if bid > 0 and ask > 0
            else 0.0
        )

        midpoint = (
            (bid + ask) / 2
            if bid > 0 and ask > 0
            else 0.0
        )

        spread_bps = (
            spread / midpoint * 10_000
            if midpoint > 0
            else 0.0
        )

        total_book = (
            bid_size + ask_size
        )

        orderbook_imbalance = (
            (
                bid_size
                - ask_size
            )
            / total_book
            if total_book > 0
            else 0.0
        )

        total_flow = (
            self.buy_volume
            + self.sell_volume
        )

        trade_flow = (
            (
                self.buy_volume
                - self.sell_volume
            )
            / total_flow
            if total_flow > 0
            else 0.0
        )

        self.flow_history.append(
            trade_flow
        )

        flow_acceleration = 0.0

        if len(
            self.flow_history
        ) >= 6:

            previous_flow = (
                self.flow_history[-6]
            )

            flow_acceleration = clamp(
                trade_flow
                - previous_flow,
                -1.0,
                1.0,
            )

        trend = self._calculate_trend(
            1
        )

        momentum = self._calculate_momentum(
            1
        )

        mean_reversion = (
            self._calculate_mean_reversion(
                close
            )
        )

        regime = self._detect_regime(
            trend,
            volatility,
            momentum,
        )

        mtf_score = (
            self._multi_timeframe_score()
        )

        data_quality = (
            self._calculate_quality(
                bid,
                ask,
                spread_bps,
            )
        )

        warmup = (
            self.warmup_complete()
        )

        features = MarketFeatures(
            symbol=symbol,
            timestamp=timestamp,
            timeframe=1,
            price=close,
            previous_price=previous_price,
            return_1=return_1,
            return_5=return_5,
            return_15=return_15,
            return_60=return_60,
            rsi=rsi,
            volatility=volatility,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            mark_price=safe_float(self.latest_market_context.get("mark_price"), price),
            funding_rate=safe_float(self.latest_market_context.get("funding_rate")),
            open_interest=safe_float(self.latest_market_context.get("open_interest")),
            open_interest_change=safe_float(self.latest_market_context.get("open_interest_change")),
            spread=spread,
            spread_bps=spread_bps,
            orderbook_imbalance=(
                orderbook_imbalance
            ),
            buy_volume=self.buy_volume,
            sell_volume=self.sell_volume,
            trade_flow=trade_flow,
            trade_flow_acceleration=(
                flow_acceleration
            ),
            trend=trend,
            momentum=momentum,
            mean_reversion=mean_reversion,
            regime=regime,
            data_quality=data_quality,
            multi_timeframe_score=mtf_score,
            warmup_complete=warmup,
        )

        features.signal_quality = (
            self._signal_quality(
                features
            )
        )

        self.rsi_history.append(
            rsi
        )

        self.last_feature_timestamp = (
            timestamp
        )

        # Start a fresh flow measurement
        # for the next execution candle.
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.trade_count = 0

        return features

    # --------------------------------------------------------

    def warmup_complete(self) -> bool:

        # Approximately three minutes of
        # real multi-timeframe history.
        requirements = {
            1: 30,
            5: 10,
            15: 5,
            60: 3,
        }

        return all(
            len(
                self.price_history[
                    timeframe
                ]
            ) >= minimum
            for timeframe, minimum
            in requirements.items()
        )

    # --------------------------------------------------------

    def _timeframe_return(
        self,
        timeframe: int,
        periods: int,
    ) -> float:

        history = self.price_history.get(
            timeframe,
            (),
        )

        if len(history) <= periods:
            return 0.0

        old = history[
            -periods - 1
        ]

        current = history[-1]

        if old <= 0:
            return 0.0

        return (
            current - old
        ) / old

    # --------------------------------------------------------

    def _get_microstructure(self):

        book = (
            self.latest_live_microstructure
        )

        return (
            safe_float(
                book.get("bid")
            ),
            safe_float(
                book.get("ask")
            ),
            safe_float(
                book.get("bid_size")
            ),
            safe_float(
                book.get("ask_size")
            ),
        )

    # --------------------------------------------------------

    def _fresh_book(self) -> bool:

        book = (
            self.latest_live_microstructure
        )

        if not book:
            return False

        received = safe_float(
            book.get("received_time")
        )

        if received <= 0:
            return False

        return (
            time.monotonic()
            - received
            <= self.stale_data_timeout_seconds
        )

    # --------------------------------------------------------

    def _calculate_rsi(
        self,
        timeframe,
        period=14,
    ) -> float:

        history = (
            self.price_history[
                timeframe
            ]
        )

        if len(history) <= period:
            return 50.0

        prices = list(history)[
            -(period + 1):
        ]

        gains = []
        losses = []

        for index in range(
            1,
            len(prices),
        ):

            change = (
                prices[index]
                - prices[index - 1]
            )

            if change > 0:

                gains.append(change)
                losses.append(0.0)

            else:

                gains.append(0.0)
                losses.append(
                    abs(change)
                )

        average_gain = (
            sum(gains)
            / period
        )

        average_loss = (
            sum(losses)
            / period
        )

        if average_loss == 0:

            if average_gain > 0:
                return 100.0

            return 50.0

        rs = (
            average_gain
            / average_loss
        )

        return 100 - (
            100
            / (1 + rs)
        )

    # --------------------------------------------------------

    def _calculate_volatility(
        self,
        timeframe,
        periods=30,
    ) -> float:

        history = (
            self.price_history[
                timeframe
            ]
        )

        if len(history) < 10:
            return 0.0

        prices = list(history)[
            -(periods + 1):
        ]

        returns = []

        for index in range(
            1,
            len(prices),
        ):

            previous = (
                prices[index - 1]
            )

            current = (
                prices[index]
            )

            if previous <= 0:
                continue

            returns.append(
                (
                    current - previous
                )
                / previous
            )

        if len(returns) < 2:
            return 0.0

        mean = (
            sum(returns)
            / len(returns)
        )

        variance = (
            sum(
                (
                    value - mean
                ) ** 2
                for value in returns
            )
            / len(returns)
        )

        return math.sqrt(
            max(
                variance,
                0.0,
            )
        )

    # --------------------------------------------------------

    def _calculate_trend(
        self,
        timeframe=1,
    ) -> float:

        history = (
            self.price_history[
                timeframe
            ]
        )

        if len(history) < 20:
            return 0.0

        prices = list(history)

        short_average = (
            sum(prices[-5:])
            / 5
        )

        long_average = (
            sum(prices[-20:])
            / 20
        )

        if long_average <= 0:
            return 0.0

        return clamp(
            (
                short_average
                - long_average
            )
            / long_average,
            -1.0,
            1.0,
        )

    # --------------------------------------------------------

    def _calculate_momentum(
        self,
        timeframe=1,
    ) -> float:

        history = (
            self.price_history[
                timeframe
            ]
        )

        if len(history) < 6:
            return 0.0

        old = history[-6]
        current = history[-1]

        if old <= 0:
            return 0.0

        return clamp(
            (
                current - old
            )
            / old,
            -1.0,
            1.0,
        )

    # --------------------------------------------------------

    def _calculate_mean_reversion(
        self,
        price,
    ) -> float:

        history = (
            self.price_history[1]
        )

        if len(history) < 20:
            return 0.0

        prices = list(history)[
            -20:
        ]

        average = (
            sum(prices)
            / len(prices)
        )

        if average <= 0:
            return 0.0

        return clamp(
            -(
                price - average
            )
            / average,
            -1.0,
            1.0,
        )

    # --------------------------------------------------------

    def _multi_timeframe_score(self) -> float:

        if not self.warmup_complete():
            return 0.0

        r1 = self._timeframe_return(
            1,
            3,
        )

        r5 = self._timeframe_return(
            5,
            2,
        )

        r15 = self._timeframe_return(
            15,
            2,
        )

        r60 = self._timeframe_return(
            60,
            1,
        )

        score = (
            r1 * 0.10
            + r5 * 0.20
            + r15 * 0.30
            + r60 * 0.40
        )

        return clamp(
            score * 500,
            -1.0,
            1.0,
        )

    # --------------------------------------------------------

    def _detect_regime(
        self,
        trend,
        volatility,
        momentum,
    ) -> str:

        if volatility > 0.003:
            return "HIGH_VOLATILITY"

        if abs(trend) > 0.0008:

            if trend > 0:
                return "TREND_UP"

            return "TREND_DOWN"

        if abs(momentum) > 0.0005:

            if momentum > 0:
                return "MOMENTUM_UP"

            return "MOMENTUM_DOWN"

        return "RANGE"

    # --------------------------------------------------------

    def _calculate_quality(
        self,
        bid,
        ask,
        spread_bps,
    ) -> str:

        if bid <= 0 or ask <= 0:
            return "INVALID"

        if not self._fresh_book():
            return "PARTIAL"

        if spread_bps > 20:
            return "PARTIAL"

        return "GOOD"

    # --------------------------------------------------------

    def _signal_quality(
        self,
        features,
    ) -> float:

        if not features.warmup_complete:
            return 0.0

        score = 0.0

        if features.data_quality == "GOOD":
            score += 0.30

        elif features.data_quality == "PARTIAL":
            score += 0.10

        if features.spread_bps <= 5:
            score += 0.20

        elif features.spread_bps <= 10:
            score += 0.10

        score += (
            abs(
                features.orderbook_imbalance
            )
            * 0.20
        )

        score += (
            abs(
                features.trade_flow
            )
            * 0.15
        )

        score += (
            abs(
                features.multi_timeframe_score
            )
            * 0.15
        )

        return clamp(
            score,
            0.0,
            1.0,
        )


# ============================================================
# STRATEGY ENGINE
# ============================================================
#
# PHASE 9 INTEGRATION ADAPTER
#
# The main TradingEngine historically used an internal strategy
# implementation.  Phase 9 introduced the real strategy package
# under app/strategies/.  This adapter makes Phase 9 the source of
# strategy decisions while preserving the legacy signal interface
# expected by the validator, learning, risk and paper-execution
# layers in this engine.
#
# No market data is fabricated here.
# ============================================================


class StrategyEngine:

    STRATEGIES = (
        "TREND",
        "MOMENTUM",
        "MEAN_REVERSION",
        "BREAKOUT",
    )

    def __init__(self):

        self.logger = logging.getLogger(
            "StrategyEngine"
        )

        self.latest_signal = None

        self.strategy_weights = {
            strategy: 1.0
            for strategy in self.STRATEGIES
        }

        try:
            self.phase9 = Phase9StrategyEngine()
            self.phase9_available = True

            # Keep the public strategy list synchronized when the
            # Phase 9 engine exposes its own strategy names.
            phase9_names = getattr(
                self.phase9,
                "STRATEGIES",
                None,
            )

            if phase9_names:
                normalized = []

                for name in phase9_names:
                    name = str(name).upper()

                    if name not in normalized:
                        normalized.append(name)

                if normalized:
                    self.STRATEGIES = tuple(normalized)

                    self.strategy_weights = {
                        name: 1.0
                        for name in self.STRATEGIES
                    }

            self.logger.info(
                "Phase 9 strategy engine loaded | strategies=%s",
                ",".join(self.STRATEGIES),
            )

        except Exception as exc:

            self.phase9 = None
            self.phase9_available = False

            self.logger.exception(
                "Phase 9 strategy engine could not be loaded: %s",
                exc,
            )

    # --------------------------------------------------------

    @staticmethod
    def _action_from_value(value):
        value = getattr(
            value,
            "value",
            value,
        )

        value = str(
            value or "HOLD"
        ).upper()

        if value.endswith(".BUY"):
            value = "BUY"

        elif value.endswith(".SELL"):
            value = "SELL"

        elif value.endswith(".HOLD"):
            value = "HOLD"

        if value == "BUY":
            return SignalAction.BUY

        if value == "SELL":
            return SignalAction.SELL

        return SignalAction.HOLD

    # --------------------------------------------------------

    @staticmethod
    def _safe_attr(
        obj,
        names,
        default=None,
    ):
        for name in names:

            try:
                value = getattr(
                    obj,
                    name,
                    None,
                )

            except Exception:
                value = None

            if value is not None:
                return value

        return default

    # --------------------------------------------------------

    def _call_phase9(
        self,
        candles,
        features,
    ):
        """
        Call the Phase 9 engine without assuming one exact method
        signature.  This keeps the integration compatible with the
        Phase 9 strategy-engine implementation while making the
        required inputs explicit.
        """

        if self.phase9 is None:
            return None

        # Phase 9 is a candle-buffer engine.  Its public API is
        # add_candle(candle) -> AggregatedSignal; it does not expose
        # evaluate().  Feed it the real candle emitted by the
        # CandleAggregator.
        if not candles:
            return None

        candle = candles[-1]

        add_candle = getattr(
            self.phase9,
            "add_candle",
            None,
        )

        if not callable(add_candle):
            raise AttributeError(
                "Phase 9 StrategyEngine has no callable add_candle()."
            )

        return add_candle(candle)

    # --------------------------------------------------------

    def _convert_votes(
        self,
        raw_signal,
        action,
        confidence,
        timestamp,
    ):
        """
        Convert Phase 9 vote/contributor metadata into the legacy
        StrategySignal list used by StrategyValidator.

        Phase 9 may expose votes as strings, dictionaries or objects.
        All forms are normalized without inventing a separate market
        signal.
        """

        raw_votes = self._safe_attr(
            raw_signal,
            (
                "strategy_votes",
                "votes",
                "strategy_names",
            ),
            [],
        )

        contributors = self._safe_attr(
            raw_signal,
            (
                "contributing_strategies",
                "contributors",
            ),
            [],
        )

        if raw_votes is None:
            raw_votes = []

        if contributors is None:
            contributors = []

        # Phase 9 AggregatedSignal.strategy_votes is a mapping of
        # strategy name -> SignalAction.  Preserve that information
        # instead of iterating only over dictionary keys.
        if isinstance(raw_votes, dict):
            raw_votes = list(raw_votes.items())
        elif not isinstance(raw_votes, (list, tuple, set)):
            raw_votes = [raw_votes]

        if not isinstance(contributors, (list, tuple, set)):
            contributors = [contributors]

        samples = []
        names_seen = set()

        def add_sample(
            name,
            sample_action=None,
            sample_confidence=None,
            reason="Phase 9 strategy contribution.",
        ):
            name = str(
                name or "PHASE9"
            ).strip()

            if not name:
                return

            # Normalize common names while preserving custom Phase 9
            # strategy names.
            normalized_name = name.upper()

            if ":" in normalized_name:
                base, suffix = normalized_name.rsplit(
                    ":",
                    1,
                )

                if suffix in (
                    "BUY",
                    "SELL",
                    "HOLD",
                ):
                    normalized_name = base

                    if sample_action is None:
                        sample_action = (
                            self._action_from_value(
                                suffix
                            )
                        )

            if normalized_name in names_seen:
                return

            names_seen.add(
                normalized_name
            )

            if sample_action is None:
                sample_action = action

            if sample_confidence is None:
                sample_confidence = confidence

            samples.append(
                StrategySignal(
                    strategy=normalized_name,
                    action=sample_action,
                    confidence=clamp(
                        safe_float(
                            sample_confidence
                        ),
                        0.0,
                        1.0,
                    ),
                    reason=str(
                        reason
                        or "Phase 9 strategy contribution."
                    ),
                    timestamp=timestamp,
                )
            )

        # Parse strategy vote entries.
        for item in raw_votes:

            if isinstance(item, tuple) and len(item) == 2:
                name, vote_value = item
                add_sample(
                    name,
                    self._action_from_value(vote_value),
                    confidence,
                    "Phase 9 strategy vote.",
                )

            elif isinstance(
                item,
                dict,
            ):
                name = (
                    item.get("strategy")
                    or item.get("name")
                    or item.get("strategy_name")
                )

                item_action = self._action_from_value(
                    item.get(
                        "action",
                        action,
                    )
                )

                item_confidence = item.get(
                    "confidence",
                    confidence,
                )

                reason = item.get(
                    "reason",
                    "Phase 9 strategy vote.",
                )

                add_sample(
                    name,
                    item_action,
                    item_confidence,
                    reason,
                )

            else:
                text = str(item)

                if ":" in text:
                    name, suffix = text.rsplit(
                        ":",
                        1,
                    )

                    parsed_action = self._action_from_value(
                        suffix
                    )

                    add_sample(
                        name,
                        parsed_action,
                        confidence,
                        "Phase 9 strategy vote.",
                    )

                else:
                    add_sample(
                        text,
                        action,
                        confidence,
                        "Phase 9 strategy vote.",
                    )

        # Contributors are normally strategy names.  Add them when
        # they were not already represented by strategy_votes.
        for item in contributors:

            if isinstance(
                item,
                dict,
            ):
                add_sample(
                    item.get(
                        "strategy",
                        item.get(
                            "name",
                            "PHASE9",
                        ),
                    ),
                    self._action_from_value(
                        item.get(
                            "action",
                            action,
                        )
                    ),
                    item.get(
                        "confidence",
                        confidence,
                    ),
                    item.get(
                        "reason",
                        "Phase 9 contributing strategy.",
                    ),
                )

            else:
                add_sample(
                    item,
                    action,
                    confidence,
                    "Phase 9 contributing strategy.",
                )

        # A valid Phase 9 signal should normally expose contributors.
        # If it does not, retain one explicit aggregated sample so the
        # downstream validator still receives a coherent signal.
        if not samples:

            add_sample(
                "PHASE9_AGGREGATED",
                action,
                confidence,
                "Aggregated Phase 9 strategy signal.",
            )

        return samples

    # --------------------------------------------------------

    def _convert_signal(
        self,
        raw_signal,
        features,
    ):

        if raw_signal is None:
            return self._result(
                SignalAction.HOLD,
                0.0,
                "Phase 9 returned no signal.",
                features,
                [],
            )

        action = self._action_from_value(
            self._safe_attr(
                raw_signal,
                ("action",),
                "HOLD",
            )
        )

        confidence = clamp(
            safe_float(
                self._safe_attr(
                    raw_signal,
                    ("confidence",),
                    0.0,
                )
            ),
            0.0,
            1.0,
        )

        score = safe_float(
            self._safe_attr(
                raw_signal,
                ("score",),
                0.0,
            )
        )

        quality = clamp(
            safe_float(
                self._safe_attr(
                    raw_signal,
                    (
                        "quality",
                        "quality_score",
                    ),
                    getattr(
                        features,
                        "signal_quality",
                        0.0,
                    ),
                )
            ),
            0.0,
            1.0,
        )

        regime = str(
            self._safe_attr(
                raw_signal,
                ("regime",),
                getattr(
                    features,
                    "regime",
                    "UNKNOWN",
                ),
            )
            or "UNKNOWN"
        )

        reason = str(
            self._safe_attr(
                raw_signal,
                ("reason",),
                "Phase 9 strategy evaluation.",
            )
            or "Phase 9 strategy evaluation."
        )

        timestamp = safe_float(
            self._safe_attr(
                raw_signal,
                ("timestamp",),
                features.timestamp,
            ),
            features.timestamp,
        )

        samples = self._convert_votes(
            raw_signal,
            action,
            confidence,
            timestamp,
        )

        votes = [
            sample.strategy
            for sample in samples
        ]

        result = AggregatedSignal(
            action=action,
            confidence=confidence,
            reason=reason,
            timestamp=timestamp,
            votes=votes,
            strategy_samples=samples,
            score=score,
            regime=regime,
            timeframe_confirmation=safe_float(
                getattr(
                    features,
                    "multi_timeframe_score",
                    0.0,
                )
            ),
            quality_score=quality,
        )

        self.latest_signal = result

        self.logger.info(
            "Strategy signal | symbol=%s | interval=%ss | action=%s | "
            "confidence=%.4f | score=%.4f | quality=%.4f | regime=%s | "
            "votes=%s | contributors=%s",
            getattr(
                features,
                "symbol",
                "UNKNOWN",
            ),
            getattr(
                features,
                "timeframe",
                1,
            ),
            action.value,
            confidence,
            score,
            quality,
            regime,
            ",".join(votes) if votes else "NONE",
            ",".join(votes) if votes else "NONE",
        )

        return result

    # --------------------------------------------------------

    def evaluate(
        self,
        features,
        candles=None,
    ) -> AggregatedSignal:
        """Compatibility entry point for the legacy core engine.

        The Phase 9 engine itself is driven through add_candle().
        This wrapper preserves the legacy TradingEngine call site so
        the rest of the Phase 1-14 pipeline does not need to change.
        """

        timestamp = safe_float(
            getattr(
                features,
                "timestamp",
                time.time(),
            ),
            time.time(),
        )

        # Phase 9 has its own candle-history warm-up. Do NOT block it on
        # FeatureEngine.warmup_complete(), because that broader MTF gate
        # requires 5s/15s/60s history and can take several minutes.
        # Phase 9 strategies only need the real 1-second candles supplied
        # below; each strategy decides when it has enough candles.

        if not self.phase9_available:
            return self._result(
                SignalAction.HOLD,
                0.0,
                "Phase 9 strategy engine unavailable.",
                features,
                [],
            )

        if candles is None:
            candles = []

        try:

            if not candles:
                self.logger.debug(
                    "Phase 9 waiting for execution candles | symbol=%s",
                    getattr(features, "symbol", "UNKNOWN"),
                )
                return self._result(
                    SignalAction.HOLD,
                    0.0,
                    "Phase 9 waiting for execution candles.",
                    features,
                    [],
                )

            candle = candles[-1]
            candle_count = len(candles)

            self.logger.info(
                "Phase 9 candle feed | symbol=%s | candle_count=%d | "
                "timestamp=%s | close=%s | core_mtf_warmup=%s",
                getattr(candle, "symbol", getattr(features, "symbol", "UNKNOWN")),
                candle_count,
                getattr(candle, "start_timestamp", "UNKNOWN"),
                getattr(candle, "close", "UNKNOWN"),
                getattr(features, "warmup_complete", False),
            )

            raw_signal = self._call_phase9(
                candles,
                features,
            )

            if raw_signal is not None:
                self.logger.info(
                    "Phase 9 result | symbol=%s | candle_count=%d | "
                    "action=%s | confidence=%s | reason=%s",
                    getattr(features, "symbol", "UNKNOWN"),
                    candle_count,
                    getattr(getattr(raw_signal, "action", None), "value", getattr(raw_signal, "action", "HOLD")),
                    getattr(raw_signal, "confidence", 0.0),
                    getattr(raw_signal, "reason", ""),
                )

            return self._convert_signal(
                raw_signal,
                features,
            )

        except Exception as exc:

            self.logger.exception(
                "Phase 9 strategy evaluation failed: %s",
                exc,
            )

            return self._result(
                SignalAction.HOLD,
                0.0,
                "Phase 9 strategy evaluation failed safely.",
                features,
                [],
            )

    # --------------------------------------------------------

    def update_strategy_weights(
        self,
        performance,
    ) -> None:

        strategies = (
            performance.get(
                "strategies",
                {},
            )
            if isinstance(
                performance,
                dict,
            )
            else {}
        )

        for name in self.STRATEGIES:

            stats = strategies.get(
                name
            )

            if not stats:
                continue

            trades = int(
                safe_float(
                    stats.get(
                        "trades",
                        0,
                    )
                )
            )

            if trades < 5:
                continue

            win_rate = safe_float(
                stats.get(
                    "win_rate"
                )
            )

            pnl = safe_float(
                stats.get(
                    "net_pnl"
                )
            )

            weight = 1.0

            if win_rate > 0.60:
                weight += 0.20

            elif win_rate < 0.40:
                weight -= 0.20

            if pnl > 0:
                weight += 0.10

            elif pnl < 0:
                weight -= 0.10

            self.strategy_weights[
                name
            ] = clamp(
                weight,
                0.50,
                1.50,
            )

        # Pass weights into Phase 9 if it exposes a compatible public
        # weight mapping.  This is optional and never blocks trading.
        try:

            phase9_weights = getattr(
                self.phase9,
                "strategy_weights",
                None,
            )

            if isinstance(
                phase9_weights,
                dict,
            ):
                phase9_weights.update(
                    self.strategy_weights
                )

        except Exception:
            pass

    # --------------------------------------------------------

    def _result(
        self,
        action,
        confidence,
        reason,
        features,
        samples,
    ):

        result = AggregatedSignal(
            action=action,
            confidence=clamp(
                safe_float(
                    confidence
                ),
                0.0,
                1.0,
            ),
            reason=str(
                reason
            ),
            timestamp=safe_float(
                getattr(
                    features,
                    "timestamp",
                    time.time(),
                ),
                time.time(),
            ),
            votes=[
                sample.strategy
                for sample in samples
            ],
            strategy_samples=samples,
            regime=str(
                getattr(
                    features,
                    "regime",
                    "UNKNOWN",
                )
                or "UNKNOWN"
            ),
            timeframe_confirmation=safe_float(
                getattr(
                    features,
                    "multi_timeframe_score",
                    0.0,
                )
            ),
            quality_score=clamp(
                safe_float(
                    getattr(
                        features,
                        "signal_quality",
                        0.0,
                    )
                ),
                0.0,
                1.0,
            ),
        )

        self.latest_signal = result

        return result


# ============================================================
# STRATEGY VALIDATOR
# ============================================================


class StrategyValidator:

    def __init__(self):

        self.logger = logging.getLogger(
            "StrategyValidator"
        )

        self.total_signals = 0

        self.buy_signals = 0
        self.sell_signals = 0
        self.hold_signals = 0

        self.strategy_stats = {}

        self.trade_results = []

        self.wins = 0
        self.losses = 0
        self.breakeven = 0

        self.gross_profit = 0.0
        self.gross_loss = 0.0

        self.equity_curve = [0.0]

        self.max_drawdown = 0.0
        self.peak_pnl = 0.0

        self.consecutive_wins = 0
        self.consecutive_losses = 0

        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0

    # --------------------------------------------------------

    def _new_stats(self):

        return {
            "signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "confidence_sum": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_pnl": 0.0,
        }

    # --------------------------------------------------------

    def _get_stats(
        self,
        name,
    ):

        if name not in self.strategy_stats:

            self.strategy_stats[
                name
            ] = self._new_stats()

        return self.strategy_stats[
            name
        ]

    # --------------------------------------------------------

    def record_signal(
        self,
        signal,
    ):

        self.total_signals += 1

        if signal.action == SignalAction.BUY:
            self.buy_signals += 1

        elif signal.action == SignalAction.SELL:
            self.sell_signals += 1

        else:
            self.hold_signals += 1

        for sample in (
            signal.strategy_samples
        ):

            stats = self._get_stats(
                sample.strategy
            )

            stats["signals"] += 1

            stats["confidence_sum"] += (
                sample.confidence
            )

            if (
                sample.action
                == SignalAction.BUY
            ):
                stats["buy_signals"] += 1

            elif (
                sample.action
                == SignalAction.SELL
            ):
                stats["sell_signals"] += 1

            else:
                stats["hold_signals"] += 1

    # --------------------------------------------------------

    def record_trade_result(
        self,
        strategy_names,
        pnl,
    ):

        pnl = safe_float(pnl)

        self.trade_results.append(
            pnl
        )

        if pnl > 0:

            self.wins += 1

            self.gross_profit += pnl

            self.consecutive_wins += 1
            self.consecutive_losses = 0

            self.max_consecutive_wins = max(
                self.max_consecutive_wins,
                self.consecutive_wins,
            )

        elif pnl < 0:

            self.losses += 1

            self.gross_loss += abs(pnl)

            self.consecutive_losses += 1
            self.consecutive_wins = 0

            self.max_consecutive_losses = max(
                self.max_consecutive_losses,
                self.consecutive_losses,
            )

        else:

            self.breakeven += 1

            self.consecutive_wins = 0
            self.consecutive_losses = 0

        net = (
            self.gross_profit
            - self.gross_loss
        )

        self.equity_curve.append(
            net
        )

        self.peak_pnl = max(
            self.peak_pnl,
            net,
        )

        drawdown = (
            self.peak_pnl
            - net
        )

        self.max_drawdown = max(
            self.max_drawdown,
            drawdown,
        )

        # Count the completed trade once globally, while also
        # attributing its outcome to each strategy that contributed
        # to the entry. This makes Phase 12 adaptive weighting
        # meaningful without multiplying the global trade count.
        names = [
            str(name).upper()
            for name in (strategy_names or [])
            if str(name).strip()
        ]
        if not names:
            names = ["PHASE9_AGGREGATED"]

        for name in dict.fromkeys(names):
            stats = self._get_stats(name)
            stats["trades"] += 1
            stats["net_pnl"] += pnl

            if pnl > 0:
                stats["wins"] += 1
                stats["gross_profit"] += pnl
            elif pnl < 0:
                stats["losses"] += 1
                stats["gross_loss"] += abs(pnl)
            else:
                stats["breakeven"] += 1

    # --------------------------------------------------------

    def get_stats(self):

        total_trades = (
            self.wins
            + self.losses
            + self.breakeven
        )

        net = (
            self.gross_profit
            - self.gross_loss
        )

        average_trade = (
            sum(self.trade_results)
            / len(self.trade_results)
            if self.trade_results
            else 0.0
        )

        strategies = {}

        for (
            name,
            stats,
        ) in self.strategy_stats.items():

            trades = stats["trades"]

            strategies[name] = {
                **stats,

                "average_confidence": (
                    stats[
                        "confidence_sum"
                    ]
                    / stats["signals"]
                    if stats["signals"]
                    else 0.0
                ),

                "win_rate": (
                    stats["wins"]
                    / trades
                    if trades
                    else 0.0
                ),

                "loss_rate": (
                    stats["losses"]
                    / trades
                    if trades
                    else 0.0
                ),

                "profit_factor": (
                    stats["gross_profit"]
                    / stats["gross_loss"]
                    if stats["gross_loss"] > 0
                    else 0.0
                ),
            }

        return {
            "signals": self.total_signals,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "hold_signals": self.hold_signals,

            "trades": total_trades,

            "wins": self.wins,
            "losses": self.losses,
            "breakeven": self.breakeven,

            "win_rate": (
                self.wins
                / total_trades
                if total_trades
                else 0.0
            ),

            "loss_rate": (
                self.losses
                / total_trades
                if total_trades
                else 0.0
            ),

            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "net_pnl": net,

            "profit_factor": (
                self.gross_profit
                / self.gross_loss
                if self.gross_loss > 0
                else 0.0
            ),

            "average_trade": average_trade,
            "expectancy": average_trade,

            "max_drawdown": (
                self.max_drawdown
            ),

            "max_consecutive_wins": (
                self.max_consecutive_wins
            ),

            "max_consecutive_losses": (
                self.max_consecutive_losses
            ),

            "strategies": strategies,
        }


# ============================================================
# RISK ENGINE
# ============================================================


class RiskEngine:

    def __init__(
        self,
        account_equity,
        max_risk_per_trade,
        max_daily_loss,
        max_trades_per_day,
        max_leverage,
    ):

        self.logger = logging.getLogger(
            "RiskEngine"
        )

        self.account_equity = max(
            safe_float(
                account_equity
            ),
            0.0,
        )

        self.starting_equity = (
            self.account_equity
        )

        self.max_risk_per_trade = (
            max_risk_per_trade
        )

        self.max_daily_loss = (
            max_daily_loss
        )

        self.max_trades_per_day = (
            max_trades_per_day
        )

        self.max_leverage = (
            max_leverage
        )

        self.daily_realized_pnl = 0.0
        self.trades_today = 0

        self.current_day = date.today()

    # --------------------------------------------------------

    def _reset_if_new_day(self):

        today = date.today()

        if today != self.current_day:

            self.current_day = today

            self.daily_realized_pnl = 0.0
            self.trades_today = 0

            self.starting_equity = (
                self.account_equity
            )

    # --------------------------------------------------------

    def update_equity(
        self,
        equity,
    ):

        self.account_equity = max(
            safe_float(equity),
            0.0,
        )

    # --------------------------------------------------------

    def record_trade(self):

        self._reset_if_new_day()

        self.trades_today += 1

    # --------------------------------------------------------

    def record_pnl(
        self,
        pnl,
    ):

        self._reset_if_new_day()

        pnl = safe_float(pnl)

        self.daily_realized_pnl += pnl

        self.account_equity = max(
            self.account_equity + pnl,
            0.0,
        )

    # --------------------------------------------------------

    def daily_loss_limit_reached(self):

        self._reset_if_new_day()

        if self.starting_equity <= 0:
            return True

        loss = max(
            -self.daily_realized_pnl,
            0.0,
        )

        return (
            loss
            >= (
                self.starting_equity
                * self.max_daily_loss
            )
        )

    # --------------------------------------------------------

    def trade_limit_reached(self):

        self._reset_if_new_day()

        return (
            self.trades_today
            >= self.max_trades_per_day
        )

    # --------------------------------------------------------

    def check_signal(
        self,
        signal,
        features,
    ):

        self._reset_if_new_day()

        if (
            signal.action
            == SignalAction.HOLD
        ):
            return False, "SIGNAL_HOLD"

        if not features.warmup_complete:

            return False, (
                "MARKET_DATA_WARMUP"
            )

        if signal.confidence < 0.55:

            return False, (
                "CONFIDENCE_TOO_LOW:"
                f"{signal.confidence:.4f}"
            )

        if features.data_quality != "GOOD":

            return False, (
                "DATA_QUALITY_NOT_GOOD:"
                f"{features.data_quality}"
            )

        if features.signal_quality < 0.45:

            return False, (
                "SIGNAL_QUALITY_TOO_LOW:"
                f"{features.signal_quality:.4f}"
            )

        if self.daily_loss_limit_reached():

            return False, (
                "DAILY_LOSS_LIMIT_REACHED"
            )

        if self.trade_limit_reached():

            return False, (
                "MAX_TRADES_PER_DAY_REACHED"
            )

        if self.account_equity <= 0:

            return False, (
                "ACCOUNT_EQUITY_ZERO_OR_NEGATIVE"
            )

        if features.price <= 0:

            return False, (
                "INVALID_ENTRY_PRICE"
            )

        if (
            features.bid <= 0
            or features.ask <= 0
        ):

            return False, (
                "INVALID_ORDERBOOK"
            )

        if features.ask < features.bid:

            return False, (
                "INVALID_BID_ASK"
            )

        if features.spread <= 0:

            return False, (
                "INVALID_SPREAD"
            )

        if features.spread_bps > 20:

            return False, (
                "SPREAD_TOO_WIDE:"
                f"{features.spread_bps:.2f}BPS"
            )

        if (
            signal.action
            == SignalAction.BUY
            and features.multi_timeframe_score
            < -0.40
        ):

            return False, (
                "MTF_STRONGLY_BEARISH"
            )

        if (
            signal.action
            == SignalAction.SELL
            and features.multi_timeframe_score
            > 0.40
        ):

            return False, (
                "MTF_STRONGLY_BULLISH"
            )

        return True, (
            "PASSED_ALL_RISK_FILTERS"
        )

    # --------------------------------------------------------

    def calculate_quantity(
        self,
        price,
        stop_distance,
    ):

        price = safe_float(
            price
        )

        stop_distance = safe_float(
            stop_distance
        )

        if price <= 0:

            return (
                0.0,
                "INVALID_PRICE",
            )

        if stop_distance <= 0:

            return (
                0.0,
                "INVALID_STOP_DISTANCE",
            )

        risk_capital = (
            self.account_equity
            * self.max_risk_per_trade
        )

        quantity = (
            risk_capital
            / stop_distance
        )

        maximum_notional = (
            self.account_equity
            * self.max_leverage
        )

        maximum_quantity = (
            maximum_notional
            / price
        )

        quantity = min(
            quantity,
            maximum_quantity,
        )

        if quantity <= 0:

            return (
                0.0,
                "CALCULATED_QUANTITY_ZERO",
            )

        return (
            quantity,
            "QUANTITY_CALCULATED",
        )

    # --------------------------------------------------------

    def get_stats(self):

        self._reset_if_new_day()

        return {
            "account_equity": (
                self.account_equity
            ),

            "starting_equity": (
                self.starting_equity
            ),

            "daily_realized_pnl": (
                self.daily_realized_pnl
            ),

            "daily_loss_limit": (
                self.starting_equity
                * self.max_daily_loss
            ),

            "trades_today": (
                self.trades_today
            ),

            "max_trades_per_day": (
                self.max_trades_per_day
            ),

            "max_risk_per_trade": (
                self.max_risk_per_trade
            ),

            "max_leverage": (
                self.max_leverage
            ),

            "daily_loss_limit_reached": (
                self.daily_loss_limit_reached()
            ),

            "trade_limit_reached": (
                self.trade_limit_reached()
            ),
        }


# ============================================================
# PAPER EXECUTION ENGINE
# ============================================================


class PaperExecutionEngine:

    def __init__(
        self,
        risk_engine,
        trade_result_handler=None,
        cooldown_seconds=5,
    ):

        self.logger = logging.getLogger(
            "PaperExecutionEngine"
        )

        self.risk_engine = risk_engine

        self.trade_result_handler = (
            trade_result_handler
        )

        self.positions = {}

        self.entries = 0
        self.exits = 0
        self.completed_trades = 0

        self.wins = 0
        self.losses = 0
        self.breakeven = 0

        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.realized_pnl = 0.0

        self.trade_pnls = []

        self.holding_times = []

        self.largest_win = 0.0
        self.largest_loss = 0.0

        self.total_volume = 0.0
        self.total_notional = 0.0

        self.exit_reasons = {}

        self.cooldown_seconds = (
            cooldown_seconds
        )

        self.last_exit_time = {}

        self._latest_bid = 0.0
        self._latest_ask = 0.0

    # --------------------------------------------------------

    def update_book(
        self,
        bid,
        ask,
    ):

        self._latest_bid = safe_float(
            bid
        )

        self._latest_ask = safe_float(
            ask
        )

    # --------------------------------------------------------

    def process_candle(
        self,
        candle,
    ):

        symbol = getattr(
            candle,
            "symbol",
            None,
        )

        if not symbol:
            return

        timeframe = int(
            safe_float(
                getattr(
                    candle,
                    "interval_seconds",
                    1,
                ),
                1,
            )
        )

        if timeframe != 1:
            return

        position = self.positions.get(
            symbol
        )

        if position is None:
            return

        close = safe_float(
            getattr(
                candle,
                "close",
                0.0,
            )
        )

        if close <= 0:
            return

        bid = (
            self._latest_bid
            if self._latest_bid > 0
            else close
        )

        ask = (
            self._latest_ask
            if self._latest_ask > 0
            else close
        )

        if position.side == "BUY":

            mark_price = bid

            position.highest_price = max(
                position.highest_price,
                close,
            )

            position.unrealized_pnl = (
                mark_price
                - position.entry_price
            ) * position.quantity

            position.max_unrealized_pnl = max(
                position.max_unrealized_pnl,
                position.unrealized_pnl,
            )

            position.min_unrealized_pnl = min(
                position.min_unrealized_pnl,
                position.unrealized_pnl,
            )

            if (
                not position.trailing_activated
                and position.initial_risk > 0
                and (
                    close
                    - position.entry_price
                )
                >= position.initial_risk
            ):

                position.trailing_activated = True

            if position.trailing_activated:

                trail_distance = max(
                    position.initial_risk
                    * 0.75,
                    position.entry_price
                    * 0.0005,
                )

                new_trailing = (
                    position.highest_price
                    - trail_distance
                )

                position.trailing_stop = max(
                    position.trailing_stop,
                    new_trailing,
                )

            if (
                mark_price
                <= position.stop_loss
            ):

                self._close(
                    symbol,
                    bid,
                    "STOP_LOSS",
                )

                return

            if (
                mark_price
                >= position.take_profit
            ):

                self._close(
                    symbol,
                    bid,
                    "TAKE_PROFIT",
                )

                return

            if (
                position.trailing_activated
                and position.trailing_stop > 0
                and mark_price
                <= position.trailing_stop
            ):

                self._close(
                    symbol,
                    bid,
                    "TRAILING_STOP",
                )

                return

        else:

            mark_price = ask

            if position.lowest_price <= 0:

                position.lowest_price = close

            position.lowest_price = min(
                position.lowest_price,
                close,
            )

            position.unrealized_pnl = (
                position.entry_price
                - mark_price
            ) * position.quantity

            position.max_unrealized_pnl = max(
                position.max_unrealized_pnl,
                position.unrealized_pnl,
            )

            position.min_unrealized_pnl = min(
                position.min_unrealized_pnl,
                position.unrealized_pnl,
            )

            if (
                not position.trailing_activated
                and position.initial_risk > 0
                and (
                    position.entry_price
                    - close
                )
                >= position.initial_risk
            ):

                position.trailing_activated = True

            if position.trailing_activated:

                trail_distance = max(
                    position.initial_risk
                    * 0.75,
                    position.entry_price
                    * 0.0005,
                )

                new_trailing = (
                    position.lowest_price
                    + trail_distance
                )

                if (
                    position.trailing_stop
                    <= 0
                ):

                    position.trailing_stop = (
                        new_trailing
                    )

                else:

                    position.trailing_stop = min(
                        position.trailing_stop,
                        new_trailing,
                    )

            if (
                mark_price
                >= position.stop_loss
            ):

                self._close(
                    symbol,
                    ask,
                    "STOP_LOSS",
                )

                return

            if (
                mark_price
                <= position.take_profit
            ):

                self._close(
                    symbol,
                    ask,
                    "TAKE_PROFIT",
                )

                return

            if (
                position.trailing_activated
                and position.trailing_stop > 0
                and mark_price
                >= position.trailing_stop
            ):

                self._close(
                    symbol,
                    ask,
                    "TRAILING_STOP",
                )

    # --------------------------------------------------------

    def process_signal(
        self,
        signal,
        features,
    ):

        symbol = features.symbol

        if (
            signal.action
            == SignalAction.HOLD
        ):

            return (
                False,
                "SIGNAL_HOLD",
            )

        allowed, reason = (
            self.risk_engine.check_signal(
                signal,
                features,
            )
        )

        if not allowed:

            self.logger.info(
                "PAPER ENTRY REJECTED | "
                "symbol=%s | action=%s | "
                "confidence=%.4f | reason=%s",
                symbol,
                signal.action.value,
                signal.confidence,
                reason,
            )

            return (
                False,
                reason,
            )

        now = time.time()

        last_exit = self.last_exit_time.get(
            symbol,
            0.0,
        )

        if (
            last_exit > 0
            and (
                now - last_exit
            ) < self.cooldown_seconds
        ):

            return (
                False,
                "COOLDOWN_ACTIVE",
            )

        side = (
            "BUY"
            if signal.action
            == SignalAction.BUY
            else "SELL"
        )

        existing = self.positions.get(
            symbol
        )

        if existing:

            if existing.side == side:

                return (
                    False,
                    "POSITION_ALREADY_OPEN",
                )

            exit_price = (
                features.bid
                if existing.side == "BUY"
                else features.ask
            )

            self._close(
                symbol,
                exit_price,
                "OPPOSITE_SIGNAL",
            )

            return (
                False,
                "OPPOSITE_SIGNAL_CLOSED",
            )

        volatility = max(
            features.volatility,
            0.0005,
        )

        stop_pct = clamp(
            volatility * 2.0,
            0.0010,
            0.0030,
        )

        target_pct = clamp(
            stop_pct * 2.5,
            0.0025,
            0.0075,
        )

        entry_price = (
            features.ask
            if side == "BUY"
            else features.bid
        )

        if entry_price <= 0:

            return (
                False,
                "INVALID_EXECUTION_PRICE",
            )

        stop_distance = (
            entry_price
            * stop_pct
        )

        quantity, quantity_reason = (
            self.risk_engine.calculate_quantity(
                entry_price,
                stop_distance,
            )
        )

        if quantity <= 0:

            return (
                False,
                quantity_reason,
            )

        if side == "BUY":

            stop = (
                entry_price
                * (1 - stop_pct)
            )

            target = (
                entry_price
                * (1 + target_pct)
            )

        else:

            stop = (
                entry_price
                * (1 + stop_pct)
            )

            target = (
                entry_price
                * (1 - target_pct)
            )

        strategies = (
            signal.votes
            or ["AGGREGATED"]
        )

        position = PaperPosition(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_timestamp=signal.timestamp,
            entry_wall_time=time.time(),
            stop_loss=stop,
            take_profit=target,
            trailing_stop=0.0,
            highest_price=entry_price,
            lowest_price=entry_price,
            strategy_names=strategies,
            entry_confidence=signal.confidence,
            entry_notional=(
                quantity * entry_price
            ),
            initial_risk=stop_distance,
            learning_id=getattr(signal, "learning_id", ""),
            entry_signal_action=signal.action.value,
        )

        self.positions[symbol] = position

        self.entries += 1

        self.total_volume += quantity

        self.total_notional += (
            position.entry_notional
        )

        self.risk_engine.record_trade()

        self.logger.info(
            "PAPER ENTRY | "
            "symbol=%s | side=%s | "
            "price=%.2f | quantity=%.8f | "
            "notional=%.8f | "
            "confidence=%.4f | "
            "stop=%.2f | target=%.2f | "
            "risk_distance=%.2f | "
            "regime=%s | strategies=%s",
            symbol,
            side,
            entry_price,
            quantity,
            position.entry_notional,
            signal.confidence,
            stop,
            target,
            stop_distance,
            features.regime,
            strategies,
        )

        return (
            True,
            "PAPER_ENTRY_ACCEPTED",
        )

    # --------------------------------------------------------

    def _close(
        self,
        symbol,
        exit_price,
        reason,
    ):

        position = self.positions.get(
            symbol
        )

        if not position:
            return

        exit_price = safe_float(
            exit_price
        )

        if exit_price <= 0:
            return

        if position.side == "BUY":

            pnl = (
                exit_price
                - position.entry_price
            ) * position.quantity

        else:

            pnl = (
                position.entry_price
                - exit_price
            ) * position.quantity

        holding_time = max(
            time.time()
            - (
                position.entry_wall_time
                if position.entry_wall_time > 0
                else time.time()
            ),
            0.0,
        )

        self.holding_times.append(
            holding_time
        )

        self.exit_reasons[
            reason
        ] = (
            self.exit_reasons.get(
                reason,
                0,
            )
            + 1
        )

        self.exits += 1
        self.completed_trades += 1

        self.realized_pnl += pnl

        self.trade_pnls.append(
            pnl
        )

        if pnl > 0:

            self.wins += 1

            self.gross_profit += pnl

            self.largest_win = max(
                self.largest_win,
                pnl,
            )

        elif pnl < 0:

            self.losses += 1

            self.gross_loss += abs(
                pnl
            )

            if (
                self.largest_loss == 0
                or pnl
                < self.largest_loss
            ):

                self.largest_loss = pnl

        else:

            self.breakeven += 1

        self.risk_engine.record_pnl(
            pnl
        )

        self.last_exit_time[
            symbol
        ] = time.time()

        if self.trade_result_handler:

            try:

                self.trade_result_handler(
                    list(position.strategy_names),
                    pnl,
                    {
                        "learning_id": position.learning_id,
                        "symbol": position.symbol,
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "entry_notional": position.entry_notional,
                        "exit_price": exit_price,
                        "entry_timestamp": position.entry_timestamp,
                        "exit_timestamp": time.time(),
                        "entry_signal_action": position.entry_signal_action,
                    },
                )

            except Exception:

                self.logger.exception(
                    "Trade-result callback failed."
                )

        self.logger.info(
            "PAPER EXIT | "
            "symbol=%s | side=%s | "
            "entry=%.2f | exit=%.2f | "
            "quantity=%.8f | "
            "pnl=%+.8f | "
            "holding=%.2fs | "
            "reason=%s",
            symbol,
            position.side,
            position.entry_price,
            exit_price,
            position.quantity,
            pnl,
            holding_time,
            reason,
        )

        del self.positions[symbol]

    # --------------------------------------------------------

    def snapshot_state(self):
        positions = {}
        for symbol, position in self.positions.items():
            positions[symbol] = {
                "symbol": position.symbol,
                "side": position.side,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "entry_timestamp": position.entry_timestamp,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "trailing_stop": position.trailing_stop,
                "entry_wall_time": position.entry_wall_time,
                "highest_price": position.highest_price,
                "lowest_price": position.lowest_price,
                "strategy_names": list(position.strategy_names),
                "entry_confidence": position.entry_confidence,
                "entry_notional": position.entry_notional,
                "unrealized_pnl": position.unrealized_pnl,
                "max_unrealized_pnl": position.max_unrealized_pnl,
                "min_unrealized_pnl": position.min_unrealized_pnl,
                "initial_risk": position.initial_risk,
                "trailing_activated": position.trailing_activated,
                "learning_id": position.learning_id,
                "entry_signal_action": position.entry_signal_action,
            }
        return {
            "entries": self.entries,
            "exits": self.exits,
            "completed_trades": self.completed_trades,
            "wins": self.wins,
            "losses": self.losses,
            "breakeven": self.breakeven,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "realized_pnl": self.realized_pnl,
            "trade_pnls": list(self.trade_pnls[-1000:]),
            "holding_times": list(self.holding_times[-1000:]),
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "total_volume": self.total_volume,
            "total_notional": self.total_notional,
            "exit_reasons": dict(self.exit_reasons),
            "last_exit_time": dict(self.last_exit_time),
            "positions": positions,
        }

    def restore_state(self, state):
        if not isinstance(state, dict):
            return
        for name in (
            "entries", "exits", "completed_trades", "wins", "losses", "breakeven",
            "gross_profit", "gross_loss", "realized_pnl", "largest_win",
            "largest_loss", "total_volume", "total_notional",
        ):
            if name in state:
                setattr(self, name, safe_float(state[name]) if name not in {"entries","exits","completed_trades","wins","losses","breakeven"} else int(safe_float(state[name])))
        self.trade_pnls = [safe_float(x) for x in state.get("trade_pnls", [])][-1000:]
        self.holding_times = [safe_float(x) for x in state.get("holding_times", [])][-1000:]
        self.exit_reasons = {str(k): int(safe_float(v)) for k, v in state.get("exit_reasons", {}).items()}
        self.last_exit_time = {str(k): safe_float(v) for k, v in state.get("last_exit_time", {}).items()}
        self.positions.clear()
        for symbol, raw in state.get("positions", {}).items():
            try:
                self.positions[symbol] = PaperPosition(
                    symbol=str(raw["symbol"]),
                    side=str(raw["side"]),
                    quantity=safe_float(raw["quantity"]),
                    entry_price=safe_float(raw["entry_price"]),
                    entry_timestamp=safe_float(raw["entry_timestamp"]),
                    stop_loss=safe_float(raw["stop_loss"]),
                    take_profit=safe_float(raw["take_profit"]),
                    trailing_stop=safe_float(raw.get("trailing_stop")),
                    entry_wall_time=safe_float(raw.get("entry_wall_time")),
                    highest_price=safe_float(raw.get("highest_price")),
                    lowest_price=safe_float(raw.get("lowest_price")),
                    strategy_names=list(raw.get("strategy_names", [])),
                    entry_confidence=safe_float(raw.get("entry_confidence")),
                    entry_notional=safe_float(raw.get("entry_notional")),
                    unrealized_pnl=safe_float(raw.get("unrealized_pnl")),
                    max_unrealized_pnl=safe_float(raw.get("max_unrealized_pnl")),
                    min_unrealized_pnl=safe_float(raw.get("min_unrealized_pnl")),
                    initial_risk=safe_float(raw.get("initial_risk")),
                    trailing_activated=bool(raw.get("trailing_activated", False)),
                    learning_id=str(raw.get("learning_id", "")),
                    entry_signal_action=str(raw.get("entry_signal_action", "HOLD")),
                )
            except Exception:
                self.logger.exception("Unable to restore paper position: %s", symbol)

    def get_stats(self):

        total = (
            self.wins
            + self.losses
            + self.breakeven
        )

        return {
            "entries": self.entries,
            "exits": self.exits,
            "open_positions": len(
                self.positions
            ),

            "completed_trades": total,

            "wins": self.wins,
            "losses": self.losses,
            "breakeven": self.breakeven,

            "win_rate": (
                self.wins / total
                if total
                else 0.0
            ),

            "loss_rate": (
                self.losses / total
                if total
                else 0.0
            ),

            "gross_profit": (
                self.gross_profit
            ),

            "gross_loss": (
                self.gross_loss
            ),

            "realized_pnl": (
                self.realized_pnl
            ),

            "net_pnl": (
                self.realized_pnl
            ),

            "average_win": (
                self.gross_profit
                / self.wins
                if self.wins
                else 0.0
            ),

            "average_loss": (
                self.gross_loss
                / self.losses
                if self.losses
                else 0.0
            ),

            "profit_factor": (
                self.gross_profit
                / self.gross_loss
                if self.gross_loss > 0
                else 0.0
            ),

            "average_holding_seconds": (
                sum(
                    self.holding_times
                )
                / len(
                    self.holding_times
                )
                if self.holding_times
                else 0.0
            ),

            "largest_win": (
                self.largest_win
            ),

            "largest_loss": (
                self.largest_loss
            ),

            "total_volume": (
                self.total_volume
            ),

            "total_notional": (
                self.total_notional
            ),

            "exit_reasons": dict(
                self.exit_reasons
            ),

            "positions": {
                symbol: {
                    "side": position.side,
                    "quantity": position.quantity,
                    "entry_price": (
                        position.entry_price
                    ),
                    "stop_loss": (
                        position.stop_loss
                    ),
                    "take_profit": (
                        position.take_profit
                    ),
                    "trailing_stop": (
                        position.trailing_stop
                    ),
                    "trailing_activated": (
                        position.trailing_activated
                    ),
                    "unrealized_pnl": (
                        position.unrealized_pnl
                    ),
                    "max_unrealized_pnl": (
                        position.max_unrealized_pnl
                    ),
                    "min_unrealized_pnl": (
                        position.min_unrealized_pnl
                    ),
                    "entry_confidence": (
                        position.entry_confidence
                    ),
                    "strategies": (
                        position.strategy_names
                    ),
                    "holding_seconds": max(
                        time.time()
                        - (
                            position.entry_wall_time
                            if position.entry_wall_time > 0
                            else time.time()
                        ),
                        0.0,
                    ),
                }
                for (
                    symbol,
                    position,
                )
                in self.positions.items()
            },
        }


# ============================================================
# LEARNING ENGINE
# ============================================================


class LearningEngine:
    """Integrated online + offline ML learning layer.

    Responsibilities:
    - Store live signal samples and pending trade labels.
    - Perform lightweight online logistic updates after completed paper trades.
    - Optionally load the historical RandomForest model used by the backtest.
    - Keep online and offline models separate so one cannot silently overwrite
      the other.
    - Expose defensive status information so monitoring/shutdown can never fail
      because an optional ML artifact is missing.
    """

    FEATURE_NAMES = (
        "return_1", "return_5", "return_15", "return_60",
        "rsi", "volatility", "trend", "momentum", "mean_reversion",
        "orderbook_imbalance", "trade_flow", "trade_flow_acceleration",
        "spread_bps", "multi_timeframe_score", "signal_confidence",
        "signal_direction",
    )

    DEFAULT_ONLINE_MODEL_PATH = "data/learning/phase9_online_model.json"
    DEFAULT_OFFLINE_MODEL_PATH = "data/models/market_direction_model.joblib"

    def __init__(
        self,
        max_samples=10000,
        model_path=DEFAULT_ONLINE_MODEL_PATH,
        offline_model_path=DEFAULT_OFFLINE_MODEL_PATH,
    ):
        self.logger = logging.getLogger("LearningEngine")

        self.samples = deque(maxlen=max_samples)
        self.pending = {}

        self.enabled = True

        # Online model state.
        self.model_loaded = False
        self.model_name = "NONE"
        self.predictions = 0
        self.correct_predictions = 0
        self.labeled_outcomes = 0
        self.learning_rate = 0.08
        self.weights = [0.0] * (len(self.FEATURE_NAMES) + 1)

        self.model_path = str(model_path)

        # Historical/offline supervised model used as an optional
        # confirmation layer by TradingEngine.
        self.offline_model_path = str(offline_model_path)
        self.offline_model = None
        self.offline_model_load_error = ""

        self._load_model()
        self._load_offline_model()

    def _vector(self, features, signal):
        direction = (
            1.0 if signal.action == SignalAction.BUY
            else -1.0 if signal.action == SignalAction.SELL
            else 0.0
        )
        return [
            clamp(safe_float(features.return_1) * 100.0, -5.0, 5.0),
            clamp(safe_float(features.return_5) * 100.0, -10.0, 10.0),
            clamp(safe_float(features.return_15) * 100.0, -15.0, 15.0),
            clamp(safe_float(features.return_60) * 100.0, -20.0, 20.0),
            clamp((safe_float(features.rsi) - 50.0) / 50.0, -1.0, 1.0),
            clamp(safe_float(features.volatility) * 100.0, 0.0, 5.0),
            clamp(safe_float(features.trend), -5.0, 5.0),
            clamp(safe_float(features.momentum), -5.0, 5.0),
            clamp(safe_float(features.mean_reversion), -5.0, 5.0),
            clamp(safe_float(features.orderbook_imbalance), -1.0, 1.0),
            clamp(safe_float(features.trade_flow), -1.0, 1.0),
            clamp(safe_float(features.trade_flow_acceleration), -1.0, 1.0),
            clamp(safe_float(features.spread_bps) / 10.0, 0.0, 5.0),
            clamp(safe_float(features.multi_timeframe_score), -1.0, 1.0),
            clamp(safe_float(signal.confidence), 0.0, 1.0),
            direction,
        ]

    @staticmethod
    def _sigmoid(z):
        z = clamp(z, -40.0, 40.0)
        return 1.0 / (1.0 + math.exp(-z))

    def _predict_probability(self, vector):
        z = self.weights[0]
        for w, x in zip(self.weights[1:], vector):
            z += w * x
        return self._sigmoid(z)

    def record(self, features, signal):
        if not self.enabled:
            return ""

        sample_id = hashlib.sha1(
            f"{features.symbol}|{features.timestamp}|{signal.action.value}|"
            f"{signal.confidence:.8f}".encode()
        ).hexdigest()[:20]

        vector = self._vector(features, signal)
        probability = (
            self._predict_probability(vector)
            if self.model_loaded
            else 0.5
        )

        sample = {
            "id": sample_id,
            "timestamp": features.timestamp,
            "symbol": features.symbol,
            "price": features.price,
            "signal": signal.action.value,
            "confidence": signal.confidence,
            "strategy_names": list(signal.votes),
            "vector": vector,
            "predicted_probability": probability,
            "outcome": None,
            "pnl": None,
        }

        self.samples.append(sample)

        if signal.action != SignalAction.HOLD:
            self.pending[sample_id] = sample

        return sample_id

    def record_outcome(self, learning_id, pnl):
        if not learning_id:
            return

        sample = self.pending.pop(learning_id, None)
        if sample is None:
            return

        pnl = safe_float(pnl)
        label = 1.0 if pnl > 0 else 0.0
        probability = safe_float(
            sample.get("predicted_probability"),
            0.5,
        )

        self.labeled_outcomes += 1

        if (probability >= 0.5) == bool(label):
            self.correct_predictions += 1

        self.predictions += 1

        error = label - probability
        vector = sample["vector"]

        self.weights[0] += self.learning_rate * error

        for i, x in enumerate(vector, start=1):
            self.weights[i] += (
                self.learning_rate * error * x
            )

        sample["outcome"] = (
            "WIN" if label else "LOSS"
        )
        sample["pnl"] = pnl

        self.model_loaded = (
            self.labeled_outcomes >= 5
        )
        self.model_name = (
            "ONLINE_LOGISTIC_V1"
            if self.model_loaded
            else "WARMING_UP"
        )

        self._save_model()

        self.logger.info(
            "ML LEARNING UPDATE | id=%s | outcome=%s | "
            "pnl=%+.8f | labeled=%d | model=%s",
            learning_id,
            sample["outcome"],
            pnl,
            self.labeled_outcomes,
            self.model_name,
        )

    def predict_confidence(self, features, signal):
        confidence = signal.confidence

        if (
            self.model_loaded
            and signal.action != SignalAction.HOLD
        ):
            probability = self._predict_probability(
                self._vector(features, signal)
            )

            confidence = clamp(
                confidence * 0.75
                + probability * 0.25,
                0.0,
                1.0,
            )

            self.logger.debug(
                "Online ML prediction | action=%s | probability=%.4f",
                signal.action.value,
                probability,
            )

        else:
            if features.signal_quality > 0.75:
                confidence += 0.04

            if abs(features.multi_timeframe_score) > 0.40:
                confidence += 0.03

            if features.data_quality != "GOOD":
                confidence -= 0.10

        return clamp(
            confidence,
            0.0,
            1.0,
        )

    def _save_model(self):
        try:
            directory = os.path.dirname(
                self.model_path
            )

            if directory:
                os.makedirs(
                    directory,
                    exist_ok=True,
                )

            tmp = (
                self.model_path
                + ".tmp"
            )

            with open(
                tmp,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "model": self.model_name,
                        "labeled_outcomes": self.labeled_outcomes,
                        "weights": self.weights,
                        "learning_rate": self.learning_rate,
                        "feature_names": list(self.FEATURE_NAMES),
                        "version": 2,
                    },
                    f,
                )

            os.replace(
                tmp,
                self.model_path,
            )

        except Exception:
            self.logger.exception(
                "Unable to save ML model."
            )

    def _load_model(self):
        try:
            if not os.path.exists(
                self.model_path
            ):
                return

            with open(
                self.model_path,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            weights = data.get(
                "weights"
            )

            if (
                isinstance(weights, list)
                and len(weights)
                == len(self.weights)
            ):
                self.weights = [
                    safe_float(x)
                    for x in weights
                ]

                self.labeled_outcomes = int(
                    safe_float(
                        data.get(
                            "labeled_outcomes"
                        ),
                        0,
                    )
                )

                self.learning_rate = clamp(
                    safe_float(
                        data.get(
                            "learning_rate"
                        ),
                        self.learning_rate,
                    ),
                    0.0001,
                    1.0,
                )

                self.model_loaded = (
                    self.labeled_outcomes >= 5
                )

                self.model_name = (
                    "ONLINE_LOGISTIC_V1"
                    if self.model_loaded
                    else "WARMING_UP"
                )

                self.logger.info(
                    "Online ML model loaded | "
                    "model=%s | labeled=%d",
                    self.model_name,
                    self.labeled_outcomes,
                )

        except Exception:
            self.logger.exception(
                "Unable to load ML model; "
                "starting fresh."
            )

    def _load_offline_model(self):
        """Load the historical RandomForest model if it exists.

        Missing offline models are normal during early development. They must
        not prevent the live/paper engine from starting or shutting down.
        """

        self.offline_model = None
        self.offline_model_load_error = ""

        path = os.path.expanduser(
            self.offline_model_path
        )

        if not os.path.exists(path):
            self.logger.info(
                "Offline ML model not present | path=%s",
                self.offline_model_path,
            )
            return

        try:
            from app.ml.model import MLModel

            self.offline_model = MLModel.load(
                path
            )

            self.logger.info(
                "Offline ML model loaded | path=%s | "
                "features=%d",
                self.offline_model_path,
                len(
                    getattr(
                        self.offline_model,
                        "feature_names",
                        [],
                    )
                ),
            )

        except Exception as exc:
            self.offline_model = None
            self.offline_model_load_error = (
                f"{type(exc).__name__}: {exc}"
            )

            self.logger.warning(
                "Offline ML model unavailable | path=%s | "
                "reason=%s",
                self.offline_model_path,
                self.offline_model_load_error,
            )

    def reload_offline_model(self):
        """Reload the historical model after a new backtest/training run."""

        self._load_offline_model()
        return self.offline_model is not None

    def offline_model_available(self):
        return self.offline_model is not None

    def get_stats(self):
        """Return a shutdown-safe ML status snapshot.

        This method intentionally contains no direct attribute access to
        optional objects that may not have been initialized.
        """

        return {
            "enabled": bool(self.enabled),
            "model_loaded": bool(self.model_loaded),
            "model_name": str(self.model_name),
            "samples": len(self.samples),
            "pending": len(self.pending),
            "predictions": int(self.predictions),
            "correct_predictions": int(
                self.correct_predictions
            ),
            "prediction_accuracy": (
                self.correct_predictions
                / self.predictions
                if self.predictions
                else 0.0
            ),
            "labeled_outcomes": int(
                self.labeled_outcomes
            ),
            "model_path": self.model_path,
            "offline_model_path": self.offline_model_path,
            "offline_model_loaded": (
                self.offline_model is not None
            ),
            "offline_model_error": (
                self.offline_model_load_error
            ),
            "feature_count": len(
                self.FEATURE_NAMES
            ),
        }


# ============================================================
# PORTFOLIO ANALYTICS
# ============================================================


class PortfolioAnalytics:

    def __init__(self):

        self.realized_pnl = 0.0

        self.peak_equity = 0.0

        self.max_drawdown = 0.0

        self.current_drawdown = 0.0

        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0

    # --------------------------------------------------------

    def record_trade(self, pnl, entry_notional=0.0):
        pnl = safe_float(pnl)
        self.trade_count += 1

        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(
                self.max_consecutive_wins,
                self.consecutive_wins,
            )
        elif pnl < 0:
            self.losing_trades += 1
            self.total_loss += abs(pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(
                self.max_consecutive_losses,
                self.consecutive_losses,
            )

    # --------------------------------------------------------

    def update(
        self,
        equity,
        realized_pnl,
    ):

        equity = safe_float(
            equity
        )

        realized_pnl = safe_float(
            realized_pnl
        )

        self.realized_pnl = (
            realized_pnl
        )

        self.peak_equity = max(
            self.peak_equity,
            equity,
        )

        if self.peak_equity > 0:

            drawdown = (
                self.peak_equity
                - equity
            ) / self.peak_equity

        else:

            drawdown = 0.0

        self.current_drawdown = max(
            drawdown,
            0.0,
        )

        self.max_drawdown = max(
            self.max_drawdown,
            self.current_drawdown,
        )

    # --------------------------------------------------------

    def get_stats(self):

        total = self.trade_count
        average_trade = (
            self.realized_pnl / total
            if total else 0.0
        )
        expectancy = (
            (
                self.total_profit
                - self.total_loss
            ) / total
            if total else 0.0
        )
        profit_factor = (
            self.total_profit / self.total_loss
            if self.total_loss > 0 else 0.0
        )

        return {
            "realized_pnl": (
                self.realized_pnl
            ),
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": (
                self.winning_trades / total
                if total else 0.0
            ),
            "average_trade": average_trade,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,

            "peak_equity": (
                self.peak_equity
            ),

            "current_drawdown": (
                self.current_drawdown
            ),

            "max_drawdown": (
                self.max_drawdown
            ),
        }


# ============================================================
# TRADING ENGINE
# ============================================================


class TradingEngine:

    def __init__(
        self,
        settings: Settings,
    ):

        self.settings = settings

        self.logger = logging.getLogger(
            "TradingEngine"
        )

        self.state = (
            EngineState.CREATED
        )

        self.shutting_down = False

        self.initialized = False

        self.delta = DeltaClient(
            settings
        )

        # ----------------------------------------------------
        # Existing optional market state manager.
        # ----------------------------------------------------

        try:

            from app.market_data.state_manager import (
                MarketDataStateManager
            )

            self.market_state = (
                MarketDataStateManager()
            )

        except Exception:

            try:

                from app.market_data.state_manager import (
                    MarketStateManager
                )

                self.market_state = (
                    MarketStateManager()
                )

            except Exception:

                self.market_state = None

        self.feature_engine = (
            FeatureEngine(
                settings.stale_data_timeout_seconds
            )
        )

        self.strategy_engine = (
            StrategyEngine()
        )

        self.strategy_validator = (
            StrategyValidator()
        )

        self.signal_performance_tracker = SignalPerformanceTracker()
        self.persistence = EnginePersistence()
        self.recovery_state = self.persistence.load()

        self.learning_engine = (
            LearningEngine()
        )

        # Offline ML model trained from the historical pipeline is an
        # additional confirmation layer. It never places orders itself.
        self.ml_predictor = MLPredictor(
            "data/models/market_direction_model.joblib"
        )

        self.portfolio = (
            PortfolioAnalytics()
        )

        self.risk_engine = None

        self.paper_execution = None

        self.candle_aggregator = None

        self.collector = None

        self.collector_thread = None

        self.watchdog = Watchdog(
            settings.heartbeat_interval_seconds
            * 2,
            self.logger,
        )

        self.latest_features = None
        self.latest_signal = None

        # Phase 9 requires actual candle history.  The candle
        # aggregator already produces real trade-derived 1-second
        # candles, so retain those candles for strategy evaluation.
        self.strategy_candle_history = {}
        self.strategy_candle_maxlen = 5000

        self.last_rejection_reason = "NONE"

        self.rejection_counts = {}

        # ----------------------------------------------------
        # HARD SAFETY:
        # LIVE ORDERS ARE NOT ENABLED.
        # ----------------------------------------------------

        self.order_execution_enabled = False

        self.engine_cycle_count = 0

        self.last_event_time = 0.0

        self.last_candle_time = 0.0

        self.last_signal_timestamp = 0.0

        self.kill_switch = False

        self.kill_switch_reason = "NONE"

        self.max_portfolio_drawdown = 0.10

        self._lock = threading.RLock()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize(self):

        if self.initialized:
            return

        self.state = (
            EngineState.INITIALIZING
        )

        self.logger.info(
            "Integrated Trading Engine "
            "initialization started."
        )

        try:

            self.settings.validate_startup()

            self.state = (
                EngineState.CONNECTING
            )

            self.delta.test_rest_connection()

            balances = (
                self.delta
                .test_private_rest_connection()
            )

            equity = (
                self._extract_equity(
                    balances
                )
            )

            self.logger.info(
                "Account equity initialized | "
                "equity=%.8f",
                equity,
            )

            if equity <= 0:

                raise RuntimeError(
                    "Unable to initialize "
                    "positive account equity."
                )

            self.risk_engine = (
                RiskEngine(
                    account_equity=equity,
                    max_risk_per_trade=(
                        self.settings
                        .max_risk_per_trade
                    ),
                    max_daily_loss=(
                        self.settings
                        .max_daily_loss
                    ),
                    max_trades_per_day=(
                        self.settings
                        .max_trades_per_day
                    ),
                    max_leverage=(
                        self.settings
                        .max_leverage
                    ),
                )
            )

            self.paper_execution = (
                PaperExecutionEngine(
                    risk_engine=(
                        self.risk_engine
                    ),
                    trade_result_handler=(
                        self._record_trade_result
                    ),
                )
            )

            if self.recovery_state.get("paper_execution"):
                self.paper_execution.restore_state(self.recovery_state["paper_execution"])

            restored_weights = self.recovery_state.get("strategy_weights")
            if isinstance(restored_weights, dict):
                for name, weight in restored_weights.items():
                    if name in self.strategy_engine.strategy_weights:
                        self.strategy_engine.strategy_weights[name] = clamp(safe_float(weight, 1.0), 0.50, 1.50)

            self.delta.load_instruments()

            self.candle_aggregator = (
                CandleAggregator(
                    intervals_seconds=[
                        1,
                        5,
                        15,
                        60,
                    ],
                    output_directory=(
                        "data/market_data"
                    ),
                    candle_handler=(
                        self._candle_event_handler
                    ),
                )
            )

            self.collector = (
                MarketDataCollector(
                    websocket_url=(
                        self.settings
                        .delta_public_ws_url
                    ),
                    event_handler=(
                        self._collector_event_handler
                    ),
                )
            )

            self.initialized = True

            self.state = (
                EngineState.RUNNING
            )

            self.logger.info(
                "Integrated Trading Engine "
                "initialized."
            )

            self.logger.info(
                "Execution mode | "
                "PAPER=true | "
                "LIVE=false | "
                "ORDER_EXECUTION_ENABLED=%s",
                self.order_execution_enabled,
            )

        except Exception:

            self.state = (
                EngineState.SAFE_MODE
            )

            self.logger.exception(
                "Trading engine initialization failed."
            )

            raise

    # ========================================================
    # EQUITY EXTRACTION
    # ========================================================

    def _extract_equity(
        self,
        balances,
    ) -> float:

        if not isinstance(
            balances,
            dict,
        ):
            return 0.0

        result = balances.get(
            "result"
        )

        if isinstance(
            result,
            dict,
        ):

            for key in (
                "equity",
                "total_equity",
                "wallet_equity",
                "balance",
                "available_balance",
            ):

                value = safe_float(
                    result.get(key),
                    -1,
                )

                if value >= 0:
                    return value

            balance_list = (
                result.get(
                    "balances"
                )
            )

            if isinstance(
                balance_list,
                list,
            ):

                for item in balance_list:

                    if not isinstance(
                        item,
                        dict,
                    ):
                        continue

                    asset = str(
                        item.get(
                            "asset_symbol",
                            item.get(
                                "symbol",
                                "",
                            ),
                        )
                    ).upper()

                    if asset in (
                        "USDT",
                        "USD",
                    ):

                        for key in (
                            "equity",
                            "balance",
                            "available_balance",
                            "wallet_balance",
                        ):

                            value = safe_float(
                                item.get(
                                    key
                                ),
                                -1,
                            )

                            if value >= 0:
                                return value

        elif isinstance(
            result,
            list,
        ):

            for item in result:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                asset = str(
                    item.get(
                        "asset_symbol",
                        item.get(
                            "symbol",
                            "",
                        ),
                    )
                ).upper()

                if asset in (
                    "USDT",
                    "USD",
                ):

                    for key in (
                        "equity",
                        "balance",
                        "available_balance",
                        "wallet_balance",
                    ):

                        value = safe_float(
                            item.get(
                                key
                            ),
                            -1,
                        )

                        if value >= 0:
                            return value

        self.logger.warning(
            "Unable to extract account equity."
        )

        return 0.0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    def run(self):

        if not self.initialized:
            self.initialize()

        if self.collector is None:

            raise RuntimeError(
                "Market-data collector "
                "is not initialized."
            )

        self.shutting_down = False

        self.watchdog.start()

        self.collector_thread = (
            self.collector.start()
        )

        self.delta.start_public_market_data()

        self.logger.info(
            "Trading engine running | "
            "Phase 1-14 architecture active."
        )

        try:

            while not self.shutting_down:

                self.watchdog.heartbeat()

                self._autonomous_cycle()

                time.sleep(1)

        except KeyboardInterrupt:

            self.logger.info(
                "Keyboard interrupt received."
            )

            self.shutdown()

        except Exception:

            self.logger.exception(
                "Trading engine main loop failed."
            )

            self.state = (
                EngineState.ERROR
            )

            self.shutdown()

    # ========================================================
    # AUTONOMOUS ORCHESTRATION
    # ========================================================

    def _autonomous_cycle(self):

        self.engine_cycle_count += 1

        if self.risk_engine is None:
            return

        if self.paper_execution:

            execution_stats = (
                self.paper_execution
                .get_stats()
            )

            unrealized = sum(
                safe_float(
                    position.get(
                        "unrealized_pnl"
                    )
                )
                for position
                in execution_stats[
                    "positions"
                ].values()
            )

            effective_equity = (
                self.risk_engine.account_equity
                + unrealized
            )

            self.portfolio.update(
                effective_equity,
                execution_stats[
                    "realized_pnl"
                ],
            )

        if (
            self.portfolio.current_drawdown
            >= self.max_portfolio_drawdown
        ):

            self.kill_switch = True

            self.kill_switch_reason = (
                "PORTFOLIO_DRAWDOWN_LIMIT"
            )

        if (
            self.risk_engine
            .daily_loss_limit_reached()
        ):

            self.kill_switch = True

            self.kill_switch_reason = (
                "DAILY_LOSS_LIMIT"
            )

        # No fresh market data means new entries are unsafe.
        # Existing paper positions are still managed by the
        # candle/book callbacks when data resumes.
        if self.last_event_time > 0:
            stale_for = time.time() - self.last_event_time
            if stale_for > self.settings.stale_data_timeout_seconds:
                self.kill_switch = True
                self.kill_switch_reason = "MARKET_DATA_STALE"
        elif self.engine_cycle_count > self.settings.stale_data_timeout_seconds:
            self.kill_switch = True
            self.kill_switch_reason = "MARKET_DATA_NOT_RECEIVED"

        # ----------------------------------------------------
        # Strategy adaptation.
        # ----------------------------------------------------

        if (
            self.engine_cycle_count
            % 30
            == 0
        ):

            performance = (
                self.strategy_validator
                .get_stats()
            )

            self.strategy_engine.update_strategy_weights(
                performance
            )

            self.logger.info(
                "AUTONOMOUS STATUS | "
                "cycle=%d | "
                "kill_switch=%s | "
                "reason=%s | "
                "drawdown=%.6f | "
                "weights=%s",
                self.engine_cycle_count,
                self.kill_switch,
                self.kill_switch_reason,
                self.portfolio.current_drawdown,
                self.strategy_engine.strategy_weights,
            )

    # ========================================================
    # MARKET DATA CALLBACK
    # ========================================================

    def _collector_event_handler(
        self,
        event,
    ):

        if self.shutting_down:
            return

        try:

            self.last_event_time = (
                time.time()
            )

            if self.market_state is not None:

                try:

                    self.market_state.update(
                        event
                    )

                except AttributeError:
                    pass

            self.feature_engine.process_event(
                event
            )

            payload = extract_payload(
                event
            )

            event_type = getattr(
                event,
                "event_type",
                None,
            )

            if event_type is None:

                event_type = payload.get(
                    "type"
                )

            if event_type == "ob_l1":

                bid = safe_float(
                    payload.get("bp")
                )

                ask = safe_float(
                    payload.get("ap")
                )

                if self.paper_execution:

                    self.paper_execution.update_book(
                        bid,
                        ask,
                    )

            if self.candle_aggregator:

                self.candle_aggregator.process_event(
                    event
                )

            self.watchdog.heartbeat()

        except Exception:

            self.logger.exception(
                "Market event processing failed."
            )

    # ========================================================
    # CANDLE CALLBACK
    # ========================================================

    def _candle_event_handler(
        self,
        candle,
    ):

        if self.shutting_down:
            return

        try:

            self.last_candle_time = (
                time.time()
            )

            timeframe = int(
                safe_float(
                    getattr(
                        candle,
                        "interval_seconds",
                        1,
                    ),
                    1,
                )
            )

            features = (
                self.feature_engine
                .process_candle(
                    candle
                )
            )

            # ------------------------------------------------
            # 5s / 15s / 60s candles update MTF history.
            # Only 1s candles create execution decisions.
            # ------------------------------------------------

            if timeframe != 1:
                return

            self.latest_features = (
                features
            )

            if self.paper_execution:

                self.paper_execution.process_candle(
                    candle
                )

            # ------------------------------------------------
            # PHASE 9 STRATEGY HISTORY
            # ------------------------------------------------
            # Use the actual candle object emitted by
            # CandleAggregator.  No synthetic prices or candles
            # are created here.
            symbol = str(
                getattr(
                    candle,
                    "symbol",
                    getattr(
                        features,
                        "symbol",
                        "UNKNOWN",
                    ),
                )
                or "UNKNOWN"
            )

            history = self.strategy_candle_history.setdefault(
                symbol,
                deque(
                    maxlen=self.strategy_candle_maxlen
                ),
            )

            candle_start = safe_float(
                getattr(
                    candle,
                    "start_timestamp",
                    0.0,
                )
            )

            # Avoid duplicate candle insertion if a callback is
            # delivered more than once.
            if history:

                previous = history[-1]

                previous_start = safe_float(
                    getattr(
                        previous,
                        "start_timestamp",
                        0.0,
                    )
                )

                if candle_start > 0 and candle_start == previous_start:
                    history[-1] = candle

                else:
                    history.append(candle)

            else:
                history.append(candle)

            strategy_candles = list(history)

            self.logger.info(
                "Strategy candle dispatch | symbol=%s | 1s_candles=%d | "
                "core_warmup=%s",
                symbol,
                len(strategy_candles),
                features.warmup_complete,
            )

            self.signal_performance_tracker.process_candle(candle)

            signal = (
                self.strategy_engine
                .evaluate(
                    features,
                    strategy_candles,
                )
            )

            self.signal_performance_tracker.record_signal(signal, candle)

            learning_id = self.learning_engine.record(
                features,
                signal,
            )
            signal.learning_id = learning_id

            model_confidence = (
                self.learning_engine
                .predict_confidence(
                    features,
                    signal,
                )
            )

            # ------------------------------------------------
            # OFFLINE ML CONFIRMATION
            # ------------------------------------------------
            # The production model predicts the short-horizon market
            # direction from the same live-compatible feature family
            # used during historical training. It is confirmation only.
            offline_ml = None
            if (
                signal.action != SignalAction.HOLD
                and self.ml_predictor.is_available()
            ):
                ml_features = {
                    "mid_close": features.price,
                    "mid_range": 0.0,
                    "mid_change": features.price - features.previous_price,
                    "mid_return": features.return_1,
                    "one_second_return": features.return_1,
                    "intrasecond_volatility": features.volatility,
                    "bid_close": features.bid,
                    "ask_close": features.ask,
                    "spread_mean": features.spread,
                    "spread_close": features.spread,
                    "imbalance_mean": features.orderbook_imbalance,
                    "trade_count": self.feature_engine.trade_count,
                    "trade_volume": features.buy_volume + features.sell_volume,
                    "trade_price_change": features.momentum,
                    "l1_update_count": 0.0,
                    "mid_change_count": 0.0,
                    "bid_change_count": 0.0,
                    "ask_change_count": 0.0,
                    "l1_gap_seconds": 0.0,
                }
                try:
                    offline_ml = self.ml_predictor.predict(ml_features)
                    predicted = int(offline_ml["prediction"])
                    action_direction = (
                        1 if signal.action == SignalAction.BUY
                        else -1 if signal.action == SignalAction.SELL
                        else 0
                    )
                    if predicted == action_direction:
                        model_confidence = clamp(
                            model_confidence * 0.70
                            + offline_ml["confidence"] * 0.30,
                            0.0,
                            1.0,
                        )
                    elif predicted != 0:
                        model_confidence = clamp(
                            model_confidence * 0.85,
                            0.0,
                            1.0,
                        )
                    else:
                        model_confidence = clamp(
                            model_confidence * 0.95,
                            0.0,
                            1.0,
                        )
                    self.logger.info(
                        "OFFLINE ML CONFIRMATION | action=%s | prediction=%s | "
                        "confidence=%.4f | label=%s",
                        signal.action.value,
                        predicted,
                        offline_ml["confidence"],
                        offline_ml["label"],
                    )
                except Exception:
                    self.logger.exception(
                        "Offline ML confirmation failed; continuing with online learner."
                    )

            signal.confidence = clamp(
                (
                    signal.confidence
                    * 0.80
                )
                + (
                    model_confidence
                    * 0.20
                ),
                0.0,
                1.0,
            )

            self.latest_signal = (
                signal
            )

            self.strategy_validator.record_signal(
                signal
            )

            self._log_signal_diagnostics(
                features,
                signal,
            )

            if self.paper_execution is None:
                return

            if self.kill_switch:

                self._record_rejection(
                    self.kill_switch_reason
                )

                return

            if (
                features.timestamp
                <= self.last_signal_timestamp
            ):

                return

            self.last_signal_timestamp = (
                features.timestamp
            )

            accepted, reason = (
                self.paper_execution
                .process_signal(
                    signal,
                    features,
                )
            )

            if not accepted:

                self._record_rejection(
                    reason
                )

            else:

                self.last_rejection_reason = (
                    "NONE"
                )

                self.logger.info(
                    "PAPER SIGNAL ACCEPTED | "
                    "symbol=%s | "
                    "action=%s | "
                    "confidence=%.4f | "
                    "score=%.4f | "
                    "regime=%s | "
                    "strategies=%s",
                    features.symbol,
                    signal.action.value,
                    signal.confidence,
                    signal.score,
                    features.regime,
                    signal.votes,
                )

            self.watchdog.heartbeat()

        except Exception:

            self.logger.exception(
                "Candle processing failed."
            )

    # ========================================================
    # SIGNAL DIAGNOSTICS
    # ========================================================

    def _log_signal_diagnostics(
        self,
        features,
        signal,
    ):

        samples = ", ".join(
            (
                f"{sample.strategy}:"
                f"{sample.action.value}:"
                f"{sample.confidence:.3f}"
            )
            for sample
            in signal.strategy_samples
        )

        self.logger.info(
            "SIGNAL DIAGNOSTICS | "
            "symbol=%s | "
            "action=%s | "
            "confidence=%.4f | "
            "score=%.4f | "
            "quality=%.4f | "
            "price=%.2f | "
            "spread_bps=%.2f | "
            "OB=%.4f | "
            "flow=%.4f | "
            "flow_accel=%.4f | "
            "MTF=%.4f | "
            "RSI=%.2f | "
            "volatility=%.6f | "
            "regime=%s | "
            "warmup=%s | "
            "votes=%s | "
            "samples=[%s]",
            features.symbol,
            signal.action.value,
            signal.confidence,
            signal.score,
            features.signal_quality,
            features.price,
            features.spread_bps,
            features.orderbook_imbalance,
            features.trade_flow,
            features.trade_flow_acceleration,
            features.multi_timeframe_score,
            features.rsi,
            features.volatility,
            features.regime,
            features.warmup_complete,
            signal.votes,
            samples,
        )

    # ========================================================

    def _record_rejection(
        self,
        reason,
    ):

        self.last_rejection_reason = (
            reason
        )

        self.rejection_counts[
            reason
        ] = (
            self.rejection_counts.get(
                reason,
                0,
            )
            + 1
        )

        self.logger.debug(
            "SIGNAL REJECTED | "
            "reason=%s",
            reason,
        )

    # ========================================================

    def _record_trade_result(
        self,
        strategies,
        pnl,
        metadata=None,
    ):

        # Attribute the completed trade to the exact aggregated signal
        # that opened the position. The strategy validator separately
        # tracks signal participation; it must not count one trade once
        # per vote.
        self.strategy_validator.record_trade_result(
            strategies,
            pnl,
        )

        entry_notional = 0.0
        if isinstance(metadata, dict):
            entry_notional = safe_float(
                metadata.get(
                    "entry_notional",
                    0.0,
                )
            )

        self.portfolio.record_trade(
            pnl,
            entry_notional,
        )

        self._persist_runtime_state()

        if isinstance(metadata, dict):
            self.learning_engine.record_outcome(
                metadata.get("learning_id", ""),
                pnl,
            )

    # ========================================================
    # PERSISTENCE
    # ========================================================

    def _persist_runtime_state(self):
        try:
            self.persistence.save({
                "saved_at": time.time(),
                "engine_state": self.state.value,
                "strategy_weights": dict(self.strategy_engine.strategy_weights),
                "paper_execution": self.paper_execution.snapshot_state() if self.paper_execution else {},
                "risk": self.risk_engine.get_stats() if self.risk_engine else {},
            })
        except Exception:
            self.logger.exception("Runtime state persistence failed.")

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):

        feature_status = None

        if self.latest_features:

            features = (
                self.latest_features
            )

            feature_status = {
                "symbol": features.symbol,

                "timeframe": (
                    features.timeframe
                ),

                "price": features.price,

                "previous_price": (
                    features.previous_price
                ),

                "return_1": (
                    features.return_1
                ),

                "return_5": (
                    features.return_5
                ),

                "return_15": (
                    features.return_15
                ),

                "return_60": (
                    features.return_60
                ),

                "rsi": features.rsi,

                "volatility": (
                    features.volatility
                ),

                "bid": features.bid,
                "ask": features.ask,

                "bid_size": (
                    features.bid_size
                ),

                "ask_size": (
                    features.ask_size
                ),

                "mark_price": features.mark_price,
                "funding_rate": features.funding_rate,
                "open_interest": features.open_interest,
                "open_interest_change": features.open_interest_change,

                "spread": (
                    features.spread
                ),

                "spread_bps": (
                    features.spread_bps
                ),

                "orderbook_imbalance": (
                    features.orderbook_imbalance
                ),

                "buy_volume": (
                    features.buy_volume
                ),

                "sell_volume": (
                    features.sell_volume
                ),

                "trade_flow": (
                    features.trade_flow
                ),

                "trade_flow_acceleration": (
                    features.trade_flow_acceleration
                ),

                "trend": (
                    features.trend
                ),

                "momentum": (
                    features.momentum
                ),

                "mean_reversion": (
                    features.mean_reversion
                ),

                "regime": (
                    features.regime
                ),

                "multi_timeframe_score": (
                    features.multi_timeframe_score
                ),

                "signal_quality": (
                    features.signal_quality
                ),

                "data_quality": (
                    features.data_quality
                ),

                "warmup_complete": (
                    features.warmup_complete
                ),
            }

        signal_status = None

        if self.latest_signal:

            signal = (
                self.latest_signal
            )

            signal_status = {
                "action": (
                    signal.action.value
                ),

                "confidence": (
                    signal.confidence
                ),

                "score": (
                    signal.score
                ),

                "reason": (
                    signal.reason
                ),

                "votes": (
                    signal.votes
                ),

                "regime": (
                    signal.regime
                ),

                "timeframe_confirmation": (
                    signal.timeframe_confirmation
                ),

                "quality_score": (
                    signal.quality_score
                ),
            }

        return {
            "state": (
                self.state.value
            ),

            "initialized": (
                self.initialized
            ),

            "execution_mode": "PAPER",

            "live_order_execution": (
                self.order_execution_enabled
            ),

            "engine_cycle_count": (
                self.engine_cycle_count
            ),

            "last_event_time": (
                self.last_event_time
            ),

            "last_candle_time": (
                self.last_candle_time
            ),

            "kill_switch": (
                self.kill_switch
            ),

            "kill_switch_reason": (
                self.kill_switch_reason
            ),

            "risk": (
                self.risk_engine.get_stats()
                if self.risk_engine
                else {}
            ),

            "paper_execution": (
                self.paper_execution.get_stats()
                if self.paper_execution
                else {}
            ),

            "strategy": (
                self.strategy_validator
                .get_stats()
            ),

            "strategy_weights": dict(
                self.strategy_engine
                .strategy_weights
            ),

            "signal_performance": self.signal_performance_tracker.get_stats(),

            "learning": (
                self.learning_engine
                .get_stats()
            ),

            "portfolio": (
                self.portfolio
                .get_stats()
            ),

            "features": feature_status,

            "latest_signal": signal_status,

            "last_rejection_reason": (
                self.last_rejection_reason
            ),

            "rejection_counts": dict(
                self.rejection_counts
            ),
        }

    # ========================================================
    # FINAL PERFORMANCE
    # ========================================================

    def _log_final_performance(self):

        if self.paper_execution is None:
            return

        execution_stats = (
            self.paper_execution
            .get_stats()
        )

        strategy_stats = (
            self.strategy_validator
            .get_stats()
        )

        # IMPORTANT:
        # This log previously had 12 format
        # placeholders but only 11 arguments.
        # The portfolio max drawdown argument
        # below fixes the shutdown traceback.
        self.logger.info(
            "FINAL PERFORMANCE | "
            "entries=%d | "
            "exits=%d | "
            "completed=%d | "
            "open=%d | "
            "wins=%d | "
            "losses=%d | "
            "win_rate=%.2f%% | "
            "gross_profit=%.8f | "
            "gross_loss=%.8f | "
            "net_pnl=%.8f | "
            "profit_factor=%.4f | "
            "max_drawdown=%.6f",
            execution_stats[
                "entries"
            ],
            execution_stats[
                "exits"
            ],
            execution_stats[
                "completed_trades"
            ],
            execution_stats[
                "open_positions"
            ],
            execution_stats[
                "wins"
            ],
            execution_stats[
                "losses"
            ],
            execution_stats[
                "win_rate"
            ]
            * 100,
            execution_stats[
                "gross_profit"
            ],
            execution_stats[
                "gross_loss"
            ],
            execution_stats[
                "net_pnl"
            ],
            execution_stats[
                "profit_factor"
            ],
            self.portfolio.max_drawdown,
        )

        for (
            name,
            stats,
        ) in strategy_stats[
            "strategies"
        ].items():

            self.logger.info(
                "STRATEGY PERFORMANCE | "
                "strategy=%s | "
                "signals=%d | "
                "trades=%d | "
                "wins=%d | "
                "losses=%d | "
                "win_rate=%.2f%% | "
                "net_pnl=%.8f | "
                "profit_factor=%.4f",
                name,
                stats["signals"],
                stats["trades"],
                stats["wins"],
                stats["losses"],
                stats["win_rate"]
                * 100,
                stats["net_pnl"],
                stats["profit_factor"],
            )

        self.logger.info(
            "ADVANCED PERFORMANCE | "
            "expectancy=%.8f | "
            "average_trade=%.8f | "
            "max_drawdown=%.8f | "
            "max_consecutive_wins=%d | "
            "max_consecutive_losses=%d",
            strategy_stats[
                "expectancy"
            ],
            strategy_stats[
                "average_trade"
            ],
            strategy_stats[
                "max_drawdown"
            ],
            strategy_stats[
                "max_consecutive_wins"
            ],
            strategy_stats[
                "max_consecutive_losses"
            ],
        )

        if execution_stats[
            "exit_reasons"
        ]:

            for (
                reason,
                count,
            ) in sorted(
                execution_stats[
                    "exit_reasons"
                ].items(),
                key=lambda item: item[1],
                reverse=True,
            ):

                self.logger.info(
                    "EXIT SUMMARY | "
                    "reason=%s | "
                    "count=%d",
                    reason,
                    count,
                )

        for (
            reason,
            count,
        ) in sorted(
            self.rejection_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            self.logger.info(
                "SIGNAL REJECTION SUMMARY | "
                "reason=%s | "
                "count=%d",
                reason,
                count,
            )

        learning = (
            self.learning_engine
            .get_stats()
        )

        self.logger.info(
            "LEARNING STATUS | "
            "samples=%d | "
            "model=%s | "
            "loaded=%s",
            learning["samples"],
            learning["model_name"],
            learning["model_loaded"],
        )

        self.logger.info(
            "PORTFOLIO STATUS | "
            "realized_pnl=%.8f | "
            "current_drawdown=%.6f | "
            "max_drawdown=%.6f | "
            "kill_switch=%s | "
            "reason=%s",
            self.portfolio.realized_pnl,
            self.portfolio.current_drawdown,
            self.portfolio.max_drawdown,
            self.kill_switch,
            self.kill_switch_reason,
        )

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self):

        if self.state in (
            EngineState.SHUTDOWN,
            EngineState.SHUTTING_DOWN,
        ):

            return

        self.shutting_down = True

        previous_state = self.state

        self.logger.info(
            "Trading engine shutdown initiated."
        )

        self.state = (
            EngineState.SHUTTING_DOWN
        )

        try:

            if self.collector is not None:

                self.collector.stop()

            if (
                self.candle_aggregator
                is not None
            ):

                try:

                    self.candle_aggregator.flush()

                except Exception:

                    self.logger.exception(
                        "Candle aggregator "
                        "flush failed."
                    )

            try:

                self.delta.stop()

            except Exception:

                self.logger.exception(
                    "Delta connection "
                    "shutdown failed."
                )

            try:

                self.watchdog.stop()

            except Exception:

                self.logger.exception(
                    "Watchdog shutdown failed."
                )

            self._persist_runtime_state()
            self._log_final_performance()

            self.state = (
                EngineState.SHUTDOWN
            )

            self.logger.info(
                "Engine state: %s -> %s",
                previous_state.value,
                self.state.value,
            )

            self.logger.info(
                "Trading engine shutdown "
                "completed."
            )

        except Exception:

            self.state = (
                EngineState.ERROR
            )

            self.logger.exception(
                "Trading engine shutdown failed."
            )