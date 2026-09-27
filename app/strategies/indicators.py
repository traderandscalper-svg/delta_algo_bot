from __future__ import annotations

import math
from typing import List, Optional, Sequence

from app.market_data.candle_aggregator import Candle


def _clean(values: Sequence[float]) -> List[float]:
    result: List[float] = []

    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(number):
            result.append(number)

    return result


def closes(candles: List[Candle]) -> List[float]:
    return _clean([float(c.close) for c in candles])


def opens(candles: List[Candle]) -> List[float]:
    return _clean([float(c.open) for c in candles])


def highs(candles: List[Candle]) -> List[float]:
    return _clean([float(c.high) for c in candles])


def lows(candles: List[Candle]) -> List[float]:
    return _clean([float(c.low) for c in candles])


def volumes(candles: List[Candle]) -> List[float]:
    return _clean([float(c.volume) for c in candles])


def typical_prices(candles: List[Candle]) -> List[float]:
    result: List[float] = []

    for candle in candles:
        try:
            value = (
                float(candle.high)
                + float(candle.low)
                + float(candle.close)
            ) / 3.0

            if math.isfinite(value):
                result.append(value)

        except (TypeError, ValueError):
            continue

    return result


def sma(
    values: List[float],
    period: int,
) -> Optional[float]:

    if period <= 0 or len(values) < period:
        return None

    window = values[-period:]

    if not window:
        return None

    return sum(window) / len(window)


def ema(
    values: List[float],
    period: int,
) -> Optional[float]:

    if period <= 0 or len(values) < period:
        return None

    multiplier = 2.0 / (period + 1.0)

    result = sum(values[:period]) / period

    for value in values[period:]:
        result = (
            (value - result) * multiplier
            + result
        )

    return result


def highest(
    values: List[float],
    period: int,
) -> Optional[float]:

    if period <= 0 or len(values) < period:
        return None

    return max(values[-period:])


def lowest(
    values: List[float],
    period: int,
) -> Optional[float]:

    if period <= 0 or len(values) < period:
        return None

    return min(values[-period:])


def returns(
    values: List[float],
    period: int,
) -> Optional[float]:

    if period <= 0 or len(values) <= period:
        return None

    previous = values[-period - 1]
    current = values[-1]

    if previous == 0:
        return None

    return (current - previous) / previous


def log_returns(
    values: List[float],
    period: int,
) -> Optional[float]:

    if period <= 0 or len(values) <= period:
        return None

    previous = values[-period - 1]
    current = values[-1]

    if previous <= 0 or current <= 0:
        return None

    return math.log(current / previous)


def volatility(
    values: List[float],
    period: int,
) -> Optional[float]:

    if period <= 1 or len(values) < period + 1:
        return None

    recent = values[-period - 1:]

    returns_list: List[float] = []

    for i in range(1, len(recent)):
        previous = recent[i - 1]
        current = recent[i]

        if previous <= 0:
            continue

        value = (
            (current - previous)
            / previous
        )

        if math.isfinite(value):
            returns_list.append(value)

    if len(returns_list) < 2:
        return None

    mean = (
        sum(returns_list)
        / len(returns_list)
    )

    variance = (
        sum(
            (value - mean) ** 2
            for value in returns_list
        )
        / len(returns_list)
    )

    return math.sqrt(
        max(variance, 0.0)
    )


def standard_deviation(
    values: List[float],
    period: int,
) -> Optional[float]:

    if period <= 1 or len(values) < period:
        return None

    window = values[-period:]

    mean = sum(window) / period

    variance = (
        sum(
            (value - mean) ** 2
            for value in window
        )
        / period
    )

    return math.sqrt(
        max(variance, 0.0)
    )


def rsi(
    values: List[float],
    period: int = 14,
) -> Optional[float]:

    if period <= 0 or len(values) < period + 1:
        return None

    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

    recent = changes[-period:]

    gains = [
        max(change, 0.0)
        for change in recent
    ]

    losses = [
        max(-change, 0.0)
        for change in recent
    ]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:

        if average_gain == 0:
            return 50.0

        return 100.0

    rs = average_gain / average_loss

    return 100.0 - (
        100.0 / (1.0 + rs)
    )


def true_ranges(
    candles: List[Candle],
) -> List[float]:

    if not candles:
        return []

    result: List[float] = []

    previous_close: Optional[float] = None

    for candle in candles:

        high = float(candle.high)
        low = float(candle.low)

        if previous_close is None:
            true_range = high - low

        else:
            true_range = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )

        result.append(
            max(true_range, 0.0)
        )

        previous_close = float(candle.close)

    return result


def atr(
    candles: List[Candle],
    period: int = 14,
) -> Optional[float]:

    ranges = true_ranges(candles)

    if len(ranges) < period:
        return None

    return (
        sum(ranges[-period:])
        / period
    )


def atr_percent(
    candles: List[Candle],
    period: int = 14,
) -> Optional[float]:

    value = atr(candles, period)

    if value is None:
        return None

    price = float(candles[-1].close)

    if price <= 0:
        return None

    return value / price


def momentum(
    values: List[float],
    period: int = 10,
) -> Optional[float]:

    return returns(values, period)


def rate_of_change(
    values: List[float],
    period: int = 10,
) -> Optional[float]:

    return returns(values, period)


def bollinger_bands(
    values: List[float],
    period: int = 20,
    deviations: float = 2.0,
) -> Optional[tuple[float, float, float]]:

    if period <= 0 or len(values) < period:
        return None

    middle = sma(values, period)
    deviation = standard_deviation(
        values,
        period,
    )

    if middle is None or deviation is None:
        return None

    upper = middle + (
        deviation * deviations
    )

    lower = middle - (
        deviation * deviations
    )

    return (
        lower,
        middle,
        upper,
    )


def bollinger_position(
    values: List[float],
    period: int = 20,
    deviations: float = 2.0,
) -> Optional[float]:

    bands = bollinger_bands(
        values,
        period,
        deviations,
    )

    if bands is None:
        return None

    lower, _, upper = bands
    price = values[-1]

    width = upper - lower

    if width <= 0:
        return 0.5

    return (
        (price - lower)
        / width
    )


def vwap(
    candles: List[Candle],
    period: int = 20,
) -> Optional[float]:

    if period <= 0 or len(candles) < period:
        return None

    recent = candles[-period:]

    total_value = 0.0
    total_volume = 0.0

    for candle in recent:

        typical = (
            float(candle.high)
            + float(candle.low)
            + float(candle.close)
        ) / 3.0

        volume = max(
            float(candle.volume),
            0.0,
        )

        total_value += (
            typical * volume
        )

        total_volume += volume

    if total_volume <= 0:
        return None

    return (
        total_value
        / total_volume
    )


def slope(
    values: List[float],
    period: int = 10,
) -> Optional[float]:

    if period < 2 or len(values) < period:
        return None

    window = values[-period:]

    x_mean = (
        (period - 1)
        / 2.0
    )

    y_mean = (
        sum(window)
        / period
    )

    numerator = 0.0
    denominator = 0.0

    for index, value in enumerate(window):

        x = float(index)

        numerator += (
            (x - x_mean)
            * (value - y_mean)
        )

        denominator += (
            x - x_mean
        ) ** 2

    if denominator == 0:
        return None

    return numerator / denominator


def zscore(
    values: List[float],
    period: int = 20,
) -> Optional[float]:

    if period <= 1 or len(values) < period:
        return None

    window = values[-period:]

    mean = sum(window) / period

    deviation = standard_deviation(
        values,
        period,
    )

    if deviation is None or deviation == 0:
        return 0.0

    return (
        (values[-1] - mean)
        / deviation
    )


def volume_ratio(
    candles: List[Candle],
    period: int = 20,
) -> Optional[float]:

    vols = volumes(candles)

    if len(vols) < period + 1:
        return None

    average = (
        sum(vols[-period - 1:-1])
        / period
    )

    if average <= 0:
        return None

    return vols[-1] / average


def candle_body_ratio(
    candle: Candle,
) -> Optional[float]:

    high = float(candle.high)
    low = float(candle.low)
    open_price = float(candle.open)
    close_price = float(candle.close)

    range_value = high - low

    if range_value <= 0:
        return 0.0

    return abs(
        close_price - open_price
    ) / range_value


def trend_strength(
    values: List[float],
    period: int = 20,
) -> Optional[float]:

    if period < 3 or len(values) < period:
        return None

    window = values[-period:]

    start = window[0]
    end = window[-1]

    if start == 0:
        return None

    net_move = abs(
        (end - start)
        / start
    )

    movement = 0.0

    for i in range(1, len(window)):

        previous = window[i - 1]
        current = window[i]

        if previous == 0:
            continue

        movement += abs(
            (current - previous)
            / previous
        )

    if movement <= 0:
        return 0.0

    return min(
        1.0,
        net_move / movement,
    )