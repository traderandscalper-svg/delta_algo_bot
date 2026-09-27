
from __future__ import annotations

from app.market_data.candle_aggregator import Candle
from app.strategies.engine import StrategyEngine


BASE_TIMESTAMP = 1790418000000000
BASE_PRICE = 84000.0


def make_candle(
    index: int,
    open_price: float,
    close_price: float,
    high_price: float,
    low_price: float,
    volume: float,
) -> Candle:

    timestamp = (
        BASE_TIMESTAMP
        + index * 1_000_000
    )

    return Candle(
        symbol="BTCUSD",
        interval_seconds=1,
        start_timestamp=timestamp,
        end_timestamp=timestamp + 1_000_000,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        trade_count=10,
        first_trade_timestamp=timestamp,
        last_trade_timestamp=timestamp + 999_999,
    )


def generate_uptrend() -> list[Candle]:

    candles = []

    price = BASE_PRICE

    for i in range(80):

        open_price = price
        close_price = price + 12.0

        candles.append(
            make_candle(
                index=i,
                open_price=open_price,
                close_price=close_price,
                high_price=close_price + 5.0,
                low_price=open_price - 2.0,
                volume=100.0 + i * 2.0,
            )
        )

        price = close_price

    return candles


def generate_downtrend() -> list[Candle]:

    candles = []

    price = BASE_PRICE

    for i in range(80):

        open_price = price
        close_price = price - 12.0

        candles.append(
            make_candle(
                index=i,
                open_price=open_price,
                close_price=close_price,
                high_price=open_price + 2.0,
                low_price=close_price - 5.0,
                volume=100.0 + i * 2.0,
            )
        )

        price = close_price

    return candles


def generate_sideways() -> list[Candle]:

    candles = []

    price = BASE_PRICE

    movements = [
        4.0,
        -4.0,
        3.0,
        -3.0,
        2.0,
        -2.0,
    ]

    for i in range(80):

        movement = movements[
            i % len(movements)
        ]

        open_price = price
        close_price = price + movement

        candles.append(
            make_candle(
                index=i,
                open_price=open_price,
                close_price=close_price,
                high_price=max(
                    open_price,
                    close_price,
                ) + 3.0,
                low_price=min(
                    open_price,
                    close_price,
                ) - 3.0,
                volume=100.0,
            )
        )

        price = close_price

    return candles


def generate_breakout() -> list[Candle]:

    candles = []

    price = BASE_PRICE

    # Long consolidation period.
    for i in range(60):

        movement = (
            1.5
            if i % 2 == 0
            else -1.5
        )

        open_price = price
        close_price = price + movement

        candles.append(
            make_candle(
                index=i,
                open_price=open_price,
                close_price=close_price,
                high_price=max(
                    open_price,
                    close_price,
                ) + 2.0,
                low_price=min(
                    open_price,
                    close_price,
                ) - 2.0,
                volume=80.0,
            )
        )

        price = close_price

    # Strong upward breakout.
    for i in range(60, 80):

        open_price = price
        close_price = price + 35.0

        candles.append(
            make_candle(
                index=i,
                open_price=open_price,
                close_price=close_price,
                high_price=close_price + 8.0,
                low_price=open_price - 3.0,
                volume=250.0,
            )
        )

        price = close_price

    return candles


def run_scenario(
    name: str,
    candles: list[Candle],
) -> None:

    engine = StrategyEngine()

    results = []

    for candle in candles:

        result = engine.add_candle(
            candle
        )

        results.append(result)

    final_signal = results[-1]

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Candles:",
        len(candles),
    )

    print(
        "Buffer:",
        engine.get_buffer_size(
            "BTCUSD",
            1,
        ),
    )

    print(
        "Action:",
        final_signal.action.value,
    )

    print(
        "Confidence:",
        round(
            final_signal.confidence,
            4,
        ),
    )

    print(
        "Score:",
        round(
            final_signal.score,
            4,
        ),
    )

    print(
        "Quality:",
        round(
            final_signal.quality,
            4,
        ),
    )

    print(
        "Regime:",
        final_signal.regime,
    )

    print(
        "Votes:",
        {
            name: signal.value
            for name, signal
            in final_signal.strategy_votes.items()
        },
    )

    print(
        "Contributors:",
        final_signal.contributing_strategies,
    )

    print(
        "Reason:",
        final_signal.reason,
    )


def main() -> None:

    print()
    print(
        "DELTA ALGORITHM BOT"
    )
    print(
        "PHASE 7.2 STRATEGY REGIME TEST"
    )

    scenarios = [
        (
            "UPTREND",
            generate_uptrend(),
        ),
        (
            "DOWNTREND",
            generate_downtrend(),
        ),
        (
            "SIDEWAYS",
            generate_sideways(),
        ),
        (
            "BREAKOUT",
            generate_breakout(),
        ),
    ]

    for name, candles in scenarios:

        run_scenario(
            name,
            candles,
        )

    print()
    print("=" * 70)
    print("STRATEGY REGIME TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

