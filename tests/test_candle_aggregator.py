
from pathlib import Path

import pytest

from app.market_data.candle_aggregator import (
    CandleAggregator,
)
from app.market_data.models import MarketDataEvent


def create_trade_event(
    price: float,
    quantity: float,
    timestamp: int,
    symbol: str = "BTCUSD",
) -> MarketDataEvent:
    return MarketDataEvent(
        event_type="trades",
        symbol=symbol,
        exchange_timestamp=timestamp,
        received_timestamp=timestamp,
        payload={
            "sy": symbol,
            "p": price,
            "s": quantity,
        },
    )


def test_single_trade_creates_active_builder(
    tmp_path: Path,
) -> None:
    aggregator = CandleAggregator(
        intervals_seconds=[1],
        output_directory=str(tmp_path),
    )

    event = create_trade_event(
        price=100.0,
        quantity=2.0,
        timestamp=1_000_000,
    )

    aggregator.process_event(event)

    stats = aggregator.get_stats()

    assert stats["active_builders"] == 1
    assert stats["closed_candles"] == 0


def test_ohlcv_values_are_calculated(
    tmp_path: Path,
) -> None:
    completed_candles = []

    aggregator = CandleAggregator(
        intervals_seconds=[1],
        output_directory=str(tmp_path),
        candle_handler=completed_candles.append,
    )

    aggregator.process_event(
        create_trade_event(
            price=100.0,
            quantity=2.0,
            timestamp=1_000_000,
        )
    )

    aggregator.process_event(
        create_trade_event(
            price=110.0,
            quantity=3.0,
            timestamp=1_200_000,
        )
    )

    aggregator.process_event(
        create_trade_event(
            price=90.0,
            quantity=4.0,
            timestamp=1_800_000,
        )
    )

    # A trade in the next one-second bucket closes
    # the previous candle.
    aggregator.process_event(
        create_trade_event(
            price=105.0,
            quantity=1.0,
            timestamp=2_000_000,
        )
    )

    assert len(completed_candles) == 1

    candle = completed_candles[0]

    assert candle.symbol == "BTCUSD"
    assert candle.interval_seconds == 1

    assert candle.open == 100.0
    assert candle.high == 110.0
    assert candle.low == 90.0
    assert candle.close == 90.0

    assert candle.volume == 9.0
    assert candle.trade_count == 3


def test_invalid_trade_is_ignored(
    tmp_path: Path,
) -> None:
    aggregator = CandleAggregator(
        intervals_seconds=[1],
        output_directory=str(tmp_path),
    )

    invalid_event = MarketDataEvent(
        event_type="trades",
        symbol="BTCUSD",
        exchange_timestamp=1_000_000,
        received_timestamp=1_000_000,
        payload={
            "sy": "BTCUSD",
            "p": -10,
            "s": 2,
        },
    )

    aggregator.process_event(invalid_event)

    stats = aggregator.get_stats()

    assert stats["active_builders"] == 0
    assert stats["ignored_events"] == 1


def test_non_trade_event_is_ignored(
    tmp_path: Path,
) -> None:
    aggregator = CandleAggregator(
        intervals_seconds=[1],
        output_directory=str(tmp_path),
    )

    event = MarketDataEvent(
        event_type="ob_l1",
        symbol="BTCUSD",
        exchange_timestamp=1_000_000,
        received_timestamp=1_000_000,
        payload={
            "sy": "BTCUSD",
            "bp": 100.0,
            "ap": 101.0,
        },
    )

    aggregator.process_event(event)

    stats = aggregator.get_stats()

    assert stats["active_builders"] == 0
    assert stats["ignored_events"] == 0


def test_multiple_intervals_are_supported(
    tmp_path: Path,
) -> None:
    aggregator = CandleAggregator(
        intervals_seconds=[1, 5, 15, 60],
        output_directory=str(tmp_path),
    )

    event = create_trade_event(
        price=100.0,
        quantity=1.0,
        timestamp=1_000_000,
    )

    aggregator.process_event(event)

    stats = aggregator.get_stats()

    assert stats["active_builders"] == 4


def test_flush_closes_active_candles(
    tmp_path: Path,
) -> None:
    completed_candles = []

    aggregator = CandleAggregator(
        intervals_seconds=[1],
        output_directory=str(tmp_path),
        candle_handler=completed_candles.append,
    )

    aggregator.process_event(
        create_trade_event(
            price=100.0,
            quantity=1.0,
            timestamp=1_000_000,
        )
    )

    aggregator.flush()

    assert len(completed_candles) == 1
    assert completed_candles[0].close == 100.0

    stats = aggregator.get_stats()

    assert stats["active_builders"] == 0
    assert stats["closed_candles"] == 1


def test_candle_is_saved_as_jsonl(
    tmp_path: Path,
) -> None:
    aggregator = CandleAggregator(
        intervals_seconds=[1],
        output_directory=str(tmp_path),
    )

    aggregator.process_event(
        create_trade_event(
            price=100.0,
            quantity=1.0,
            timestamp=1_000_000,
        )
    )

    aggregator.process_event(
        create_trade_event(
            price=101.0,
            quantity=1.0,
            timestamp=2_000_000,
        )
    )

    candle_file = (
        tmp_path / "candles.jsonl"
    )

    assert candle_file.exists()

    content = candle_file.read_text(
        encoding="utf-8"
    )

    assert content.strip() != ""
    assert '"symbol":"BTCUSD"' in content
    assert '"open":100.0' in content