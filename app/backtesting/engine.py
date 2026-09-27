"""Historical candle backtesting engine for delta_algo_bot.

Consumes the raw JSONL market-data files produced by MarketDataCollector,
rebuilds 1-second candles from real trade events, applies a deterministic
multi-filter strategy, and reports equity, trades, drawdown and trade-level
features suitable for ML training.

This module never sends exchange orders.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime, timezone
from typing import Iterable


@dataclass
class Bar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestTrade:
    entry_timestamp: int
    exit_timestamp: int
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    hold_seconds: float
    features: dict
    outcome: int


class BacktestEngine:
    """Event-driven 1-second historical backtester with realistic bid/ask use."""

    def __init__(
        self,
        initial_equity: float = 1000.0,
        risk_per_trade: float = 0.0025,
        max_leverage: float = 1.0,
        fee_rate: float = 0.0005,
        slippage_bps: float = 1.0,
        stop_atr_multiple: float = 1.0,
        target_r_multiple: float = 2.0,
    ):
        self.initial_equity = float(initial_equity)
        self.equity = float(initial_equity)
        self.risk_per_trade = float(risk_per_trade)
        self.max_leverage = float(max_leverage)
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)
        self.stop_atr_multiple = float(stop_atr_multiple)
        self.target_r_multiple = float(target_r_multiple)

        self.bars: list[Bar] = []
        self.trades: list[BacktestTrade] = []
        self._trade_buffer: list[dict] = []
        self._bid = None
        self._ask = None
        self._last_trade_price = None
        self._last_trade_side = 0
        self._bar_start = None
        self._bar = None
        self._closes = deque(maxlen=500)
        self._highs = deque(maxlen=500)
        self._lows = deque(maxlen=500)
        self._volumes = deque(maxlen=500)
        self._position = None
        self._peak_equity = self.equity
        self.max_drawdown = 0.0

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @staticmethod
    def _number(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _timestamp(event):
        value = event.get("exchange_timestamp") or event.get("timestamp") or event.get("ts") or event.get("t")
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _payload(record):
        if isinstance(record.get("payload"), dict):
            return record["payload"]
        return record

    def load_jsonl(self, path: str) -> int:
        count = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    self.process_event(json.loads(line))
                    count += 1
                except Exception:
                    continue
        self._flush_bar()
        return count

    def process_event(self, record: dict) -> None:
        event = self._payload(record)
        event_type = event.get("type") or record.get("event_type")
        ts = self._timestamp(event)
        if not ts:
            ts = self._timestamp(record)

        if event_type == "ob_l1":
            self._bid = self._number(event.get("bp"), self._bid or 0.0)
            self._ask = self._number(event.get("ap"), self._ask or 0.0)
            return

        if event_type != "trades":
            return

        price = self._number(event.get("p", event.get("price")))
        size = abs(self._number(event.get("s", event.get("size", event.get("q", 0.0)))))
        if price <= 0:
            return

        second = ts // 1_000_000
        if self._bar_start is None:
            self._bar_start = second
            self._bar = Bar(ts, price, price, price, price, size)
        elif second != self._bar_start:
            self._flush_bar()
            self._bar_start = second
            self._bar = Bar(ts, price, price, price, price, size)
        else:
            self._bar.high = max(self._bar.high, price)
            self._bar.low = min(self._bar.low, price)
            self._bar.close = price
            self._bar.volume += size

        if self._last_trade_price is not None:
            if price > self._last_trade_price:
                self._last_trade_side = 1
            elif price < self._last_trade_price:
                self._last_trade_side = -1
        self._last_trade_price = price

    def _flush_bar(self) -> None:
        if self._bar is None:
            return
        bar = self._bar
        self.bars.append(bar)
        self._closes.append(bar.close)
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._volumes.append(bar.volume)
        self._on_bar(bar)
        self._bar = None

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------

    def _ema(self, period: int) -> float:
        values = list(self._closes)
        if not values:
            return 0.0
        period = min(period, len(values))
        alpha = 2.0 / (period + 1.0)
        value = values[0]
        for price in values[1:]:
            value = alpha * price + (1.0 - alpha) * value
        return value

    def _rsi(self, period: int = 14) -> float:
        values = list(self._closes)
        if len(values) <= period:
            return 50.0
        gains = 0.0
        losses = 0.0
        for a, b in zip(values[-period-1:-1], values[-period:]):
            change = b - a
            if change >= 0:
                gains += change
            else:
                losses -= change
        if losses == 0:
            return 100.0 if gains else 50.0
        rs = gains / losses
        return 100.0 - 100.0 / (1.0 + rs)

    def _atr(self, period: int = 14) -> float:
        highs = list(self._highs)
        lows = list(self._lows)
        closes = list(self._closes)
        if len(closes) < 2:
            return 0.0
        trs = []
        start = max(1, len(closes) - period)
        for i in range(start, len(closes)):
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
        return sum(trs) / len(trs) if trs else 0.0

    def _features(self) -> dict:
        closes = list(self._closes)
        price = closes[-1]
        ret1 = (price / closes[-2] - 1.0) if len(closes) >= 2 and closes[-2] else 0.0
        ret5 = (price / closes[-6] - 1.0) if len(closes) >= 6 and closes[-6] else 0.0
        ret15 = (price / closes[-16] - 1.0) if len(closes) >= 16 and closes[-16] else 0.0
        ema_fast = self._ema(8)
        ema_slow = self._ema(21)
        atr = self._atr()
        spread_bps = 0.0
        imbalance = 0.0
        if self._bid and self._ask and self._ask >= self._bid:
            mid = (self._bid + self._ask) / 2.0
            if mid:
                spread_bps = (self._ask - self._bid) / mid * 10000.0
        return {
            "return_1": ret1,
            "return_5": ret5,
            "return_15": ret15,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "trend": (ema_fast / ema_slow - 1.0) if ema_slow else 0.0,
            "rsi": self._rsi(),
            "atr": atr,
            "volatility": (atr / price) if price else 0.0,
            "orderbook_imbalance": imbalance,
            "trade_flow": float(self._last_trade_side),
            "spread_bps": spread_bps,
        }

    def _signal(self, features: dict) -> int:
        if len(self._closes) < 30:
            return 0
        score = 0.0
        if features["ema_fast"] > features["ema_slow"]:
            score += 1.0
        else:
            score -= 1.0
        if features["return_5"] > 0:
            score += 0.5
        elif features["return_5"] < 0:
            score -= 0.5
        if features["rsi"] < 70 and features["rsi"] > 52:
            score += 0.5
        elif features["rsi"] > 30 and features["rsi"] < 48:
            score -= 0.5
        if features["trade_flow"] > 0:
            score += 0.25
        elif features["trade_flow"] < 0:
            score -= 0.25
        if features["spread_bps"] > 10:
            return 0
        return 1 if score >= 1.25 else -1 if score <= -1.25 else 0

    # ------------------------------------------------------------------
    # Execution/risk
    # ------------------------------------------------------------------

    def _on_bar(self, bar: Bar) -> None:
        if self._position is not None:
            self._manage_position(bar)

        if self._position is None:
            features = self._features()
            signal = self._signal(features)
            if signal:
                self._open(bar, signal, features)

        self._peak_equity = max(self._peak_equity, self.equity)
        if self._peak_equity:
            dd = (self._peak_equity - self.equity) / self._peak_equity
            self.max_drawdown = max(self.max_drawdown, dd)

    def _open(self, bar: Bar, side: int, features: dict) -> None:
        atr = max(features["atr"], bar.close * 0.0005)
        stop_distance = atr * self.stop_atr_multiple
        risk_cash = self.equity * self.risk_per_trade
        quantity = risk_cash / stop_distance if stop_distance > 0 else 0.0
        max_notional = self.equity * self.max_leverage
        quantity = min(quantity, max_notional / bar.close) if bar.close else 0.0
        if quantity <= 0:
            return

        spread = (self._ask - self._bid) if self._bid and self._ask else 0.0
        slip = bar.close * self.slippage_bps / 10000.0 + spread / 2.0
        entry = bar.close + slip if side > 0 else bar.close - slip
        stop = entry - stop_distance if side > 0 else entry + stop_distance
        target = entry + stop_distance * self.target_r_multiple if side > 0 else entry - stop_distance * self.target_r_multiple
        self._position = {
            "side": side,
            "entry": entry,
            "quantity": quantity,
            "stop": stop,
            "target": target,
            "timestamp": bar.timestamp,
            "features": dict(features),
        }

    def _manage_position(self, bar: Bar) -> None:
        p = self._position
        exit_price = None
        if p["side"] > 0:
            if bar.low <= p["stop"]:
                exit_price = p["stop"]
            elif bar.high >= p["target"]:
                exit_price = p["target"]
        else:
            if bar.high >= p["stop"]:
                exit_price = p["stop"]
            elif bar.low <= p["target"]:
                exit_price = p["target"]
        if exit_price is not None:
            self._close(bar, exit_price)

    def _close(self, bar: Bar, exit_price: float) -> None:
        p = self._position
        gross = (exit_price - p["entry"]) * p["quantity"] * p["side"]
        fees = (abs(p["entry"] * p["quantity"]) + abs(exit_price * p["quantity"])) * self.fee_rate
        pnl = gross - fees
        self.equity += pnl
        entry_notional = abs(p["entry"] * p["quantity"])
        return_pct = pnl / entry_notional if entry_notional else 0.0
        trade = BacktestTrade(
            entry_timestamp=p["timestamp"],
            exit_timestamp=bar.timestamp,
            side="BUY" if p["side"] > 0 else "SELL",
            entry_price=p["entry"],
            exit_price=exit_price,
            quantity=p["quantity"],
            pnl=pnl,
            return_pct=return_pct,
            hold_seconds=max(0.0, (bar.timestamp - p["timestamp"]) / 1_000_000.0),
            features=p["features"],
            outcome=1 if pnl > 0 else 0,
        )
        self.trades.append(trade)
        self._trade_buffer.append(asdict(trade))
        self._position = None

    # ------------------------------------------------------------------
    # Results / ML dataset
    # ------------------------------------------------------------------

    def result(self) -> dict:
        wins = sum(1 for t in self.trades if t.pnl > 0)
        losses = sum(1 for t in self.trades if t.pnl < 0)
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = sum(abs(t.pnl) for t in self.trades if t.pnl < 0)
        return {
            "initial_equity": self.initial_equity,
            "final_equity": self.equity,
            "net_pnl": self.equity - self.initial_equity,
            "trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(self.trades) if self.trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss else 0.0,
            "max_drawdown": self.max_drawdown,
            "bars": len(self.bars),
        }

    def save_ml_dataset(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for trade in self._trade_buffer:
                row = dict(trade["features"])
                row["label"] = trade["outcome"]
                row["pnl"] = trade["pnl"]
                row["side"] = 1 if trade["side"] == "BUY" else -1
                handle.write(json.dumps(row) + "\n")

    def run(self, paths: Iterable[str]) -> dict:
        for path in paths:
            self.load_jsonl(path)
        return self.result()


def discover_market_data(root="data/market_data") -> list[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.endswith(".jsonl")
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Delta historical backtest.")
    parser.add_argument("--data", nargs="*", default=None)
    parser.add_argument("--root", default="data/market_data")
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--risk", type=float, default=0.0025)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--dataset", default="data/learning/backtest_dataset.jsonl")
    args = parser.parse_args()

    paths = args.data or discover_market_data(args.root)
    engine = BacktestEngine(
        initial_equity=args.equity,
        risk_per_trade=args.risk,
        max_leverage=args.leverage,
    )
    result = engine.run(paths)
    engine.save_ml_dataset(args.dataset)
    print(json.dumps(result, indent=2))
    print(f"ML dataset: {args.dataset}")
