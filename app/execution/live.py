from __future__ import annotations

import logging
import time
import uuid
from typing import Any


class LiveExecutionEngine:
    """Small, explicit live-order adapter.

    Strategy/risk decisions remain in the core engine. This class only
    translates an accepted entry/exit into Delta REST orders and keeps
    the exchange order IDs for recovery/audit.
    """

    def __init__(self, rest_client, instrument_manager, risk_engine, result_handler=None):
        self.rest = rest_client
        self.instruments = instrument_manager
        self.risk_engine = risk_engine
        self.result_handler = result_handler
        self.logger = logging.getLogger("LiveExecutionEngine")
        self.positions: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _client_order_id(prefix: str) -> str:
        return f"dalgo-{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"

    def _product_id(self, symbol: str) -> int:
        product = self.instruments.get(symbol)
        if not product:
            raise RuntimeError(f"Delta product not loaded: {symbol}")
        product_id = product.get("id")
        if product_id is None:
            raise RuntimeError(f"Delta product id missing: {symbol}")
        return int(product_id)

    def _place_market(self, symbol: str, side: str, size: float, reduce_only: bool = False):
        if size <= 0:
            raise ValueError("Order size must be positive")
        payload = {
            "product_id": self._product_id(symbol),
            "size": int(size) if float(size).is_integer() else size,
            "side": side.lower(),
            "order_type": "market_order",
            "reduce_only": bool(reduce_only),
            "client_order_id": self._client_order_id("mkt"),
        }
        response = self.rest.place_order(payload)
        self.orders[str(response.get("result", {}).get("id", payload["client_order_id"]))] = response
        return response

    def process_signal(self, signal, features):
        if signal.action.value == "HOLD":
            return False, "SIGNAL_HOLD"

        allowed, reason = self.risk_engine.check_signal(signal, features)
        if not allowed:
            return False, reason

        symbol = features.symbol
        side = "BUY" if signal.action.value == "BUY" else "SELL"

        existing = self.positions.get(symbol)
        if existing:
            if existing["side"] == side:
                return False, "POSITION_ALREADY_OPEN"
            self.close_position(symbol, features, "OPPOSITE_SIGNAL")
            return False, "OPPOSITE_SIGNAL_CLOSED"

        volatility = max(float(features.volatility), 0.0005)
        stop_pct = min(max(volatility * 2.0, 0.0010), 0.0030)
        target_pct = min(max(stop_pct * 2.5, 0.0025), 0.0075)
        entry_reference = float(features.ask if side == "BUY" else features.bid)
        if entry_reference <= 0:
            return False, "INVALID_EXECUTION_PRICE"

        stop_distance = entry_reference * stop_pct
        quantity, quantity_reason = self.risk_engine.calculate_quantity(entry_reference, stop_distance)
        if quantity <= 0:
            return False, quantity_reason

        response = self._place_market(symbol, side, quantity, False)
        result = response.get("result", {}) if isinstance(response, dict) else {}
        order_id = str(result.get("id", ""))

        entry_price = float(result.get("average_fill_price") or result.get("price") or entry_reference)
        if side == "BUY":
            stop = entry_price * (1 - stop_pct)
            target = entry_price * (1 + target_pct)
        else:
            stop = entry_price * (1 + stop_pct)
            target = entry_price * (1 - target_pct)

        self.positions[symbol] = {
            "symbol": symbol,
            "side": side,
            "quantity": float(quantity),
            "entry_price": entry_price,
            "stop_loss": stop,
            "take_profit": target,
            "order_id": order_id,
            "entry_timestamp": float(signal.timestamp),
            "learning_id": getattr(signal, "learning_id", ""),
            "strategy_names": list(signal.votes or ["AGGREGATED"]),
        }
        self.risk_engine.record_trade()
        self.logger.warning(
            "LIVE ENTRY SENT | symbol=%s side=%s quantity=%s entry=%.8f order_id=%s stop=%.8f target=%.8f",
            symbol, side, quantity, entry_price, order_id, stop, target,
        )
        return True, "LIVE_ENTRY_ACCEPTED"

    def close_position(self, symbol: str, features, reason: str = "EXIT"):
        position = self.positions.get(symbol)
        if not position:
            return False, "NO_POSITION"
        side = "SELL" if position["side"] == "BUY" else "BUY"
        response = self._place_market(symbol, side, position["quantity"], True)
        result = response.get("result", {}) if isinstance(response, dict) else {}
        exit_price = float(result.get("average_fill_price") or result.get("price") or (features.bid if side == "SELL" else features.ask))
        if position["side"] == "BUY":
            pnl = (exit_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - exit_price) * position["quantity"]
        self.risk_engine.record_pnl(pnl)
        metadata = dict(position)
        metadata.update({"exit_price": exit_price, "reason": reason, "exit_timestamp": time.time()})
        if self.result_handler:
            self.result_handler(position["strategy_names"], pnl, metadata)
        self.logger.warning(
            "LIVE EXIT SENT | symbol=%s side=%s quantity=%s exit=%.8f pnl=%+.8f reason=%s",
            symbol, side, position["quantity"], exit_price, pnl, reason,
        )
        del self.positions[symbol]
        return True, "LIVE_EXIT_ACCEPTED"

    def process_market_price(self, symbol: str, bid: float, ask: float):
        position = self.positions.get(symbol)
        if not position:
            return
        mark = float(bid if position["side"] == "BUY" else ask)
        if position["side"] == "BUY":
            if mark <= position["stop_loss"]:
                self.close_position(symbol, type("F", (), {"bid": bid, "ask": ask})(), "STOP_LOSS")
            elif mark >= position["take_profit"]:
                self.close_position(symbol, type("F", (), {"bid": bid, "ask": ask})(), "TAKE_PROFIT")
        else:
            if mark >= position["stop_loss"]:
                self.close_position(symbol, type("F", (), {"bid": bid, "ask": ask})(), "STOP_LOSS")
            elif mark <= position["take_profit"]:
                self.close_position(symbol, type("F", (), {"bid": bid, "ask": ask})(), "TAKE_PROFIT")

    def get_stats(self):
        return {"live": True, "positions": {k: dict(v) for k, v in self.positions.items()}, "orders": len(self.orders)}

    def snapshot_state(self):
        return {"positions": self.positions, "orders": self.orders}
