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

    def __init__(self, rest_client, instrument_manager, risk_engine, result_handler=None, execution_mode="LIVE", taker_fee_rate=0.0005, maker_fee_rate=0.0002, gst_rate=0.18):
        self.rest = rest_client
        self.instruments = instrument_manager
        self.risk_engine = risk_engine
        self.result_handler = result_handler
        self.execution_mode = str(execution_mode).upper()
        self.taker_fee_rate = max(float(taker_fee_rate), 0.0)
        self.maker_fee_rate = max(float(maker_fee_rate), 0.0)
        self.gst_rate = max(float(gst_rate), 0.0)
        self.logger = logging.getLogger("ExchangeExecutionEngine")
        self.positions: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.total_trading_fees = 0.0
        self.total_fee_gst = 0.0
        self.total_fees = 0.0

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

    def _contract_value(self, symbol: str) -> float:
        product = self.instruments.get(symbol)
        if not product:
            raise RuntimeError(f"Delta product not loaded: {symbol}")
        value = float(product.get("contract_value") or 0.0)
        if value <= 0:
            raise RuntimeError(f"Delta contract_value missing/invalid: {symbol}")
        return value

    def _notional(self, symbol: str, contracts: int, price: float) -> float:
        return float(contracts) * self._contract_value(symbol) * float(price)

    def _estimate_trading_fee(self, symbol: str, contracts: int, price: float, maker: bool = False) -> float:
        notional = self._notional(symbol, contracts, price)
        rate = self.maker_fee_rate if maker else self.taker_fee_rate
        return notional * rate

    def _extract_trading_fee(self, response: dict[str, Any], fallback: float) -> float:
        """Use a fee/commission returned by Delta when available; otherwise use the estimate."""
        if not isinstance(response, dict):
            return fallback
        candidates = []
        result = response.get("result")
        if isinstance(result, dict):
            candidates.append(result)
        candidates.append(response)
        for item in candidates:
            for key in ("commission", "paid_commission", "commission_amount", "fee"):
                value = item.get(key)
                try:
                    fee = float(value)
                except (TypeError, ValueError):
                    continue
                if fee >= 0:
                    return fee
        return fallback

    def _contract_size(self, symbol: str, base_quantity: float, price: float) -> int:
        """Convert risk-engine base-asset quantity into Delta derivative contracts."""
        contract_value = self._contract_value(symbol)
        if base_quantity <= 0 or price <= 0:
            return 0

        # RiskEngine quantities are expressed in underlying units (e.g. BTC).
        # Delta derivative orders use integer contract counts.
        contracts_from_risk = int(base_quantity / contract_value)

        # Preserve the leverage/notional cap after integer rounding.
        max_notional = float(self.risk_engine.account_equity) * float(self.risk_engine.max_leverage)
        max_contracts = int(max_notional / (price * contract_value))

        contracts = min(contracts_from_risk, max_contracts)

        # Never force a minimum one-contract order. If one Delta contract
        # would exceed the risk-engine quantity, the trade must be rejected.
        return max(contracts, 0)

    def reconcile_exchange_state(self) -> dict[str, Any]:
        """Reconcile local execution state with Delta before new signals."""
        positions_response = self.rest.get_positions()
        orders_response = self.rest.get_open_orders()

        if not isinstance(positions_response, dict):
            raise RuntimeError("Invalid Delta positions response during reconciliation.")
        if not isinstance(orders_response, dict):
            raise RuntimeError("Invalid Delta open-orders response during reconciliation.")
        if positions_response.get("success") is False:
            raise RuntimeError(f"Delta position reconciliation failed: {positions_response}")
        if orders_response.get("success") is False:
            raise RuntimeError(f"Delta open-order reconciliation failed: {orders_response}")

        raw_positions = positions_response.get("result", [])
        if isinstance(raw_positions, dict):
            raw_positions = [raw_positions]
        if not isinstance(raw_positions, list):
            raise RuntimeError("Unexpected Delta positions result format.")

        product_by_id = {}
        for product in getattr(self.instruments, "products", []):
            try:
                product_by_id[int(product.get("id"))] = product
            except (TypeError, ValueError):
                continue

        exchange_positions: dict[str, dict[str, Any]] = {}
        for raw in raw_positions:
            if not isinstance(raw, dict):
                continue
            size = int(float(raw.get("size") or 0))
            if size == 0:
                continue
            product_id = raw.get("product_id")
            product = product_by_id.get(int(product_id)) if product_id is not None else None
            symbol = str(raw.get("product_symbol") or (product or {}).get("symbol") or "").upper()
            if not symbol:
                raise RuntimeError(f"Delta returned an open position without a symbol: {raw}")
            if product is None:
                product = self.instruments.get(symbol)
            if not product:
                raise RuntimeError(f"Open Delta position has unknown product: {symbol}")

            side = "BUY" if size > 0 else "SELL"
            contracts = abs(size)
            entry_price = float(raw.get("entry_price") or 0.0)
            previous = self.positions.get(symbol)
            managed = bool(previous and previous.get("managed_by_bot"))
            self.orders = self.orders if isinstance(self.orders, dict) else {}
            exchange_positions[symbol] = {
                **(previous if previous else {}),
                "symbol": symbol,
                "side": side,
                "quantity": contracts,
                "base_quantity": contracts * self._contract_value(symbol),
                "contract_value": self._contract_value(symbol),
                "entry_price": entry_price,
                "entry_notional": self._notional(symbol, contracts, entry_price) if entry_price > 0 else 0.0,
                "managed_by_bot": managed,
                "reconciled": True,
            }

        stale_local = set(self.positions) - set(exchange_positions)
        for symbol in stale_local:
            self.logger.warning("POSITION RECONCILIATION | local position absent on Delta; removing local state | symbol=%s", symbol)

        self.positions = exchange_positions

        raw_orders = orders_response.get("result", [])
        if isinstance(raw_orders, list):
            self.orders = {str(o.get("id", o.get("client_order_id", i))): o for i, o in enumerate(raw_orders) if isinstance(o, dict)}
        else:
            self.orders = {}

        summary = {
            "exchange_positions": len(self.positions),
            "open_orders": len(self.orders),
            "symbols": sorted(self.positions),
        }
        self.logger.warning(
            "POSITION RECONCILIATION COMPLETE | positions=%d | open_orders=%d | symbols=%s",
            summary["exchange_positions"], summary["open_orders"], ",".join(summary["symbols"]) or "NONE",
        )
        return summary

    def _place_market(self, symbol: str, side: str, size: float, reduce_only: bool = False, stop_loss: float = 0.0, take_profit: float = 0.0, trail_amount: float = 0.0):
        if size <= 0:
            raise ValueError("Order size must be positive")
        payload = {
            "product_id": self._product_id(symbol),
            "size": int(size),
            "side": side.lower(),
            "order_type": "market_order",
            "reduce_only": bool(reduce_only),
            "client_order_id": self._client_order_id("mkt"),
        }
        if not reduce_only and stop_loss > 0 and take_profit > 0:
            payload["bracket_stop_trigger_method"] = "mark_price"
            payload["bracket_stop_loss_price"] = str(stop_loss)
            payload["bracket_take_profit_price"] = str(take_profit)
            if trail_amount > 0:
                payload["bracket_trail_amount"] = str(trail_amount)
        response = self.rest.place_order(payload)
        if not isinstance(response, dict) or response.get("success") is False:
            raise RuntimeError(f"Delta order rejected: {response}")
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
            if existing.get("reconciled") and not existing.get("managed_by_bot"):
                return False, "EXCHANGE_POSITION_RECONCILED_MANUAL_INTERVENTION_REQUIRED"
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

        reference_stop = entry_reference * (1 - stop_pct) if side == "BUY" else entry_reference * (1 + stop_pct)
        reference_target = entry_reference * (1 + target_pct) if side == "BUY" else entry_reference * (1 - target_pct)
        contract_size = self._contract_size(symbol, quantity, entry_reference)
        if contract_size <= 0:
            return False, "ORDER_SIZE_BELOW_DELTA_MINIMUM_OR_MARGIN_CAP"

        estimated_entry_fee = self._estimate_trading_fee(symbol, contract_size, entry_reference, maker=False)
        estimated_entry_gst = estimated_entry_fee * self.gst_rate
        response = self._place_market(
            symbol,
            side,
            contract_size,
            False,
            reference_stop,
            reference_target,
            max(entry_reference * 0.0005, 0.01),
        )
        result = response.get("result", {}) if isinstance(response, dict) else {}
        order_id = str(result.get("id", ""))

        entry_price = float(result.get("average_fill_price") or result.get("price") or entry_reference)
        actual_entry_fee = self._extract_trading_fee(response, estimated_entry_fee)
        actual_entry_gst = actual_entry_fee * self.gst_rate
        if side == "BUY":
            stop = entry_price * (1 - stop_pct)
            target = entry_price * (1 + target_pct)
        else:
            stop = entry_price * (1 + stop_pct)
            target = entry_price * (1 - target_pct)

        self.positions[symbol] = {
            "symbol": symbol,
            "side": side,
            "quantity": int(contract_size),
            "base_quantity": float(quantity),
            "contract_value": self._contract_value(symbol),
            "entry_price": entry_price,
            "stop_loss": stop,
            "take_profit": target,
            "order_id": order_id,
            "entry_timestamp": float(signal.timestamp),
            "entry_notional": self._notional(symbol, int(contract_size), entry_price),
            "entry_trading_fee": float(actual_entry_fee),
            "entry_fee_gst": float(actual_entry_gst),
            "entry_fee_total": float(actual_entry_fee + actual_entry_gst),
            "learning_id": getattr(signal, "learning_id", ""),
            "strategy_names": list(signal.votes or ["AGGREGATED"]),
            "managed_by_bot": True,
            "reconciled": False,
        }
        self.total_trading_fees += actual_entry_fee
        self.total_fee_gst += actual_entry_gst
        self.total_fees += actual_entry_fee + actual_entry_gst
        self.risk_engine.record_trade()
        self.logger.warning(
            "%s ENTRY SENT | symbol=%s side=%s contracts=%s base_quantity=%.8f entry=%.8f order_id=%s stop=%.8f target=%.8f entry_fee=%.8f entry_gst=%.8f",
            self.execution_mode,
            symbol,
            side,
            contract_size,
            quantity,
            entry_price,
            order_id,
            stop,
            target,
            actual_entry_fee,
            actual_entry_gst,
        )
        return True, f"{self.execution_mode}_ENTRY_ACCEPTED"

    def close_position(self, symbol: str, features, reason: str = "EXIT"):
        position = self.positions.get(symbol)
        if not position:
            return False, "NO_POSITION"
        side = "SELL" if position["side"] == "BUY" else "BUY"
        exit_contracts = int(position["quantity"])
        response = self._place_market(symbol, side, exit_contracts, True)
        result = response.get("result", {}) if isinstance(response, dict) else {}
        exit_price = float(result.get("average_fill_price") or result.get("price") or (features.bid if side == "SELL" else features.ask))
        contract_value = float(position.get("contract_value") or self._contract_value(symbol))
        estimated_exit_fee = self._estimate_trading_fee(symbol, exit_contracts, exit_price, maker=False)
        exit_trading_fee = self._extract_trading_fee(response, estimated_exit_fee)
        exit_fee_gst = exit_trading_fee * self.gst_rate
        if position["side"] == "BUY":
            pnl = (exit_price - position["entry_price"]) * position["quantity"] * contract_value
        else:
            pnl = (position["entry_price"] - exit_price) * position["quantity"] * contract_value
        gross_pnl = pnl
        total_trading_fee = float(position.get("entry_trading_fee", 0.0)) + exit_trading_fee
        total_fee_gst = float(position.get("entry_fee_gst", 0.0)) + exit_fee_gst
        total_fees = total_trading_fee + total_fee_gst
        net_pnl = gross_pnl - total_fees
        self.total_trading_fees += exit_trading_fee
        self.total_fee_gst += exit_fee_gst
        self.total_fees += exit_trading_fee + exit_fee_gst
        self.risk_engine.record_pnl(net_pnl)
        metadata = dict(position)
        metadata.update({
            "exit_price": exit_price,
            "exit_notional": self._notional(symbol, exit_contracts, exit_price),
            "exit_trading_fee": exit_trading_fee,
            "exit_fee_gst": exit_fee_gst,
            "exit_fee_total": exit_trading_fee + exit_fee_gst,
            "gross_pnl": gross_pnl,
            "total_trading_fees": total_trading_fee,
            "total_fee_gst": total_fee_gst,
            "total_fees": total_fees,
            "net_pnl": net_pnl,
            "reason": reason,
            "exit_timestamp": time.time(),
        })
        if self.result_handler:
            self.result_handler(position["strategy_names"], net_pnl, metadata)
        self.logger.warning(
            "%s EXIT SENT | symbol=%s side=%s quantity=%s exit=%.8f gross_pnl=%+.8f fees=%.8f net_pnl=%+.8f reason=%s",
            self.execution_mode, symbol, side, position["quantity"], exit_price, gross_pnl, total_fees, net_pnl, reason,
        )
        del self.positions[symbol]
        return True, f"{self.execution_mode}_EXIT_ACCEPTED"

    def process_market_price(self, symbol: str, bid: float, ask: float):
        position = self.positions.get(symbol)
        if not position:
            return
        if position.get("reconciled") and not position.get("managed_by_bot"):
            return
        if float(position.get("stop_loss") or 0.0) <= 0 or float(position.get("take_profit") or 0.0) <= 0:
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

    def close_all(self, reason: str, features) -> None:
        for symbol in list(self.positions):
            try:
                self.close_position(symbol, features, reason)
            except Exception:
                self.logger.exception("LIVE emergency close failed | symbol=%s", symbol)

    def get_stats(self):
        return {
            "exchange_execution": True,
            "execution_mode": self.execution_mode,
            "positions": {k: dict(v) for k, v in self.positions.items()},
            "orders": len(self.orders),
            "total_trading_fees": self.total_trading_fees,
            "total_fee_gst": self.total_fee_gst,
            "total_fees": self.total_fees,
        }

    def snapshot_state(self):
        return {"positions": self.positions, "orders": self.orders}
