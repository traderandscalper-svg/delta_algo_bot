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

    def __init__(self, rest_client, instrument_manager, risk_engine, result_handler=None, execution_mode="LIVE", taker_fee_rate=0.0005, maker_fee_rate=0.0002, gst_rate=0.18, allow_exchange_orders=True):
        self.rest = rest_client
        self.instruments = instrument_manager
        self.risk_engine = risk_engine
        self.result_handler = result_handler
        self.execution_mode = str(execution_mode).upper()
        self.allow_exchange_orders = bool(allow_exchange_orders)
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
        """Reconcile local execution state with Delta before new signals.

        The current trading universe is BTCUSD. Delta's real-time positions
        endpoint requires either a product_id or an underlying_asset_symbol,
        so resolve BTCUSD through the loaded instrument table rather than
        making an invalid unfiltered /v2/positions request.
        """
        primary_product = self.instruments.get("BTCUSD")
        if not primary_product:
            raise RuntimeError(
                "BTCUSD product is required for exchange-state reconciliation "
                "but could not be resolved from Delta."
            )

        product_id = primary_product.get("id")
        if product_id is None:
            raise RuntimeError(
                "BTCUSD product id is missing; cannot reconcile Delta position."
            )

        positions_response = self.rest.get_positions(
            product_id=int(product_id)
        )
        orders_response = self.rest.get_open_orders()

        raw_positions = positions_response.get("result", []) if isinstance(positions_response, dict) else []
        if isinstance(raw_positions, dict):
            raw_positions = [raw_positions]

        exchange_positions = 0
        for raw in raw_positions if isinstance(raw_positions, list) else []:
            if not isinstance(raw, dict):
                continue
            size = int(float(raw.get("size") or 0))
            if size == 0:
                continue
            symbol = str(
                raw.get("product_symbol")
                or raw.get("symbol")
                or primary_product.get("symbol")
                or "BTCUSD"
            ).upper()
            self.positions[symbol] = {
                "symbol": symbol,
                "side": "BUY" if size > 0 else "SELL",
                "quantity": abs(size),
                "entry_price": float(raw.get("entry_price") or 0.0),
                "managed_by_bot": False,
                "reconciled": True,
                "exchange_position": dict(raw),
            }
            exchange_positions += 1
            self.logger.warning(
                "EXCHANGE POSITION RECONCILED | symbol=%s | side=%s | contracts=%d | entry_price=%s | manual_management_required=true",
                symbol,
                self.positions[symbol]["side"],
                abs(size),
                raw.get("entry_price"),
            )

        raw_orders = orders_response.get("result", []) if isinstance(orders_response, dict) else []
        if isinstance(raw_orders, dict):
            raw_orders = [raw_orders]
        open_orders = len(raw_orders) if isinstance(raw_orders, list) else 0

        return {
            "exchange_positions": exchange_positions,
            "open_orders": open_orders,
        }

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
        self.logger.warning(
            "ORDER REQUEST | mode=%s | symbol=%s | side=%s | contracts=%d | reduce_only=%s | client_order_id=%s | payload=%s",
            self.execution_mode, symbol, side, int(size), bool(reduce_only), payload["client_order_id"], payload,
        )
        response = self.rest.place_order(payload)
        if not isinstance(response, dict) or response.get("success") is False:
            self.logger.error("ORDER REJECTED | mode=%s | symbol=%s | client_order_id=%s | response=%s",
                              self.execution_mode, symbol, payload["client_order_id"], response)
            raise RuntimeError(f"Delta order rejected: {response}")

        result = response.get("result") if isinstance(response.get("result"), dict) else {}
        order_id = str(result.get("id", payload["client_order_id"]))
        self.orders[order_id] = response
        self.logger.warning(
            "ORDER ACCEPTED | mode=%s | symbol=%s | side=%s | contracts=%d | order_id=%s | state=%s | avg_fill=%s | unfilled=%s",
            self.execution_mode, symbol, side, int(size), order_id,
            result.get("state"), result.get("average_fill_price"), result.get("unfilled_size"),
        )

        verified = result
        try:
            vr = self.rest.get_order_by_client_order_id(payload["client_order_id"])
            if isinstance(vr, dict) and vr.get("success") is not False and isinstance(vr.get("result"), dict):
                verified = vr["result"]
                self.orders[order_id] = verified
                self.logger.warning(
                    "ORDER VERIFIED | mode=%s | symbol=%s | order_id=%s | state=%s | avg_fill=%s | size=%s | unfilled=%s | commission=%s",
                    self.execution_mode, symbol, str(verified.get("id", order_id)),
                    verified.get("state"), verified.get("average_fill_price"),
                    verified.get("size"), verified.get("unfilled_size"),
                    verified.get("commission") or verified.get("paid_commission"),
                )
        except Exception:
            self.logger.exception("ORDER VERIFICATION FAILED | mode=%s | symbol=%s | client_order_id=%s",
                                  self.execution_mode, symbol, payload["client_order_id"])

        state = str(verified.get("state") or "").lower()
        requested_size = int(float(verified.get("size") or size))
        unfilled_size = int(float(verified.get("unfilled_size") or 0))
        filled_size = max(requested_size - unfilled_size, 0)
        fill_price = verified.get("average_fill_price") or verified.get("price")

        if filled_size > 0 and state == "closed":
            self.logger.warning("FILL VERIFIED | mode=%s | symbol=%s | side=%s | order_id=%s | filled_contracts=%d | average_fill_price=%s",
                                self.execution_mode, symbol, side, str(verified.get("id", order_id)), filled_size, fill_price)
        elif filled_size > 0:
            self.logger.warning("PARTIAL/PENDING FILL | mode=%s | symbol=%s | side=%s | order_id=%s | filled_contracts=%d | average_fill_price=%s | state=%s",
                                self.execution_mode, symbol, side, str(verified.get("id", order_id)), filled_size, fill_price, state)
        else:
            self.logger.warning("NO FILL VERIFIED | mode=%s | symbol=%s | side=%s | order_id=%s | state=%s | unfilled=%s",
                                self.execution_mode, symbol, side, str(verified.get("id", order_id)), state, unfilled_size)

        return {
            "response": response,
            "verified_order": verified,
            "filled_size": filled_size,
            "fill_price": float(fill_price) if fill_price not in (None, "") else 0.0,
            "client_order_id": payload["client_order_id"],
            "order_id": str(verified.get("id", order_id)),
        }

    def process_signal(self, signal, features):
        if signal.action.value == "HOLD":
            return False, "SIGNAL_HOLD"

        symbol = features.symbol
        side = "BUY" if signal.action.value == "BUY" else "SELL"

        self.logger.warning(
            "SIGNAL | symbol=%s | action=%s | confidence=%.4f | score=%.4f | quality=%.4f | price=%.8f | volatility=%.8f | regime=%s | reason=%s | votes=%s",
            symbol, signal.action.value, float(getattr(signal, "confidence", 0.0)),
            float(getattr(signal, "score", 0.0)), float(getattr(signal, "quality_score", 0.0)),
            float(getattr(features, "price", 0.0)), float(getattr(features, "volatility", 0.0)),
            getattr(features, "regime", "UNKNOWN"), getattr(signal, "reason", ""), list(signal.votes or []),
        )
        allowed, reason = self.risk_engine.check_signal(signal, features)
        self.logger.warning(
            "RISK | symbol=%s | action=%s | allowed=%s | reason=%s | equity=%.8f | max_risk=%.6f | max_leverage=%s",
            symbol, signal.action.value, allowed, reason,
            float(self.risk_engine.account_equity), float(self.risk_engine.max_risk_per_trade), self.risk_engine.max_leverage,
        )
        if not allowed:
            return False, reason

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
        self.logger.warning(
            "CONTRACT | symbol=%s | side=%s | reference_price=%.8f | risk_quantity=%.8f | contract_value=%.8f | contracts=%d | estimated_notional=%.8f",
            symbol, side, entry_reference, quantity, self._contract_value(symbol), contract_size,
            self._notional(symbol, contract_size, entry_reference) if contract_size > 0 else 0.0,
        )
        if contract_size <= 0:
            return False, "ORDER_SIZE_BELOW_DELTA_MINIMUM_OR_MARGIN_CAP"

        estimated_entry_fee = self._estimate_trading_fee(symbol, contract_size, entry_reference, maker=False)
        estimated_entry_gst = estimated_entry_fee * self.gst_rate

        if not self.allow_exchange_orders:
            self.logger.warning(
                "EXCHANGE EXECUTION DISABLED | mode=%s | symbol=%s | side=%s | no Delta order will be sent",
                self.execution_mode, symbol, side,
            )
            self.logger.warning(
                "WOULD SEND ORDER | symbol=%s | side=%s | contracts=%d | entry_reference=%.8f | stop_loss=%.8f | take_profit=%.8f | estimated_notional=%.8f | estimated_fee=%.8f | estimated_fee_gst=%.8f",
                symbol, side, int(contract_size), entry_reference, reference_stop, reference_target,
                self._notional(symbol, contract_size, entry_reference), estimated_entry_fee, estimated_entry_gst,
            )
            return False, "EXCHANGE_EXECUTION_DISABLED_DRY_RUN"

        execution = self._place_market(
            symbol, side, contract_size, False, reference_stop, reference_target,
            max(entry_reference * 0.0005, 0.01),
        )
        response = execution["response"]
        result = execution["verified_order"] if isinstance(execution.get("verified_order"), dict) else {}
        order_id = str(execution["order_id"])
        filled_contracts = int(execution.get("filled_size") or 0)
        if filled_contracts <= 0:
            self.logger.error("POSITION NOT CREATED | symbol=%s | order_id=%s | reason=NO_VERIFIED_FILL", symbol, order_id)
            return False, "ORDER_ACCEPTED_BUT_NO_FILL_VERIFIED"
        if filled_contracts != int(contract_size):
            self.logger.error("POSITION NOT CREATED | symbol=%s | order_id=%s | requested_contracts=%d | verified_filled_contracts=%d",
                              symbol, order_id, int(contract_size), filled_contracts)
            return False, "PARTIAL_FILL_NOT_YET_SUPPORTED"

        entry_price = float(execution.get("fill_price") or result.get("average_fill_price") or result.get("price") or entry_reference)

        position_verified = False
        try:
            pr = self.rest.get_positions(
                product_id=self._product_id(symbol)
            )
            raw_positions = pr.get("result", []) if isinstance(pr, dict) else []
            if isinstance(raw_positions, dict):
                raw_positions = [raw_positions]
            for rp in raw_positions if isinstance(raw_positions, list) else []:
                if str(rp.get("product_symbol") or "").upper() == symbol and int(float(rp.get("size") or 0)) != 0:
                    exchange_size = abs(int(float(rp.get("size") or 0)))
                    position_verified = exchange_size >= int(contract_size)
                    self.logger.warning(
                        "POSITION VERIFIED | mode=%s | symbol=%s | side=%s | contracts=%d | exchange_size=%d | entry_price=%s | order_id=%s",
                        self.execution_mode, symbol, side, int(contract_size), exchange_size, rp.get("entry_price"), order_id,
                    )
                    break
        except Exception:
            self.logger.exception("POSITION VERIFICATION FAILED | mode=%s | symbol=%s | order_id=%s",
                                  self.execution_mode, symbol, order_id)
        if not position_verified:
            self.logger.error("POSITION NOT VERIFIED | mode=%s | symbol=%s | order_id=%s | filled_contracts=%d",
                              self.execution_mode, symbol, order_id, filled_contracts)
            return False, "FILL_VERIFIED_BUT_POSITION_NOT_VERIFIED"
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
            "%s ENTRY COMPLETE | symbol=%s side=%s contracts=%s base_quantity=%.8f entry=%.8f order_id=%s stop=%.8f target=%.8f entry_fee=%.8f entry_gst=%.8f | SIGNAL->RISK->CONTRACT->ORDER->FILL->POSITION=VERIFIED",
            self.execution_mode, symbol, side, contract_size, quantity, entry_price,
            order_id, stop, target, actual_entry_fee, actual_entry_gst,
        )
        return True, f"{self.execution_mode}_ENTRY_FILLED_AND_POSITION_VERIFIED"

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
