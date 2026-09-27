from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional\n\nfrom app.config.settings import Settings


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = Path(
    "data/ml/final_training_dataset.csv"
)

RESULT_FILE = Path(
    "data/ml/backtest_engine_test.csv"
)

SYMBOL = "BTCUSD"


# ------------------------------------------------------------
# Execution assumptions
# ------------------------------------------------------------

# These are configurable simulation assumptions.
# They are NOT claims about Delta's current live fees.

DEFAULT_INITIAL_CAPITAL = 10_000.0

DEFAULT_FEE_RATE = 0.0005

DEFAULT_SLIPPAGE_BPS = 1.0

DEFAULT_STOP_LOSS_PCT = 0.0025

DEFAULT_TAKE_PROFIT_PCT = 0.0025

DEFAULT_MAX_HOLD_SECONDS = 60


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(
    "BacktestEngine"
)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MarketBar:
    """
    One chronological market observation.
    """

    timestamp: float

    symbol: str

    bid: float

    ask: float

    mid: float

    spread: float

    trade_price: Optional[float]

    trade_volume: float

    imbalance: float

    volatility: float


@dataclass
class Position:
    """
    Simulated open position.
    """

    side: str

    entry_time: float

    entry_price: float

    quantity: float

    stop_loss: Optional[float]

    take_profit: Optional[float]

    max_exit_time: float


@dataclass
class Trade:
    """
    Completed simulated trade.
    """

    entry_time: float

    exit_time: float

    side: str

    entry_price: float

    exit_price: float

    quantity: float

    gross_pnl: float

    fees: float

    slippage_cost: float

    net_pnl: float

    exit_reason: str
    holding_seconds: float


@dataclass
class BacktestConfig:
    """
    Backtesting configuration.
    """

    initial_capital: float = (
        DEFAULT_INITIAL_CAPITAL
    )

    fee_rate: float = (
        DEFAULT_FEE_RATE
    )

    slippage_bps: float = (
        DEFAULT_SLIPPAGE_BPS
    )

    stop_loss_pct: float = (
        DEFAULT_STOP_LOSS_PCT
    )

    take_profit_pct: float = (
        DEFAULT_TAKE_PROFIT_PCT
    )

    max_hold_seconds: int = (
        DEFAULT_MAX_HOLD_SECONDS
    )

    allow_long: bool = True

    allow_short: bool = True


# ============================================================
# NUMERIC HELPERS
# ============================================================

def safe_float(
    value,
) -> Optional[float]:

    if value is None:
        return None

    if value == "":
        return None

    try:

        result = float(value)

        if not math.isfinite(result):

            return None

        return result

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# DATASET LOADING
# ============================================================

def load_market_data(
    path: Path = DATASET_FILE,
    test_only: bool = True,
) -> List[MarketBar]:

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        rows = list(reader)

    rows = [
        row for row in rows
        if (row.get("symbol") or SYMBOL) == SYMBOL
    ]
    rows.sort(
        key=lambda row: safe_float(
            row.get("timestamp_seconds")
            or row.get("feature_timestamp_seconds"),
            0.0,
        )
    )

    if test_only:
        split_index = int(len(rows) * 0.80)
        rows = rows[split_index:]
        logger.info(
            "Using chronological ML holdout | rows=%d | start_index=%d",
            len(rows),
            split_index,
        )

    bars: List[MarketBar] = []

    for row in rows:

        symbol = (
            row.get("symbol")
            or SYMBOL
        )

        if symbol != SYMBOL:
            continue

        timestamp = safe_float(
            row.get(
                "timestamp_seconds"
            )
            or row.get(
                "feature_timestamp_seconds"
            )
        )

        bid = safe_float(
            row.get("bid_close")
        )

        ask = safe_float(
            row.get("ask_close")
        )

        mid = safe_float(
            row.get("mid_close")
        )

        spread = safe_float(
            row.get("spread_close")
        )

        if (
            timestamp is None
            or bid is None
            or ask is None
            or mid is None
            or spread is None
        ):

            continue

        trade_price = safe_float(
            row.get(
                "trade_price_close"
            )
        )

        trade_volume = safe_float(
            row.get(
                "trade_volume"
            )
        )

        imbalance = safe_float(
            row.get(
                "imbalance_mean"
            )
        )

        volatility = safe_float(
            row.get(
                "intrasecond_volatility"
            )
        )

        bars.append(
            MarketBar(
                timestamp=timestamp,
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                spread=spread,
                trade_price=trade_price,
                trade_volume=(
                    trade_volume
                    if trade_volume is not None
                    else 0.0
                ),
                imbalance=(
                    imbalance
                    if imbalance is not None
                    else 0.0
                ),
                volatility=(
                    volatility
                    if volatility is not None
                    else 0.0
                ),
            )
        )

    bars.sort(
        key=lambda bar:
        bar.timestamp
    )

    logger.info(
        "Market bars loaded | %d",
        len(bars),
    )

    return bars


# ============================================================
# ML SIGNAL GENERATION
# ============================================================

def generate_ml_signals(
    dataset_path: Path = DATASET_FILE,
    model_path: Path = DATA_ROOT / "models" / "market_direction_model.joblib",
    test_only: bool = True,
) -> List[Optional[str]]:
    """Generate backtest signals from the trained production ML model."""
    from app.ml.model import MLModel

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained ML model not found: {model_path}. "
            "Run the ML training pipeline first."
        )

    model = MLModel.load(model_path)

    with dataset_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    filtered = [
        row for row in rows
        if (
            (row.get("symbol") or SYMBOL) == SYMBOL
            and safe_float(row.get("timestamp_seconds") or row.get("feature_timestamp_seconds")) is not None
            and safe_float(row.get("bid_close")) is not None
            and safe_float(row.get("ask_close")) is not None
            and safe_float(row.get("mid_close")) is not None
            and safe_float(row.get("spread_close")) is not None
        )
    ]
    filtered.sort(
        key=lambda row: safe_float(
            row.get("timestamp_us")
            or row.get("timestamp")
            or row.get("timestamp_seconds")
            or row.get("feature_timestamp_seconds"),
            0.0,
        )
    )

    if test_only:
        split_index = int(len(filtered) * 0.80)
        filtered = filtered[split_index:]

    signals: List[Optional[str]] = []

    for row in filtered:
        features = {}
        for name in model.feature_names:
            value = safe_float(row.get(name))
            features[name] = value if value is not None else 0.0

        prediction = model.predict_single(features)["prediction"]

        if prediction > 0:
            signals.append("LONG")
        elif prediction < 0:
            signals.append("SHORT")
        else:
            signals.append(None)

    if len(signals) == 0:
        raise RuntimeError("The trained ML model produced no backtest observations.")

    return signals


# ============================================================
# EXECUTION PRICE MODEL
# ============================================================

def apply_slippage(
    price: float,
    side: str,
    slippage_bps: float,
) -> float:

    adjustment = (
        slippage_bps / 10_000.0
    )

    if side == "BUY":

        return price * (
            1.0 + adjustment
        )

    return price * (
        1.0 - adjustment
    )


# ============================================================
# ENTRY EXECUTION
# ============================================================

def execute_entry(
    bar: MarketBar,
    side: str,
    config: BacktestConfig,
) -> float:

    if side == "LONG":

        # A long market entry consumes the ask.
        raw_price = bar.ask

        execution_side = "BUY"

    elif side == "SHORT":

        # A short market entry consumes the bid.
        raw_price = bar.bid

        execution_side = "SELL"

    else:

        raise ValueError(
            f"Invalid position side: {side}"
        )

    return apply_slippage(
        raw_price,
        execution_side,
        config.slippage_bps,
    )


# ============================================================
# EXIT EXECUTION
# ============================================================

def execute_exit(
    bar: MarketBar,
    side: str,
    config: BacktestConfig,
) -> float:

    if side == "LONG":

        # Closing a long sells into the bid.
        raw_price = bar.bid

        execution_side = "SELL"

    elif side == "SHORT":

        # Closing a short buys from the ask.
        raw_price = bar.ask

        execution_side = "BUY"

    else:

        raise ValueError(
            f"Invalid position side: {side}"
        )

    return apply_slippage(
        raw_price,
        execution_side,
        config.slippage_bps,
    )


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_quantity(
    capital: float,
    price: float,
    capital_fraction: float = 1.0,
) -> float:

    if capital <= 0:
        return 0.0

    if price <= 0:
        return 0.0

    notional = (
        capital
        * capital_fraction
    )

    return (
        notional / price
    )


# ============================================================
# PNL
# ============================================================

def calculate_gross_pnl(
    position: Position,
    exit_price: float,
) -> float:

    if position.side == "LONG":

        return (
            exit_price
            - position.entry_price
        ) * position.quantity

    if position.side == "SHORT":

        return (
            position.entry_price
            - exit_price
        ) * position.quantity

    raise ValueError(
        f"Invalid position side: "
        f"{position.side}"
    )


def calculate_fees(
    entry_price: float,
    exit_price: float,
    quantity: float,
    fee_rate: float,
) -> float:

    entry_notional = (
        abs(entry_price)
        * quantity
    )

    exit_notional = (
        abs(exit_price)
        * quantity
    )

    return (
        entry_notional
        + exit_notional
    ) * fee_rate


def calculate_slippage_cost(
    position: Position,
    exit_price: float,
    bar: MarketBar,
) -> float:

    # The actual execution prices already include
    # slippage. This method estimates the explicit
    # difference between theoretical bid/ask execution
    # and the simulated execution price.

    if position.side == "LONG":

        theoretical_exit = bar.bid

        theoretical_entry = bar.ask

    else:

        theoretical_exit = bar.ask

        theoretical_entry = bar.bid

    entry_cost = abs(
        position.entry_price
        - theoretical_entry
    ) * position.quantity

    exit_cost = abs(
        exit_price
        - theoretical_exit
    ) * position.quantity

    return (
        entry_cost
        + exit_cost
    )


# ============================================================
# STOP / TARGET CALCULATION
# ============================================================

def calculate_stop_loss(
    side: str,
    entry_price: float,
    percentage: float,
) -> float:

    if side == "LONG":

        return entry_price * (
            1.0 - percentage
        )

    return entry_price * (
        1.0 + percentage
    )


def calculate_take_profit(
    side: str,
    entry_price: float,
    percentage: float,
) -> float:

    if side == "LONG":

        return entry_price * (
            1.0 + percentage
        )

    return entry_price * (
        1.0 - percentage
    )


# ============================================================
# EXIT REASON
# ============================================================

def determine_exit_reason(
    position: Position,
    bar: MarketBar,
) -> Optional[str]:

    if position.side == "LONG":

        # Conservative rule:
        # if both levels could be touched inside the
        # same observation, stop loss is processed first.

        if (
            position.stop_loss is not None
            and bar.low if False else False
        ):
            pass

        if (
            position.stop_loss is not None
            and bar.bid
            <= position.stop_loss
        ):

            return "STOP_LOSS"

        if (
            position.take_profit is not None
            and bar.bid
            >= position.take_profit
        ):

            return "TAKE_PROFIT"

    elif position.side == "SHORT":

        if (
            position.stop_loss is not None
            and bar.ask
            >= position.stop_loss
        ):

            return "STOP_LOSS"

        if (
            position.take_profit is not None
            and bar.ask
            <= position.take_profit
        ):

            return "TAKE_PROFIT"

    if (
        bar.timestamp
        >= position.max_exit_time
    ):

        return "MAX_HOLD"

    return None


# ============================================================
# BACKTEST ENGINE
# ============================================================

class BacktestEngine:

    def __init__(
        self,
        config: Optional[
            BacktestConfig
        ] = None,
    ):

        self.config = (
            config
            or BacktestConfig()
        )

        self.capital = (
            self.config.initial_capital
        )

        self.starting_capital = (
            self.config.initial_capital
        )

        self.position: Optional[
            Position
        ] = None

        self.trades: List[
            Trade
        ] = []

        self.equity_curve: List[
            Dict[str, float]
        ] = []

    # --------------------------------------------------------
    # OPEN
    # --------------------------------------------------------

    def open_position(
        self,
        bar: MarketBar,
        side: str,
    ) -> bool:

        if self.position is not None:

            return False

        if side == "LONG":

            if not self.config.allow_long:
                return False

        elif side == "SHORT":

            if not self.config.allow_short:
                return False

        else:

            return False

        entry_price = execute_entry(
            bar,
            side,
            self.config,
        )

        quantity = calculate_quantity(
            self.capital,
            entry_price,
        )

        if quantity <= 0:

            return False

        stop_loss = calculate_stop_loss(
            side,
            entry_price,
            self.config.stop_loss_pct,
        )

        take_profit = (
            calculate_take_profit(
                side,
                entry_price,
                self.config.take_profit_pct,
            )
        )

        self.position = Position(
            side=side,
            entry_time=bar.timestamp,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_exit_time=(
                bar.timestamp
                + self.config.max_hold_seconds
            ),
        )

        return True

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close_position(
        self,
        bar: MarketBar,
        reason: str,
    ) -> Optional[Trade]:

        if self.position is None:

            return None

        position = self.position

        exit_price = execute_exit(
            bar,
            position.side,
            self.config,
        )

        gross_pnl = calculate_gross_pnl(
            position,
            exit_price,
        )

        fees = calculate_fees(
            position.entry_price,
            exit_price,
            position.quantity,
            self.config.fee_rate,
        )

        slippage_cost = (
            calculate_slippage_cost(
                position,
                exit_price,
                bar,
            )
        )

        net_pnl = (
            gross_pnl
            - fees
        )

        self.capital += net_pnl

        holding_seconds = (
            bar.timestamp
            - position.entry_time
        )

        trade = Trade(
            entry_time=position.entry_time,
            exit_time=bar.timestamp,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            gross_pnl=gross_pnl,
            fees=fees,
            slippage_cost=slippage_cost,
            net_pnl=net_pnl,
            exit_reason=reason,
            holding_seconds=holding_seconds,
        )

        self.trades.append(
            trade
        )

        self.position = None

        return trade

    # --------------------------------------------------------
    # EQUITY
    # --------------------------------------------------------

    def calculate_equity(
        self,
        bar: MarketBar,
    ) -> float:

        if self.position is None:

            return self.capital

        position = self.position

        if position.side == "LONG":

            unrealized = (
                bar.bid
                - position.entry_price
            ) * position.quantity

        else:

            unrealized = (
                position.entry_price
                - bar.ask
            ) * position.quantity

        return (
            self.capital
            + unrealized
        )

    # --------------------------------------------------------
    # PROCESS BAR
    # --------------------------------------------------------

    def process_bar(
        self,
        bar: MarketBar,
        signal: Optional[str],
    ) -> None:

        # ----------------------------------------------------
        # Existing position
        # ----------------------------------------------------

        if self.position is not None:

            exit_reason = (
                determine_exit_reason(
                    self.position,
                    bar,
                )
            )

            if exit_reason is not None:

                self.close_position(
                    bar,
                    exit_reason,
                )

            else:

                # Optional reversal.
                if (
                    signal == "LONG"
                    and self.position.side
                    == "SHORT"
                ):

                    self.close_position(
                        bar,
                        "SIGNAL_REVERSAL",
                    )

                    self.open_position(
                        bar,
                        "LONG",
                    )

                elif (
                    signal == "SHORT"
                    and self.position.side
                    == "LONG"
                ):

                    self.close_position(
                        bar,
                        "SIGNAL_REVERSAL",
                    )

                    self.open_position(
                        bar,
                        "SHORT",
                    )

        # ----------------------------------------------------
        # New position
        # ----------------------------------------------------

        if self.position is None:

            if signal == "LONG":

                self.open_position(
                    bar,
                    "LONG",
                )

            elif signal == "SHORT":

                self.open_position(
                    bar,
                    "SHORT",
                )

        # ----------------------------------------------------
        # Equity
        # ----------------------------------------------------

        equity = self.calculate_equity(
            bar
        )

        self.equity_curve.append(
            {
                "timestamp": bar.timestamp,
                "equity": equity,
            }
        )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
        bars: List[MarketBar],
        signals: List[Optional[str]],
    ) -> None:

        if len(bars) != len(signals):

            raise ValueError(
                "bars and signals must have "
                "the same length."
            )

        previous_timestamp = None

        for bar, signal in zip(
            bars,
            signals,
        ):

            # Never process time backwards.
            if (
                previous_timestamp is not None
                and bar.timestamp
                < previous_timestamp
            ):

                raise ValueError(
                    "Market data is not chronological."
                )

            previous_timestamp = (
                bar.timestamp
            )

            self.process_bar(
                bar,
                signal,
            )

        # Force-close any remaining position
        # at the final available market observation.
        if (
            self.position is not None
            and bars
        ):

            self.close_position(
                bars[-1],
                "END_OF_DATA",
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    def metrics(self) -> Dict[str, float]:

        total_trades = len(
            self.trades
        )

        winning_trades = sum(
            1
            for trade in self.trades
            if trade.net_pnl > 0
        )

        losing_trades = sum(
            1
            for trade in self.trades
            if trade.net_pnl < 0
        )

        breakeven_trades = (
            total_trades
            - winning_trades
            - losing_trades
        )

        gross_profit = sum(
            trade.net_pnl
            for trade in self.trades
            if trade.net_pnl > 0
        )

        gross_loss = sum(
            trade.net_pnl
            for trade in self.trades
            if trade.net_pnl < 0
        )

        total_fees = sum(
            trade.fees
            for trade in self.trades
        )

        total_slippage = sum(
            trade.slippage_cost
            for trade in self.trades
        )

        net_pnl = (
            self.capital
            - self.starting_capital
        )

        win_rate = (
            winning_trades
            / total_trades
            if total_trades > 0
            else 0.0
        )

        profit_factor = (
            gross_profit
            / abs(gross_loss)
            if gross_loss < 0
            else (
                math.inf
                if gross_profit > 0
                else 0.0
            )
        )

        # ----------------------------------------------------
        # Drawdown
        # ----------------------------------------------------

        peak = (
            self.starting_capital
        )

        max_drawdown = 0.0

        for point in (
            self.equity_curve
        ):

            equity = point[
                "equity"
            ]

            if equity > peak:

                peak = equity

            drawdown = (
                peak - equity
            )

            if drawdown > max_drawdown:

                max_drawdown = drawdown

        max_drawdown_pct = (
            max_drawdown
            / peak
            if peak > 0
            else 0.0
        )

        average_trade = (
            net_pnl / total_trades
            if total_trades > 0
            else 0.0
        )

        return {
            "starting_capital":
                self.starting_capital,

            "ending_capital":
                self.capital,

            "net_pnl":
                net_pnl,

            "return_pct":
                (
                    net_pnl
                    / self.starting_capital
                    * 100.0
                    if self.starting_capital > 0
                    else 0.0
                ),

            "total_trades":
                total_trades,

            "winning_trades":
                winning_trades,

            "losing_trades":
                losing_trades,

            "breakeven_trades":
                breakeven_trades,

            "win_rate":
                win_rate,

            "profit_factor":
                profit_factor,

            "gross_profit":
                gross_profit,

            "gross_loss":
                gross_loss,

            "total_fees":
                total_fees,

            "total_slippage":
                total_slippage,

            "max_drawdown":
                max_drawdown,

            "max_drawdown_pct":
                max_drawdown_pct * 100.0,

            "average_trade":
                average_trade,
        }


# ============================================================
# TEST SIGNAL GENERATOR
# ============================================================

def generate_test_signals(
    bars: List[MarketBar],
) -> List[Optional[str]]:

    """
    Temporary smoke-test strategy.

    This is NOT a trading strategy recommendation.

    It only verifies that the backtesting engine can
    process LONG / SHORT / FLAT signals.

    The real strategy modules will be created separately
    in Phase 7.2+.
    """

    signals: List[
        Optional[str]
    ] = []

    previous_mid = None

    for bar in bars:

        if previous_mid is None:

            signals.append(None)

            previous_mid = bar.mid

            continue

        change = (
            bar.mid
            - previous_mid
        )

        # Extremely small test threshold.
        # This is deliberately only a technical smoke test.
        threshold = (
            max(
                bar.spread * 0.25,
                0.01,
            )
        )

        if change > threshold:

            signals.append("LONG")

        elif change < -threshold:

            signals.append("SHORT")

        else:

            signals.append(None)

        previous_mid = bar.mid

    return signals


# ============================================================
# SAVE TRADES
# ============================================================

def save_trades(
    trades: List[Trade],
    path: Path = RESULT_FILE,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "quantity",
        "gross_pnl",
        "fees",
        "slippage_cost",
        "net_pnl",
        "exit_reason",
        "holding_seconds",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for trade in trades:

            writer.writerow(
                {
                    "entry_time":
                        trade.entry_time,

                    "exit_time":
                        trade.exit_time,

                    "side":
                        trade.side,

                    "entry_price":
                        trade.entry_price,

                    "exit_price":
                        trade.exit_price,

                    "quantity":
                        trade.quantity,

                    "gross_pnl":
                        trade.gross_pnl,

                    "fees":
                        trade.fees,

                    "slippage_cost":
                        trade.slippage_cost,

                    "net_pnl":
                        trade.net_pnl,

                    "exit_reason":
                        trade.exit_reason,

                    "holding_seconds":
                        trade.holding_seconds,
                }
            )


# ============================================================
# SMOKE TEST
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Delta historical backtest."
    )
    parser.add_argument("--dataset", default=str(DATASET_FILE))
    parser.add_argument("--model", default="data/models/market_direction_model.joblib")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--stop-loss", type=float, default=DEFAULT_STOP_LOSS_PCT)
    parser.add_argument("--take-profit", type=float, default=DEFAULT_TAKE_PROFIT_PCT)
    parser.add_argument("--max-hold", type=int, default=DEFAULT_MAX_HOLD_SECONDS)
    parser.add_argument(
        "--smoke-signals",
        action="store_true",
        help="Use the legacy technical smoke-test signals instead of the trained ML model.",
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Evaluate the model on all rows instead of the default chronological holdout.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)

    print()
    print("=" * 72)
    print("       DELTA ML-INTEGRATED BACKTEST ENGINE")
    print("=" * 72)
    print(f"Dataset:       {dataset_path}")
    print(f"ML model:      {model_path}")
    print("Mode:          HISTORICAL SIMULATION ONLY")
    print("Live orders:   DISABLED")
    print()

    test_only = not args.all_data
    bars = load_market_data(dataset_path, test_only=test_only)

    if len(bars) < 2:
        raise RuntimeError("Not enough market observations for backtesting.")

    if args.smoke_signals:
        signals = generate_test_signals(bars)
        signal_mode = "LEGACY_SMOKE_TEST"
    else:
        signals = generate_ml_signals(
            dataset_path,
            model_path,
            test_only=test_only,
        )
        signal_mode = "TRAINED_ML"

    if len(signals) != len(bars):
        raise RuntimeError(
            f"Signal/bar length mismatch: {len(signals)} != {len(bars)}"
        )

    config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_hold_seconds=args.max_hold,
    )

    engine = BacktestEngine(config)
    engine.run(bars, signals)
    metrics = engine.metrics()
    save_trades(engine.trades)

    print("=" * 72)
    print(f"Signal mode:        {signal_mode}")
    print("Evaluation set:     " + ("HOLDOUT 20%" if test_only else "ALL DATA"))
    print(f"Starting capital:   {metrics['starting_capital']:.8f}")
    print(f"Ending capital:     {metrics['ending_capital']:.8f}")
    print(f"Net PnL:            {metrics['net_pnl']:.8f}")
    print(f"Return:             {metrics['return_pct']:.4f}%")
    print(f"Trades:             {int(metrics['total_trades'])}")
    print(f"Win rate:           {metrics['win_rate'] * 100.0:.2f}%")
    print(f"Profit factor:      {metrics['profit_factor']}")
    print(f"Max drawdown:       {metrics['max_drawdown']:.8f}")
    print(f"Max drawdown %:     {metrics['max_drawdown_pct']:.4f}%")
    print(f"Fees:               {metrics['total_fees']:.8f}")
    print(f"Slippage cost:      {metrics['total_slippage']:.8f}")
    print(f"Trade output:       {RESULT_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
