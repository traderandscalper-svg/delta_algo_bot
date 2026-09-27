import argparse
import logging
import signal
import sys

from app.config.settings import Settings
from app.core.engine import TradingEngine
from app.backtesting.engine import BacktestEngine, discover_market_data
from app.ml.trainer import MLTrainer


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(settings.log_level.upper(), "INFO"),
        format=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"{settings.log_directory}/engine.log",
                encoding="utf-8",
            ),
        ],
    )


def run_backtest(settings: Settings, paths: list[str] | None = None) -> str:
    equity = 1000.0
    engine = BacktestEngine(
        initial_equity=equity,
        risk_per_trade=settings.max_risk_per_trade,
        max_leverage=float(settings.max_leverage),
    )
    data_paths = paths or discover_market_data("data/market_data")
    if not data_paths:
        raise FileNotFoundError(
            "No market-data JSONL files found in data/market_data. "
            "Run the bot in PAPER/DEMO mode first to collect historical data."
        )
    result = engine.run(data_paths)
    dataset = "data/learning/backtest_dataset.jsonl"
    engine.save_ml_dataset(dataset)

    print("\n" + "=" * 60)
    print("BACKTEST RESULT")
    print("=" * 60)
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"ml_dataset: {dataset}")
    print("=" * 60)
    return dataset


def run_ml_training(dataset: str = "data/learning/backtest_dataset.jsonl") -> str:
    model_path = "data/learning/backtest_model.joblib"
    result = MLTrainer(model_path).train(dataset)

    print("\n" + "=" * 60)
    print("ML TRAINING RESULT")
    print("=" * 60)
    for key, value in result.__dict__.items():
        print(f"{key}: {value}")
    print("=" * 60)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delta Algorithmic Trading Bot"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run the historical backtester and generate an ML dataset.",
    )
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Train the offline ML model from the backtest dataset.",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run backtest followed by offline ML training.",
    )
    parser.add_argument(
        "--data",
        nargs="*",
        default=None,
        help="Optional JSONL market-data files for --backtest.",
    )
    parser.add_argument(
        "--dataset",
        default="data/learning/backtest_dataset.jsonl",
        help="ML dataset path.",
    )

    args = parser.parse_args()

    if args.backtest or args.pipeline or args.train_ml:
        settings = Settings()
        if args.backtest or args.pipeline:
            dataset = run_backtest(settings, args.data)
        else:
            dataset = args.dataset

        if args.train_ml or args.pipeline:
            run_ml_training(dataset)
        return

    print("=" * 60)
    print("        DELTA ALGORITHMIC TRADING BOT")
    print("=" * 60)

    try:
        settings = Settings()
        settings.validate_startup()
        configure_logging(settings)

        logger = logging.getLogger("Main")
        logger.info("Configuration loaded successfully.")

        engine = TradingEngine(settings)

        def handle_signal(signum, frame):
            logger.info("Shutdown signal received: %s", signum)
            engine.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        engine.initialize()
        engine.run()

    except Exception as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger("Main").exception(
            "Fatal startup error: %s",
            exc,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
