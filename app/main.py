import argparse
import logging
import signal
import subprocess
import sys
from pathlib import Path

from app.config.settings import Settings
from app.core.engine import TradingEngine


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
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


def run_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n$ " + " ".join(command))
    subprocess.run(command, check=True)


def run_research_pipeline() -> None:
    """Build historical data -> align it -> train ML -> backtest ML signals."""
    print("=" * 72)
    print("        DELTA RESEARCH PIPELINE")
    print("=" * 72)
    print("Live orders: DISABLED")
    print()

    # 1. Build one-second intrasecond features from raw Delta JSONL.
    run_module("app.ml.intrasecond_features")

    # 2. Build multi-horizon future targets.
    run_module("app.ml.multi_horizon_dataset")

    # 3. Align current features with future targets and run leakage checks.
    run_module("app.ml.dataset_alignment")

    # 4. Train the production-compatible ML model.
    run_module(
        "app.ml.trainer",
        "--dataset",
        "data/ml/final_training_dataset.csv",
        "--model",
        "data/models/market_direction_model.joblib",
    )

    # 5. Backtest the trained model on the same chronological market dataset.
    run_module(
        "app.backtest.engine",
        "--dataset",
        "data/ml/final_training_dataset.csv",
        "--model",
        "data/models/market_direction_model.joblib",
    )

    print()
    print("=" * 72)
    print("        RESEARCH PIPELINE COMPLETE")
    print("=" * 72)
    print("ML model:  data/models/market_direction_model.joblib")
    print("Metadata:  data/models/model_metadata.json")
    print("Backtest:  data/ml/backtest_engine_test.csv")
    print("=" * 72)


def run_backtest() -> None:
    run_module(
        "app.backtest.engine",
        "--dataset",
        "data/ml/final_training_dataset.csv",
        "--model",
        "data/models/market_direction_model.joblib",
    )


def run_ml_training() -> None:
    run_module(
        "app.ml.trainer",
        "--dataset",
        "data/ml/final_training_dataset.csv",
        "--model",
        "data/models/market_direction_model.joblib",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delta Algorithmic Trading Bot"
    )
    parser.add_argument(
        "--research",
        action="store_true",
        help="Run dataset construction, leakage validation, ML training and ML backtest.",
    )
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Train the production ML model from the aligned dataset.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run the historical backtest using the trained ML model.",
    )

    args = parser.parse_args()

    if args.research:
        run_research_pipeline()
        return

    if args.train_ml:
        run_ml_training()
        return

    if args.backtest:
        run_backtest()
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
