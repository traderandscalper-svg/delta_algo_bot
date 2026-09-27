import logging
import signal
import sys

from app.config.settings import Settings
from app.core.engine import TradingEngine


def configure_logging(settings: Settings) -> None:

    logging.basicConfig(
        level=getattr(
            logging,
            settings.log_level.upper(),
            logging.INFO,
        ),
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


def main() -> None:

    print("=" * 60)
    print("        DELTA ALGORITHMIC TRADING BOT")
    print("=" * 60)

    try:

        settings = Settings()

        settings.validate_startup()

        configure_logging(settings)

        logger = logging.getLogger("Main")

        logger.info(
            "Configuration loaded successfully."
        )

        engine = TradingEngine(settings)

        def handle_signal(signum, frame):
            logger.info(
                "Shutdown signal received: %s",
                signum,
            )

            engine.shutdown()

            sys.exit(0)

        signal.signal(
            signal.SIGINT,
            handle_signal,
        )

        signal.signal(
            signal.SIGTERM,
            handle_signal,
        )

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