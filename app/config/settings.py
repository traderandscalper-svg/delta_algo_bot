from enum import Enum
from pathlib import Path
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    DEMO = "DEMO"
    PAPER = "PAPER"
    LIVE = "LIVE"
    BACKTEST = "BACKTEST"
    TRAINING = "TRAINING"


class Settings(BaseSettings):
    """
    Central configuration for the trading engine.

    No trading subsystem should read .env directly.
    Everything should go through this settings object.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    trading_mode: TradingMode = TradingMode.DEMO

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    delta_rest_url: str = ""
    delta_public_ws_url: str = ""
    delta_private_ws_url: str = ""

    delta_api_key: str = ""
    delta_api_secret: str = ""

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    max_risk_per_trade: float = Field(
        default=0.0025,
        gt=0,
        le=0.01,
    )

    max_leverage: int = Field(
        default=1,
        ge=1,
        le=5,
    )

    max_daily_loss: float = Field(
        default=0.02,
        gt=0,
        le=0.25,
    )

    max_trades_per_day: int = Field(
        default=100,
        ge=1,
    )

    paper_starting_equity: float = Field(
        default=10_000.0,
        gt=0,
    )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    enable_demo_trading: bool = False

    enable_live_trading: bool = False

    heartbeat_interval_seconds: int = Field(
        default=10,
        ge=1,
    )

    stale_data_timeout_seconds: int = Field(
        default=5,
        ge=1,
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log_level: str = "INFO"

    log_directory: str = "logs"

    # Persistent market/ML/runtime data. On Windows this defaults to E:\\delta_algo_bot_data.
    data_directory: str = r"E:\\delta_algo_bot_data"

    @field_validator("delta_rest_url", "delta_public_ws_url", "delta_private_ws_url")
    @classmethod
    def remove_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    def validate_startup(self) -> None:
        """
        Validate settings before the engine is allowed to start.
        """

        if self.trading_mode in {TradingMode.DEMO, TradingMode.PAPER}:
            if not self.delta_rest_url:
                raise ValueError("DELTA_REST_URL is required in DEMO/PAPER mode.")

            if not self.delta_public_ws_url:
                raise ValueError(
                    "DELTA_PUBLIC_WS_URL is required in DEMO/PAPER mode."
                )

        if self.trading_mode == TradingMode.DEMO:
            if not self.delta_private_ws_url:
                raise ValueError(
                    "DELTA_PRIVATE_WS_URL is required in DEMO mode."
                )
            if not self.delta_api_key:
                raise ValueError(
                    "DELTA_API_KEY is required in DEMO mode."
                )
            if not self.delta_api_secret:
                raise ValueError(
                    "DELTA_API_SECRET is required in DEMO mode."
                )

        if self.trading_mode == TradingMode.LIVE:
            if not self.enable_live_trading:
                raise ValueError(
                    "LIVE mode requested but ENABLE_LIVE_TRADING=false."
                )
            if not self.delta_rest_url or "api.india.delta.exchange" not in self.delta_rest_url:
                raise ValueError("LIVE mode requires Delta India production REST URL.")
            if not self.delta_public_ws_url or "public-socket.india.delta.exchange" not in self.delta_public_ws_url:
                raise ValueError("LIVE mode requires Delta India production public WebSocket URL.")
            if not self.delta_private_ws_url or "socket.india.delta.exchange" not in self.delta_private_ws_url:
                raise ValueError("LIVE mode requires Delta India production private WebSocket URL.")
            if not self.delta_api_key or not self.delta_api_secret:
                raise ValueError("LIVE mode requires DELTA_API_KEY and DELTA_API_SECRET.")

        if self.max_risk_per_trade > self.max_daily_loss:
            raise ValueError(
                "MAX_RISK_PER_TRADE cannot exceed MAX_DAILY_LOSS."
            )

        Path(self.data_directory).mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(self.log_directory).mkdir(
            parents=True,
            exist_ok=True,
        )