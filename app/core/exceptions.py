class TradingEngineError(Exception):
    """Base exception for the trading engine."""


class ConfigurationError(TradingEngineError):
    """Invalid configuration."""


class SafetyError(TradingEngineError):
    """Safety system prevented an operation."""


class ExchangeConnectionError(TradingEngineError):
    """Exchange connection failed."""


class StateMismatchError(TradingEngineError):
    """Internal state differs from exchange state."""


class EmergencyStopError(TradingEngineError):
    """Emergency stop activated."""