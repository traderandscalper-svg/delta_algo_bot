# Delta Algo Bot

Advanced Delta Exchange India DEMO/TESTNET algorithmic trading bot.

## Safety

- Default mode is DEMO.
- Live order execution is disabled.
- Paper execution is used by the trading engine.
- Never commit real API credentials.
- The included .env contains placeholders only.

## Features

- Real Delta REST authentication
- Real Delta public WebSocket market data
- L1 order book and trade-flow features
- Multi-timeframe strategy confirmation
- Adaptive strategy weighting
- Risk management and portfolio protection
- Paper execution with bid/ask-aware fills
- Historical market-data storage
- Intrasecond ML feature generation
- Multi-horizon target construction
- Dataset alignment and quality checks
- Random Forest ML model training
- Chronological holdout evaluation
- ML-driven historical backtesting
- Offline ML confirmation in the paper trading engine
- Watchdog and stale-data protection
- Portfolio and strategy analytics

## Windows setup

Open PowerShell in this folder:

    py -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Then edit .env and replace YOUR_DELTA_API_KEY and YOUR_DELTA_API_SECRET with your Delta Exchange India DEMO/TESTNET credentials.

## Collect real market data

    .\.venv\Scripts\python.exe -m app.main

Stop it with Ctrl+C after collecting enough data.

## Build ML + backtest pipeline

    .\.venv\Scripts\python.exe -m app.main --research

The research pipeline performs:

1. Intrasecond feature construction
2. Multi-horizon target construction
3. Dataset alignment/leakage checks
4. ML training
5. Chronological holdout backtest

## Individual commands

    .\.venv\Scripts\python.exe -m app.ml.intrasecond_features
    .\.venv\Scripts\python.exe -m app.ml.multi_horizon_dataset
    .\.venv\Scripts\python.exe -m app.ml.dataset_alignment
    .\.venv\Scripts\python.exe -m app.ml.trainer
    .\.venv\Scripts\python.exe -m app.backtest.engine

## Tests

    .\.venv\Scripts\python.exe -m pytest -q

## Important

The ML/backtest results are only meaningful after collecting a sufficiently large and representative real-market dataset. A few seconds of data is not enough for reliable model training or strategy validation.

The project is designed for research and DEMO/PAPER execution. Do not enable live trading until the complete system has been independently validated.
