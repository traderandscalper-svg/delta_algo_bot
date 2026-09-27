"""
Build an ML-ready dataset from real Delta Exchange
market-data JSONL files.

Phase 6:
    Historical Data & ML Dataset Pipeline

Primary target price:
    L1 MID PRICE = (BID + ASK) / 2

Important:
    - L1 mid-price is the primary target/ML price.
    - Trade price is retained separately as a feature.
    - FeatureEngine receives L1 mid-price as its price.
    - Future returns are calculated from future L1 mid-price.
    - Historical validation metadata is NOT trusted.
    - Sampling prevents future-event leakage.
    - Ticker, L1 and trade data are reconstructed from
      the historical JSONL records.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from app.ml.features import FeatureEngine
from app.ml.labels import calculate_future_return


logger = logging.getLogger("MLDatasetBuilder")


class MLDatasetBuilder:
    """
    Convert persistent Delta Exchange JSONL market data
    into a regular supervised-learning time series.

    PRIMARY TARGET PRICE:

        mid_price = (bid + ask) / 2

    Trade price is kept separately and is never used as
    the primary future-return target when valid L1 data exists.
    """

    def __init__(
        self,
        horizon_seconds: float = 10.0,
        label_threshold: Optional[float] = None,
        sample_interval_seconds: float = 1.0,
        target_directional_ratio: float = 0.30,
    ) -> None:

        self.horizon_seconds = float(horizon_seconds)

        self.label_threshold = (
            None
            if label_threshold is None
            else float(label_threshold)
        )

        self.sample_interval_seconds = float(
            sample_interval_seconds
        )

        self.target_directional_ratio = float(
            target_directional_ratio
        )

        if self.horizon_seconds <= 0:
            raise ValueError(
                "horizon_seconds must be greater than 0."
            )

        if self.sample_interval_seconds <= 0:
            raise ValueError(
                "sample_interval_seconds must be greater than 0."
            )

        if not (
            0.05
            <= self.target_directional_ratio
            <= 0.80
        ):
            raise ValueError(
                "target_directional_ratio must be between "
                "0.05 and 0.80."
            )

    # ============================================================
    # BUILD
    # ============================================================

    def build(
        self,
        input_files: Iterable[Path],
        output_file: Path,
    ) -> pd.DataFrame:

        observations: List[Dict[str, Any]] = []

        for input_file in input_files:

            logger.info(
                "Reading market data | file=%s",
                input_file,
            )

            observations.extend(
                self._read_file(input_file)
            )

        if not observations:
            raise RuntimeError(
                "No usable historical market observations found."
            )

        observations.sort(
            key=lambda item: (
                item["symbol"],
                item["timestamp"],
                item["event_order"],
            )
        )

        logger.info(
            "Total historical observations: %d",
            len(observations),
        )

        samples = self._create_feature_samples(
            observations
        )

        logger.info(
            "Feature samples created: %d",
            len(samples),
        )

        if not samples:
            raise RuntimeError(
                "Unable to create ML feature samples."
            )

        df = pd.DataFrame(samples)

        df = df.sort_values(
            by=[
                "symbol",
                "timestamp_us",
            ]
        ).reset_index(drop=True)

        # --------------------------------------------------------
        # Safety check:
        # The primary target price must be the L1 mid-price.
        # --------------------------------------------------------

        if "mid_price" not in df.columns:
            raise RuntimeError(
                "mid_price column was not created."
            )

        if "price" not in df.columns:
            raise RuntimeError(
                "Primary price column was not created."
            )

        mid_available = int(
            df["mid_price"].notna().sum()
        )

        if mid_available != len(df):
            raise RuntimeError(
                "Some samples do not have an L1 mid-price. "
                "The primary target price must be L1 mid-price."
            )

        # Explicitly force primary price to L1 mid-price.
        df["price"] = (
            df["mid_price"]
            .astype(float)
        )

        # --------------------------------------------------------
        # Diagnostics BEFORE future-return calculation.
        # --------------------------------------------------------

        self._log_mid_price_diagnostics(
            df
        )

        # --------------------------------------------------------
        # Future returns.
        #
        # IMPORTANT:
        # Future return is calculated from MID PRICE,
        # not trade price.
        # --------------------------------------------------------

        df = self._calculate_future_returns(
            df
        )

        if df.empty:
            raise RuntimeError(
                "No samples with valid future returns."
            )

        # --------------------------------------------------------
        # Return distribution.
        # --------------------------------------------------------

        self._print_threshold_analysis(
            df
        )

        # --------------------------------------------------------
        # Threshold.
        # --------------------------------------------------------

        threshold = self._select_label_threshold(
            df
        )

        logger.info(
            "Selected label threshold: %.8f (%.4f%%)",
            threshold,
            threshold * 100.0,
        )

        # --------------------------------------------------------
        # Labels.
        # --------------------------------------------------------

        df = self._apply_labels(
            df,
            threshold,
        )

        if df.empty:
            raise RuntimeError(
                "No labelled ML samples were created."
            )

        df["label_threshold"] = float(
            threshold
        )

        df["label_horizon_seconds"] = float(
            self.horizon_seconds
        )

        df["sample_interval_seconds"] = float(
            self.sample_interval_seconds
        )

        # --------------------------------------------------------
        # Preserve existing dataset before replacing it.
        # --------------------------------------------------------

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_file.exists():

            backup_file = (
                output_file.parent
                / "training_dataset_previous.csv"
            )

            try:

                shutil.copy2(
                    output_file,
                    backup_file,
                )

                logger.info(
                    "Previous dataset backed up | file=%s",
                    backup_file,
                )

            except OSError as exc:

                logger.warning(
                    "Could not backup previous dataset | error=%s",
                    exc,
                )

        # --------------------------------------------------------
        # Save.
        # --------------------------------------------------------

        df.to_csv(
            output_file,
            index=False,
        )

        logger.info(
            "ML dataset created | rows=%d | columns=%d | file=%s",
            len(df),
            len(df.columns),
            output_file,
        )

        self._log_dataset_summary(
            df,
            threshold,
        )

        return df

    # ============================================================
    # READ JSONL
    # ============================================================

    def _read_file(
        self,
        path: Path,
    ) -> List[Dict[str, Any]]:

        result: List[Dict[str, Any]] = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    record = json.loads(
                        line
                    )

                except json.JSONDecodeError:

                    logger.warning(
                        "Invalid JSON | file=%s | line=%d",
                        path,
                        line_number,
                    )

                    continue

                observation = (
                    self._record_to_observation(
                        record,
                        line_number,
                    )
                )

                if observation is not None:
                    result.append(
                        observation
                    )

        return result

    # ============================================================
    # RECORD -> HISTORICAL OBSERVATION
    # ============================================================

    def _record_to_observation(
        self,
        record: Dict[str, Any],
        line_number: int = 0,
    ) -> Optional[Dict[str, Any]]:

        if not isinstance(
            record,
            dict,
        ):
            return None

        event_type = record.get(
            "event_type"
        )

        if event_type not in {
            "ticker",
            "ob_l1",
            "trades",
        }:
            return None

        normalized = record.get(
            "normalized"
        )

        if not isinstance(
            normalized,
            dict,
        ):
            normalized = {}

        # --------------------------------------------------------
        # IMPORTANT:
        # Do NOT trust historical "valid".
        # --------------------------------------------------------

        raw_message = record.get(
            "message"
        )

        if not isinstance(
            raw_message,
            dict,
        ):
            raw_message = {}

        payload = self._extract_payload(
            raw_message
        )

        # --------------------------------------------------------
        # Reconstruct event.
        # --------------------------------------------------------

        if event_type == "trades":

            event = (
                self._normalize_historical_trade(
                    normalized,
                    payload,
                    raw_message,
                )
            )

        elif event_type == "ob_l1":

            event = (
                self._normalize_historical_l1(
                    normalized,
                    payload,
                    raw_message,
                )
            )

        elif event_type == "ticker":

            event = (
                self._normalize_historical_ticker(
                    normalized,
                    payload,
                    raw_message,
                )
            )

        else:
            event = None

        if event is None:
            return None

        # --------------------------------------------------------
        # Timestamp.
        # --------------------------------------------------------

        timestamp_candidates = [
            record.get("exchange_timestamp"),
            event.get("exchange_timestamp"),
            normalized.get("exchange_timestamp"),
            payload.get("ts"),
            payload.get("t"),
            raw_message.get("ts"),
            raw_message.get("t"),
            record.get("received_at_us"),
        ]

        timestamp_us = None

        for candidate in timestamp_candidates:

            timestamp_us = (
                self._to_microseconds(
                    candidate
                )
            )

            if timestamp_us is not None:
                break

        if timestamp_us is None:
            return None

        symbol = (
            event.get("symbol")
            or record.get("symbol")
            or normalized.get("symbol")
            or payload.get("sy")
            or payload.get("symbol")
            or raw_message.get("sy")
            or raw_message.get("symbol")
        )

        if not symbol:
            return None

        event["symbol"] = str(
            symbol
        )

        return {
            "timestamp": int(
                timestamp_us
            ),
            "symbol": str(
                symbol
            ),
            "event_type": event_type,
            "normalized": event,
            "event_order": line_number,
        }

    # ============================================================
    # RAW PAYLOAD EXTRACTION
    # ============================================================

    @staticmethod
    def _extract_payload(
        message: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not isinstance(
            message,
            dict,
        ):
            return {}

        data = message.get(
            "d"
        )

        if isinstance(
            data,
            list,
        ):

            if not data:
                return {}

            first = data[0]

            if isinstance(
                first,
                dict,
            ):
                return first

            return {}

        if isinstance(
            data,
            dict,
        ):
            return data

        return message

    # ============================================================
    # HISTORICAL TRADE NORMALIZATION
    # ============================================================

    def _normalize_historical_trade(
        self,
        normalized: Dict[str, Any],
        payload: Dict[str, Any],
        message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:

        price = self._first_numeric(
            normalized.get("price"),
            payload.get("p"),
            payload.get("price"),
            payload.get("trade_price"),
            message.get("p"),
            message.get("price"),
        )

        size = self._first_numeric(
            normalized.get("size"),
            payload.get("s"),
            payload.get("size"),
            payload.get("quantity"),
            payload.get("trade_size"),
            payload.get("qty"),
            payload.get("q"),
            payload.get("volume"),
            message.get("s"),
            message.get("size"),
        )

        if (
            price is None
            or price <= 0
        ):
            return None

        if (
            size is None
            or size <= 0
        ):
            return None

        symbol = (
            normalized.get("symbol")
            or payload.get("sy")
            or payload.get("symbol")
            or message.get("sy")
            or message.get("symbol")
        )

        if not symbol:
            return None

        side = (
            normalized.get("side")
            or payload.get("side")
        )

        trade_id = (
            normalized.get("trade_id")
            or payload.get("trade_id")
            or payload.get("id")
        )

        exchange_timestamp = (
            normalized.get(
                "exchange_timestamp"
            )
            or payload.get("ts")
            or payload.get("t")
            or message.get("ts")
            or message.get("t")
        )

        return {
            "price": float(
                price
            ),
            "size": float(
                size
            ),
            "side": side,
            "trade_id": trade_id,
            "symbol": str(
                symbol
            ),
            "exchange_timestamp": (
                exchange_timestamp
            ),
        }

    # ============================================================
    # HISTORICAL L1 NORMALIZATION
    # ============================================================

    def _normalize_historical_l1(
        self,
        normalized: Dict[str, Any],
        payload: Dict[str, Any],
        message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:

        # Stored normalized L1 is preferred because it has
        # already been parsed successfully by the collector.

        bid = self._first_numeric(
            normalized.get("bid"),
            payload.get("bp"),
            payload.get("bid"),
            payload.get("bid_price"),
            message.get("bp"),
        )

        ask = self._first_numeric(
            normalized.get("ask"),
            payload.get("ap"),
            payload.get("ask"),
            payload.get("ask_price"),
            message.get("ap"),
        )

        bid_size = self._first_numeric(
            normalized.get("bid_size"),
            payload.get("bs"),
            payload.get("bid_size"),
            payload.get("bid_quantity"),
            message.get("bs"),
        )

        ask_size = self._first_numeric(
            normalized.get("ask_size"),
            payload.get("as"),
            payload.get("ask_size"),
            payload.get("ask_quantity"),
            message.get("as"),
        )

        if (
            bid is None
            and ask is None
        ):
            return None

        symbol = (
            normalized.get("symbol")
            or payload.get("sy")
            or payload.get("symbol")
            or message.get("sy")
            or message.get("symbol")
        )

        if not symbol:
            return None

        return {
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "symbol": str(
                symbol
            ),
            "exchange_timestamp": (
                normalized.get(
                    "exchange_timestamp"
                )
                or payload.get("ts")
                or payload.get("t")
                or message.get("ts")
                or message.get("t")
            ),
        }

    # ============================================================
    # HISTORICAL TICKER NORMALIZATION
    # ============================================================

    def _normalize_historical_ticker(
        self,
        normalized: Dict[str, Any],
        payload: Dict[str, Any],
        message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:

        symbol = (
            normalized.get("symbol")
            or payload.get("sy")
            or payload.get("symbol")
            or message.get("sy")
            or message.get("symbol")
        )

        if not symbol:
            return None

        last_price = self._first_numeric(
            normalized.get("last_price"),
            normalized.get("price"),
            payload.get("last_price"),
            payload.get("last"),
            payload.get("lp"),
        )

        mark_price = self._first_numeric(
            normalized.get("mark_price"),
            payload.get("mark_price"),
            payload.get("mark"),
            payload.get("m"),
        )

        bid = self._first_numeric(
            normalized.get("bid"),
            payload.get("bp"),
            payload.get("bid"),
        )

        ask = self._first_numeric(
            normalized.get("ask"),
            payload.get("ap"),
            payload.get("ask"),
        )

        bid_size = self._first_numeric(
            normalized.get("bid_size"),
            payload.get("bs"),
        )

        ask_size = self._first_numeric(
            normalized.get("ask_size"),
            payload.get("as"),
        )

        open_price = self._first_numeric(
            normalized.get("open"),
            payload.get("open"),
        )

        high_price = self._first_numeric(
            normalized.get("high"),
            payload.get("high"),
        )

        low_price = self._first_numeric(
            normalized.get("low"),
            payload.get("low"),
        )

        open_interest = self._first_numeric(
            normalized.get("open_interest"),
            payload.get("oi"),
        )

        open_interest_change = self._first_numeric(
            normalized.get(
                "open_interest_change"
            ),
            payload.get("oi_change"),
        )

        change_24h = self._first_numeric(
            normalized.get("change_24h"),
            payload.get("m24hc"),
        )

        if last_price is None:

            last_price = (
                self._extract_ticker_ohlc_close(
                    payload
                )
            )

        if (
            last_price is None
            and mark_price is None
            and bid is None
            and ask is None
        ):
            return None

        return {
            "last_price": last_price,
            "mark_price": mark_price,
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "open_interest": open_interest,
            "open_interest_change": (
                open_interest_change
            ),
            "change_24h": change_24h,
            "symbol": str(
                symbol
            ),
            "exchange_timestamp": (
                normalized.get(
                    "exchange_timestamp"
                )
                or payload.get("ts")
                or payload.get("t")
                or message.get("ts")
                or message.get("t")
            ),
        }

    # ============================================================
    # TICKER OHLC CLOSE
    # ============================================================

    @staticmethod
    def _extract_ticker_ohlc_close(
        payload: Dict[str, Any],
    ) -> Optional[float]:

        ohlc = payload.get(
            "ohlc"
        )

        if isinstance(
            ohlc,
            dict,
        ):

            for key in (
                "close",
                "c",
            ):

                value = ohlc.get(
                    key
                )

                numeric = (
                    MLDatasetBuilder._safe_float(
                        value
                    )
                )

                if numeric is not None:
                    return numeric

        if isinstance(
            ohlc,
            list,
        ):

            if len(ohlc) >= 4:

                numeric = (
                    MLDatasetBuilder._safe_float(
                        ohlc[3]
                    )
                )

                if numeric is not None:
                    return numeric

        return None

    # ============================================================
    # NUMERIC HELPERS
    # ============================================================

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> Optional[float]:

        try:

            if value is None:
                return None

            result = float(
                value
            )

            if not np.isfinite(
                result
            ):
                return None

            return result

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _first_numeric(
        *values: Any,
    ) -> Optional[float]:

        for value in values:

            result = (
                MLDatasetBuilder._safe_float(
                    value
                )
            )

            if result is not None:
                return result

        return None

    # ============================================================
    # TIMESTAMP NORMALIZATION
    # ============================================================

    @staticmethod
    def _to_microseconds(
        value: Any,
    ) -> Optional[int]:

        try:

            timestamp = int(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

        if timestamp <= 0:
            return None

        # Seconds.
        if timestamp < 100_000_000_000:
            return timestamp * 1_000_000

        # Milliseconds.
        if timestamp < 100_000_000_000_000:
            return timestamp * 1_000

        # Microseconds.
        if timestamp < 100_000_000_000_000_000:
            return timestamp

        # Nanoseconds.
        return timestamp // 1_000

    # ============================================================
    # CREATE FEATURE SAMPLES
    # ============================================================

    def _create_feature_samples(
        self,
        observations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        observations_by_symbol: Dict[
            str,
            List[Dict[str, Any]],
        ] = {}

        for observation in observations:

            symbol = str(
                observation["symbol"]
            )

            observations_by_symbol.setdefault(
                symbol,
                [],
            ).append(
                observation
            )

        all_samples: List[
            Dict[str, Any]
        ] = []

        for (
            symbol,
            symbol_observations,
        ) in observations_by_symbol.items():

            logger.info(
                "Building samples | symbol=%s | observations=%d",
                symbol,
                len(symbol_observations),
            )

            symbol_samples = (
                self._create_symbol_samples(
                    symbol,
                    symbol_observations,
                )
            )

            all_samples.extend(
                symbol_samples
            )

        return all_samples

    # ============================================================
    # SYMBOL SAMPLE CREATION
    # ============================================================

    def _create_symbol_samples(
        self,
        symbol: str,
        observations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        observations = sorted(
            observations,
            key=lambda item: (
                item["timestamp"],
                item["event_order"],
            ),
        )

        feature_engine = FeatureEngine()

        latest: Dict[str, Any] = {
            "price": None,

            "mark_price": None,

            "bid": None,
            "ask": None,
            "bid_size": None,
            "ask_size": None,

            "open": None,
            "high": None,
            "low": None,

            "open_interest": None,
            "open_interest_change": None,
            "change_24h": None,

            "trade_price": None,
            "trade_size": None,
            "trade_side": None,

            "mid_price": None,
            "ticker_price": None,
        }

        snapshots: List[
            Dict[str, Any]
        ] = []

        # --------------------------------------------------------
        # Process events chronologically.
        # --------------------------------------------------------

        for observation in observations:

            event_type = observation[
                "event_type"
            ]

            normalized = observation[
                "normalized"
            ]

            timestamp = int(
                observation["timestamp"]
            )

            # ----------------------------------------------------
            # TICKER
            # ----------------------------------------------------

            if event_type == "ticker":

                ticker_price = (
                    self._safe_float(
                        normalized.get(
                            "last_price"
                        )
                    )
                )

                mark_price = (
                    self._safe_float(
                        normalized.get(
                            "mark_price"
                        )
                    )
                )

                if ticker_price is not None:

                    latest[
                        "ticker_price"
                    ] = ticker_price

                if mark_price is not None:

                    latest[
                        "mark_price"
                    ] = mark_price

                for key in (
                    "bid",
                    "ask",
                    "bid_size",
                    "ask_size",
                    "open",
                    "high",
                    "low",
                    "open_interest",
                    "open_interest_change",
                    "change_24h",
                ):

                    value = (
                        self._safe_float(
                            normalized.get(
                                key
                            )
                        )
                    )

                    if value is not None:
                        latest[key] = value

            # ----------------------------------------------------
            # L1 ORDER BOOK
            # ----------------------------------------------------

            elif event_type == "ob_l1":

                for key in (
                    "bid",
                    "ask",
                    "bid_size",
                    "ask_size",
                ):

                    value = (
                        self._safe_float(
                            normalized.get(
                                key
                            )
                        )
                    )

                    if value is not None:
                        latest[key] = value

            # ----------------------------------------------------
            # TRADES
            # ----------------------------------------------------

            elif event_type == "trades":

                trade_price = (
                    self._safe_float(
                        normalized.get(
                            "price"
                        )
                    )
                )

                trade_size = (
                    self._safe_float(
                        normalized.get(
                            "size"
                        )
                    )
                )

                if trade_price is not None:

                    latest[
                        "trade_price"
                    ] = trade_price

                if trade_size is not None:

                    latest[
                        "trade_size"
                    ] = trade_size

                if (
                    normalized.get(
                        "side"
                    )
                    is not None
                ):

                    latest[
                        "trade_side"
                    ] = normalized.get(
                        "side"
                    )

            # ----------------------------------------------------
            # Calculate L1 mid-price.
            # ----------------------------------------------------

            bid = self._safe_float(
                latest.get("bid")
            )

            ask = self._safe_float(
                latest.get("ask")
            )

            if (
                bid is not None
                and ask is not None
                and bid > 0
                and ask > 0
                and ask >= bid
            ):

                latest[
                    "mid_price"
                ] = (
                    bid + ask
                ) / 2.0

            # ----------------------------------------------------
            # IMPORTANT:
            #
            # L1 MID PRICE IS NOW THE PRIMARY PRICE.
            #
            # We do NOT use trade price here.
            #
            # If no valid L1 mid exists, the event cannot create
            # a target sample.
            # ----------------------------------------------------

            mid_price = self._safe_float(
                latest.get(
                    "mid_price"
                )
            )

            if (
                mid_price is None
                or mid_price <= 0
            ):
                continue

            latest[
                "price"
            ] = mid_price

            # ----------------------------------------------------
            # Snapshot.
            # ----------------------------------------------------

            snapshots.append(
                {
                    "timestamp": timestamp,
                    "state": dict(
                        latest
                    ),
                }
            )

        if not snapshots:
            return []

        # --------------------------------------------------------
        # Regular 1-second grid.
        # --------------------------------------------------------

        first_timestamp = int(
            snapshots[0][
                "timestamp"
            ]
        )

        last_timestamp = int(
            snapshots[-1][
                "timestamp"
            ]
        )

        sample_interval_us = int(
            self.sample_interval_seconds
            * 1_000_000
        )

        if sample_interval_us <= 0:
            return []

        next_sample_timestamp = (
            first_timestamp
        )

        snapshot_index = 0

        latest_state: Optional[
            Dict[str, Any]
        ] = None

        samples: List[
            Dict[str, Any]
        ] = []

        while (
            next_sample_timestamp
            <= last_timestamp
        ):

            # ----------------------------------------------------
            # Only events <= sample timestamp.
            #
            # This prevents future-event leakage.
            # ----------------------------------------------------

            while (
                snapshot_index
                < len(snapshots)
                and int(
                    snapshots[
                        snapshot_index
                    ][
                        "timestamp"
                    ]
                )
                <= next_sample_timestamp
            ):

                latest_state = dict(
                    snapshots[
                        snapshot_index
                    ][
                        "state"
                    ]
                )

                snapshot_index += 1

            if latest_state is None:

                next_sample_timestamp += (
                    sample_interval_us
                )

                continue

            mid_price = self._safe_float(
                latest_state.get(
                    "mid_price"
                )
            )

            if (
                mid_price is None
                or mid_price <= 0
            ):

                next_sample_timestamp += (
                    sample_interval_us
                )

                continue

            # ----------------------------------------------------
            # PRIMARY PRICE = L1 MID
            # ----------------------------------------------------

            latest_state[
                "price"
            ] = mid_price

            feature_input = dict(
                latest_state
            )

            try:

                features = (
                    feature_engine.update(
                        feature_input
                    )
                )

            except Exception as exc:

                logger.warning(
                    "Feature calculation failed | "
                    "symbol=%s | timestamp=%d | error=%s",
                    symbol,
                    next_sample_timestamp,
                    exc,
                )

                next_sample_timestamp += (
                    sample_interval_us
                )

                continue

            if not features:

                next_sample_timestamp += (
                    sample_interval_us
                )

                continue

            # ----------------------------------------------------
            # Create sample.
            # ----------------------------------------------------

            sample = {
                "timestamp_us": int(
                    next_sample_timestamp
                ),

                "timestamp_seconds": (
                    next_sample_timestamp
                    / 1_000_000.0
                ),

                "symbol": symbol,

                # =================================================
                # PRIMARY ML TARGET PRICE
                # =================================================
                "price": float(
                    mid_price
                ),

                # =================================================
                # MARKET DATA FEATURES
                # =================================================

                # Actual executed market trade price.
                "trade_price": (
                    self._safe_float(
                        latest_state.get(
                            "trade_price"
                        )
                    )
                ),

                # L1 target/mark-to-market price.
                "mid_price": float(
                    mid_price
                ),

                "ticker_price": (
                    self._safe_float(
                        latest_state.get(
                            "ticker_price"
                        )
                    )
                ),

                "mark_price": (
                    self._safe_float(
                        latest_state.get(
                            "mark_price"
                        )
                    )
                ),

                "bid": (
                    self._safe_float(
                        latest_state.get(
                            "bid"
                        )
                    )
                ),

                "ask": (
                    self._safe_float(
                        latest_state.get(
                            "ask"
                        )
                    )
                ),

                "bid_size": (
                    self._safe_float(
                        latest_state.get(
                            "bid_size"
                        )
                    )
                ),

                "ask_size": (
                    self._safe_float(
                        latest_state.get(
                            "ask_size"
                        )
                    )
                ),

                "trade_size": (
                    self._safe_float(
                        latest_state.get(
                            "trade_size"
                        )
                    )
                ),

                "open_interest": (
                    self._safe_float(
                        latest_state.get(
                            "open_interest"
                        )
                    )
                ),

                "open_interest_change": (
                    self._safe_float(
                        latest_state.get(
                            "open_interest_change"
                        )
                    )
                ),

                "change_24h": (
                    self._safe_float(
                        latest_state.get(
                            "change_24h"
                        )
                    )
                ),
            }

            # ----------------------------------------------------
            # FeatureEngine-generated columns.
            # ----------------------------------------------------

            sample.update(
                features
            )

            # ----------------------------------------------------
            # HARD SAFETY:
            #
            # FeatureEngine must never overwrite our primary
            # target price with trade price.
            # ----------------------------------------------------

            sample[
                "price"
            ] = float(
                mid_price
            )

            sample[
                "mid_price"
            ] = float(
                mid_price
            )

            samples.append(
                sample
            )

            next_sample_timestamp += (
                sample_interval_us
            )

        return samples

    # ============================================================
    # MID-PRICE DIAGNOSTICS
    # ============================================================

    def _log_mid_price_diagnostics(
        self,
        df: pd.DataFrame,
    ) -> None:

        if df.empty:
            return

        total = len(df)

        mid = (
            df["mid_price"]
            .astype(float)
        )

        unique_mid = int(
            mid.nunique()
        )

        unchanged = int(
            (
                mid.diff()
                .fillna(0.0)
                == 0.0
            ).sum()
        )

        changed = (
            total
            - unchanged
        )

        logger.info(
            "PRIMARY TARGET = L1 MID PRICE"
        )

        logger.info(
            "L1 mid diagnostics | "
            "unique_mid_prices=%d | "
            "changed_samples=%d/%d (%.2f%%) | "
            "unchanged_samples=%d/%d (%.2f%%)",
            unique_mid,
            changed,
            total,
            changed / max(total, 1) * 100.0,
            unchanged,
            total,
            unchanged / max(total, 1) * 100.0,
        )

        if (
            "trade_price"
            in df.columns
        ):

            trade = (
                df[
                    "trade_price"
                ]
                .astype(float)
            )

            trade_unique = int(
                trade.nunique(
                    dropna=True
                )
            )

            logger.info(
                "Trade-price diagnostics | "
                "unique_trade_prices=%d",
                trade_unique,
            )

        # --------------------------------------------------------
        # Show actual first/last target movement.
        # --------------------------------------------------------

        if total >= 2:

            first_mid = float(
                mid.iloc[0]
            )

            last_mid = float(
                mid.iloc[-1]
            )

            total_return = (
                (
                    last_mid
                    - first_mid
                )
                / first_mid
            )

            logger.info(
                "L1 mid total movement | "
                "first=%.8f | last=%.8f | return=%.8f (%.4f%%)",
                first_mid,
                last_mid,
                total_return,
                total_return * 100.0,
            )

    # ============================================================
    # FUTURE RETURNS
    # ============================================================

    def _calculate_future_returns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if df.empty:
            return df

        output_frames: List[
            pd.DataFrame
        ] = []

        horizon_us = int(
            self.horizon_seconds
            * 1_000_000
        )

        for symbol, group in df.groupby(
            "symbol",
            sort=False,
        ):

            group = (
                group
                .sort_values(
                    "timestamp_us"
                )
                .reset_index(
                    drop=True
                )
            )

            timestamps = (
                group[
                    "timestamp_us"
                ].to_numpy()
            )

            # ----------------------------------------------------
            # CRITICAL:
            #
            # Future target uses MID PRICE.
            # ----------------------------------------------------

            prices = (
                group[
                    "mid_price"
                ]
                .astype(float)
                .to_numpy()
            )

            future_returns: List[
                Optional[float]
            ] = []

            future_prices: List[
                Optional[float]
            ] = []

            future_timestamps: List[
                Optional[int]
            ] = []

            future_index = 1

            for index in range(
                len(group)
            ):

                current_timestamp = int(
                    timestamps[index]
                )

                target_timestamp = (
                    current_timestamp
                    + horizon_us
                )

                if future_index <= index:
                    future_index = (
                        index + 1
                    )

                while (
                    future_index
                    < len(group)
                    and int(
                        timestamps[
                            future_index
                        ]
                    )
                    < target_timestamp
                ):

                    future_index += 1

                if (
                    future_index
                    >= len(group)
                ):

                    future_returns.append(
                        None
                    )

                    future_prices.append(
                        None
                    )

                    future_timestamps.append(
                        None
                    )

                    continue

                current_price = float(
                    prices[index]
                )

                future_price = float(
                    prices[
                        future_index
                    ]
                )

                future_return = (
                    calculate_future_return(
                        current_price,
                        future_price,
                    )
                )

                if future_return is None:

                    future_returns.append(
                        None
                    )

                else:

                    future_returns.append(
                        float(
                            future_return
                        )
                    )

                future_prices.append(
                    future_price
                )

                future_timestamps.append(
                    int(
                        timestamps[
                            future_index
                        ]
                    )
                )

            group[
                "future_mid_price"
            ] = future_prices

            group[
                "future_timestamp_us"
            ] = future_timestamps

            group[
                "future_return"
            ] = future_returns

            group = group.dropna(
                subset=[
                    "future_return",
                    "future_mid_price",
                ]
            ).copy()

            output_frames.append(
                group
            )

        if not output_frames:
            return df.iloc[
                0:0
            ].copy()

        return pd.concat(
            output_frames,
            ignore_index=True,
        )

    # ============================================================
    # RETURN / THRESHOLD ANALYSIS
    # ============================================================

    def _print_threshold_analysis(
        self,
        df: pd.DataFrame,
    ) -> None:

        returns = (
            df[
                "future_return"
            ]
            .astype(float)
        )

        absolute_returns = (
            returns.abs()
        )

        print()

        print(
            "============================================================"
        )

        print(
            "       10-SECOND L1 MID-PRICE RETURN ANALYSIS"
        )

        print(
            "============================================================"
        )

        print(
            f"Samples with future return: {len(df):,}"
        )

        print()

        print(
            "Threshold      DOWN        FLAT        UP      Directional"
        )

        print(
            "------------------------------------------------------------"
        )

        candidate_thresholds = [
            0.00001,
            0.00002,
            0.00003,
            0.00005,
            0.00007,
            0.00010,
            0.00015,
            0.00020,
            0.00030,
            0.00050,
        ]

        for threshold in candidate_thresholds:

            down = int(
                (
                    returns
                    <= -threshold
                ).sum()
            )

            up = int(
                (
                    returns
                    >= threshold
                ).sum()
            )

            flat = int(
                len(returns)
                - down
                - up
            )

            directional = (
                (down + up)
                / max(
                    len(returns),
                    1,
                )
            )

            print(
                f"{threshold * 100:>7.4f}%     "
                f"{down:>6,}      "
                f"{flat:>6,}      "
                f"{up:>6,}       "
                f"{directional * 100:>7.2f}%"
            )

        print()

        # --------------------------------------------------------
        # Exact return distribution.
        # --------------------------------------------------------

        print(
            "10-second return statistics:"
        )

        print(
            f"  Min: {returns.min():.8f} "
            f"({returns.min() * 100:.4f}%)"
        )

        print(
            f"  Max: {returns.max():.8f} "
            f"({returns.max() * 100:.4f}%)"
        )

        print(
            f"  Mean: {returns.mean():.8f} "
            f"({returns.mean() * 100:.4f}%)"
        )

        print(
            f"  Median: {returns.median():.8f} "
            f"({returns.median() * 100:.4f}%)"
        )

        print()

        zero_returns = int(
            (
                returns
                == 0.0
            ).sum()
        )

        positive_returns = int(
            (
                returns
                > 0.0
            ).sum()
        )

        negative_returns = int(
            (
                returns
                < 0.0
            ).sum()
        )

        print(
            "Return direction:"
        )

        print(
            f"  Positive: {positive_returns:,} "
            f"({positive_returns / len(returns) * 100:.2f}%)"
        )

        print(
            f"  Negative: {negative_returns:,} "
            f"({negative_returns / len(returns) * 100:.2f}%)"
        )

        print(
            f"  Zero:     {zero_returns:,} "
            f"({zero_returns / len(returns) * 100:.2f}%)"
        )

        print()

        quantiles = [
            0.50,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            0.99,
        ]

        print(
            "Absolute-return quantiles:"
        )

        for quantile in quantiles:

            value = float(
                absolute_returns.quantile(
                    quantile
                )
            )

            print(
                f"  Q{int(quantile * 100):02d}: "
                f"{value:.8f} "
                f"({value * 100:.4f}%)"
            )

        # --------------------------------------------------------
        # Non-zero return distribution.
        #
        # This tells us whether the zero-heavy distribution is
        # caused by genuinely unchanged L1 mid-price.
        # --------------------------------------------------------

        nonzero_absolute = (
            absolute_returns[
                absolute_returns > 0
            ]
        )

        print()

        print(
            "Non-zero absolute-return statistics:"
        )

        if nonzero_absolute.empty:

            print(
                "  No non-zero 10-second returns found."
            )

        else:

            print(
                f"  Non-zero samples: "
                f"{len(nonzero_absolute):,}"
            )

            for quantile in (
                0.50,
                0.70,
                0.80,
                0.90,
                0.95,
                0.99,
            ):

                value = float(
                    nonzero_absolute.quantile(
                        quantile
                    )
                )

                print(
                    f"  Moving Q{int(quantile * 100):02d}: "
                    f"{value:.8f} "
                    f"({value * 100:.4f}%)"
                )

        print(
            "============================================================"
        )

    # ============================================================
    # AUTOMATIC THRESHOLD SELECTION
    # ============================================================

    def _select_label_threshold(
        self,
        df: pd.DataFrame,
    ) -> float:

        if self.label_threshold is not None:

            if self.label_threshold <= 0:
                raise ValueError(
                    "label_threshold must be greater than 0."
                )

            logger.info(
                "Using manually supplied label threshold: "
                "%.8f (%.4f%%)",
                self.label_threshold,
                self.label_threshold * 100.0,
            )

            return self.label_threshold

        returns = (
            df[
                "future_return"
            ]
            .astype(float)
        )

        absolute_returns = (
            returns.abs()
        )

        # --------------------------------------------------------
        # First inspect the complete distribution.
        #
        # If Q70 is zero, the dataset does NOT contain enough
        # moving samples to create 30% directional labels using
        # a positive threshold.
        #
        # Therefore we do NOT pretend that 30% directional
        # coverage has been achieved.
        # --------------------------------------------------------

        percentile = (
            1.0
            - self.target_directional_ratio
        )

        raw_threshold = float(
            absolute_returns.quantile(
                percentile
            )
        )

        minimum_threshold = 0.00001

        if raw_threshold > 0:

            threshold = max(
                raw_threshold,
                minimum_threshold,
            )

            return threshold

        # --------------------------------------------------------
        # Q70 is zero.
        #
        # Select the smallest configured threshold that produces
        # actual directional labels.
        #
        # This is deliberately conservative and data-driven.
        # --------------------------------------------------------

        candidate_thresholds = [
            0.00001,
            0.00002,
            0.00003,
            0.00005,
            0.00007,
            0.00010,
            0.00015,
            0.00020,
            0.00030,
            0.00050,
        ]

        best_threshold = (
            minimum_threshold
        )

        best_directional = -1.0

        for candidate in (
            candidate_thresholds
        ):

            directional = (
                (
                    absolute_returns
                    >= candidate
                ).sum()
                / max(
                    len(absolute_returns),
                    1,
                )
            )

            if (
                directional > 0
                and directional
                > best_directional
            ):

                best_directional = (
                    float(
                        directional
                    )
                )

                best_threshold = (
                    candidate
                )

        logger.warning(
            "Q%.0f absolute return is zero. "
            "The dataset cannot support %.0f%% directional "
            "labels with a positive threshold. "
            "Using threshold %.8f (%.4f%%), actual "
            "directional coverage will be %.2f%%.",
            percentile * 100,
            self.target_directional_ratio * 100,
            best_threshold,
            best_threshold * 100,
            best_directional * 100,
        )

        return best_threshold

    # ============================================================
    # APPLY LABELS
    # ============================================================

    @staticmethod
    def _apply_labels(
        df: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:

        if df.empty:
            return df

        returns = (
            df[
                "future_return"
            ]
            .astype(float)
        )

        labels = np.where(
            returns >= threshold,
            1,
            np.where(
                returns <= -threshold,
                -1,
                0,
            ),
        )

        df = df.copy()

        df[
            "label"
        ] = labels.astype(
            int
        )

        return df

    # ============================================================
    # DATASET SUMMARY
    # ============================================================

    def _log_dataset_summary(
        self,
        df: pd.DataFrame,
        threshold: float,
    ) -> None:

        if df.empty:
            return

        down_count = int(
            (
                df["label"]
                == -1
            ).sum()
        )

        flat_count = int(
            (
                df["label"]
                == 0
            ).sum()
        )

        up_count = int(
            (
                df["label"]
                == 1
            ).sum()
        )

        total = len(df)

        directional = (
            down_count
            + up_count
        )

        first_timestamp = int(
            df[
                "timestamp_us"
            ].min()
        )

        last_timestamp = int(
            df[
                "timestamp_us"
            ].max()
        )

        duration_seconds = (
            last_timestamp
            - first_timestamp
        ) / 1_000_000.0

        average_interval = (
            duration_seconds
            / max(
                len(df) - 1,
                1,
            )
        )

        logger.info(
            "Dataset time span: %.2f seconds",
            duration_seconds,
        )

        logger.info(
            "Dataset average sample interval: %.4f seconds",
            average_interval,
        )

        logger.info(
            "PRIMARY TARGET PRICE: L1 MID PRICE"
        )

        logger.info(
            "Future-return horizon: %.2f seconds",
            self.horizon_seconds,
        )

        logger.info(
            "Final label threshold: %.8f (%.4f%%)",
            threshold,
            threshold * 100.0,
        )

        logger.info(
            "Dataset label distribution | "
            "DOWN=%d | FLAT=%d | UP=%d",
            down_count,
            flat_count,
            up_count,
        )

        logger.info(
            "Directional samples: %d / %d (%.2f%%)",
            directional,
            total,
            (
                directional
                / max(
                    total,
                    1,
                )
                * 100.0
            ),
        )

        # --------------------------------------------------------
        # Price source diagnostics.
        # --------------------------------------------------------

        for column in (
            "trade_price",
            "mid_price",
            "ticker_price",
            "mark_price",
        ):

            if column not in df.columns:
                continue

            available = int(
                df[
                    column
                ].notna().sum()
            )

            percentage = (
                available
                / max(
                    total,
                    1,
                )
                * 100.0
            )

            logger.info(
                "Price source availability | "
                "%s=%d/%d (%.2f%%)",
                column,
                available,
                total,
                percentage,
            )

        # --------------------------------------------------------
        # PRIMARY MID PRICE diagnostics.
        # --------------------------------------------------------

        if "mid_price" in df.columns:

            mid = (
                df[
                    "mid_price"
                ]
                .astype(float)
            )

            unique_mid = int(
                mid.nunique()
            )

            unchanged_mid = int(
                (
                    mid.diff()
                    .fillna(0.0)
                    == 0.0
                ).sum()
            )

            logger.info(
                "PRIMARY L1 MID diagnostics | "
                "unique_prices=%d | "
                "unchanged_samples=%d/%d (%.2f%%)",
                unique_mid,
                unchanged_mid,
                total,
                (
                    unchanged_mid
                    / max(
                        total,
                        1,
                    )
                    * 100.0
                ),
            )

        # --------------------------------------------------------
        # Future-return diagnostics.
        # --------------------------------------------------------

        if "future_return" in df.columns:

            returns = (
                df[
                    "future_return"
                ]
                .astype(float)
            )

            nonzero = int(
                (
                    returns
                    != 0.0
                ).sum()
            )

            logger.info(
                "10-second return diagnostics | "
                "nonzero=%d/%d (%.2f%%)",
                nonzero,
                total,
                (
                    nonzero
                    / max(
                        total,
                        1,
                    )
                    * 100.0
                ),
            )


# ================================================================
# FIND INPUT FILES
# ================================================================

def find_input_files(
    input_path: Path,
) -> List[Path]:

    if input_path.is_file():
        return [
            input_path
        ]

    if not input_path.exists():
        return []

    return sorted(
        input_path.glob(
            "market_data_*.jsonl"
        )
    )


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build an ML dataset from real Delta Exchange "
            "market data using L1 mid-price as the primary "
            "prediction target."
        )
    )

    parser.add_argument(
        "--input",
        default="data/market_data",
        help=(
            "JSONL file or directory "
            "containing market data."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/ml/"
            "training_dataset.csv"
        ),
        help="Output CSV path.",
    )

    parser.add_argument(
        "--horizon",
        type=float,
        default=10.0,
        help=(
            "Future prediction horizon "
            "in seconds. Default: 10."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Manual UP/DOWN return threshold. "
            "Example: 0.0003 = 0.03%%. "
            "If omitted, threshold is calibrated "
            "from the actual return distribution."
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help=(
            "Sampling interval in seconds. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--directional-ratio",
        type=float,
        default=0.30,
        help=(
            "Desired directional ratio for automatic "
            "threshold calibration. The actual ratio "
            "depends on the observed market movement."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    input_path = Path(
        args.input
    )

    input_files = find_input_files(
        input_path
    )

    if not input_files:

        raise SystemExit(
            "No market-data JSONL files found. "
            "Run the collector first."
        )

    builder = MLDatasetBuilder(
        horizon_seconds=args.horizon,
        label_threshold=args.threshold,
        sample_interval_seconds=args.interval,
        target_directional_ratio=(
            args.directional_ratio
        ),
    )

    df = builder.build(
        input_files=input_files,
        output_file=Path(
            args.output
        ),
    )

    down_count = int(
        (
            df["label"]
            == -1
        ).sum()
    )

    flat_count = int(
        (
            df["label"]
            == 0
        ).sum()
    )

    up_count = int(
        (
            df["label"]
            == 1
        ).sum()
    )

    threshold = float(
        df[
            "label_threshold"
        ].iloc[0]
    )

    print()

    print(
        "============================================================"
    )

    print(
        "       DELTA ML DATASET CREATED - L1 MID TARGET"
    )

    print(
        "============================================================"
    )

    print(
        f"Rows:              {len(df):,}"
    )

    print(
        f"Columns:           {len(df.columns):,}"
    )

    print(
        f"UP:                {up_count:,}"
    )

    print(
        f"FLAT:              {flat_count:,}"
    )

    print(
        f"DOWN:              {down_count:,}"
    )

    print(
        f"Threshold:         {threshold:.8f} "
        f"({threshold * 100:.4f}%)"
    )

    print(
        "Primary target:    L1 MID PRICE"
    )

    print(
        "Trade price:       Separate feature"
    )

    print(
        f"Horizon:           {args.horizon:.2f} seconds"
    )

    print(
        f"Interval:           {args.interval:.2f} seconds"
    )

    print(
        f"Output:             {args.output}"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()