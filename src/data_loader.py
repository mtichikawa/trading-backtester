"""Load OHLCV data from T1 PostgreSQL, CSV, or generate synthetic data."""

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


class DataLoader:
    """Loads OHLCV price data for backtesting."""

    def __init__(self):
        self.db_url = os.getenv("DB_URL", "")

    def load_from_db(
        self,
        pair: str = "BTC/USD",
        timeframe: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read OHLCV from T1 PostgreSQL ohlcv table via SQLAlchemy."""
        from sqlalchemy import create_engine, text

        if not self.db_url:
            raise ValueError("DB_URL environment variable not set")

        engine = create_engine(self.db_url)

        query = text(
            "SELECT open_time, open, high, low, close, volume "
            "FROM ohlcv WHERE pair = :pair AND timeframe = :timeframe "
            "ORDER BY open_time"
        )
        params = {"pair": pair, "timeframe": timeframe}

        if start:
            query = text(
                "SELECT open_time, open, high, low, close, volume "
                "FROM ohlcv WHERE pair = :pair AND timeframe = :timeframe "
                "AND open_time >= :start ORDER BY open_time"
            )
            params["start"] = start

        if end:
            query = text(
                "SELECT open_time, open, high, low, close, volume "
                "FROM ohlcv WHERE pair = :pair AND timeframe = :timeframe "
                "AND open_time >= :start AND open_time <= :end ORDER BY open_time"
            )
            params["end"] = end

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)

        return df

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Load OHLCV data from a CSV file."""
        df = pd.read_csv(path)

        expected_cols = ["open_time", "open", "high", "low", "close", "volume"]
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        df["open_time"] = pd.to_datetime(df["open_time"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def generate_synthetic(n_candles: int = 500, seed: int = 42) -> pd.DataFrame:
        """Generate realistic synthetic OHLCV data for demos and testing.

        Creates a random walk with trend, mean reversion, and realistic
        high/low/volume behavior.
        """
        rng = np.random.RandomState(seed)

        # Start price around 40000 (BTC-like)
        price = 40000.0
        prices = []

        for _ in range(n_candles):
            # Random return with slight upward drift
            ret = rng.normal(0.0002, 0.015)
            price *= (1 + ret)

            # Generate OHLC from close
            close = price
            high = close * (1 + abs(rng.normal(0, 0.005)))
            low = close * (1 - abs(rng.normal(0, 0.005)))
            open_price = close * (1 + rng.normal(0, 0.003))

            # Ensure high >= max(open, close) and low <= min(open, close)
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            volume = abs(rng.normal(100, 30))

            prices.append({
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": round(volume, 4),
            })

        start_time = datetime(2026, 1, 1)
        times = [start_time + timedelta(hours=i) for i in range(n_candles)]

        df = pd.DataFrame(prices)
        df.insert(0, "open_time", times)

        return df
