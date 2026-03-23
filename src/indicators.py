"""Technical indicator computations for parameter sweeping.

Re-implements T3's indicator logic so the backtester can sweep parameters
independently without modifying T3 at sweep time.
"""

import numpy as np
import pandas as pd

from .config import IndicatorParams


def compute_ema_crossover(
    df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26
) -> pd.Series:
    """Compute EMA crossover signal.

    Signal = (EMA_fast - EMA_slow) / EMA_slow, clipped to [-1, +1].
    Positive = bullish (fast above slow), negative = bearish.
    """
    ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()

    signal = (ema_fast - ema_slow) / (ema_slow + 1e-10)
    return signal.clip(-1, 1)


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI-based signal.

    Standard RSI mapped to [-1, +1]: signal = -(rsi - 50) / 50.
    RSI < 30 (oversold) -> positive (bullish).
    RSI > 70 (overbought) -> negative (bearish).
    """
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    signal = -(rsi - 50) / 50
    return signal.clip(-1, 1)


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """Compute MACD histogram signal.

    MACD histogram normalized by close price, clipped to [-1, +1].
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    # Normalize by close price for scale-independent signal
    signal = histogram / (df["close"] + 1e-10)
    # Scale up to make signals more meaningful
    signal = signal * 100
    return signal.clip(-1, 1)


def compute_bollinger(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """Compute Bollinger Band signal.

    Signal = -(close - middle) / (upper - middle), clipped to [-1, +1].
    Above upper band (overbought) -> negative (bearish).
    Below lower band (oversold) -> positive (bullish).
    """
    middle = df["close"].rolling(window=period, min_periods=period).mean()
    std = df["close"].rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    half_width = upper - middle

    signal = -(df["close"] - middle) / (half_width + 1e-10)
    return signal.clip(-1, 1)


def compute_composite_signal(
    df: pd.DataFrame,
    params: IndicatorParams = None,
    fusion_weight_technical: float = 1.0,
) -> pd.DataFrame:
    """Compute composite signal from all four indicators.

    Since the backtester operates without sentiment, the signal is the
    mean of all 4 technical indicators scaled by fusion_weight_technical.
    Confidence = 1.0 - std(indicators), higher when indicators agree.

    Returns DataFrame with 'signal' and 'confidence' columns added.
    """
    if params is None:
        params = IndicatorParams()

    ema = compute_ema_crossover(df, params.ema_fast, params.ema_slow)
    rsi = compute_rsi(df, params.rsi_period)
    macd = compute_macd(df, params.macd_fast, params.macd_slow, params.macd_signal)
    bb = compute_bollinger(df, params.bb_period, params.bb_std)

    # Stack indicators for aggregation
    indicators = pd.DataFrame({
        "ema": ema,
        "rsi": rsi,
        "macd": macd,
        "bb": bb,
    })

    # Composite signal: mean of indicators * fusion weight
    raw_signal = indicators.mean(axis=1)
    signal = raw_signal * fusion_weight_technical

    # Confidence: higher when indicators agree (low std)
    indicator_std = indicators.std(axis=1)
    confidence = (1.0 - indicator_std).clip(0, 1)

    result = df.copy()
    result["signal"] = signal
    result["confidence"] = confidence

    # Fill NaN from indicator warmup periods
    result["signal"] = result["signal"].fillna(0.0)
    result["confidence"] = result["confidence"].fillna(0.0)

    return result
