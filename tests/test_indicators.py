"""Tests for technical indicator computations."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.config import IndicatorParams
from src.indicators import (
    compute_bollinger,
    compute_composite_signal,
    compute_ema_crossover,
    compute_macd,
    compute_rsi,
)


@pytest.fixture
def synthetic_df():
    return DataLoader.generate_synthetic(n_candles=100, seed=123)


class TestEMACrossover:
    def test_output_range(self, synthetic_df):
        signal = compute_ema_crossover(synthetic_df, fast_period=12, slow_period=26)
        assert signal.min() >= -1.0
        assert signal.max() <= 1.0

    def test_output_length(self, synthetic_df):
        signal = compute_ema_crossover(synthetic_df)
        assert len(signal) == len(synthetic_df)

    def test_different_periods_different_signals(self, synthetic_df):
        s1 = compute_ema_crossover(synthetic_df, 9, 21)
        s2 = compute_ema_crossover(synthetic_df, 15, 34)
        # Different params should produce different signals
        assert not s1.equals(s2)


class TestRSI:
    def test_output_range(self, synthetic_df):
        signal = compute_rsi(synthetic_df, period=14)
        # After warmup period, values should be in [-1, 1]
        valid = signal.dropna()
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_output_length(self, synthetic_df):
        signal = compute_rsi(synthetic_df)
        assert len(signal) == len(synthetic_df)


class TestMACD:
    def test_output_range(self, synthetic_df):
        signal = compute_macd(synthetic_df, fast=12, slow=26, signal_period=9)
        assert signal.min() >= -1.0
        assert signal.max() <= 1.0

    def test_output_length(self, synthetic_df):
        signal = compute_macd(synthetic_df)
        assert len(signal) == len(synthetic_df)


class TestBollinger:
    def test_output_range(self, synthetic_df):
        signal = compute_bollinger(synthetic_df, period=20, std_dev=2.0)
        valid = signal.dropna()
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_output_length(self, synthetic_df):
        signal = compute_bollinger(synthetic_df)
        assert len(signal) == len(synthetic_df)


class TestCompositeSignal:
    def test_adds_signal_and_confidence_columns(self, synthetic_df):
        result = compute_composite_signal(synthetic_df)
        assert "signal" in result.columns
        assert "confidence" in result.columns

    def test_signal_range(self, synthetic_df):
        result = compute_composite_signal(synthetic_df)
        assert result["signal"].min() >= -1.0
        assert result["signal"].max() <= 1.0

    def test_confidence_range(self, synthetic_df):
        result = compute_composite_signal(synthetic_df)
        assert result["confidence"].min() >= 0.0
        assert result["confidence"].max() <= 1.0

    def test_fusion_weight_affects_signal(self, synthetic_df):
        r1 = compute_composite_signal(synthetic_df, fusion_weight_technical=0.3)
        r2 = compute_composite_signal(synthetic_df, fusion_weight_technical=0.8)
        # Different weights should produce different signal magnitudes
        assert not r1["signal"].equals(r2["signal"])

    def test_no_nans_in_output(self, synthetic_df):
        result = compute_composite_signal(synthetic_df)
        assert result["signal"].isna().sum() == 0
        assert result["confidence"].isna().sum() == 0

    def test_preserves_original_columns(self, synthetic_df):
        result = compute_composite_signal(synthetic_df)
        for col in ["open_time", "open", "high", "low", "close", "volume"]:
            assert col in result.columns
