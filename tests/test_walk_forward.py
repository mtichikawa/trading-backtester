"""Tests for walk-forward out-of-sample validation and the buy-and-hold benchmark."""

from typing import Any
import numpy as np
import pandas as pd
import pytest

from src.config import BacktestConfig
from src.data_loader import DataLoader
from src.walk_forward import WalkForwardValidator, buy_and_hold


@pytest.fixture
def synthetic_df():
    return DataLoader.generate_synthetic(n_candles=500, seed=7)


def _df_from_closes(closes: Any):
    return pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=len(closes), freq="D").astype(str),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1.0] * len(closes),
    })


class TestBuyAndHold:
    def test_rising_series_positive_return(self):
        bh = buy_and_hold(_df_from_closes([100, 110, 120, 130]), periods_per_year=365.0)
        assert bh["total_return_pct"] == pytest.approx(30.0, rel=1e-3)

    def test_flat_series_zero_return(self):
        bh = buy_and_hold(_df_from_closes([100, 100, 100, 100]), periods_per_year=365.0)
        assert bh["total_return_pct"] == 0.0
        assert bh["sharpe_ratio"] == 0.0

    def test_too_short_is_safe(self):
        assert buy_and_hold(_df_from_closes([100]), periods_per_year=365.0)["total_return_pct"] == 0.0


class TestWalkForward:
    def test_produces_oos_result(self, synthetic_df):
        wf = WalkForwardValidator(BacktestConfig(timeframe="1d"), train_bars=200, test_bars=60)
        out = wf.run(synthetic_df)
        for key in ("oos_metrics", "buy_and_hold", "beats_buy_and_hold", "folds", "n_folds"):
            assert key in out
        assert out["n_folds"] >= 1
        assert isinstance(out["beats_buy_and_hold"], bool)

    def test_stitched_curve_starts_at_initial_equity(self, synthetic_df):
        wf = WalkForwardValidator(BacktestConfig(timeframe="1d"), train_bars=200, test_bars=60)
        out = wf.run(synthetic_df)
        assert out["oos_equity_curve"][0] == BacktestConfig().initial_equity

    def test_each_fold_optimizes_then_tests_unseen(self, synthetic_df):
        """Every fold must carry a frozen param set and a disjoint OOS range."""
        wf = WalkForwardValidator(BacktestConfig(timeframe="1d"), train_bars=200, test_bars=60)
        out = wf.run(synthetic_df)
        for fold in out["folds"]:
            assert "best_params" in fold
            assert fold["is_range"][1] <= fold["oos_range"][0]  # train ends before test starts

    def test_deterministic(self, synthetic_df):
        wf = WalkForwardValidator(BacktestConfig(timeframe="1d"), train_bars=200, test_bars=60)
        a = wf.run(synthetic_df)["oos_metrics"]
        b = wf.run(synthetic_df)["oos_metrics"]
        assert a == b

    def test_short_series_shrinks_windows(self):
        df = DataLoader.generate_synthetic(n_candles=120, seed=1)
        out = WalkForwardValidator(BacktestConfig(timeframe="1d")).run(df)
        assert out["n_folds"] >= 1
        assert out["train_bars"] + out["test_bars"] <= 120
