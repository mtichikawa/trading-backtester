"""Tests for the core backtest engine."""

import numpy as np
import pytest

from src.backtester import Backtester
from src.config import BacktestConfig
from src.data_loader import DataLoader


@pytest.fixture
def synthetic_df():
    """Generate synthetic OHLCV data for testing."""
    return DataLoader.generate_synthetic(n_candles=200, seed=42)


@pytest.fixture
def backtester():
    """Create a backtester with default config."""
    return Backtester(BacktestConfig())


class TestBacktestRun:
    def test_produces_valid_result(self, backtester, synthetic_df):
        result = backtester.run(synthetic_df)
        assert "parameters" in result
        assert "metrics" in result
        assert "trades" in result
        assert "equity_curve" in result

    def test_metrics_have_all_keys(self, backtester, synthetic_df):
        result = backtester.run(synthetic_df)
        expected_keys = {
            "total_return_pct", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "win_rate", "profit_factor", "total_trades",
        }
        assert set(result["metrics"].keys()) == expected_keys

    def test_equity_curve_has_values(self, backtester, synthetic_df):
        result = backtester.run(synthetic_df)
        assert len(result["equity_curve"]) >= 1
        assert result["equity_curve"][0] == 10000.0  # initial equity

    def test_trades_are_list(self, backtester, synthetic_df):
        result = backtester.run(synthetic_df)
        assert isinstance(result["trades"], list)

    def test_win_rate_in_valid_range(self, backtester, synthetic_df):
        result = backtester.run(synthetic_df)
        wr = result["metrics"]["win_rate"]
        assert 0.0 <= wr <= 1.0


class TestParameterOverrides:
    def test_custom_fusion_weight(self, synthetic_df):
        bt = Backtester(BacktestConfig())
        result = bt.run(synthetic_df, {"fusion_weight_technical": 0.8})
        assert result["parameters"]["fusion_weight_technical"] == 0.8

    def test_custom_entry_threshold(self, synthetic_df):
        bt = Backtester(BacktestConfig())
        result = bt.run(synthetic_df, {"entry_threshold": 0.5})
        assert result["parameters"]["entry_threshold"] == 0.5

    def test_higher_threshold_fewer_trades(self, synthetic_df):
        bt = Backtester(BacktestConfig())
        result_low = bt.run(synthetic_df, {"entry_threshold": 0.1, "min_confidence": 0.1})
        result_high = bt.run(synthetic_df, {"entry_threshold": 0.8, "min_confidence": 0.1})
        # Higher threshold should produce fewer or equal trades
        assert result_high["metrics"]["total_trades"] <= result_low["metrics"]["total_trades"]


class TestEquityCurve:
    def test_no_trades_means_flat_equity(self, synthetic_df):
        """With very high thresholds, no trades should occur."""
        bt = Backtester(BacktestConfig())
        result = bt.run(synthetic_df, {"entry_threshold": 10.0})
        assert result["metrics"]["total_trades"] == 0
        assert len(result["equity_curve"]) == 1
        assert result["equity_curve"][0] == 10000.0

    def test_pair_and_timeframe_in_result(self, backtester, synthetic_df):
        result = backtester.run(synthetic_df)
        assert result["pair"] == "BTC/USD"
        assert result["timeframe"] == "1h"
