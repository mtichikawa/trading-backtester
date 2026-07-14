"""Tests for the core backtest engine."""

from typing import Any
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
    def test_produces_valid_result(self, backtester: Any, synthetic_df):
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
        """With very high thresholds, no trades should occur and equity stays flat.

        The time-based equity curve has one point per bar, so with no trades it is
        a flat line at the initial equity (not a single point).
        """
        bt = Backtester(BacktestConfig())
        result = bt.run(synthetic_df, {"entry_threshold": 10.0})
        assert result["metrics"]["total_trades"] == 0
        curve = result["equity_curve"]
        assert all(v == 10000.0 for v in curve)
        assert result["metrics"]["total_return_pct"] == 0.0

    def test_pair_and_timeframe_in_result(self, backtester, synthetic_df):
        result = backtester.run(synthetic_df)
        assert result["pair"] == "BTC/USD"
        assert result["timeframe"] == "1h"


class TestTradingCosts:
    def test_trades_carry_gross_net_and_cost(self, backtester, synthetic_df):
        """Each trade keeps gross PnL, the cost charged, and net PnL = gross - cost."""
        result = backtester.run(synthetic_df, {"entry_threshold": 0.05, "min_confidence": 0.0})
        trades = result["trades"]
        assert trades, "expected at least one trade for this test"
        expected_cost = 2 * (40.0 + 10.0) / 10000.0  # round-trip at default fees
        for t in trades:
            assert t["cost_pct"] == pytest.approx(expected_cost)
            assert t["pnl_pct"] == pytest.approx(t["pnl_pct_gross"] - expected_cost)

    def test_costs_reduce_total_return(self, synthetic_df):
        """Zero-cost config must produce a total return >= the cost-laden one."""
        params = {"entry_threshold": 0.05, "min_confidence": 0.0}
        free = Backtester(BacktestConfig(taker_fee_bps=0.0, slippage_bps=0.0)).run(synthetic_df, params)
        paid = Backtester(BacktestConfig()).run(synthetic_df, params)
        assert paid["metrics"]["total_trades"] == free["metrics"]["total_trades"]
        assert free["metrics"]["total_return_pct"] >= paid["metrics"]["total_return_pct"]

    def test_periods_per_year_tracks_timeframe(self):
        assert Backtester(BacktestConfig(timeframe="1d"))._periods_per_year() == 365.0
        assert Backtester(BacktestConfig(timeframe="4h"))._periods_per_year() == 2190.0
        assert Backtester(BacktestConfig(timeframe="1h"))._periods_per_year() == 8760.0

    def test_daily_sharpe_not_hourly_inflated(self, synthetic_df):
        """Same returns, daily vs hourly annualization differ by sqrt(8760/365) ~ 4.9x.

        Guards against the old hardcoded-8760 bug silently inflating daily Sharpe.
        """
        params = {"entry_threshold": 0.05, "min_confidence": 0.0}
        hourly = Backtester(BacktestConfig(timeframe="1h")).run(synthetic_df, params)
        daily = Backtester(BacktestConfig(timeframe="1d")).run(synthetic_df, params)
        sh, sd = hourly["metrics"]["sharpe_ratio"], daily["metrics"]["sharpe_ratio"]
        if sd != 0:
            assert abs(sh / sd) == pytest.approx(np.sqrt(8760.0 / 365.0), rel=0.01)
