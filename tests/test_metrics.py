"""Tests for backtest performance metrics."""

import numpy as np
import pytest

from src.metrics import (
    compute_all_metrics,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


class TestSharpeRatio:
    def test_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.005])
        result = sharpe_ratio(returns)
        assert result > 0

    def test_negative_returns(self):
        returns = np.array([-0.01, -0.02, -0.01, -0.015, -0.005])
        result = sharpe_ratio(returns)
        assert result < 0

    def test_zero_volatility(self):
        returns = np.array([0.01, 0.01, 0.01])
        result = sharpe_ratio(returns)
        assert result == 0.0

    def test_empty_returns(self):
        result = sharpe_ratio(np.array([]))
        assert result == 0.0

    def test_single_return(self):
        result = sharpe_ratio(np.array([0.05]))
        assert result == 0.0  # std with ddof=1 on single value is 0


class TestSortinoRatio:
    def test_positive_returns(self):
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        result = sortino_ratio(returns)
        assert result > 0

    def test_all_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.015])
        result = sortino_ratio(returns)
        assert result == float("inf")

    def test_all_negative_returns(self):
        returns = np.array([-0.01, -0.02, -0.015])
        result = sortino_ratio(returns)
        assert result < 0

    def test_empty_returns(self):
        result = sortino_ratio(np.array([]))
        assert result == 0.0

    def test_sortino_greater_than_sharpe_with_few_downside(self):
        """Sortino should be higher than Sharpe when downside is limited."""
        returns = np.array([0.02, 0.03, -0.005, 0.01, 0.02, 0.015])
        s = sharpe_ratio(returns)
        so = sortino_ratio(returns)
        # Sortino penalizes only downside, should be larger when few losses
        assert so > s


class TestMaxDrawdown:
    def test_known_drawdown(self):
        equity = np.array([100, 110, 105, 90, 95, 100])
        dd = max_drawdown(equity)
        # Peak was 110, trough was 90 → dd = (90-110)/110 = -18.18%
        assert pytest.approx(dd, rel=1e-3) == -20 / 110

    def test_no_drawdown(self):
        equity = np.array([100, 110, 120, 130])
        dd = max_drawdown(equity)
        assert dd == 0.0

    def test_monotonic_decline(self):
        equity = np.array([100, 90, 80, 70])
        dd = max_drawdown(equity)
        assert pytest.approx(dd, rel=1e-3) == -0.30

    def test_empty_curve(self):
        dd = max_drawdown(np.array([]))
        assert dd == 0.0

    def test_single_value(self):
        dd = max_drawdown(np.array([100]))
        assert dd == 0.0


class TestWinRate:
    def test_all_winners(self):
        trades = [{"pnl_pct": 0.05}, {"pnl_pct": 0.02}, {"pnl_pct": 0.01}]
        assert win_rate(trades) == 1.0

    def test_all_losers(self):
        trades = [{"pnl_pct": -0.05}, {"pnl_pct": -0.02}]
        assert win_rate(trades) == 0.0

    def test_mixed(self):
        trades = [
            {"pnl_pct": 0.05},
            {"pnl_pct": -0.02},
            {"pnl_pct": 0.01},
            {"pnl_pct": -0.01},
        ]
        assert win_rate(trades) == 0.5

    def test_no_trades(self):
        assert win_rate([]) == 0.0

    def test_breakeven_not_counted_as_win(self):
        trades = [{"pnl_pct": 0.0}]
        assert win_rate(trades) == 0.0


class TestProfitFactor:
    def test_positive_factor(self):
        trades = [
            {"pnl_pct": 0.10},
            {"pnl_pct": -0.05},
            {"pnl_pct": 0.08},
        ]
        # gross profit = 0.18, gross loss = 0.05
        pf = profit_factor(trades)
        assert pytest.approx(pf, rel=1e-3) == 0.18 / 0.05

    def test_all_winners(self):
        trades = [{"pnl_pct": 0.05}, {"pnl_pct": 0.02}]
        assert profit_factor(trades) == float("inf")

    def test_all_losers(self):
        trades = [{"pnl_pct": -0.05}, {"pnl_pct": -0.02}]
        assert profit_factor(trades) == 0.0

    def test_no_trades(self):
        assert profit_factor([]) == 0.0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        trades = [{"pnl_pct": 0.05}, {"pnl_pct": -0.02}]
        equity = np.array([10000, 10500, 10290])
        result = compute_all_metrics(trades, equity)
        expected_keys = {
            "total_return_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_pct",
            "win_rate",
            "profit_factor",
            "total_trades",
        }
        assert set(result.keys()) == expected_keys

    def test_total_trades_count(self):
        trades = [{"pnl_pct": 0.01}] * 5
        equity = np.array([10000, 10100, 10200, 10300, 10400, 10500])
        result = compute_all_metrics(trades, equity)
        assert result["total_trades"] == 5
