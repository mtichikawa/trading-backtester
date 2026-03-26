"""Tests for the parameter sweep module."""

import pytest

from src.backtester import Backtester
from src.config import BacktestConfig
from src.data_loader import DataLoader
from src.parameter_sweep import ParameterSweep


@pytest.fixture
def synthetic_df():
    return DataLoader.generate_synthetic(n_candles=150, seed=99)


@pytest.fixture
def sweep(synthetic_df):
    config = BacktestConfig()
    # Use small sweep ranges for fast tests
    config.sweep_fusion_weights = [0.4, 0.6, 0.8]
    config.sweep_ema_fast = [9, 12]
    config.sweep_ema_slow = [26]
    config.sweep_rsi_period = [14]
    config.sweep_bb_period = [20]
    config.sweep_entry_threshold = [0.1, 0.15]
    config.sweep_min_confidence = [0.3, 0.5]
    return ParameterSweep(Backtester(config))


class TestFusionWeightSweep:
    def test_returns_ranked_results(self, sweep, synthetic_df):
        results = sweep.sweep_fusion_weights(synthetic_df)
        assert len(results) == 3  # 3 weights
        # Check descending Sharpe order
        sharpes = [r["metrics"]["sharpe_ratio"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_each_result_has_metrics(self, sweep, synthetic_df):
        results = sweep.sweep_fusion_weights(synthetic_df)
        for r in results:
            assert "metrics" in r
            assert "parameters" in r


class TestIndicatorSweep:
    def test_returns_results(self, sweep, synthetic_df):
        results = sweep.sweep_indicators(synthetic_df, best_weight=0.6)
        assert len(results) > 0

    def test_results_are_ranked(self, sweep, synthetic_df):
        results = sweep.sweep_indicators(synthetic_df, best_weight=0.6)
        sharpes = [r["metrics"]["sharpe_ratio"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)


class TestThresholdSweep:
    def test_returns_results(self, sweep, synthetic_df):
        results = sweep.sweep_thresholds(
            synthetic_df, {"fusion_weight_technical": 0.6}
        )
        # 2 entry thresholds * 2 min confidence = 4
        assert len(results) == 4

    def test_results_are_ranked(self, sweep, synthetic_df):
        results = sweep.sweep_thresholds(
            synthetic_df, {"fusion_weight_technical": 0.6}
        )
        sharpes = [r["metrics"]["sharpe_ratio"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)


class TestFullStagedSweep:
    def test_returns_best_parameters(self, sweep, synthetic_df):
        result = sweep.full_staged_sweep(synthetic_df)
        assert "best_parameters" in result
        assert "best_metrics" in result
        assert "sweep_summary" in result

    def test_summary_has_total_combos(self, sweep, synthetic_df):
        result = sweep.full_staged_sweep(synthetic_df)
        assert result["sweep_summary"]["total_combinations_tested"] > 0

    def test_best_sharpe_is_max(self, sweep, synthetic_df):
        result = sweep.full_staged_sweep(synthetic_df)
        assert result["sweep_summary"]["best_sharpe"] >= result["sweep_summary"]["worst_sharpe"]

    def test_params_within_sweep_range(self, sweep, synthetic_df):
        result = sweep.full_staged_sweep(synthetic_df)
        best = result["best_parameters"]
        assert best["entry_threshold"] in [0.1, 0.15]
        assert best["min_confidence"] in [0.3, 0.5]
