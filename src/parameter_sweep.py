"""Staged parameter sweep for finding optimal backtest configurations.

Uses a three-stage approach to avoid exhaustive grid search:
1. Sweep fusion weights with default indicators
2. Sweep indicator params with best weight
3. Sweep entry/confidence thresholds with best params
"""

from typing import Dict, List, Optional

import pandas as pd

from .backtester import Backtester
from .config import BacktestConfig


class ParameterSweep:
    """Staged parameter sweep over backtest configurations."""

    def __init__(self, backtester: Backtester = None):
        self.backtester = backtester or Backtester()

    def sweep_fusion_weights(
        self,
        df: pd.DataFrame,
        weights: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Sweep fusion weight values, return results ranked by Sharpe.

        Args:
            df: OHLCV DataFrame.
            weights: List of fusion weights to test.

        Returns:
            List of result dicts sorted by Sharpe ratio (descending).
        """
        if weights is None:
            weights = self.backtester.config.sweep_fusion_weights

        results = []
        for w in weights:
            result = self.backtester.run(df, {"fusion_weight_technical": w})
            results.append(result)

        results.sort(key=lambda r: r["metrics"]["sharpe_ratio"], reverse=True)
        return results

    def sweep_indicators(
        self,
        df: pd.DataFrame,
        best_weight: float,
    ) -> List[Dict]:
        """Sweep indicator parameters with a fixed fusion weight.

        Tests combinations of EMA fast/slow, RSI period, and BB period.

        Args:
            df: OHLCV DataFrame.
            best_weight: Best fusion weight from stage 1.

        Returns:
            List of result dicts sorted by Sharpe ratio (descending).
        """
        config = self.backtester.config
        results = []

        for ema_fast in config.sweep_ema_fast:
            for ema_slow in config.sweep_ema_slow:
                if ema_fast >= ema_slow:
                    continue  # fast must be shorter than slow
                for rsi_period in config.sweep_rsi_period:
                    for bb_period in config.sweep_bb_period:
                        params = {
                            "fusion_weight_technical": best_weight,
                            "ema_fast": ema_fast,
                            "ema_slow": ema_slow,
                            "rsi_period": rsi_period,
                            "bb_period": bb_period,
                        }
                        result = self.backtester.run(df, params)
                        results.append(result)

        results.sort(key=lambda r: r["metrics"]["sharpe_ratio"], reverse=True)
        return results

    def sweep_thresholds(
        self,
        df: pd.DataFrame,
        best_params: Dict,
    ) -> List[Dict]:
        """Sweep entry threshold and min confidence with best indicator params.

        Args:
            df: OHLCV DataFrame.
            best_params: Best parameters from stages 1 + 2.

        Returns:
            List of result dicts sorted by Sharpe ratio (descending).
        """
        config = self.backtester.config
        results = []

        for entry_thresh in config.sweep_entry_threshold:
            for min_conf in config.sweep_min_confidence:
                params = dict(best_params)
                params["entry_threshold"] = entry_thresh
                params["min_confidence"] = min_conf
                result = self.backtester.run(df, params)
                results.append(result)

        results.sort(key=lambda r: r["metrics"]["sharpe_ratio"], reverse=True)
        return results

    def full_staged_sweep(self, df: pd.DataFrame) -> Dict:
        """Run all 3 sweep stages and return the best parameters.

        Returns:
            Dict with best_parameters, sweep_summary, and stage results.
        """
        # Stage 1: fusion weights
        stage1_results = self.sweep_fusion_weights(df)
        best_weight = stage1_results[0]["parameters"]["fusion_weight_technical"]

        # Stage 2: indicator parameters
        stage2_results = self.sweep_indicators(df, best_weight)
        best_indicator_params = stage2_results[0]["parameters"]

        # Stage 3: thresholds
        stage3_params = {
            "fusion_weight_technical": best_indicator_params["fusion_weight_technical"],
            "ema_fast": best_indicator_params["ema_fast"],
            "ema_slow": best_indicator_params["ema_slow"],
            "rsi_period": best_indicator_params["rsi_period"],
            "bb_period": best_indicator_params["bb_period"],
        }
        stage3_results = self.sweep_thresholds(df, stage3_params)

        best_result = stage3_results[0]
        total_combos = len(stage1_results) + len(stage2_results) + len(stage3_results)

        # Collect all sharpes for summary
        all_sharpes = (
            [r["metrics"]["sharpe_ratio"] for r in stage1_results]
            + [r["metrics"]["sharpe_ratio"] for r in stage2_results]
            + [r["metrics"]["sharpe_ratio"] for r in stage3_results]
        )

        return {
            "best_parameters": best_result["parameters"],
            "best_metrics": best_result["metrics"],
            "sweep_summary": {
                "total_combinations_tested": total_combos,
                "best_sharpe": max(all_sharpes),
                "worst_sharpe": min(all_sharpes),
                "stages": {
                    "stage1_fusion_weights": len(stage1_results),
                    "stage2_indicators": len(stage2_results),
                    "stage3_thresholds": len(stage3_results),
                },
            },
        }
