"""Core backtest engine: runs strategy on OHLCV data and computes metrics."""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig, IndicatorParams
from .indicators import compute_composite_signal
from .metrics import compute_all_metrics
from .strategy import Strategy


class Backtester:
    """Backtests trading signals against OHLCV price data.

    Applies technical indicators with given parameters, runs the strategy,
    tracks equity curve, and computes performance metrics.
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        df_ohlcv: pd.DataFrame,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Run a backtest on the given OHLCV data.

        Args:
            df_ohlcv: DataFrame with open_time, open, high, low, close, volume.
            params: Optional parameter overrides. Keys can include:
                - fusion_weight_technical
                - ema_fast, ema_slow, rsi_period
                - macd_fast, macd_slow, macd_signal
                - bb_period, bb_std
                - entry_threshold, exit_threshold, stop_loss_pct, min_confidence
                - short_enabled

        Returns:
            BacktestResult dict with params, metrics, trades, equity_curve.
        """
        # Build indicator params from config + overrides
        indicator_params = IndicatorParams(
            ema_fast=self._get_param(params, "ema_fast", self.config.indicator_params.ema_fast),
            ema_slow=self._get_param(params, "ema_slow", self.config.indicator_params.ema_slow),
            rsi_period=self._get_param(params, "rsi_period", self.config.indicator_params.rsi_period),
            macd_fast=self._get_param(params, "macd_fast", self.config.indicator_params.macd_fast),
            macd_slow=self._get_param(params, "macd_slow", self.config.indicator_params.macd_slow),
            macd_signal=self._get_param(params, "macd_signal", self.config.indicator_params.macd_signal),
            bb_period=self._get_param(params, "bb_period", self.config.indicator_params.bb_period),
            bb_std=self._get_param(params, "bb_std", self.config.indicator_params.bb_std),
        )

        fusion_weight = self._get_param(
            params, "fusion_weight_technical", self.config.fusion_weight_technical
        )

        # Compute signals
        df_signals = compute_composite_signal(df_ohlcv, indicator_params, fusion_weight)

        # Build strategy
        strategy = Strategy(
            entry_threshold=self._get_param(params, "entry_threshold", self.config.entry_threshold),
            exit_threshold=self._get_param(params, "exit_threshold", self.config.exit_threshold),
            stop_loss_pct=self._get_param(params, "stop_loss_pct", self.config.stop_loss_pct),
            min_confidence=self._get_param(params, "min_confidence", self.config.min_confidence),
            short_enabled=self._get_param(params, "short_enabled", self.config.short_enabled),
        )

        # Generate trades
        trades = strategy.generate_trades(df_signals)

        # Build equity curve
        equity_curve = self._build_equity_curve(trades, self.config.initial_equity)

        # Compute metrics
        metrics = compute_all_metrics(trades, equity_curve)

        # Build used params dict for reporting
        used_params = {
            "fusion_weight_technical": fusion_weight,
            "ema_fast": indicator_params.ema_fast,
            "ema_slow": indicator_params.ema_slow,
            "rsi_period": indicator_params.rsi_period,
            "macd_fast": indicator_params.macd_fast,
            "macd_slow": indicator_params.macd_slow,
            "macd_signal": indicator_params.macd_signal,
            "bb_period": indicator_params.bb_period,
            "bb_std": indicator_params.bb_std,
            "entry_threshold": strategy.entry_threshold,
            "exit_threshold": strategy.exit_threshold,
            "stop_loss_pct": strategy.stop_loss_pct,
            "min_confidence": strategy.min_confidence,
            "short_enabled": strategy.short_enabled,
        }

        return {
            "pair": self.config.pair,
            "timeframe": self.config.timeframe,
            "parameters": used_params,
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve.tolist(),
        }

    def _build_equity_curve(self, trades, initial_equity: float) -> np.ndarray:
        """Build equity curve from trade list."""
        if not trades:
            return np.array([initial_equity])

        equity = initial_equity
        curve = [equity]

        for trade in trades:
            pnl = equity * trade["pnl_pct"] * self.config.position_size_fraction
            equity += pnl
            curve.append(equity)

        return np.array(curve)

    @staticmethod
    def _get_param(params: Optional[Dict], key: str, default):
        """Get parameter from override dict or fall back to default."""
        if params and key in params:
            return params[key]
        return default
