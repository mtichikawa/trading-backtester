"""Core backtest engine: runs strategy on OHLCV data and computes metrics.

Sharpe and Sortino ratios are annualized assuming hourly candles (8760 periods/year).
"""

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

        # Generate trades (gross, before costs)
        trades = strategy.generate_trades(df_signals)

        # Apply researched trading costs (fees + slippage) to each round-trip trade
        trades = self._apply_costs(trades)

        # Build a TIME-BASED (per-bar, mark-to-market) equity curve so Sharpe is
        # computed on calendar returns, annualized by the candle's true periods/year.
        equity_curve = self._build_equity_curve_timebased(df_signals, trades)

        # Compute metrics, annualizing by the actual timeframe (not a hardcoded 8760)
        metrics = compute_all_metrics(
            trades, equity_curve, periods_per_year=self._periods_per_year()
        )

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

    # Periods per year for Sharpe/Sortino annualization, keyed by candle timeframe.
    _PERIODS_PER_YEAR = {
        "1m": 525600.0, "5m": 105120.0, "15m": 35040.0, "30m": 17520.0,
        "1h": 8760.0, "4h": 2190.0, "1d": 365.0,
    }

    def _periods_per_year(self) -> float:
        """Annualization scale for the configured timeframe (defaults to hourly)."""
        return self._PERIODS_PER_YEAR.get(self.config.timeframe, 8760.0)

    def _per_side_cost(self) -> float:
        """Fee + slippage charged on each side of a trade, as a fraction."""
        return (self.config.taker_fee_bps + self.config.slippage_bps) / 10000.0

    def _apply_costs(self, trades):
        """Deduct round-trip trading costs from each trade's PnL.

        Keeps the gross PnL as 'pnl_pct_gross' and records 'cost_pct'; 'pnl_pct'
        becomes net of costs so win rate, profit factor, and avg win/loss all
        reflect what an account would actually keep.
        """
        round_trip = 2.0 * self._per_side_cost()
        adjusted = []
        for t in trades:
            t2 = dict(t)
            t2["pnl_pct_gross"] = t["pnl_pct"]
            t2["cost_pct"] = round_trip
            t2["pnl_pct"] = t["pnl_pct"] - round_trip
            adjusted.append(t2)
        return adjusted

    def _build_equity_curve_timebased(self, df_signals, trades) -> np.ndarray:
        """Per-bar mark-to-market equity curve with costs charged at entry/exit.

        While a position is open, equity compounds by the position-sized close-to-
        close return each bar. The per-side cost is charged at the entry bar and
        again at the exit bar. This yields calendar (per-candle) returns, which is
        what Sharpe/Sortino should be computed on.
        """
        n = len(df_signals)
        if n == 0:
            return np.array([self.config.initial_equity])

        close = df_signals["close"].to_numpy(dtype=float)
        times = df_signals["open_time"].astype(str).to_numpy()
        time_to_idx = {t: i for i, t in enumerate(times)}

        side_per_bar = np.zeros(n)   # position held during (i-1 -> i): +1 long, -1 short
        cost_per_bar = np.zeros(n)   # cost fraction charged at bar i
        per_side = self._per_side_cost()

        for t in trades:
            ei = time_to_idx.get(t["entry_time"])
            xi = time_to_idx.get(t["exit_time"])
            if ei is None or xi is None or xi <= ei:
                continue
            sgn = 1.0 if t["side"] == "long" else -1.0
            side_per_bar[ei + 1: xi + 1] = sgn
            cost_per_bar[ei] += per_side
            cost_per_bar[xi] += per_side

        psf = self.config.position_size_fraction
        equity = self.config.initial_equity
        curve = [equity]
        equity *= (1.0 - cost_per_bar[0])  # cost if a trade somehow opens at bar 0
        for i in range(1, n):
            bar_ret = (close[i] - close[i - 1]) / close[i - 1] if close[i - 1] else 0.0
            equity *= (1.0 + psf * side_per_bar[i] * bar_ret) * (1.0 - cost_per_bar[i])
            curve.append(equity)

        return np.array(curve)

    def _build_equity_curve(self, trades, initial_equity: float) -> np.ndarray:
        """Event-based equity curve (one point per trade). Retained for reference;
        run() uses the time-based curve above for honest, calendar-scaled metrics."""
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
