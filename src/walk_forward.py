"""Walk-forward (out-of-sample) validation for the backtester.

The single most honest thing a backtest can do: never report metrics on the same
data the parameters were optimized on. This harness rolls a train/test window
across the series. On each fold it optimizes parameters on the in-sample (train)
window, then applies those frozen parameters to the next, unseen out-of-sample
(test) window. Only the out-of-sample results are kept and stitched into one
continuous equity curve, which is what the final metrics are computed on.

A buy-and-hold benchmark over the same out-of-sample span is reported alongside,
because a strategy that underperforms simply holding the asset is not worth its
trading costs.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtester import Backtester
from .config import BacktestConfig
from .metrics import compute_all_metrics, sharpe_ratio
from .parameter_sweep import ParameterSweep


def buy_and_hold(df: pd.DataFrame, periods_per_year: float, initial_equity: float = 10000.0) -> Dict:
    """Buy at the first close, hold to the last. The honest benchmark to beat."""
    if len(df) < 2:
        return {"total_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0}
    close = df["close"].to_numpy(dtype=float)
    equity = initial_equity * (close / close[0])
    returns = np.diff(equity) / equity[:-1]
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity - peak) / peak)) * 100
    total_return = (equity[-1] - equity[0]) / equity[0] * 100
    return {
        "total_return_pct": round(float(total_return), 4),
        "sharpe_ratio": round(sharpe_ratio(returns, periods_per_year=periods_per_year), 4),
        "max_drawdown_pct": round(max_dd, 4),
    }


class WalkForwardValidator:
    """Rolling-window walk-forward optimization and out-of-sample evaluation."""

    def __init__(
        self,
        config: BacktestConfig = None,
        train_bars: int = 252,
        test_bars: int = 63,
    ):
        self.config = config or BacktestConfig()
        self.train_bars = train_bars
        self.test_bars = test_bars

    def _fit_window(self, sizes: int) -> bool:
        return sizes >= self.train_bars + self.test_bars

    def run(self, df: pd.DataFrame) -> Dict:
        """Run rolling walk-forward validation.

        Returns a dict with per-fold records, the stitched out-of-sample metrics,
        the buy-and-hold benchmark over the same span, and the parameter set used
        for each fold (so you can see how much the optimizer moved between folds).
        """
        df = df.sort_values("open_time").reset_index(drop=True)
        n = len(df)
        ppy = Backtester(self.config)._periods_per_year()

        # Shrink the windows gracefully if the series is short.
        train_bars, test_bars = self.train_bars, self.test_bars
        if not self._fit_window(n):
            train_bars = max(30, int(n * 0.6))
            test_bars = max(15, int(n * 0.15))

        folds: List[Dict] = []
        oos_returns: List[np.ndarray] = []
        oos_trades: List[Dict] = []
        oos_index_start: Optional[int] = None
        oos_index_end: int = 0

        start = 0
        while start + train_bars + test_bars <= n:
            is_lo, is_hi = start, start + train_bars
            oos_lo, oos_hi = is_hi, is_hi + test_bars

            if oos_index_start is None:
                oos_index_start = oos_lo
            oos_index_end = oos_hi

            # 1. Optimize on the in-sample window only.
            is_df = df.iloc[is_lo:is_hi].reset_index(drop=True)
            sweep = ParameterSweep(Backtester(self.config)).full_staged_sweep(is_df)
            best_params = sweep["best_parameters"]

            # 2. Apply frozen params to [is_lo : oos_hi] so indicators warm up on the
            #    in-sample bars, then keep only the out-of-sample tail.
            eval_df = df.iloc[is_lo:oos_hi].reset_index(drop=True)
            result = Backtester(self.config).run(eval_df, best_params)

            equity = np.asarray(result["equity_curve"], dtype=float)
            tail = oos_hi - oos_lo  # number of out-of-sample bars
            oos_equity = equity[-(tail + 1):] if len(equity) > tail else equity
            fold_returns = np.diff(oos_equity) / oos_equity[:-1] if len(oos_equity) > 1 else np.array([])

            # Trades whose entry falls inside the out-of-sample date range.
            oos_times = set(df.iloc[oos_lo:oos_hi]["open_time"].astype(str))
            fold_trades = [t for t in result["trades"] if t["entry_time"] in oos_times]

            oos_returns.append(fold_returns)
            oos_trades.extend(fold_trades)
            folds.append({
                "is_range": [str(df.iloc[is_lo]["open_time"]), str(df.iloc[is_hi - 1]["open_time"])],
                "oos_range": [str(df.iloc[oos_lo]["open_time"]), str(df.iloc[oos_hi - 1]["open_time"])],
                "best_params": best_params,
                "oos_metrics": compute_all_metrics(fold_trades, oos_equity, periods_per_year=ppy),
            })

            start += test_bars

        if not folds:
            return {"error": "not enough data for a single train+test fold", "n_bars": n}

        # Stitch all out-of-sample returns into one continuous equity curve.
        all_returns = np.concatenate([r for r in oos_returns if len(r)]) if oos_returns else np.array([])
        stitched = self.config.initial_equity * np.cumprod(1 + all_returns)
        stitched = np.insert(stitched, 0, self.config.initial_equity)
        oos_metrics = compute_all_metrics(oos_trades, stitched, periods_per_year=ppy)

        oos_df = df.iloc[oos_index_start:oos_index_end]
        bh = buy_and_hold(oos_df, ppy, self.config.initial_equity)

        return {
            "pair": self.config.pair,
            "timeframe": self.config.timeframe,
            "train_bars": train_bars,
            "test_bars": test_bars,
            "n_folds": len(folds),
            "oos_span": [str(df.iloc[oos_index_start]["open_time"]),
                         str(df.iloc[oos_index_end - 1]["open_time"])],
            "oos_metrics": oos_metrics,
            "buy_and_hold": bh,
            "beats_buy_and_hold": oos_metrics["total_return_pct"] > bh["total_return_pct"],
            "folds": folds,
            "oos_equity_curve": stitched.tolist(),
        }
