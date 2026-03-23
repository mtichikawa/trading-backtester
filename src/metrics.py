"""Backtest performance metrics: Sharpe, Sortino, drawdown, win rate, profit factor."""

from typing import Dict, List

import numpy as np


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns: Array of period returns.
        risk_free: Risk-free rate per period.

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if no data or zero volatility.
    """
    if len(returns) == 0:
        return 0.0

    excess = returns - risk_free
    std = np.std(excess, ddof=1) if len(excess) > 1 else 0.0

    if std == 0:
        return 0.0

    # Annualize assuming hourly candles (8760 hours/year)
    return float((np.mean(excess) / std) * np.sqrt(8760))


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute annualized Sortino ratio (downside deviation only).

    Args:
        returns: Array of period returns.
        risk_free: Risk-free rate per period.

    Returns:
        Annualized Sortino ratio. Returns 0.0 if no data or zero downside deviation.
    """
    if len(returns) == 0:
        return 0.0

    excess = returns - risk_free
    downside = excess[excess < 0]

    if len(downside) == 0:
        return 0.0 if np.mean(excess) == 0 else float("inf")

    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else abs(downside[0])

    if downside_std == 0:
        return 0.0

    return float((np.mean(excess) / downside_std) * np.sqrt(8760))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum drawdown as worst peak-to-trough percentage.

    Args:
        equity_curve: Array of equity values over time.

    Returns:
        Max drawdown as a negative percentage (e.g., -0.15 for 15% drawdown).
        Returns 0.0 if no data.
    """
    if len(equity_curve) == 0:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (value - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return float(max_dd)


def win_rate(trades: List[Dict]) -> float:
    """Compute fraction of profitable trades.

    Args:
        trades: List of trade dicts with 'pnl_pct' key.

    Returns:
        Win rate as fraction [0, 1]. Returns 0.0 if no trades.
    """
    if len(trades) == 0:
        return 0.0

    winners = sum(1 for t in trades if t["pnl_pct"] > 0)
    return float(winners / len(trades))


def profit_factor(trades: List[Dict]) -> float:
    """Compute profit factor: gross profits / gross losses.

    Args:
        trades: List of trade dicts with 'pnl_pct' key.

    Returns:
        Profit factor. Returns 0.0 if no trades or no losses.
    """
    if len(trades) == 0:
        return 0.0

    gross_profit = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
    gross_loss = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def compute_all_metrics(
    trades: List[Dict], equity_curve: np.ndarray
) -> Dict[str, float]:
    """Compute all backtest metrics.

    Args:
        trades: List of trade dicts with 'pnl_pct' key.
        equity_curve: Array of equity values over time.

    Returns:
        Dict with all metric values.
    """
    returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([])

    total_return = (
        (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        if len(equity_curve) > 0
        else 0.0
    )

    return {
        "total_return_pct": float(round(total_return, 4)),
        "sharpe_ratio": round(sharpe_ratio(returns), 4),
        "sortino_ratio": round(sortino_ratio(returns), 4),
        "max_drawdown_pct": round(max_drawdown(equity_curve) * 100, 4),
        "win_rate": round(win_rate(trades), 4),
        "profit_factor": round(profit_factor(trades), 4),
        "total_trades": len(trades),
    }
