"""Generate backtest output files and human-readable summaries."""

import json
import os
from datetime import datetime
from typing import Dict


def save_backtest_result(result: Dict, output_dir: str = "results/") -> str:
    """Save a backtest result to a timestamped JSON file.

    Args:
        result: BacktestResult dict from Backtester.run().
        output_dir: Directory to write to.

    Returns:
        Path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert for JSON serialization (equity_curve already list from backtester)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return filepath


def save_parameter_report(sweep_results: Dict, output_dir: str = "results/") -> str:
    """Save parameter sweep results as a feedback JSON for T3/T5.

    Args:
        sweep_results: Dict from ParameterSweep.full_staged_sweep().
        output_dir: Directory to write to.

    Returns:
        Path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "parameter_report.json")

    report = {
        "generated_at": datetime.now().isoformat(),
        "best_parameters": sweep_results["best_parameters"],
        "best_metrics": sweep_results["best_metrics"],
        "sweep_summary": sweep_results["sweep_summary"],
    }

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return filepath


def format_summary(result: Dict) -> str:
    """Format a backtest result as a human-readable summary string.

    Args:
        result: BacktestResult dict from Backtester.run().

    Returns:
        Formatted summary string.
    """
    metrics = result["metrics"]
    params = result["parameters"]

    lines = [
        "=" * 50,
        f"  Backtest Results — {result.get('pair', 'N/A')} ({result.get('timeframe', 'N/A')})",
        "=" * 50,
        "",
        "Parameters:",
        f"  Fusion weight (technical): {params.get('fusion_weight_technical', 'N/A')}",
        f"  EMA: {params.get('ema_fast', 'N/A')}/{params.get('ema_slow', 'N/A')}",
        f"  RSI period: {params.get('rsi_period', 'N/A')}",
        f"  BB period: {params.get('bb_period', 'N/A')}",
        f"  Entry threshold: {params.get('entry_threshold', 'N/A')}",
        f"  Min confidence: {params.get('min_confidence', 'N/A')}",
        f"  Stop loss: {params.get('stop_loss_pct', 'N/A')}",
        "",
        "Metrics:",
        f"  Total return:   {metrics['total_return_pct']:+.2f}%",
        f"  Sharpe ratio:   {metrics['sharpe_ratio']:.4f}",
        f"  Sortino ratio:  {metrics['sortino_ratio']:.4f}",
        f"  Max drawdown:   {metrics['max_drawdown_pct']:.2f}%",
        f"  Win rate:       {metrics['win_rate']:.2%}",
        f"  Profit factor:  {metrics['profit_factor']:.4f}",
        f"  Total trades:   {metrics['total_trades']}",
        "=" * 50,
    ]

    return "\n".join(lines)
