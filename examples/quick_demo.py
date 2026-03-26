"""Quick demo: runs a backtest + mini parameter sweep with synthetic data.

Works standalone — no database, no T1/T3, no external services required.

Usage:
    cd projects-hub/trading-backtester
    source venv/bin/activate
    python examples/quick_demo.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import BacktestConfig
from src.data_loader import DataLoader
from src.backtester import Backtester
from src.parameter_sweep import ParameterSweep
from src.report import format_summary


def main():
    print("=" * 60)
    print("  T4 Trading Backtester — Quick Demo")
    print("=" * 60)
    print()

    # Generate synthetic OHLCV data
    loader = DataLoader()
    df = loader.generate_synthetic(n_candles=500)
    print(f"Generated {len(df)} synthetic candles")
    print(f"  Date range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")
    print(f"  Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    print()

    # --- Single backtest with default params ---
    print("-" * 60)
    print("  Single Backtest (default parameters)")
    print("-" * 60)

    config = BacktestConfig()
    backtester = Backtester(config)
    result = backtester.run(df)

    print(format_summary(result))
    print()

    # --- Mini parameter sweep ---
    print("-" * 60)
    print("  Mini Parameter Sweep (4 fusion weights)")
    print("-" * 60)

    # Use a smaller sweep for the demo
    config.sweep_fusion_weights = [0.4, 0.5, 0.6, 0.7]
    config.sweep_ema_fast = [9, 12]
    config.sweep_ema_slow = [26, 34]
    config.sweep_rsi_period = [14]
    config.sweep_bb_period = [20]
    config.sweep_entry_threshold = [0.1, 0.15]
    config.sweep_min_confidence = [0.3, 0.5]

    sweep = ParameterSweep(Backtester(config))
    sweep_results = sweep.full_staged_sweep(df)

    best = sweep_results["best_parameters"]
    best_metrics = sweep_results["best_metrics"]
    summary = sweep_results["sweep_summary"]

    print(f"\n  Combinations tested: {summary['total_combinations_tested']}")
    print(f"  Best Sharpe: {summary['best_sharpe']:.4f}")
    print(f"  Worst Sharpe: {summary['worst_sharpe']:.4f}")
    print()
    print("  Best parameters found:")
    print(f"    Fusion weight: {best.get('fusion_weight_technical', 'N/A')}")
    print(f"    EMA: {best.get('ema_fast', 'N/A')}/{best.get('ema_slow', 'N/A')}")
    print(f"    RSI period: {best.get('rsi_period', 'N/A')}")
    print(f"    Entry threshold: {best.get('entry_threshold', 'N/A')}")
    print(f"    Min confidence: {best.get('min_confidence', 'N/A')}")
    print()
    print("  Best metrics:")
    print(f"    Total return: {best_metrics['total_return_pct']:+.2f}%")
    print(f"    Sharpe: {best_metrics['sharpe_ratio']:.4f}")
    print(f"    Win rate: {best_metrics['win_rate']:.2%}")
    print(f"    Max drawdown: {best_metrics['max_drawdown_pct']:.2f}%")
    print(f"    Trades: {best_metrics['total_trades']}")
    print()
    print("=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
