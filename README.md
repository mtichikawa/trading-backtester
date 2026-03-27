# T4 — Trading Backtester

Backtests trading signals against historical OHLCV price data. Sweeps indicator parameters (EMA, RSI, MACD, Bollinger Bands) and fusion weights to find optimal configurations. Reports Sharpe ratio, Sortino ratio, max drawdown, win rate, and profit factor per parameter set. Writes optimized parameters as JSON feedback for T3 (signal engine) and T5 (dashboard) consumption.

Part of the T1–T5 trading system arc — a portfolio showcase built with zero API costs and no paid services.

## Architecture

```
T1 ohlcv table (BTC/USD, ETH/USD, SOL/USD)
        |
        v
DataLoader ──── load_from_db() / generate_synthetic()
        |
        v
Indicators ──── EMA crossover, RSI, MACD, Bollinger Bands
        |                (pure pandas/numpy, parameter-configurable)
        v
Strategy ────── Signal threshold entry/exit + stop loss
        |
        v
Backtester ──── Iterates candles, tracks equity curve
        |
        v
ParameterSweep ─ 3-stage grid: weights → indicators → thresholds
        |
        v
Report ──────── JSON results + parameter_report.json (→ T3/T5)
```

## Quick Start

```bash
cd projects-hub/trading-backtester
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the demo (no database needed)
python examples/quick_demo.py

# Run tests
pytest tests/ -v
```

## Modules

| File | Purpose |
|---|---|
| `src/config.py` | `BacktestConfig` dataclass — parameters, sweep ranges, defaults |
| `src/data_loader.py` | Load OHLCV from T1 PostgreSQL, CSV, or synthetic generator |
| `src/indicators.py` | Technical indicators: EMA, RSI, MACD, Bollinger — all pure pandas |
| `src/strategy.py` | Signal-threshold strategy: long/short entry, exit, stop loss |
| `src/backtester.py` | Core engine: runs strategy on data, builds equity curve |
| `src/metrics.py` | Sharpe, Sortino, max drawdown, win rate, profit factor |
| `src/parameter_sweep.py` | 3-stage parameter sweep (weights → indicators → thresholds) |
| `src/report.py` | JSON output + human-readable summaries |
| `examples/quick_demo.py` | Standalone demo with synthetic data |

## Parameter Sweep Methodology

The sweep avoids exhaustive grid search (17K+ combinations) by using three sequential stages:

1. **Fusion weights** — Test 6 values (0.3–0.8) with default indicator params. Find the best technical/sentiment weight balance.

2. **Indicator parameters** — Using the best weight, sweep EMA fast/slow periods, RSI period, and Bollinger period. Find optimal indicator tuning.

3. **Entry thresholds** — Using best weight + indicators, sweep entry threshold and minimum confidence. Find the right selectivity.

Each stage ranks results by Sharpe ratio and passes the best config to the next stage. Total combinations tested: ~50–80 (vs 17K+ exhaustive).

## Output Format

### Backtest result (`results/backtest_*.json`)

```json
{
    "pair": "BTC/USD",
    "timeframe": "1h",
    "parameters": {
        "fusion_weight_technical": 0.6,
        "ema_fast": 12, "ema_slow": 26,
        "rsi_period": 14,
        "entry_threshold": 0.15,
        "min_confidence": 0.5
    },
    "metrics": {
        "total_return_pct": 8.4,
        "sharpe_ratio": 1.42,
        "sortino_ratio": 1.89,
        "max_drawdown_pct": -5.2,
        "win_rate": 0.58,
        "profit_factor": 1.73,
        "total_trades": 47
    }
}
```

### Parameter report (`results/parameter_report.json`)

Feedback artifact consumed by T3 (signal engine) to update its default parameters and by T5 (dashboard) to display optimization results.

## Connecting to the Trading Arc

- **T1 (crypto-data-pipeline)** — Provides OHLCV price data via PostgreSQL
- **T3 (trading-signal-engine)** — Produces the signals this backtester evaluates; receives optimized parameters back via `parameter_report.json`
- **T5 (trading-dashboard)** — Displays backtest results and parameter sweep summaries

## The Parameter Feedback Loop

The most interesting engineering problem in this project: T3 produces signals with default parameters, T4 backtests those parameters against historical data and discovers which configurations would have performed best, then writes a `parameter_report.json` that T3 can read to update its defaults. This creates a feedback loop where the system self-optimizes — the backtester evaluates the signal engine's output and tells it how to improve.

The staged sweep design keeps this tractable. Rather than testing every possible combination, each stage narrows the search space before the next stage begins. This mirrors how a quant would manually tune parameters: first get the big picture right (fusion weights), then fine-tune indicators, then calibrate entry/exit sensitivity.
