# trading-backtester · T4

Backtests T3 signal output against historical OHLCV data and runs a staged parameter sweep over signal configurations, with walk-forward (out-of-sample) validation. The best-scoring parameters are saved to `results/parameter_report.json`. The tests run without a database connection.

---

## Trading Arc

| Repo | Role |
|------|------|
| T1 · crypto-data-pipeline | Live OHLCV ingestion · market event tagging |
| T2 · trading-chart-generator | Candlestick PNGs + JSON sidecars |
| T3 · trading-signal-engine | Technical indicators + FinBERT sentiment |
| **T4 · trading-backtester** | Backtesting + parameter sweep |
| T5 · trading-dashboard | Streamlit oversight UI · [Live Demo](https://mtichikawa-trading.streamlit.app) |

---

## Architecture

```
T1 OHLCV data (PostgreSQL or synthetic)
        │
        ▼
DataLoader ──── load_from_db() / generate_synthetic()
        │
        ▼
Indicators ──── EMA crossover, RSI, MACD, Bollinger Bands
        │       (pure pandas/numpy, parameter-configurable)
        ▼
Strategy ────── Signal threshold entry/exit + stop loss
        │
        ▼
Backtester ──── Iterates candles, tracks equity curve
        │
        ▼
ParameterSweep ─ 3-stage grid: weights → indicators → thresholds
        │
        ▼
Report ──────── parameter_report.json
```

---

## Modules

| File | Purpose |
|------|---------|
| `src/config.py` | `BacktestConfig` + `IndicatorParams` dataclasses |
| `src/data_loader.py` | Load OHLCV from T1 PostgreSQL, CSV, or synthetic generator |
| `src/indicators.py` | EMA, RSI, MACD, Bollinger — pure pandas |
| `src/strategy.py` | Signal-threshold entry/exit with stop loss |
| `src/backtester.py` | Core engine: runs strategy, builds equity curve |
| `src/metrics.py` | Sharpe, Sortino, max drawdown, win rate, profit factor |
| `src/parameter_sweep.py` | 3-stage staged sweep |
| `src/walk_forward.py` | Walk-forward validation: optimize on each train window, score on the next unseen window, compare to buy-and-hold |
| `src/report.py` | JSON output + parameter_report.json |

---

## Parameter Sweep

Exhaustive grid search would require 17,000+ combinations. T4 uses a 3-stage staged approach testing ~60–80 configs total:

1. **Fusion weights** — Test technical/sentiment weight ratios (0.3–0.8). Find best Sharpe.
2. **Indicator parameters** — Using best weight, sweep EMA periods, RSI period, Bollinger window.
3. **Entry thresholds** — Using best weight + indicators, sweep signal threshold and stop-loss.

Each stage passes its best config to the next. Total runtime is a fraction of exhaustive search.

## Sweep Output

T4 writes `parameter_report.json`, the best-scoring configuration and its metrics, to its results directory. Nothing reads it automatically. T3's fusion weights and sentiment threshold are configurable, so a sweep result can be applied there by hand.

---

## Metrics

All metrics computed in `src/metrics.py`:

| Metric | Description |
|--------|-------------|
| Sharpe ratio | Annualized, assuming hourly candles (8,760 periods/year) |
| Sortino ratio | Annualized, downside deviation only |
| Max drawdown | Worst peak-to-trough decline |
| Win rate | Fraction of trades with positive P&L |
| Profit factor | Gross profit / gross loss |

---

## Setup

```bash
cd trading-backtester
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Standalone demo — no database needed
python examples/quick_demo.py

# Tests
pytest tests/ -v
# all run with synthetic data, no DB required
```

---

## Contact

Mike Ichikawa · [projects.ichikawa@gmail.com](mailto:projects.ichikawa@gmail.com) · [mtichikawa.github.io](https://mtichikawa.github.io)
