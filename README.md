# trading-backtester · T4

Backtests T3 signal output against historical OHLCV data and runs a staged parameter sweep to find the optimal signal configuration. The winning parameters are written back to T3 as a feedback artifact. 72/72 tests pass without a database connection.

---

## Trading Arc

| Repo | Role | Status |
|------|------|--------|
| T1 · crypto-data-pipeline | Live OHLCV ingestion · market event tagging | Shipped Mar 6 |
| T2 · trading-chart-generator | Candlestick PNGs + JSON sidecars · 25/25 tests | Shipped Mar 10 |
| T3 · trading-signal-engine | Technical indicators + FinBERT sentiment · 51/51 tests | Shipped Mar 16 |
| **T4 · trading-backtester** | Backtesting + parameter sweep · 72/72 tests | Shipped Mar 26 |
| T5 · trading-dashboard | Streamlit oversight UI · 8/8 tests | Shipped Mar 31 · [Live Demo](https://mtichikawa-trading.streamlit.app) |

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
Report ──────── parameter_report.json → T3 + T5
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
| `src/report.py` | JSON output + parameter_report.json for T3/T5 |

---

## Parameter Sweep

Exhaustive grid search would require 17,000+ combinations. T4 uses a 3-stage staged approach testing ~60–80 configs total:

1. **Fusion weights** — Test technical/sentiment weight ratios (0.3–0.8). Find best Sharpe.
2. **Indicator parameters** — Using best weight, sweep EMA periods, RSI period, Bollinger window.
3. **Entry thresholds** — Using best weight + indicators, sweep signal threshold and stop-loss.

Each stage passes its best config to the next. Total runtime is a fraction of exhaustive search.

## The Feedback Loop

T4 writes `parameter_report.json` — the winning configuration — to its results directory. T3 reads this on next run and updates its signal generation defaults. T5's Parameters page surfaces the sweep results for review before the loop closes.

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
# 72/72 — all run with synthetic data, no DB required
```

---

## Contact

Mike Ichikawa · [projects.ichikawa@gmail.com](mailto:projects.ichikawa@gmail.com) · [mtichikawa.github.io](https://mtichikawa.github.io)
