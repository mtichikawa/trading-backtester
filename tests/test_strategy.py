"""Tests for the trading strategy logic."""

import pandas as pd
import pytest

from src.strategy import Strategy, PositionState


def _make_df(signals, confidences=None, prices=None):
    """Helper to create a signal DataFrame for testing."""
    n = len(signals)
    if confidences is None:
        confidences = [0.8] * n
    if prices is None:
        prices = [100.0 + i * 0.1 for i in range(n)]

    return pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "close": prices,
        "signal": signals,
        "confidence": confidences,
    })


class TestLongEntry:
    def test_enters_long_on_strong_signal(self):
        # Signal above threshold with high confidence
        signals = [0.0, 0.0, 0.5, 0.5, -0.1]
        df = _make_df(signals)
        strategy = Strategy(entry_threshold=0.3, min_confidence=0.6)
        trades = strategy.generate_trades(df)
        assert len(trades) >= 1
        assert trades[0]["side"] == "long"

    def test_no_entry_on_weak_signal(self):
        # Signal below threshold
        signals = [0.1, 0.15, 0.1, 0.05, 0.0]
        df = _make_df(signals)
        strategy = Strategy(entry_threshold=0.3, min_confidence=0.6)
        trades = strategy.generate_trades(df)
        assert len(trades) == 0

    def test_no_entry_on_low_confidence(self):
        # Strong signal but low confidence
        signals = [0.0, 0.5, 0.5, -0.1, -0.2]
        confidences = [0.3, 0.3, 0.3, 0.3, 0.3]
        df = _make_df(signals, confidences)
        strategy = Strategy(entry_threshold=0.3, min_confidence=0.6)
        trades = strategy.generate_trades(df)
        assert len(trades) == 0


class TestExitLogic:
    def test_exit_on_signal_reversal(self):
        # Enter long, then signal drops below exit threshold
        signals = [0.0, 0.5, 0.4, -0.1, -0.2]
        df = _make_df(signals)
        strategy = Strategy(entry_threshold=0.3, exit_threshold=0.0, min_confidence=0.6)
        trades = strategy.generate_trades(df)
        assert len(trades) == 1
        assert trades[0]["side"] == "long"
        assert trades[0]["exit_time"] is not None

    def test_stop_loss_trigger(self):
        # Enter long, then price drops significantly
        signals = [0.0, 0.5, 0.5, 0.5, 0.5]
        prices = [100.0, 100.0, 90.0, 85.0, 80.0]  # 10%+ drop
        df = _make_df(signals, prices=prices)
        strategy = Strategy(
            entry_threshold=0.3, exit_threshold=-1.0,
            stop_loss_pct=0.05, min_confidence=0.6
        )
        trades = strategy.generate_trades(df)
        assert len(trades) >= 1
        assert trades[0]["pnl_pct"] < 0

    def test_profitable_long_trade(self):
        signals = [0.0, 0.5, 0.4, 0.3, -0.1]
        prices = [100.0, 100.0, 105.0, 108.0, 110.0]
        df = _make_df(signals, prices=prices)
        strategy = Strategy(entry_threshold=0.3, exit_threshold=0.0, min_confidence=0.6)
        trades = strategy.generate_trades(df)
        assert len(trades) == 1
        assert trades[0]["pnl_pct"] > 0


class TestShortEntry:
    def test_short_entry_when_enabled(self):
        signals = [0.0, -0.5, -0.4, 0.1, 0.2]
        df = _make_df(signals)
        strategy = Strategy(
            entry_threshold=0.3, exit_threshold=0.0,
            min_confidence=0.6, short_enabled=True
        )
        trades = strategy.generate_trades(df)
        assert len(trades) >= 1
        assert trades[0]["side"] == "short"

    def test_no_short_when_disabled(self):
        signals = [0.0, -0.5, -0.4, 0.1, 0.2]
        df = _make_df(signals)
        strategy = Strategy(
            entry_threshold=0.3, min_confidence=0.6, short_enabled=False
        )
        trades = strategy.generate_trades(df)
        assert len(trades) == 0


class TestEdgeCases:
    def test_no_signals_above_threshold(self):
        signals = [0.0, 0.1, -0.1, 0.05, 0.0]
        df = _make_df(signals)
        strategy = Strategy(entry_threshold=0.5, min_confidence=0.6)
        trades = strategy.generate_trades(df)
        assert len(trades) == 0

    def test_trade_records_entry_signal_and_confidence(self):
        signals = [0.0, 0.6, 0.5, -0.1, -0.2]
        confidences = [0.5, 0.9, 0.85, 0.7, 0.6]
        df = _make_df(signals, confidences)
        strategy = Strategy(entry_threshold=0.3, exit_threshold=0.0, min_confidence=0.6)
        trades = strategy.generate_trades(df)
        assert len(trades) == 1
        assert trades[0]["signal_strength"] == pytest.approx(0.6)
        assert trades[0]["confidence"] == pytest.approx(0.9)

    def test_multiple_trades(self):
        # Two distinct entry/exit cycles
        signals = [0.0, 0.5, -0.1, 0.0, 0.5, -0.1]
        df = _make_df(signals)
        strategy = Strategy(entry_threshold=0.3, exit_threshold=0.0, min_confidence=0.6)
        trades = strategy.generate_trades(df)
        assert len(trades) == 2
