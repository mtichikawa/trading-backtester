"""Signal-based trading strategy with entry/exit rules."""

from enum import Enum
from typing import Dict, List

import pandas as pd


class PositionState(Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


class Strategy:
    """Signal-threshold strategy for backtesting.

    Opens long when signal exceeds entry_threshold with sufficient confidence.
    Opens short (if enabled) when signal is below -entry_threshold.
    Exits on signal reversal past exit_threshold or stop loss.
    """

    def __init__(
        self,
        entry_threshold: float = 0.3,
        exit_threshold: float = 0.0,
        stop_loss_pct: float = 0.05,
        min_confidence: float = 0.6,
        short_enabled: bool = False,
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_pct = stop_loss_pct
        self.min_confidence = min_confidence
        self.short_enabled = short_enabled

    def generate_trades(self, df: pd.DataFrame) -> List[Dict]:
        """Generate round-trip trades from a DataFrame with signal and confidence columns.

        Args:
            df: DataFrame with columns: open_time, close, signal, confidence

        Returns:
            List of trade dicts with entry/exit info and PnL.
        """
        trades = []
        state = PositionState.FLAT
        entry_price = 0.0
        entry_time = None
        entry_signal = 0.0
        entry_confidence = 0.0

        for i in range(len(df)):
            row = df.iloc[i]
            signal = row["signal"]
            confidence = row["confidence"]
            price = row["close"]
            time = row["open_time"]

            if state == PositionState.FLAT:
                # Check for long entry
                if signal > self.entry_threshold and confidence > self.min_confidence:
                    state = PositionState.LONG
                    entry_price = price
                    entry_time = time
                    entry_signal = signal
                    entry_confidence = confidence
                # Check for short entry
                elif (
                    self.short_enabled
                    and signal < -self.entry_threshold
                    and confidence > self.min_confidence
                ):
                    state = PositionState.SHORT
                    entry_price = price
                    entry_time = time
                    entry_signal = signal
                    entry_confidence = confidence

            elif state == PositionState.LONG:
                unrealized_pnl = (price - entry_price) / entry_price

                # Exit conditions: signal reversal or stop loss
                exit_trade = False
                if signal < self.exit_threshold:
                    exit_trade = True
                elif unrealized_pnl < -self.stop_loss_pct:
                    exit_trade = True

                if exit_trade:
                    pnl_pct = (price - entry_price) / entry_price
                    trades.append({
                        "entry_time": str(entry_time),
                        "exit_time": str(time),
                        "entry_price": float(entry_price),
                        "exit_price": float(price),
                        "side": "long",
                        "pnl_pct": float(pnl_pct),
                        "signal_strength": float(entry_signal),
                        "confidence": float(entry_confidence),
                    })
                    state = PositionState.FLAT

            elif state == PositionState.SHORT:
                unrealized_pnl = (entry_price - price) / entry_price

                exit_trade = False
                if signal > -self.exit_threshold:
                    exit_trade = True
                elif unrealized_pnl < -self.stop_loss_pct:
                    exit_trade = True

                if exit_trade:
                    pnl_pct = (entry_price - price) / entry_price
                    trades.append({
                        "entry_time": str(entry_time),
                        "exit_time": str(time),
                        "entry_price": float(entry_price),
                        "exit_price": float(price),
                        "side": "short",
                        "pnl_pct": float(pnl_pct),
                        "signal_strength": float(entry_signal),
                        "confidence": float(entry_confidence),
                    })
                    state = PositionState.FLAT

        return trades
