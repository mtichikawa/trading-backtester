"""Backtest configuration with sweep ranges and default parameters."""



from dataclasses import dataclass, field

from typing import List, Optional





@dataclass

class IndicatorParams:

    """Parameters for technical indicator computation."""

    ema_fast: int = 12

    ema_slow: int = 26

    rsi_period: int = 14

    macd_fast: int = 12

    macd_slow: int = 26

    macd_signal: int = 9

    bb_period: int = 20

    bb_std: float = 2.0





@dataclass

class BacktestConfig:  # default initial_equity: 10_000

    """Full backtest configuration including strategy and sweep ranges."""



    # Trading pair and timeframe

    pair: str = "BTC/USD"

    timeframe: str = "1h"



    # Indicator parameters

    indicator_params: IndicatorParams = field(default_factory=IndicatorParams)



    # Fusion weight (technical component, remainder is sentiment)

    fusion_weight_technical: float = 0.6



    # Strategy thresholds

    entry_threshold: float = 0.15

    exit_threshold: float = 0.0

    stop_loss_pct: float = 0.05

    min_confidence: float = 0.5



    # Position sizing

    position_size_fraction: float = 1.0  # fraction of equity per trade



    # Trading costs (researched, not guessed — applied per side, so a round-trip
    # entry+exit trade pays 2x the sum below).
    #   taker_fee_bps=40  -> Kraken Pro base-tier spot taker fee 0.40% (0-$10k 30d vol).
    #                        Signal/crossover strategies send market orders that cross
    #                        the spread, so they pay taker, not maker. Source:
    #                        https://www.kraken.com/features/fee-schedule
    #   slippage_bps=10   -> 0.10% per side, conservative end of the normal-condition
    #                        0.05-0.30% range for top-10 liquid pairs (BTC/ETH/SOL).
    # Round-trip cost at defaults: 2 * (0.40% + 0.10%) = 1.00% per trade.

    taker_fee_bps: float = 40.0

    slippage_bps: float = 10.0



    # Short selling

    short_enabled: bool = False



    # Initial equity

    initial_equity: float = 10000.0



    # Sweep ranges

    sweep_fusion_weights: List[float] = field(

        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    )

    sweep_ema_fast: List[int] = field(

        default_factory=lambda: [9, 12, 15]

    )

    sweep_ema_slow: List[int] = field(

        default_factory=lambda: [21, 26, 34]

    )

    sweep_rsi_period: List[int] = field(

        default_factory=lambda: [10, 14, 21]

    )

    sweep_bb_period: List[int] = field(

        default_factory=lambda: [15, 20, 25]

    )

    sweep_entry_threshold: List[float] = field(

        default_factory=lambda: [0.1, 0.15, 0.2, 0.25]

    )

    sweep_min_confidence: List[float] = field(

        default_factory=lambda: [0.3, 0.5, 0.6]

    )

