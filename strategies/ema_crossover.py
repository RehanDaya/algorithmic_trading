#!/usr/bin/env python3
"""
EMA Crossover Strategy
=====================

This module implements an EMA (Exponential Moving Average) crossover strategy
that generates buy signals when a fast EMA crosses above a slow EMA and sell
signals when the fast EMA crosses below the slow EMA.

EMA gives more weight to recent prices, making it more responsive to price changes
compared to Simple Moving Average (SMA). This makes it better for capturing
trend changes earlier, but also more prone to false signals.

Popular EMA combinations:
- 12/26 EMA (used in MACD)
- 8/21 EMA (short-term trading)
- 50/200 EMA (long-term trend following)

Author: Trading Bot System
Date: 2024
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy
    
    This strategy implements EMA crossover signals:
    - Long when fast EMA crosses above slow EMA (golden cross)
    - Close when fast EMA crosses below slow EMA (death cross)
    - Uses 12/26 EMA by default (same as MACD components)
    
    The strategy is more responsive than SMA crossover due to EMA's
    emphasis on recent price action, making it suitable for capturing
    trend changes earlier.
    """
    
    params = (
        ('ema_fast', 12),       # Fast EMA period
        ('ema_slow', 26),       # Slow EMA period
        ('printlog', False),
    )
    
    def init_indicators(self):
        """Initialize EMA indicators"""
        self.ema_fast = bt.indicators.EMA(
            self.data.close,
            period=self.params.ema_fast
        )
        
        self.ema_slow = bt.indicators.EMA(
            self.data.close,
            period=self.params.ema_slow
        )
        
        # Crossover indicator for cleaner signal detection
        self.ema_crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)
    
    def get_strategy_name(self) -> str:
        return "EMA Crossover"
    
    def get_strategy_description(self) -> str:
        return f"EMA({self.params.ema_fast}/{self.params.ema_slow}): Buy on golden cross, sell on death cross"
    
    def should_buy(self) -> bool:
        """Check if fast EMA crosses above slow EMA (golden cross)"""
        return self.ema_crossover[0] > 0
    
    def should_sell(self) -> bool:
        """Check if fast EMA crosses below slow EMA (death cross)"""
        return self.ema_crossover[0] < 0 