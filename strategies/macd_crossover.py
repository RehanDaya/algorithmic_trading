#!/usr/bin/env python3
"""
MACD Crossover Strategy
======================

This module implements a MACD (Moving Average Convergence Divergence) crossover strategy
that generates buy signals when the MACD line crosses above the signal line and sell
signals when the MACD line crosses below the signal line.

MACD Components:
- MACD Line: 12-period EMA - 26-period EMA
- Signal Line: 9-period EMA of MACD Line
- Histogram: MACD Line - Signal Line

This is a momentum-based trend-following strategy that works well in trending markets.

Author: Trading Bot System
Date: 2024
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class MACDCrossoverStrategy(BaseStrategy):
    """
    MACD Crossover Strategy
    
    This strategy implements MACD crossover signals:
    - Long when MACD line crosses above the signal line (bullish crossover)
    - Close when MACD line crosses below the signal line (bearish crossover)
    - Uses standard MACD parameters: 12, 26, 9
    
    The strategy captures momentum changes and trend reversals by monitoring
    the relationship between the MACD line and its signal line.
    """
    
    params = (
        ('macd_fast', 12),      # Fast EMA period
        ('macd_slow', 26),      # Slow EMA period  
        ('macd_signal', 9),     # Signal line EMA period
        ('printlog', False),
    )
    
    def init_indicators(self):
        """Initialize MACD indicator"""
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        
        # Store individual lines for easier access
        self.macd_line = self.macd.lines.macd
        self.signal_line = self.macd.lines.signal
        
        # Crossover indicators for cleaner signal detection
        self.macd_crossover = bt.indicators.CrossOver(self.macd_line, self.signal_line)
    
    def get_strategy_name(self) -> str:
        return "MACD Crossover"
    
    def get_strategy_description(self) -> str:
        return f"MACD({self.params.macd_fast}, {self.params.macd_slow}, {self.params.macd_signal}): Buy on bullish crossover, sell on bearish crossover"
    
    def should_buy(self) -> bool:
        """Check if MACD line crosses above signal line (bullish crossover)"""
        return self.macd_crossover[0] > 0
    
    def should_sell(self) -> bool:
        """Check if MACD line crosses below signal line (bearish crossover)"""
        return self.macd_crossover[0] < 0 