#!/usr/bin/env python3
"""
Stochastic Oscillator Strategy
=============================

This module implements a Stochastic Oscillator strategy that generates buy signals
when the oscillator is in oversold territory and crosses upward, and sell signals
when it's in overbought territory and crosses downward.

The Stochastic Oscillator compares a security's closing price to its price range
over a specific period. It consists of two lines:
- %K line: (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
- %D line: 3-period SMA of %K line

Typical parameters:
- Period: 14 (for %K calculation)
- %D: 3-period SMA of %K
- Overbought: 80
- Oversold: 20

Author: Trading Bot System
Date: 2024
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class StochasticOscillatorStrategy(BaseStrategy):
    """
    Stochastic Oscillator Strategy
    
    This strategy implements Stochastic Oscillator signals:
    - Long when %K crosses above %D in oversold territory (< 20)
    - Close when %K crosses below %D in overbought territory (> 80)
    - Uses standard parameters: 14-period with 3-period %D smoothing
    
    The strategy aims to capture momentum reversals by identifying
    oversold and overbought conditions combined with momentum shifts.
    """
    
    params = (
        ('stoch_period', 14),       # Period for %K calculation
        ('stoch_period_dfast', 3),  # Period for %D (fast) calculation
        ('stoch_period_dslow', 3),  # Period for %D (slow) calculation
        ('oversold_level', 20),     # Oversold threshold
        ('overbought_level', 80),   # Overbought threshold
        ('printlog', False),
    )
    
    def init_indicators(self):
        """Initialize Stochastic Oscillator indicator"""
        self.stochastic = bt.indicators.Stochastic(
            self.data,
            period=self.params.stoch_period,
            period_dfast=self.params.stoch_period_dfast,
            period_dslow=self.params.stoch_period_dslow
        )
        
        # Store individual lines for easier access
        self.stoch_k = self.stochastic.lines.percK
        self.stoch_d = self.stochastic.lines.percD
        
        # Crossover indicator for signal detection
        self.stoch_crossover = bt.indicators.CrossOver(self.stoch_k, self.stoch_d)
    
    def get_strategy_name(self) -> str:
        return "Stochastic Oscillator"
    
    def get_strategy_description(self) -> str:
        return f"Stochastic({self.params.stoch_period}): Buy on oversold crossover, sell on overbought crossover"
    
    def should_buy(self) -> bool:
        """Check if %K crosses above %D in oversold territory"""
        return (self.stoch_crossover[0] > 0 and 
                self.stoch_k[0] < self.params.oversold_level and
                self.stoch_d[0] < self.params.oversold_level)
    
    def should_sell(self) -> bool:
        """Check if %K crosses below %D in overbought territory"""
        return (self.stoch_crossover[0] < 0 and 
                self.stoch_k[0] > self.params.overbought_level and
                self.stoch_d[0] > self.params.overbought_level) 