#!/usr/bin/env python3
"""
Momentum Strategy
================

This module implements a Momentum strategy that generates buy signals when
price momentum is strong and positive, and sell signals when momentum
weakens or turns negative.

Momentum is calculated as the rate of change in price over a specified period.
The strategy uses multiple momentum indicators to confirm signals:
- Rate of Change (ROC): Percentage change over N periods
- Momentum Oscillator: Current price - Price N periods ago

The strategy aims to capture trending moves by identifying when momentum
is accelerating in a particular direction.

Author: Trading Bot System
Date: 2024
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy
    
    This strategy implements momentum-based signals:
    - Long when Rate of Change (ROC) is above the buy threshold
    - Close when ROC falls below the sell threshold
    - Uses 14-period ROC by default with adaptive thresholds
    
    The strategy captures trending moves by identifying when price
    momentum is accelerating upward or decelerating downward.
    """
    
    params = (
        ('williams_period', 40),    # Williams %R period (as recommended)
        ('oversold_level', -80),    # Oversold threshold
        ('overbought_level', -20),  # Overbought threshold  
        ('signal_level', -50),      # Signal line for entry/exit
        ('printlog', False),
    )
    
    def init_indicators(self):
        """Initialize Williams %R momentum indicator"""
        # Williams %R indicator (the recommended momentum indicator)
        self.williams_r = bt.indicators.WilliamsR(
            self.data,
            period=self.params.williams_period
        )
        
        # Simple trend filter using moving averages
        self.sma_fast = bt.indicators.SMA(self.data.close, period=20)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=50)
    
    def get_strategy_name(self) -> str:
        return "Momentum"
    
    def get_strategy_description(self) -> str:
        return f"Williams %R({self.params.williams_period}): Buy oversold rally above {self.params.signal_level}, sell below {self.params.signal_level}"
    
    def should_buy(self) -> bool:
        """Check for Williams %R oversold rally signal"""
        # Buy when Williams %R was oversold and now rallies above signal level
        # Also check for uptrend (fast MA > slow MA)
        return (self.williams_r[0] > self.params.signal_level and 
                self.williams_r[-1] <= self.params.oversold_level and
                self.sma_fast[0] > self.sma_slow[0])
    
    def should_sell(self) -> bool:
        """Check for Williams %R overbought or signal level break"""
        return (self.williams_r[0] < self.params.signal_level or
                self.williams_r[0] > self.params.overbought_level) 