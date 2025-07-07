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
        ('momentum_period', 14),    # Period for momentum calculation
        ('roc_period', 14),         # Period for Rate of Change calculation
        ('buy_threshold', 0.5),     # ROC threshold for buy signal (%)
        ('sell_threshold', -0.5),   # ROC threshold for sell signal (%)
        ('printlog', False),
    )
    
    def init_indicators(self):
        """Initialize momentum indicators"""
        # Rate of Change indicator
        self.roc = bt.indicators.RateOfChange(
            self.data.close,
            period=self.params.roc_period
        )
        
        # Momentum oscillator (price - price N periods ago)
        self.momentum = bt.indicators.Momentum(
            self.data.close,
            period=self.params.momentum_period
        )
        
        # Moving average of ROC for trend confirmation
        self.roc_ma = bt.indicators.SMA(
            self.roc,
            period=5
        )
    
    def get_strategy_name(self) -> str:
        return "Momentum"
    
    def get_strategy_description(self) -> str:
        return f"Momentum({self.params.roc_period}): Buy ROC > {self.params.buy_threshold}%, sell ROC < {self.params.sell_threshold}%"
    
    def should_buy(self) -> bool:
        """Check if momentum is strong and positive"""
        return self.roc[0] > self.params.buy_threshold
    
    def should_sell(self) -> bool:
        """Check if momentum is weakening or negative"""
        return self.roc[0] < self.params.sell_threshold 