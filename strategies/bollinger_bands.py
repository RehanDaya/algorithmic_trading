#!/usr/bin/env python3
"""
Bollinger Bands Mean Reversion Strategy
======================================

This module implements a Bollinger Bands mean reversion strategy that buys when price
touches the lower band and sells when price touches the upper band.

Based on the concept that prices tend to revert to the mean after extreme moves.
Bollinger Bands consist of:
- Middle Band: Simple Moving Average (typically 20 periods)
- Upper Band: Middle Band + (2 × Standard Deviation)
- Lower Band: Middle Band - (2 × Standard Deviation)

Author: Trading Bot System
Date: 2024
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy
    
    This strategy implements mean reversion using Bollinger Bands:
    - Long when price touches or goes below the lower Bollinger Band
    - Close when price touches or goes above the upper Bollinger Band
    - Uses standard Bollinger Bands parameters: 20-period SMA with 2 standard deviations
    
    The strategy assumes that extreme price moves (touching the bands) are likely
    to revert back towards the mean (middle band).
    """
    
    params = (
        ('bb_period', 20),      # Period for the moving average
        ('bb_devfactor', 2.0),  # Standard deviation factor
        ('printlog', False),
    )
    
    def init_indicators(self):
        """Initialize Bollinger Bands indicator"""
        self.bollinger = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_devfactor
        )
        
        # Store individual bands for easier access
        self.bb_top = self.bollinger.lines.top
        self.bb_mid = self.bollinger.lines.mid
        self.bb_bot = self.bollinger.lines.bot
    
    def get_strategy_name(self) -> str:
        return "Bollinger Bands Mean Reversion"
    
    def get_strategy_description(self) -> str:
        return f"Bollinger Bands({self.params.bb_period}, {self.params.bb_devfactor}): Buy at lower band, sell at upper band"
    
    def should_buy(self) -> bool:
        """Check if price is at or below the lower Bollinger Band"""
        return (self.data.close[0] <= self.bb_bot[0] and 
                self.data.close[-1] > self.bb_bot[-1])  # Price just touched or crossed below
    
    def should_sell(self) -> bool:
        """Check if price is at or above the upper Bollinger Band"""
        return (self.data.close[0] >= self.bb_top[0] and 
                self.data.close[-1] < self.bb_top[-1])  # Price just touched or crossed above 