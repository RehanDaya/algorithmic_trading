#!/usr/bin/env python3
"""
Moving Average Crossover Strategy
=================================

This module implements a moving average crossover strategy that buys on golden cross
and sells on death cross.

Author: Trading Bot System
Date: 2024
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    This strategy uses two moving averages to generate signals:
    - Buy when short MA crosses above long MA (golden cross)
    - Sell when short MA crosses below long MA (death cross)
    - Uses Simple Moving Averages (SMA) by default
    """
    
    params = (
        ('short_period', 10),
        ('long_period', 50),
        ('printlog', False),
    )
    
    def init_indicators(self):
        """Initialize moving average indicators"""
        self.short_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.short_period
        )
        self.long_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.long_period
        )
        
        # Crossover signals
        self.crossover = bt.indicators.CrossOver(
            self.short_ma,
            self.long_ma
        )
    
    def get_strategy_name(self) -> str:
        return "Moving Average Crossover"
    
    def get_strategy_description(self) -> str:
        return f"MA({self.params.short_period}) x MA({self.params.long_period}): Buy on golden cross, Sell on death cross"
    
    def should_buy(self) -> bool:
        """Check if short MA crosses above long MA"""
        # Skip if we don't have enough data
        if len(self.data) < self.params.long_period:
            return False
            
        # Check for golden cross (crossover > 0)
        return self.crossover[0] > 0
    
    def should_sell(self) -> bool:
        """Check if short MA crosses below long MA"""
        # Check for death cross (crossover < 0)
        return self.crossover[0] < 0
    
    def next(self):
        """Override next to add MA crossover logging"""
        # Skip if we have a pending order
        if self.order:
            return
        
        # Skip if we don't have enough data
        if len(self.data) < self.params.long_period:
            return
            
        short_ma = self.short_ma[0]
        long_ma = self.long_ma[0]
        crossover = self.crossover[0]
        
        # Check if we're not in a position
        if not self.position:
            # Enter long position on golden cross
            if crossover > 0:
                self.log(f'GOLDEN CROSS: MA({self.params.short_period})={short_ma:.2f} > MA({self.params.long_period})={long_ma:.2f} - BUYING')
                # Use backtrader's built-in order_target_percent
                self.order = self.order_target_percent(target=0.99)
                
        else:
            # Exit long position on death cross
            if crossover < 0:
                self.log(f'DEATH CROSS: MA({self.params.short_period})={short_ma:.2f} < MA({self.params.long_period})={long_ma:.2f} - SELLING')
                # Close position using backtrader's built-in method
                self.order = self.order_target_percent(target=0.0) 