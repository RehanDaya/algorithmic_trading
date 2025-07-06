#!/usr/bin/env python3
"""
RSI Mean Reversion Strategy
==========================

This module implements an RSI mean reversion strategy that buys when RSI is oversold
and sells when RSI is overbought.

Author: Trading Bot System
Date: 2024
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy
    
    This strategy implements a simple mean reversion approach using RSI:
    - Long when RSI < oversold_threshold (default: 30)
    - Close when RSI > overbought_threshold (default: 70)
    - Fixed position size for consistent risk management
    """
    
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('position_size', 1),
        ('printlog', False),
    )
    
    def init_indicators(self):
        """Initialize RSI indicator"""
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.params.rsi_period,
            safediv=True
        )
    
    def get_strategy_name(self) -> str:
        return "RSI Mean Reversion"
    
    def get_strategy_description(self) -> str:
        return f"RSI({self.params.rsi_period}): Buy < {self.params.rsi_oversold}, Sell > {self.params.rsi_overbought}"
    
    def should_buy(self) -> bool:
        """Check if RSI is oversold"""
        # Skip if we don't have enough data
        if len(self.data) < self.params.rsi_period:
            return False
            
        current_rsi = self.rsi[0]
        return current_rsi < self.params.rsi_oversold
    
    def should_sell(self) -> bool:
        """Check if RSI is overbought"""
        current_rsi = self.rsi[0]
        return current_rsi > self.params.rsi_overbought
    
    def next(self):
        """Override next to add RSI logging"""
        # Skip if we have a pending order
        if self.order:
            return
        
        # Skip if we don't have enough data
        if len(self.data) < self.params.rsi_period:
            return
            
        current_rsi = self.rsi[0]
        
        # Check if we're not in a position
        if not self.position:
            # Enter long position when RSI is oversold
            if current_rsi < self.params.rsi_oversold:
                self.log(f'RSI OVERSOLD: {current_rsi:.2f} - BUYING')
                self.order = self.buy(size=self.params.position_size)
                
        else:
            # Exit long position when RSI is overbought
            if current_rsi > self.params.rsi_overbought:
                self.log(f'RSI OVERBOUGHT: {current_rsi:.2f} - SELLING')
                self.order = self.sell(size=self.params.position_size) 