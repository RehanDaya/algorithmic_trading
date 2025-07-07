#!/usr/bin/env python3
"""
Parabolic SAR Strategy
=====================

This module implements a Parabolic SAR (Stop and Reverse) strategy that generates
buy and sell signals based on the Parabolic SAR indicator position relative to price.

The Parabolic SAR is a trend-following indicator that provides potential reversal
points in the price direction. It appears as dots above or below the price:
- Dots below price: Uptrend (bullish)
- Dots above price: Downtrend (bearish)

The indicator uses an acceleration factor (AF) that starts at 0.02 and increases
by 0.02 each time a new extreme is reached, up to a maximum of 0.20.

Key characteristics:
- Excellent for trending markets
- Provides clear stop-loss levels
- Can generate many false signals in sideways markets

Author: Trading Bot System
Date: 2024
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class ParabolicSARStrategy(BaseStrategy):
    """
    Parabolic SAR Strategy
    
    This strategy implements Parabolic SAR signals:
    - Long when price moves above the SAR dots (trend reversal to upside)
    - Close when price moves below the SAR dots (trend reversal to downside)
    - Uses standard SAR parameters: AF=0.02, AF_MAX=0.20
    
    The strategy is designed to capture trending moves and provides
    built-in stop-loss levels through the SAR dots.
    """
    
    params = (
        ('sar_af', 0.02),       # Acceleration Factor
        ('sar_afmax', 0.20),    # Maximum Acceleration Factor
        ('printlog', False),
    )
    
    def init_indicators(self):
        """Initialize Parabolic SAR indicator"""
        self.sar = bt.indicators.ParabolicSAR(
            self.data,
            af=self.params.sar_af,
            afmax=self.params.sar_afmax
        )
        
        # Track previous SAR position for signal detection
        self.sar_signal = bt.indicators.CrossOver(self.data.close, self.sar)
    
    def get_strategy_name(self) -> str:
        return "Parabolic SAR"
    
    def get_strategy_description(self) -> str:
        return f"Parabolic SAR(AF={self.params.sar_af}, MAX={self.params.sar_afmax}): Buy when price > SAR, sell when price < SAR"
    
    def should_buy(self) -> bool:
        """Check if price crosses above SAR (bullish reversal)"""
        return self.sar_signal[0] > 0
    
    def should_sell(self) -> bool:
        """Check if price crosses below SAR (bearish reversal)"""
        return self.sar_signal[0] < 0 