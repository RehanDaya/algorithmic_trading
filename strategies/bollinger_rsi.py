#!/usr/bin/env python3
"""
Bollinger Bands + RSI Combined Strategy
======================================

This module implements a combined strategy using both Bollinger Bands and RSI
indicators to generate more reliable trading signals by filtering out false
signals that either indicator might produce individually.

The strategy combines:
- Bollinger Bands: For identifying overbought/oversold price levels
- RSI: For confirming momentum conditions

This combination helps reduce false signals by requiring both indicators
to confirm the trading condition before generating a signal.

Entry Conditions:
- Buy: Price touches lower Bollinger Band AND RSI is oversold (< 30)
- Sell: Price touches upper Bollinger Band AND RSI is overbought (> 70)

Author: Trading Bot System
Date: 2024
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class BollingerRSIStrategy(BaseStrategy):
    """
    Bollinger Bands + RSI Combined Strategy
    
    This strategy combines Bollinger Bands and RSI for enhanced signal quality:
    - Long when price ≤ lower BB AND RSI < oversold threshold
    - Close when price ≥ upper BB AND RSI > overbought threshold
    - Uses standard parameters: BB(20,2) and RSI(14)
    
    The combination helps filter false signals by requiring both
    price extremes (Bollinger Bands) and momentum confirmation (RSI).
    """
    
    params = (
        ('bb_period', 20),          # Bollinger Bands period
        ('bb_devfactor', 2.0),      # Bollinger Bands deviation factor
        ('rsi_period', 14),         # RSI period
        ('rsi_oversold', 30),       # RSI oversold threshold
        ('rsi_overbought', 70),     # RSI overbought threshold
        ('printlog', False),
    )
    
    def init_indicators(self):
        """Initialize Bollinger Bands and RSI indicators"""
        # Bollinger Bands
        self.bollinger = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_devfactor
        )
        
        self.bb_top = self.bollinger.lines.top
        self.bb_mid = self.bollinger.lines.mid
        self.bb_bot = self.bollinger.lines.bot
        
        # RSI
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.params.rsi_period,
            safediv=True
        )
    
    def get_strategy_name(self) -> str:
        return "Bollinger Bands + RSI"
    
    def get_strategy_description(self) -> str:
        return f"BB({self.params.bb_period},{self.params.bb_devfactor}) + RSI({self.params.rsi_period}): Combined oversold/overbought signals"
    
    def should_buy(self) -> bool:
        """Check if both BB and RSI indicate oversold conditions"""
        bb_oversold = self.data.close[0] <= self.bb_bot[0]
        rsi_oversold = self.rsi[0] < self.params.rsi_oversold
        
        return bb_oversold and rsi_oversold
    
    def should_sell(self) -> bool:
        """Check if both BB and RSI indicate overbought conditions"""
        bb_overbought = self.data.close[0] >= self.bb_top[0]
        rsi_overbought = self.rsi[0] > self.params.rsi_overbought
        
        return bb_overbought and rsi_overbought 