#!/usr/bin/env python3
"""
Buy and Hold Strategy
====================

This module implements a simple buy-and-hold strategy that purchases the asset
at the beginning of the timeframe and holds it until the end.
"""

import backtrader as bt
from .base_strategy import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy and Hold Strategy
    
    This strategy implements a simple buy-and-hold approach:
    - Buy on the first available trading day
    - Hold the position until the end of the backtest period
    - Never sells unless forced to at the end
    """
    
    params = (
        ('printlog', False),
    )
    
    def __init__(self):
        """Initialize buy and hold strategy"""
        super().__init__()
        self.has_bought = False  # Track if we've made our initial purchase
    
    def init_indicators(self):
        """No indicators needed for buy and hold"""
        pass
    
    def get_strategy_name(self) -> str:
        return "Buy and Hold"
    
    def get_strategy_description(self) -> str:
        return "Buy at start, hold until end (benchmark strategy)"
    
    def should_buy(self) -> bool:
        """Buy only once at the beginning"""
        return not self.has_bought and not self.position
    
    def should_sell(self) -> bool:
        """Never sell in buy and hold strategy"""
        return False
    
    def next(self):
        """Buy and hold logic - buy once and never sell"""
        # Skip if we have a pending order
        if self.order:
            return
            
        # Buy once at the beginning if we haven't bought yet
        if not self.has_bought and not self.position:
            self.log(f'BUY AND HOLD: Initial purchase')
            # Use order_target_percent to invest 95% of capital
            self.order = self.order_target_percent(target=0.95)
            self.has_bought = True
        
        # After buying, just hold - no selling logic 