#!/usr/bin/env python3
"""
Base Strategy Class
==================

This module contains the base strategy class that all trading strategies inherit from.
It provides common functionality for logging, trade tracking, and performance monitoring.

Author: Trading Bot System
Date: 2024
"""

import backtrader as bt
from typing import Dict, List, Optional


class BaseStrategy(bt.Strategy):
    """
    Base Strategy Class
    
    Abstract base class for all trading strategies. Provides common functionality
    for logging, trade tracking, and performance monitoring.
    """
    
    params = (
        ('printlog', False),
    )
    
    def __init__(self):
        """Initialize base strategy components"""
        # Track orders and trades
        self.order = None
        self.trade_count = 0
        self.entry_price = 0
        self.entry_time = None
        self.trades_log = []
        
        # For trade duration calculation
        self.position_entry_time = None
        self.trade_durations = []
        
        # Initialize strategy-specific indicators
        self.init_indicators()
    
    def init_indicators(self):
        """Initialize strategy-specific indicators - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement init_indicators()")
    
    def get_strategy_name(self) -> str:
        """Return strategy name - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_strategy_name()")
    
    def get_strategy_description(self) -> str:
        """Return strategy description - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_strategy_description()")
    
    def should_buy(self) -> bool:
        """Check if buy condition is met - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement should_buy()")
    
    def should_sell(self) -> bool:
        """Check if sell condition is met - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement should_sell()")
    
    def calculate_position_size(self) -> int:
        """Calculate the maximum number of shares that can be bought with available cash"""
        available_cash = self.broker.getcash()
        current_price = self.data.close[0]
        
        # Calculate max shares (accounting for commission)
        # Leave a small buffer to account for commission
        max_shares = int(available_cash * 0.99 / current_price)
        
        # Ensure we buy at least 1 share if we have enough cash
        return max(1, max_shares) if available_cash >= current_price else 0
    
    def log(self, txt, dt=None):
        """Logging function for strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        if self.params.printlog:
            print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Handle order status notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
                self.entry_price = order.executed.price
                self.entry_time = self.datas[0].datetime.datetime(0)
                self.position_entry_time = len(self.data)
                
            elif order.issell():
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
                
                # Calculate trade duration
                if self.position_entry_time:
                    duration = len(self.data) - self.position_entry_time
                    self.trade_durations.append(duration)
                    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.status}')
            
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications"""
        if not trade.isclosed:
            return
            
        self.trade_count += 1
        pnl = trade.pnl
        pnl_pct = (trade.pnl / trade.price) * 100
        
        # Log trade details
        trade_info = {
            'trade_num': self.trade_count,
            'entry_date': self.entry_time,
            'exit_date': self.datas[0].datetime.datetime(0),
            'entry_price': self.entry_price,
            'exit_price': trade.price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'size': trade.size
        }
        self.trades_log.append(trade_info)
        
        self.log(f'TRADE #{self.trade_count} CLOSED: '
                f'PnL: ${pnl:.2f} ({pnl_pct:.2f}%)')
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Skip if we have a pending order
        if self.order:
            return
            
        # Check if we're not in a position
        if not self.position:
            # Check buy condition
            if self.should_buy():
                position_size = self.calculate_position_size()
                if position_size > 0:
                    self.log(f'{self.get_strategy_name()}: BUY SIGNAL - Size: {position_size} shares')
                    self.order = self.buy(size=position_size)
                else:
                    self.log(f'{self.get_strategy_name()}: BUY SIGNAL - Insufficient cash')
                
        else:
            # Check sell condition
            if self.should_sell():
                # Sell all shares we currently own
                current_position = self.position.size
                self.log(f'{self.get_strategy_name()}: SELL SIGNAL - Size: {current_position} shares')
                self.order = self.sell(size=current_position) 