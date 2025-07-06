#!/usr/bin/env python3
"""
Trading Strategies Package
==========================

This package contains various trading strategies for algorithmic trading.
Each strategy is implemented in its own module and inherits from BaseStrategy.

Available Strategies:
1. RSI Mean Reversion Strategy
2. Moving Average Crossover Strategy

Author: Trading Bot System
Date: 2024
"""

from typing import Dict, Type
from .base_strategy import BaseStrategy
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .ma_crossover import MovingAverageCrossoverStrategy

# Strategy registry for easy access
AVAILABLE_STRATEGIES: Dict[str, Type[BaseStrategy]] = {
    'rsi_mean_reversion': RSIMeanReversionStrategy,
    'ma_crossover': MovingAverageCrossoverStrategy,
}


def get_strategy_class(strategy_name: str) -> Type[BaseStrategy]:
    """
    Get strategy class by name
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Strategy class or None if not found
    """
    return AVAILABLE_STRATEGIES.get(strategy_name.lower())


def list_available_strategies() -> Dict[str, str]:
    """
    List all available strategies with descriptions
    
    Returns:
        Dictionary mapping strategy names to descriptions
    """
    strategies = {}
    for name, strategy_class in AVAILABLE_STRATEGIES.items():
        # Create temporary instance to get description
        try:
            # We need to create a minimal instance without calling __init__
            # to avoid backtrader initialization issues
            temp_instance = strategy_class.__new__(strategy_class)
            temp_instance.params = strategy_class.params
            strategies[name] = temp_instance.get_strategy_description()
        except Exception:
            # Fallback to class name if description fails
            strategies[name] = strategy_class.__name__
    
    return strategies


# Export main classes and functions
__all__ = [
    'BaseStrategy',
    'RSIMeanReversionStrategy', 
    'MovingAverageCrossoverStrategy',
    'get_strategy_class',
    'list_available_strategies',
    'AVAILABLE_STRATEGIES'
] 