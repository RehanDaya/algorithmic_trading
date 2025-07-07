#!/usr/bin/env python3
"""
Strategies Package
=================

This package contains all available trading strategies.
Each strategy inherits from BaseStrategy and implements specific trading logic.

Available Strategies:
- RSI Mean Reversion: Buy when RSI < 30, sell when RSI > 70
- Moving Average Crossover: Buy on golden cross, sell on death cross  
- Buy and Hold: Buy at start and hold until end (benchmark)
- Bollinger Bands: Mean reversion using Bollinger Bands
- MACD Crossover: Buy/sell on MACD line crossovers with signal line
- EMA Crossover: Fast/slow EMA crossover signals
- Stochastic Oscillator: Overbought/oversold signals with momentum confirmation
- Momentum: Rate of change based trend following
- Parabolic SAR: Trend following with stop and reverse signals
- Bollinger RSI: Combined Bollinger Bands and RSI for enhanced signals
"""

from typing import Dict, Type
from .base_strategy import BaseStrategy
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .ma_crossover import MovingAverageCrossoverStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .bollinger_bands import BollingerBandsStrategy
from .macd_crossover import MACDCrossoverStrategy
from .ema_crossover import EMACrossoverStrategy
from .stochastic_oscillator import StochasticOscillatorStrategy
from .momentum import MomentumStrategy
from .parabolic_sar import ParabolicSARStrategy
from .bollinger_rsi import BollingerRSIStrategy

# Registry of available strategies
AVAILABLE_STRATEGIES = {
    'rsi_mean_reversion': RSIMeanReversionStrategy,
    'ma_crossover': MovingAverageCrossoverStrategy,
    'buy_and_hold': BuyAndHoldStrategy,
    'bollinger_bands': BollingerBandsStrategy,
    'macd_crossover': MACDCrossoverStrategy,
    'ema_crossover': EMACrossoverStrategy,
    'stochastic_oscillator': StochasticOscillatorStrategy,
    'momentum': MomentumStrategy,
    'parabolic_sar': ParabolicSARStrategy,
    'bollinger_rsi': BollingerRSIStrategy,
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
            strategies[name] = "Description unavailable"
    
    return strategies


# Export main classes and functions
__all__ = [
    'BaseStrategy',
    'RSIMeanReversionStrategy', 
    'MovingAverageCrossoverStrategy',
    'BuyAndHoldStrategy',
    'BollingerBandsStrategy',
    'MACDCrossoverStrategy',
    'EMACrossoverStrategy',
    'StochasticOscillatorStrategy',
    'MomentumStrategy',
    'ParabolicSARStrategy',
    'BollingerRSIStrategy',
    'get_strategy_class',
    'list_available_strategies',
    'AVAILABLE_STRATEGIES'
] 