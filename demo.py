#!/usr/bin/env python3
"""
Demo Script - Multi-Strategy Trading System
===========================================

This script demonstrates how to use the reorganized multi-strategy framework.
Each strategy is now in its own file for better organization and maintainability.

Author: Trading Bot System
Date: 2024
"""

from trading_bot import TradingSystem
from strategies import list_available_strategies

def main():
    """Demonstrate the multi-strategy system"""
    print("🚀 MULTI-STRATEGY TRADING SYSTEM DEMO")
    print("="*50)
    
    # Show available strategies
    strategies = list_available_strategies()
    print(f"📈 Available Strategies ({len(strategies)}):")
    for i, (name, description) in enumerate(strategies.items(), 1):
        print(f"  {i}. {name}: {description}")
    
    print("\n🧪 Quick Test with RSI Strategy:")
    print("-" * 40)
    
    # Test RSI strategy quickly
    try:
        system = TradingSystem(
            symbol='AAPL', 
            benchmark='SPY', 
            strategy='rsi_mean_reversion'
        )
        
        # Use shorter data for quick demo
        system.fetch_data(period='1mo', interval='1d')
        system.run_backtest(initial_cash=10000.0)
        
        # Show quick results
        results = system.results
        print(f"✓ Strategy: RSI Mean Reversion")
        print(f"✓ Initial: ${results['initial_value']:,.2f}")
        print(f"✓ Final: ${results['final_value']:,.2f}")
        print(f"✓ Return: {results['total_return_pct']:.2f}%")
        
        print("\n💡 To run full analysis, use: python trading_bot.py")
        print("💡 To test all strategies, use: python test_strategies.py")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    print("\n📂 Project Structure:")
    print("strategies/")
    print("├── __init__.py          # Package registry")
    print("├── base_strategy.py     # Base class")
    print("├── rsi_mean_reversion.py # RSI strategy")
    print("└── ma_crossover.py      # MA crossover strategy")
    
    print("\n✅ Demo completed! Check the README.md for full documentation.")

if __name__ == "__main__":
    main() 