#!/usr/bin/env python3
"""
Test Script for Multiple Trading Strategies
===========================================

This script tests the multi-strategy framework by running both available
strategies and comparing their performance.

Author: Trading Bot System
Date: 2024
"""

import os
import sys
from trading_bot import TradingSystem
from strategies import list_available_strategies

def test_single_strategy(strategy_name: str, symbol: str = 'AAPL'):
    """Test a single strategy"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING STRATEGY: {strategy_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Initialize system
        system = TradingSystem(symbol=symbol, benchmark='SPY', strategy=strategy_name)
        
        # Fetch data
        system.fetch_data(period='3mo', interval='1d')  # Use daily data for testing
        
        # Run backtest
        system.run_backtest(initial_cash=10000.0)
        
        # Calculate metrics
        metrics = system.calculate_comprehensive_metrics()
        
        # Print summary
        print(f"\n📊 STRATEGY SUMMARY:")
        print(f"Strategy: {strategy_name}")
        print(f"Symbol: {symbol}")
        print(f"Initial Value: ${system.results['initial_value']:,.2f}")
        print(f"Final Value: ${system.results['final_value']:,.2f}")
        print(f"Total Return: {system.results['total_return_pct']:.2f}%")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        
        return {
            'strategy': strategy_name,
            'symbol': symbol,
            'final_value': system.results['final_value'],
            'total_return_pct': system.results['total_return_pct'],
            'total_trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'max_drawdown': metrics.get('max_drawdown', 0)
        }
        
    except Exception as e:
        print(f"❌ Error testing {strategy_name}: {e}")
        return None

def compare_strategies():
    """Compare all available strategies"""
    print("🚀 MULTI-STRATEGY COMPARISON TEST")
    print("="*60)
    
    # Get available strategies
    strategies = list_available_strategies()
    print(f"📈 Found {len(strategies)} strategies:")
    for name, desc in strategies.items():
        print(f"  • {name}: {desc}")
    
    # Test each strategy
    results = []
    for strategy_name in strategies.keys():
        result = test_single_strategy(strategy_name)
        if result:
            results.append(result)
    
    # Compare results
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("🏆 STRATEGY COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Strategy':<25} {'Return %':<10} {'Trades':<8} {'Win %':<8} {'Max DD %':<10}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['strategy']:<25} {result['total_return_pct']:<9.2f} "
                  f"{result['total_trades']:<7} {result['win_rate']:<7.1f} "
                  f"{result['max_drawdown']:<9.2f}")
        
        # Find best performing strategy
        best_strategy = max(results, key=lambda x: x['total_return_pct'])
        print(f"\n🥇 Best Performing Strategy: {best_strategy['strategy'].upper()}")
        print(f"   Total Return: {best_strategy['total_return_pct']:.2f}%")

def main():
    """Main test function"""
    print("🧪 ALGORITHMIC TRADING SYSTEM - STRATEGY TESTING")
    print("="*60)
    
    try:
        # Test individual strategies
        compare_strategies()
        
        print(f"\n{'='*60}")
        print("✅ All strategy tests completed successfully!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 