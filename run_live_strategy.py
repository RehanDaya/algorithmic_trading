#!/usr/bin/env python3
"""
Live Strategy Runner
====================

Example script showing how to run a backtesting strategy in live trading mode.

Usage:
    python run_live_strategy.py
"""

import signal
import sys
from live_trading import LiveTradingEngine
from strategies import list_available_strategies


def main():
    """Run a live trading strategy"""
    print("=" * 70)
    print("LIVE STRATEGY TRADING")
    print("=" * 70)
    
    # Initialize engine
    engine = LiveTradingEngine(paper_trading=True)
    
    # Connect to Alpaca
    if not engine.connect():
        print("❌ Failed to connect. Please check your API credentials.")
        return
    
    # Display available strategies (exclude buy_and_hold and base_strategy)
    print("\n📊 Available Strategies:")
    strategies = list_available_strategies()
    live_strategies = {k: v for k, v in strategies.items() 
                      if k not in ['base_strategy', 'buy_and_hold']}
    
    for i, (name, desc) in enumerate(live_strategies.items(), 1):
        print(f"  {i}. {name}: {desc}")
    
    # Get user input
    try:
        strategy_choice = input("\nSelect strategy (name or number): ").strip()
        
        # Convert number to name if needed
        strategy_list = list(live_strategies.keys())
        if strategy_choice.isdigit():
            idx = int(strategy_choice) - 1
            if 0 <= idx < len(strategy_list):
                strategy_name = strategy_list[idx]
            else:
                print("❌ Invalid strategy number")
                return
        else:
            strategy_name = strategy_choice
            if strategy_name not in live_strategies:
                if strategy_name == 'buy_and_hold':
                    print("❌ 'buy_and_hold' is not available for live trading (benchmark only)")
                else:
                    print(f"❌ Strategy '{strategy_name}' not found")
                return
        
        symbol = input("Enter symbol to trade (e.g., AAPL, SPY): ").strip().upper()
        if not symbol:
            print("❌ Symbol is required")
            return
        
        # Get strategy parameters (use defaults for now)
        print(f"\nUsing default parameters for {strategy_name}")
        print("(You can modify parameters in the code if needed)")
        
        # Default parameters based on strategy
        default_params = {
            'rsi_mean_reversion': {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70},
            'ma_crossover': {'short_period': 10, 'long_period': 50},
            'macd_crossover': {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9},
            'ema_crossover': {'ema_fast': 12, 'ema_slow': 26},
            'bollinger_bands': {'bb_period': 20, 'bb_std': 2},
            'stochastic_oscillator': {'stoch_k': 14, 'stoch_d': 3, 'stoch_oversold': 20, 'stoch_overbought': 80},
            'momentum': {'williams_period': 40, 'oversold_level': -80, 'overbought_level': -20, 'signal_level': -50},
            'parabolic_sar': {'sar_af': 0.02, 'sar_afmax': 0.20},
            'bollinger_rsi': {'bb_period': 20, 'bb_devfactor': 2.0, 'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70},
        }
        
        parameters = default_params.get(strategy_name, {})
        
        # Get check interval
        interval_input = input("Check interval in seconds (default: 60): ").strip()
        interval_seconds = int(interval_input) if interval_input.isdigit() else 60
        
        # Start strategy
        print(f"\n🚀 Starting {strategy_name} for {symbol}")
        print("   Press Ctrl+C to stop\n")
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\n\n⏹️  Stopping strategy...")
            engine.stop_strategy()
            engine.disconnect()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        engine.start_strategy(symbol, strategy_name, parameters, interval_seconds)
        
        # Keep main thread alive
        try:
            while engine.strategy_running:
                time.sleep(1)
        except KeyboardInterrupt:
            signal_handler(None, None)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        engine.disconnect()


if __name__ == "__main__":
    import time
    main()

