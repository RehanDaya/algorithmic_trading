#!/usr/bin/env python3
"""
Interactive Strategy Analysis & Comparison Tool
==============================================

This script provides a user interface for interactively analyzing and comparing
trading strategies. It orchestrates the backtesting_engine.py core functionality
with user input handling and result presentation.

RESPONSIBILITIES:
- User input collection (strategy selection, parameters)
- Multi-strategy comparison logic
- Results aggregation and comparison
- Interactive plotting options

SEPARATION OF CONCERNS:
- Core backtesting logic: backtesting_engine.py (TradingSystem class)
- User interface & orchestration: run_strategies.py (this file)
- Strategy implementations: strategies/ package


"""

import os
import sys
import pandas as pd

from backtesting_engine import TradingSystem
from strategies import list_available_strategies

def get_user_strategy_selection():
    """Get user's strategy selection"""
    print("🎯 STRATEGY SELECTION")
    print("="*50)
    
    # Get available strategies (excluding base strategy)
    all_strategies = list_available_strategies()
    available_strategies = {k: v for k, v in all_strategies.items() if k != 'base_strategy'}
    
    if len(available_strategies) < 1:
        print(f"❌ Error: No strategies available.")
        return None
    
    print(f"📈 Available Strategies ({len(available_strategies)}):")
    strategy_list = list(available_strategies.keys())
    for i, (name, desc) in enumerate(available_strategies.items(), 1):
        print(f"  {i}. {name}: {desc}")
    
    print(f"\n  {len(strategy_list) + 1}. ALL STRATEGIES")
    print("\nSelect strategies to analyze:")
    print("  - Enter a single strategy number (e.g., 1) for detailed analysis")
    print("  - Enter multiple strategy numbers separated by commas (e.g., 1,2) for comparison")
    print("  - Or enter 'all' to compare all strategies")
    
    while True:
        try:
            user_input = input(f"\nYour selection (default: all): ").strip().lower() or 'all'
            
            if user_input == 'all' or user_input == str(len(strategy_list) + 1):
                return list(available_strategies.keys())
            
            # Parse comma-separated numbers
            selected_numbers = [int(x.strip()) for x in user_input.split(',')]
            
            # Validate numbers
            if any(num < 1 or num > len(strategy_list) for num in selected_numbers):
                print(f"❌ Please enter numbers between 1 and {len(strategy_list)}.")
                continue
            
            # Convert to strategy names
            selected_strategies = [strategy_list[num - 1] for num in selected_numbers]
            
            print(f"\n✅ Selected strategies: {', '.join(selected_strategies)}")
            return selected_strategies
            
        except ValueError:
            print("❌ Please enter valid numbers separated by commas or 'all'.")
        except KeyboardInterrupt:
            print("\n\n❌ Selection cancelled.")
            return None

def get_user_parameters():
    """Get user's analysis parameters"""
    print("\n🔧 ANALYSIS PARAMETERS")
    print("="*50)
    
    # Symbol selection
    print("1. Trading Symbol:")
    print("   Popular options: AAPL, MSFT, GOOGL, AMZN, TSLA, SPY, QQQ, NVDA")
    symbol = input("   Enter symbol (default: SPY): ").strip().upper() or 'SPY'
    
    # Benchmark selection
    print("\n2. Benchmark Symbol:")
    print("   Popular options: SPY, QQQ, IWM, DIA")
    benchmark = input("   Enter benchmark (default: SPY): ").strip().upper() or 'SPY'
    
    # Period selection
    print("\n3. Data Period:")
    print("   Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
    period = input("   Enter period (default: 2y): ").strip().lower() or '2y'
    
    # Interval selection
    print("\n4. Data Interval:")
    print("   Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")
    print("   Note: Intraday data (< 1d) is limited to last 60 days")
    interval = input("   Enter interval (default: 1d): ").strip().lower() or '1d'
    
    # Initial cash
    print("\n5. Initial Capital:")
    while True:
        try:
            cash_input = input("   Enter initial cash (default: 10000): ").strip()
            initial_cash = float(cash_input) if cash_input else 10000.0
            if initial_cash <= 0:
                print("   ❌ Initial cash must be positive.")
                continue
            break
        except ValueError:
            print("   ❌ Please enter a valid number.")
    
    # Plot display option
    print("\n6. Display Plots:")
    print("   Show interactive plots for strategies? (can slow down analysis)")
    show_plots_input = input("   Display plots? (y/n) [default: n]: ").strip().lower() or 'n'
    show_plots = show_plots_input in ['y', 'yes', 'true', '1']
    
    return {
        'symbol': symbol,
        'benchmark': benchmark,
        'period': period,
        'interval': interval,
        'initial_cash': initial_cash,
        'show_plots': show_plots
    }

def print_detailed_analysis(system, metrics, fscore, strategy_name):
    """Print detailed analysis using the TradingSystem's built-in print_results method"""
    # Use the existing print_results method from TradingSystem to avoid duplication
    system.print_results(metrics, fscore)

def create_strategy_summary(strategy_name, system, metrics, parameters):
    """
    Create strategy summary dictionary - extracted common data structure
    
    Args:
        strategy_name: Name of the strategy
        system: TradingSystem instance
        metrics: Metrics dictionary
        parameters: Analysis parameters
        
    Returns:
        Dictionary with strategy summary data
    """
    # Helper function to safely get numeric values
    def safe_get(value, default=0):
        if value is None or str(value) == 'nan':
            return default
        return value
    
    return {
        'strategy': strategy_name,
        'symbol': parameters['symbol'],
        'final_value': safe_get(system.results['final_value'], 10000),
        'total_return_pct': safe_get(system.results['total_return_pct'], 0),
        'total_trades': safe_get(metrics.get('total_trades', 0), 0),
        'win_rate': safe_get(metrics.get('win_rate', 0), 0),
        'max_drawdown': safe_get(metrics.get('max_drawdown', 0), 0),
        'sharpe_ratio': safe_get(metrics.get('sharpe_ratio', 0), 0),
        'annualized_return': safe_get(metrics.get('annualized_return', 0), 0),
        'sortino_ratio': safe_get(metrics.get('sortino_ratio', 0), 0),
        'calmar_ratio': safe_get(metrics.get('calmar_ratio', 0), 0),
        'beta': safe_get(metrics.get('beta', 0), 0),
        'alpha': safe_get(metrics.get('alpha', 0), 0)
    }


def run_strategy_analysis(selected_strategies, parameters):
    """Run analysis for selected strategies with given parameters"""
    print(f"\n🚀 RUNNING STRATEGY ANALYSIS")
    print("="*60)
    print(f"Symbol: {parameters['symbol']}")
    print(f"Benchmark: {parameters['benchmark']}")
    print(f"Period: {parameters['period']}")
    print(f"Interval: {parameters['interval']}")
    print(f"Initial Cash: ${parameters['initial_cash']:,.2f}")
    print(f"Strategies: {', '.join(selected_strategies)}")
    
    # Check if single strategy or multiple
    is_single_strategy = len(selected_strategies) == 1
    
    # Fetch data once for all strategies
    print(f"\n{'='*60}")
    print(f"🧪 ANALYZING {len(selected_strategies)} STRATEGIES")
    print(f"{'='*60}")
    
    # Create a dummy system just to fetch data once
    dummy_system = TradingSystem(
        symbol=parameters['symbol'], 
        benchmark=parameters['benchmark'], 
        strategy=selected_strategies[0]
    )
    dummy_system.fetch_data(period=parameters['period'], interval=parameters['interval'])
    fscore = dummy_system.calculate_piotroski_fscore()
    
    # Run each strategy
    results = []
    strategy_systems = []
    
    for strategy_name in selected_strategies:
        
        try:
            # Initialize system
            system = TradingSystem(
                symbol=parameters['symbol'], 
                benchmark=parameters['benchmark'], 
                strategy=strategy_name
            )
            
            # Reuse the fetched data
            system.data = dummy_system.data
            system.benchmark_data = dummy_system.benchmark_data
            system.current_interval = dummy_system.current_interval
            
            # Run backtest
            system.run_backtest(initial_cash=parameters['initial_cash'])
            
            # Calculate comprehensive metrics
            metrics = system.calculate_comprehensive_metrics()
            
            # Store system for detailed analysis
            strategy_systems.append((system, metrics, fscore, strategy_name))
            
            # For single strategy, show detailed analysis immediately
            if is_single_strategy:
                print_detailed_analysis(system, metrics, fscore, strategy_name)
                # Generate plot for single strategy
                if parameters['show_plots']:
                    print(f"\n📊 GENERATING PLOTS FOR {strategy_name.upper()}")
                    print("="*60)
                system.plot_results(show_plots=parameters['show_plots'])
            else:
                # For multiple strategies, just continue without printing individual summaries
                pass
            
            results.append(create_strategy_summary(strategy_name, system, metrics, parameters))
        except Exception as e:
            print(f"❌ Error analyzing {strategy_name}: {e}")
    
    # For multiple strategies, show comparison and detailed analysis
    if not is_single_strategy and len(results) > 1:
        print(f"\n{'='*135}")
        print("🏆 STRATEGY COMPARISON SUMMARY")
        print(f"{'='*135}")
        
        # Create DataFrame from results
        df_data = []
        for result in results:
            df_data.append({
                'Strategy': result['strategy'],
                'Final Value': f"${result['final_value']:,.0f}",
                'Return %': f"{result['total_return_pct']:.2f}%",
                'Ann Ret %': f"{result['annualized_return']:.2f}%",
                'Trades': result['total_trades'],
                'Win %': f"{result['win_rate']:.1f}%",
                'Max DD %': f"{result['max_drawdown']:.2f}%",
                'Sharpe': f"{result['sharpe_ratio']:.3f}",
                'Alpha': f"{result['alpha']:.4f}",
                'Beta': f"{result['beta']:.3f}"
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Find best performing strategy by return
        best_by_return = max(results, key=lambda x: x['total_return_pct'])
        print(f"\n🥇 Best Return: {best_by_return['strategy'].upper()}")
        print(f"   Total Return: {best_by_return['total_return_pct']:.2f}%")
        
        # Find best performing strategy by Sharpe ratio
        best_by_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
        print(f"\n📈 Best Risk-Adjusted Return: {best_by_sharpe['strategy'].upper()}")
        print(f"   Sharpe Ratio: {best_by_sharpe['sharpe_ratio']:.3f}")
        
        # Find strategy with lowest drawdown
        best_by_drawdown = min(results, key=lambda x: abs(x['max_drawdown']))
        print(f"\n🛡️  Lowest Drawdown: {best_by_drawdown['strategy'].upper()}")
        print(f"   Max Drawdown: {best_by_drawdown['max_drawdown']:.2f}%")
        
        # Detailed analysis removed - only showing comparison table above
        
        # Generate plots for all selected strategies
        if parameters['show_plots']:
            print(f"\n{'='*135}")
            print("📊 GENERATING PLOTS FOR ALL SELECTED STRATEGIES")
            print(f"{'='*135}")
            for system, metrics, fscore, strategy_name in strategy_systems:
                system.plot_results(show_plots=True)
        else:
            # Don't generate plots if not requested
            pass

    elif len(results) == 0:
        print(f"\n❌ Could not analyze any strategies successfully.")

def main():
    """Main interactive analysis function"""
    print("🧪 INTERACTIVE STRATEGY ANALYSIS & COMPARISON TOOL")
    print("="*70)
    print("Welcome! This tool allows you to analyze single strategies or compare")
    print("multiple strategies with comprehensive metrics and customizable parameters.\n")
    
    try:
        # Get user strategy selection
        selected_strategies = get_user_strategy_selection()
        if not selected_strategies:
            print("❌ No strategies selected. Exiting.")
            return
        
        # Get user parameters
        parameters = get_user_parameters()
        
        # Determine analysis type
        analysis_type = "Single Strategy Analysis" if len(selected_strategies) == 1 else "Multi-Strategy Comparison"
        
        # Confirm selection
        print(f"\n{'='*60}")
        print("📋 CONFIRMATION")
        print(f"{'='*60}")
        print(f"Analysis Type: {analysis_type}")
        print(f"Selected strategies: {', '.join(selected_strategies)}")
        print(f"Symbol: {parameters['symbol']}")
        print(f"Benchmark: {parameters['benchmark']}")
        print(f"Period: {parameters['period']}")
        print(f"Interval: {parameters['interval']}")
        print(f"Initial Cash: ${parameters['initial_cash']:,.2f}")
        print(f"Display Plots: {'Yes' if parameters['show_plots'] else 'No'}")
        
        confirm = input("\nProceed with analysis? (y/n) [default: y]: ").strip().lower() or 'y'
        if confirm not in ['y', 'yes']:
            print("❌ Analysis cancelled.")
            return
        
        # Run analysis
        run_strategy_analysis(selected_strategies, parameters)
        
        print(f"\n{'='*60}")
        print("✅ Strategy analysis completed successfully!")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\n❌ Analysis cancelled by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 