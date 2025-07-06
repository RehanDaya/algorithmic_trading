#!/usr/bin/env python3
"""
Backtesting Engine - Core trading system using backtrader framework
"""

import os
import sys
import datetime
import warnings
from typing import Dict, List, Tuple, Optional
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core data manipulation and environment
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Market data and backtesting
import yfinance as yf
import backtrader as bt
from backtrader import cerebro, strategy, indicators

# Import strategies
from strategies import get_strategy_class, list_available_strategies

# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load environment variables
load_dotenv()


class TradingSystem:
    """
    Main Trading System Class
    
    Handles data fetching, backtesting, and performance analysis
    """
    
    def __init__(self, symbol: str = 'AAPL', benchmark: str = 'SPY', strategy: str = 'rsi_mean_reversion'):
        """
        Initialize the trading system
        
        Args:
            symbol: Primary trading symbol (default: AAPL)
            benchmark: Benchmark symbol for comparison (default: SPY)
            strategy: Strategy name (default: rsi_mean_reversion)
        """
        self.symbol = symbol
        self.benchmark = benchmark
        self.strategy_name = strategy
        self.strategy_class = get_strategy_class(strategy)
        self.risk_free_rate = float(os.getenv('RISK_FREE_RATE', 0.02))
        
        if not self.strategy_class:
            raise ValueError(f"Strategy '{strategy}' not found. Available strategies: {list(list_available_strategies().keys())}")
        

        
        # Data storage
        self.data = None
        self.benchmark_data = None
        self.cerebro = None
        self.strategy_instance = None
        self.current_interval = '15m'  # Store current interval for timeframe mapping
        
        # Results storage
        self.results = {}
        self.trades = []
    
    def _get_timeframe_compression(self, interval: str) -> Tuple[int, int]:
        """
        Convert interval string to backtrader timeframe and compression
        
        Args:
            interval: Data interval (e.g., '1d', '1h', '15m')
            
        Returns:
            Tuple of (timeframe, compression)
        """
        interval_map = {
            # Minutes
            '1m': (bt.TimeFrame.Minutes, 1),
            '2m': (bt.TimeFrame.Minutes, 2),
            '5m': (bt.TimeFrame.Minutes, 5),
            '15m': (bt.TimeFrame.Minutes, 15),
            '30m': (bt.TimeFrame.Minutes, 30),
            '60m': (bt.TimeFrame.Minutes, 60),
            '90m': (bt.TimeFrame.Minutes, 90),
            '1h': (bt.TimeFrame.Minutes, 60),
            
            # Days
            '1d': (bt.TimeFrame.Days, 1),
            '5d': (bt.TimeFrame.Days, 5),
            
            # Weeks
            '1wk': (bt.TimeFrame.Weeks, 1),
            '1w': (bt.TimeFrame.Weeks, 1),
            
            # Months
            '1mo': (bt.TimeFrame.Months, 1),
            '3mo': (bt.TimeFrame.Months, 3),
        }
        
        return interval_map.get(interval.lower(), (bt.TimeFrame.Days, 1))

    def fetch_data(self, period: str = '6mo', interval: str = '15m') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical data using yfinance
        
        Args:
            period: Data period (default: 6mo)
            interval: Data interval (default: 15m)
            
        Returns:
            Tuple of (primary_data, benchmark_data)
        """
        try:
            # Store interval for timeframe mapping
            self.current_interval = interval
            
            print(f"Fetching {period} of {interval} data for {self.symbol} and {self.benchmark}...")
            
            # Adjust period for intraday data limitations
            if interval in ['1m', '5m', '15m', '30m', '1h']:
                # yfinance limits intraday data to last 60 days
                if period in ['6mo', '1y', '2y', '5y', 'max']:
                    period = '60d'
                    print(f"⚠️  Adjusted period to {period} due to intraday data limitations")
            
            # Fetch primary symbol data
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=period, interval=interval)
            
            # Fetch benchmark data
            benchmark_ticker = yf.Ticker(self.benchmark)
            self.benchmark_data = benchmark_ticker.history(period=period, interval=interval)
            
            # If intraday data fails, fallback to daily data with original period
            if self.data.empty or self.benchmark_data.empty:
                print(f"⚠️  Intraday data unavailable, falling back to daily data for {period}")
                self.data = ticker.history(period='6mo', interval='1d')
                self.benchmark_data = benchmark_ticker.history(period='6mo', interval='1d')
                self.current_interval = '1d'  # Update stored interval
                
            if self.data.empty or self.benchmark_data.empty:
                raise ValueError("No data retrieved from yfinance")
                
            print(f"✓ Retrieved {len(self.data)} bars for {self.symbol}")
            print(f"✓ Retrieved {len(self.benchmark_data)} bars for {self.benchmark}")
            
            return self.data, self.benchmark_data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise
    
    def calculate_piotroski_fscore(self) -> Dict:
        """
        Calculate Piotroski F-Score for fundamental analysis
        
        METRIC DEFINITION:
        Piotroski F-Score: A 9-point financial strength score based on:
        1. Positive net income
        2. Positive return on assets (ROA)
        3. Positive operating cash flow
        4. Operating cash flow > net income
        5. Decreasing long-term debt
        6. Improving current ratio
        7. No dilution (shares outstanding not increasing)
        8. Improving gross margin
        9. Improving asset turnover
        
        Score interpretation:
        - 8-9: Strong financial position
        - 5-7: Moderate financial position
        - 0-4: Weak financial position
        
        Returns:
            Dictionary with F-Score components and total score
        """
        try:
            print(f"Calculating Piotroski F-Score for {self.symbol}...")
            
            ticker = yf.Ticker(self.symbol)
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            if financials.empty or balance_sheet.empty or cash_flow.empty:
                print("⚠️  Warning: Could not retrieve complete financial data for F-Score")
                return {"total_score": "N/A", "components": {}}
            
            # Initialize score components
            score = 0
            components = {}
            
            # Get most recent year data
            recent_col = financials.columns[0]
            prev_col = financials.columns[1] if len(financials.columns) > 1 else recent_col
            
            # 1. Positive net income
            net_income = financials.loc['Net Income', recent_col]
            components['positive_net_income'] = net_income > 0
            if net_income > 0:
                score += 1
                
            # 2. Positive return on assets (ROA)
            try:
                total_assets = balance_sheet.loc['Total Assets', recent_col]
                roa = net_income / total_assets
                components['positive_roa'] = roa > 0
                if roa > 0:
                    score += 1
            except:
                components['positive_roa'] = False
                
            # 3. Positive operating cash flow
            try:
                operating_cf = cash_flow.loc['Operating Cash Flow', recent_col]
                components['positive_operating_cf'] = operating_cf > 0
                if operating_cf > 0:
                    score += 1
            except:
                components['positive_operating_cf'] = False
                
            # 4. Operating cash flow > net income
            try:
                components['cf_greater_than_ni'] = operating_cf > net_income
                if operating_cf > net_income:
                    score += 1
            except:
                components['cf_greater_than_ni'] = False
                
            # 5. Decreasing long-term debt
            try:
                current_debt = balance_sheet.loc['Long Term Debt', recent_col]
                prev_debt = balance_sheet.loc['Long Term Debt', prev_col]
                components['decreasing_debt'] = current_debt < prev_debt
                if current_debt < prev_debt:
                    score += 1
            except:
                components['decreasing_debt'] = False
                
            # 6. Improving current ratio
            try:
                current_assets = balance_sheet.loc['Current Assets', recent_col]
                current_liabilities = balance_sheet.loc['Current Liabilities', recent_col]
                prev_current_assets = balance_sheet.loc['Current Assets', prev_col]
                prev_current_liabilities = balance_sheet.loc['Current Liabilities', prev_col]
                
                current_ratio = current_assets / current_liabilities
                prev_current_ratio = prev_current_assets / prev_current_liabilities
                
                components['improving_current_ratio'] = current_ratio > prev_current_ratio
                if current_ratio > prev_current_ratio:
                    score += 1
            except:
                components['improving_current_ratio'] = False
                
            # 7. No dilution (shares outstanding not increasing)
            try:
                info = ticker.info
                shares_outstanding = info.get('sharesOutstanding', 0)
                # Simplified check - in practice, you'd compare with previous year
                components['no_dilution'] = True  # Placeholder
                score += 1
            except:
                components['no_dilution'] = False
                
            # 8. Improving gross margin
            try:
                revenue = financials.loc['Total Revenue', recent_col]
                cost_of_revenue = financials.loc['Cost Of Revenue', recent_col]
                prev_revenue = financials.loc['Total Revenue', prev_col]
                prev_cost_of_revenue = financials.loc['Cost Of Revenue', prev_col]
                
                gross_margin = (revenue - cost_of_revenue) / revenue
                prev_gross_margin = (prev_revenue - prev_cost_of_revenue) / prev_revenue
                
                components['improving_gross_margin'] = gross_margin > prev_gross_margin
                if gross_margin > prev_gross_margin:
                    score += 1
            except:
                components['improving_gross_margin'] = False
                
            # 9. Improving asset turnover
            try:
                asset_turnover = revenue / total_assets
                prev_total_assets = balance_sheet.loc['Total Assets', prev_col]
                prev_asset_turnover = prev_revenue / prev_total_assets
                
                components['improving_asset_turnover'] = asset_turnover > prev_asset_turnover
                if asset_turnover > prev_asset_turnover:
                    score += 1
            except:
                components['improving_asset_turnover'] = False
            
            fscore_result = {
                'total_score': score,
                'max_score': 9,
                'components': components
            }
            
            print(f"✓ Piotroski F-Score calculated: {score}/9")
            return fscore_result
            
        except Exception as e:
            print(f"❌ Error calculating F-Score: {e}")
            return {"total_score": "N/A", "components": {}}
    
    def run_backtest(self, initial_cash: float = 10000.0) -> Dict:
        """
        Run the backtest using backtrader
        
        Args:
            initial_cash: Starting cash amount
            
        Returns:
            Dictionary with backtest results
        """
        try:
            print(f"Running backtest with ${initial_cash:,.2f} initial cash...")
            
            # Create cerebro engine
            self.cerebro = bt.Cerebro()
            
            # Add strategy
            self.cerebro.addstrategy(self.strategy_class, printlog=False)
            
            # Get correct timeframe and compression based on current interval
            timeframe, compression = self._get_timeframe_compression(self.current_interval)
            
            # Convert pandas DataFrame to backtrader format
            data_feed = bt.feeds.PandasData(
                dataname=self.data,
                timeframe=timeframe,
                compression=compression
            )
            
            # Add data feed
            self.cerebro.adddata(data_feed)
            
            # Set initial cash
            self.cerebro.broker.setcash(initial_cash)
            
            # Set commission (0.1% per trade)
            self.cerebro.broker.setcommission(commission=0.001)
            
            # Add analyzers for metrics
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
            
            # Store initial value
            initial_value = self.cerebro.broker.getvalue()
            
            # Run backtest
            print("Executing backtest...")
            results = self.cerebro.run()
            
            # Get final value
            final_value = self.cerebro.broker.getvalue()
            
            # Extract strategy instance
            if results and len(results) > 0:
                self.strategy_instance = results[0]
            else:
                print("❌ No results returned from backtest")
                self.strategy_instance = None
            
            # Store results
            self.results = {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': final_value - initial_value,
                'total_return_pct': ((final_value - initial_value) / initial_value) * 100,
                'analyzers': results[0].analyzers
            }
            
            print(f"✓ Backtest completed successfully")
            print(f"  Initial Value: ${initial_value:,.2f}")
            print(f"  Final Value: ${final_value:,.2f}")
            print(f"  Total Return: ${final_value - initial_value:,.2f} ({((final_value - initial_value) / initial_value) * 100:.2f}%)")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error running backtest: {e}")
            raise
    
    def calculate_comprehensive_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        METRIC DEFINITIONS:
        
        Risk-Adjusted Returns:
        - Sharpe Ratio: Risk-adjusted return using standard deviation
        - Sortino Ratio: Risk-adjusted return using downside deviation
        - Calmar Ratio: Annualized return divided by maximum drawdown
        
        Benchmark Analysis:
        - Alpha: Excess return over the benchmark (SPY)
        - Beta: Strategy's sensitivity to benchmark returns
        - Information Ratio: Excess return over benchmark divided by tracking error
        
        Trade Analysis:
        - Win/Loss Ratio: Ratio of winning to losing trades
        - Profit Factor: Gross profit divided by gross loss
        - Trade Frequency: Number of trades per year
        - Average Trade Duration: Average time positions are held
        
        Fundamental Analysis:
        - Piotroski F-Score: 9-point financial strength score using fundamental data
        
        Returns:
            Dictionary with all calculated metrics
        """
        try:
            print("Calculating comprehensive performance metrics...")
            
            if not self.results:
                raise ValueError("No backtest results available")
                
            metrics = {}
            
            # Basic metrics
            initial_value = self.results['initial_value']
            final_value = self.results['final_value']
            total_return = self.results['total_return']
            total_return_pct = self.results['total_return_pct']
            
            # Get analyzers
            analyzers = self.results['analyzers']
            
            # Time-based calculations
            start_date = self.data.index[0]
            end_date = self.data.index[-1]
            days_total = (end_date - start_date).days
            years = days_total / 365.25
            
            # 1. Annualized Return
            annualized_return = (pow(final_value / initial_value, 1/years) - 1) * 100
            metrics['annualized_return'] = annualized_return
            
            # 2. Sharpe Ratio
            sharpe_analysis = analyzers.sharpe.get_analysis()
            sharpe_ratio = sharpe_analysis.get('sharperatio', 0)
            metrics['sharpe_ratio'] = sharpe_ratio if sharpe_ratio else 0
            
            # 3. Maximum Drawdown
            drawdown_analysis = analyzers.drawdown.get_analysis()
            max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0)
            metrics['max_drawdown'] = max_drawdown
            
            # 4. Calmar Ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            metrics['calmar_ratio'] = calmar_ratio
            
            # 5. Trade Analysis
            trade_analysis = analyzers.trades.get_analysis()
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            won_trades = trade_analysis.get('won', {}).get('total', 0)
            lost_trades = trade_analysis.get('lost', {}).get('total', 0)
            
            # 6. Win/Loss Ratio
            win_loss_ratio = won_trades / lost_trades if lost_trades > 0 else float('inf')
            metrics['win_loss_ratio'] = win_loss_ratio
            
            # 7. Profit Factor
            gross_profit = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0)
            gross_loss = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            metrics['profit_factor'] = profit_factor
            
            # 8. Trade Frequency
            trade_frequency = total_trades / days_total * 252  # Annualized
            metrics['trade_frequency'] = trade_frequency
            
            # 9. Average Trade Duration
            metrics['avg_trade_duration_hours'] = 0  # Default value
            try:
                if self.strategy_instance is not None:
                    if hasattr(self.strategy_instance, 'trade_durations'):
                        trade_durations = getattr(self.strategy_instance, 'trade_durations', [])
                        if trade_durations and len(trade_durations) > 0:
                            avg_duration = np.mean(trade_durations)
                            # Convert from bars to hours based on actual interval
                            timeframe, compression = self._get_timeframe_compression(self.current_interval)
                            
                            if timeframe == bt.TimeFrame.Minutes:
                                avg_duration_hours = avg_duration * (compression / 60.0)  # minutes to hours
                            elif timeframe == bt.TimeFrame.Days:
                                avg_duration_hours = avg_duration * (compression * 24)  # days to hours
                            elif timeframe == bt.TimeFrame.Weeks:
                                avg_duration_hours = avg_duration * (compression * 24 * 7)  # weeks to hours
                            elif timeframe == bt.TimeFrame.Months:
                                avg_duration_hours = avg_duration * (compression * 24 * 30)  # approx months to hours
                            else:
                                avg_duration_hours = avg_duration  # fallback
                                
                            metrics['avg_trade_duration_hours'] = avg_duration_hours
            except Exception as e:
                # Silently handle trade duration calculation errors
                pass
                
            # 10. Calculate returns for Beta, Alpha, and other metrics
            strategy_returns = self.calculate_strategy_returns()
            benchmark_returns = self.calculate_benchmark_returns()
            
            if len(strategy_returns) > 0 and len(benchmark_returns) > 0:
                # Align returns
                aligned_returns = self.align_returns(strategy_returns, benchmark_returns)
                
                if len(aligned_returns) > 1:
                    strat_rets = aligned_returns['strategy']
                    bench_rets = aligned_returns['benchmark']
                    
                    # 11. Beta
                    beta = np.cov(strat_rets, bench_rets)[0, 1] / np.var(bench_rets)
                    metrics['beta'] = beta
                    
                    # 12. Alpha
                    alpha = np.mean(strat_rets) - (self.risk_free_rate/252 + beta * (np.mean(bench_rets) - self.risk_free_rate/252))
                    metrics['alpha'] = alpha * 252 * 100  # Annualized alpha in %
                    
                    # 13. Information Ratio
                    excess_returns = strat_rets - bench_rets
                    tracking_error = np.std(excess_returns)
                    information_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
                    metrics['information_ratio'] = information_ratio * np.sqrt(252)  # Annualized
                    
                    # 14. Sortino Ratio
                    negative_returns = strat_rets[strat_rets < 0]
                    downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
                    sortino_ratio = (np.mean(strat_rets) - self.risk_free_rate/252) / downside_deviation if downside_deviation > 0 else 0
                    metrics['sortino_ratio'] = sortino_ratio * np.sqrt(252)  # Annualized
                else:
                    metrics['beta'] = 0
                    metrics['alpha'] = 0
                    metrics['information_ratio'] = 0
                    metrics['sortino_ratio'] = 0
            else:
                metrics['beta'] = 0
                metrics['alpha'] = 0
                metrics['information_ratio'] = 0
                metrics['sortino_ratio'] = 0
            
            # Additional metrics
            metrics['total_trades'] = total_trades
            metrics['winning_trades'] = won_trades
            metrics['losing_trades'] = lost_trades
            metrics['win_rate'] = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            print("✓ Comprehensive metrics calculated successfully")
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return {}
    
    def calculate_strategy_returns(self) -> pd.Series:
        """Calculate strategy returns from backtest"""
        try:
            returns_analysis = self.results['analyzers'].returns.get_analysis()
            if returns_analysis:
                return pd.Series(list(returns_analysis.values()))
            else:
                # Fallback: calculate from equity curve
                time_return_analysis = self.results['analyzers'].time_return.get_analysis()
                if time_return_analysis:
                    return pd.Series(list(time_return_analysis.values()))
                else:
                    return pd.Series([])
        except:
            return pd.Series([])
    
    def calculate_benchmark_returns(self) -> pd.Series:
        """Calculate benchmark returns"""
        try:
            benchmark_close = self.benchmark_data['Close'].resample('D').last()
            benchmark_returns = benchmark_close.pct_change().dropna()
            return benchmark_returns
        except:
            return pd.Series([])
    
    def align_returns(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """Align strategy and benchmark returns by date"""
        try:
            # Create a simple alignment based on length
            min_length = min(len(strategy_returns), len(benchmark_returns))
            
            if min_length > 0:
                return {
                    'strategy': strategy_returns.iloc[:min_length],
                    'benchmark': benchmark_returns.iloc[:min_length]
                }
            else:
                return {'strategy': pd.Series([]), 'benchmark': pd.Series([])}
        except:
            return {'strategy': pd.Series([]), 'benchmark': pd.Series([])}
    
    def print_results(self, metrics: Dict, fscore: Dict):
        """Print comprehensive results"""
        print("\n" + "="*80)
        print("ALGORITHMIC TRADING SYSTEM - BACKTEST RESULTS")
        print("="*80)
        
        print(f"\n📊 STRATEGY OVERVIEW")
        print(f"{'Symbol:':<25} {self.symbol}")
        
        # Get strategy description
        strategy_description = "Unknown Strategy"
        try:
            if self.strategy_instance is not None:
                if hasattr(self.strategy_instance, 'get_strategy_description'):
                    strategy_description = self.strategy_instance.get_strategy_description()
                else:
                    # Fallback to strategy name if description method doesn't exist
                    strategy_description = getattr(self.strategy_instance, '__class__', type(self.strategy_instance)).__name__
        except Exception as e:
            print(f"Error getting strategy description: {e}")
            strategy_description = "Unknown Strategy"
        
        print(f"{'Strategy:':<25} {strategy_description}")
        print(f"{'Benchmark:':<25} {self.benchmark}")
        print(f"{'Period:':<25} {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"{'Data Points:':<25} {len(self.data):,}")
        
        print(f"\n📈 FUNDAMENTAL ANALYSIS")
        if fscore['total_score'] != 'N/A':
            print(f"{'Piotroski F-Score:':<25} {fscore['total_score']}/9")
            score_interpretation = "Strong" if fscore['total_score'] >= 7 else "Moderate" if fscore['total_score'] >= 4 else "Weak"
            print(f"{'Quality Rating:':<25} {score_interpretation}")
        else:
            print(f"{'Piotroski F-Score:':<25} N/A (Data unavailable)")
        
        print(f"\n💰 PERFORMANCE METRICS")
        print(f"{'Initial Capital:':<25} ${self.results['initial_value']:,.2f}")
        print(f"{'Final Value:':<25} ${self.results['final_value']:,.2f}")
        print(f"{'Total Return:':<25} ${self.results['total_return']:,.2f} ({self.results['total_return_pct']:.2f}%)")
        print(f"{'Annualized Return:':<25} {metrics.get('annualized_return', 0):.2f}%")
        
        print(f"\n📊 RISK METRICS")
        print(f"{'Sharpe Ratio:':<25} {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"{'Sortino Ratio:':<25} {metrics.get('sortino_ratio', 0):.3f}")
        print(f"{'Maximum Drawdown:':<25} {metrics.get('max_drawdown', 0):.2f}%")
        print(f"{'Calmar Ratio:':<25} {metrics.get('calmar_ratio', 0):.3f}")
        
        print(f"\n📈 BENCHMARK COMPARISON")
        print(f"{'Beta:':<25} {metrics.get('beta', 0):.3f}")
        print(f"{'Alpha:':<25} {metrics.get('alpha', 0):.2f}%")
        print(f"{'Information Ratio:':<25} {metrics.get('information_ratio', 0):.3f}")
        
        print(f"\n🎯 TRADE ANALYSIS")
        print(f"{'Total Trades:':<25} {metrics.get('total_trades', 0)}")
        print(f"{'Winning Trades:':<25} {metrics.get('winning_trades', 0)}")
        print(f"{'Losing Trades:':<25} {metrics.get('losing_trades', 0)}")
        print(f"{'Win Rate:':<25} {metrics.get('win_rate', 0):.1f}%")
        print(f"{'Win/Loss Ratio:':<25} {metrics.get('win_loss_ratio', 0):.2f}")
        print(f"{'Profit Factor:':<25} {metrics.get('profit_factor', 0):.2f}")
        
        print(f"\n⏰ TRADE FREQUENCY")
        print(f"{'Trade Frequency:':<25} {metrics.get('trade_frequency', 0):.1f} trades/year")
        print(f"{'Avg Trade Duration:':<25} {metrics.get('avg_trade_duration_hours', 0):.1f} hours")
        
        print(f"\n🔧 SYSTEM PARAMETERS")
        
        # Show strategy-specific parameters
        try:
            if self.strategy_instance is not None and hasattr(self.strategy_instance, 'params'):
                params = self.strategy_instance.params
                if hasattr(params, 'rsi_period'):
                    print(f"{'RSI Period:':<25} {params.rsi_period}")
                    print(f"{'RSI Oversold:':<25} {params.rsi_oversold}")
                    print(f"{'RSI Overbought:':<25} {params.rsi_overbought}")
                elif hasattr(params, 'short_period'):
                    print(f"{'Short MA Period:':<25} {params.short_period}")
                    print(f"{'Long MA Period:':<25} {params.long_period}")
                
                print(f"{'Position Size:':<25} {params.position_size} share")
            else:
                print(f"{'Position Size:':<25} 1 share")
        except Exception as e:
            print(f"{'Position Size:':<25} 1 share (error reading params: {e})")
        
        print(f"{'Commission:':<25} 0.1%")
        
        # Trade Log
        print(f"\n📋 TRADE LOG")
        try:
            if (self.strategy_instance is not None and 
                hasattr(self.strategy_instance, 'trades_log')):
                trades_log = getattr(self.strategy_instance, 'trades_log', [])
                if trades_log:
                    print(f"All {len(trades_log)} trades:")
                    print("-" * 80)
                    for trade in trades_log:
                        print(f"Trade #{trade['trade_num']}: {trade['entry_date'].strftime('%Y-%m-%d %H:%M')} -> "
                              f"{trade['exit_date'].strftime('%Y-%m-%d %H:%M')}: "
                              f"${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f} "
                              f"(${trade['pnl']:.2f}, {trade['pnl_pct']:.1f}%)")
                else:
                    print("No trades executed")
            else:
                print("No trades executed (strategy instance unavailable)")
        except Exception as e:
            print(f"Error accessing trade log - {e}")
        

    
    def plot_results(self):
        """Plot backtest results using backtrader's plotting"""
        try:
            print("\nGenerating plots...")
            
            # Use backtrader's built-in plotting
            self.cerebro.plot(
                style='candlestick',
                barup='green',
                bardown='red',
                volume=False,
                plotname=f'{self.symbol} - RSI Mean Reversion Strategy'
            )
            
            print("✓ Plots generated successfully")
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
            # Fallback to simple matplotlib plot
            try:
                plt.figure(figsize=(12, 8))
                plt.plot(self.data.index, self.data['Close'], label=f'{self.symbol} Price')
                plt.title(f'{self.symbol} Price Chart')
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.legend()
                plt.grid(True)
                plt.show()
            except:
                print("Could not generate fallback plot")


def main():
    """
    Main execution function
    
    This is the entry point for the trading system. It orchestrates:
    1. Strategy selection and validation
    2. Data fetching and validation
    3. Fundamental analysis (F-Score)
    4. Backtesting execution
    5. Performance analysis
    6. Results presentation
    """
    print("🚀 ALGORITHMIC TRADING SYSTEM STARTING...")
    print("="*60)
    
    # Show available strategies
    print("📈 Available Trading Strategies:")
    strategies = list_available_strategies()
    for i, (name, description) in enumerate(strategies.items(), 1):
        print(f"  {i}. {name}: {description}")
    
    try:
        # Test both strategies
        strategy_names = ['rsi_mean_reversion', 'ma_crossover']
        
        for strategy_name in strategy_names:
            print(f"\n{'='*80}")
            print(f"🎯 TESTING STRATEGY: {strategy_name.upper()}")
            print(f"{'='*80}")
            
            # Initialize trading system
            system = TradingSystem(symbol='AAPL', benchmark='SPY', strategy=strategy_name)
            
            # Fetch data
            system.fetch_data(period='6mo', interval='15m')
            
            # Calculate F-Score (only once)
            if strategy_name == strategy_names[0]:
                fscore = system.calculate_piotroski_fscore()
            
            # Run backtest
            system.run_backtest(initial_cash=10000.0)
            
            # Calculate metrics
            metrics = system.calculate_comprehensive_metrics()
            
            # Print results
            system.print_results(metrics, fscore if strategy_name == strategy_names[0] else {"total_score": "N/A", "components": {}})
            
            # Generate plots
            system.plot_results()
            
            print(f"\n✅ {strategy_name.upper()} strategy testing completed!")
        
        print(f"\n{'='*80}")
        print("🎉 ALL STRATEGY TESTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 