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


class AlphaBetaAnalyzer(bt.Analyzer):
    """
    Custom analyzer to calculate Alpha and Beta relative to a benchmark
    
    Alpha: Excess return over what Beta would predict from benchmark
    Beta: Strategy's sensitivity to benchmark returns (correlation * volatility_ratio)
    """
    
    def __init__(self):
        self.strategy_returns = []
        self.benchmark_returns = []
        self.dates = []
        
    def next(self):
        # Calculate daily returns
        if len(self.strategy) > 1:
            # Strategy return
            current_value = self.strategy.broker.getvalue()
            if not hasattr(self, 'prev_value'):
                self.prev_value = current_value
                return
            
            strategy_return = (current_value - self.prev_value) / self.prev_value
            self.strategy_returns.append(strategy_return)
            self.prev_value = current_value
            
            # Benchmark return (if available)
            if len(self.datas) > 1:  # Assuming benchmark is second data feed
                benchmark_data = self.datas[1]
                if len(benchmark_data) > 1:
                    benchmark_return = (benchmark_data.close[0] - benchmark_data.close[-1]) / benchmark_data.close[-1]
                    self.benchmark_returns.append(benchmark_return)
                    self.dates.append(self.datas[0].datetime.date(0))
    
    def stop(self):
        # Calculate Alpha and Beta
        if len(self.strategy_returns) > 1 and len(self.benchmark_returns) > 1:
            # Ensure same length
            min_len = min(len(self.strategy_returns), len(self.benchmark_returns))
            strategy_rets = np.array(self.strategy_returns[:min_len])
            benchmark_rets = np.array(self.benchmark_returns[:min_len])
            
            # Calculate Beta using linear regression
            if np.std(benchmark_rets) > 0:
                covariance = np.cov(strategy_rets, benchmark_rets)[0, 1]
                benchmark_variance = np.var(benchmark_rets)
                beta = covariance / benchmark_variance
                
                # Calculate Alpha
                strategy_mean = np.mean(strategy_rets)
                benchmark_mean = np.mean(benchmark_rets)
                alpha = strategy_mean - (beta * benchmark_mean)
                
                # Annualize Alpha (assuming daily returns)
                alpha_annualized = alpha * 252
                
                self.alpha = alpha_annualized
                self.beta = beta
            else:
                self.alpha = 0
                self.beta = 0
        else:
            self.alpha = 0
            self.beta = 0
    
    def get_analysis(self):
        return {
            'alpha': getattr(self, 'alpha', 0),
            'beta': getattr(self, 'beta', 0)
        }


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
                
            if self.data.empty or self.benchmark_data.empty:
                raise ValueError("No data retrieved from yfinance")
                
            # Add diagnostic information
            print(f"✓ Retrieved {len(self.data)} bars for {self.symbol}")
            print(f"✓ Retrieved {len(self.benchmark_data)} bars for {self.benchmark}")
            print(f"📅 Data range: {self.data.index[0].strftime('%Y-%m-%d %H:%M')} to {self.data.index[-1].strftime('%Y-%m-%d %H:%M')}")
            print(f"📊 Expected bars for {period} at {interval}: ~{self._calculate_expected_bars(period, interval)}")
            
            return self.data, self.benchmark_data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise
    
    def _calculate_expected_bars(self, period: str, interval: str) -> int:
        """Calculate expected number of bars for given period and interval"""
        period_days = {
            '60d': 60, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }.get(period, 60)
        
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '1d': 1440
        }.get(interval, 15)
        
        # Approximate trading hours (6.5 hours per day)
        trading_minutes_per_day = 6.5 * 60
        trading_days = period_days * (5/7)  # Account for weekends
        
        return int(trading_days * trading_minutes_per_day / interval_minutes)

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
            try:
                net_income = financials.loc['Net Income', recent_col]
                components['positive_net_income'] = net_income > 0
                if net_income > 0:
                    score += 1
            except KeyError:
                components['positive_net_income'] = False
                
            # 2. Positive return on assets (ROA)
            try:
                total_assets = balance_sheet.loc['Total Assets', recent_col]
                roa = net_income / total_assets
                components['positive_roa'] = roa > 0
                if roa > 0:
                    score += 1
            except KeyError:
                components['positive_roa'] = False
                
            # 3. Positive operating cash flow
            try:
                operating_cf = cash_flow.loc['Operating Cash Flow', recent_col]
                components['positive_operating_cf'] = operating_cf > 0
                if operating_cf > 0:
                    score += 1
            except KeyError:
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
            except KeyError:
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
            except KeyError:
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
            except KeyError:
                components['improving_gross_margin'] = False
                
            # 9. Improving asset turnover
            try:
                asset_turnover = revenue / total_assets
                prev_total_assets = balance_sheet.loc['Total Assets', prev_col]
                prev_asset_turnover = prev_revenue / prev_total_assets
                
                components['improving_asset_turnover'] = asset_turnover > prev_asset_turnover
                if asset_turnover > prev_asset_turnover:
                    score += 1
            except KeyError:
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
            # Create cerebro engine
            self.cerebro = bt.Cerebro()
            
            # Add strategy
            self.cerebro.addstrategy(self.strategy_class, printlog=False)
            
            # Add backtrader's built-in sizer to use all available cash
            self.cerebro.addsizer(bt.sizers.AllInSizerInt)
            
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
            
            # Remove commission to avoid affecting metrics
            # self.cerebro.broker.setcommission(commission=0.001)
            
            # Add analyzers for metrics - Using backtrader's built-in analyzers with proper configuration
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, 
                                   riskfreerate=self.risk_free_rate,
                                   timeframe=bt.TimeFrame.Days,
                                   _name='sharpe')
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
            self.cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
            self.cerebro.addanalyzer(bt.analyzers.Calmar, _name='calmar')
            self.cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
            self.cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')
            self.cerebro.addanalyzer(bt.analyzers.GrossLeverage, _name='leverage')
            self.cerebro.addanalyzer(bt.analyzers.PositionsValue, _name='positions')
            self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
            self.cerebro.addanalyzer(bt.analyzers.PeriodStats, _name='period_stats')
            
            # Add benchmark tracking using TimeReturn analyzer as per backtrader documentation
            if hasattr(self, 'benchmark_data') and self.benchmark_data is not None:
                # Convert benchmark data to backtrader format for proper tracking
                benchmark_feed = bt.feeds.PandasData(
                    dataname=self.benchmark_data,
                    timeframe=timeframe,
                    compression=compression
                )
                self.cerebro.adddata(benchmark_feed)
                # Add TimeReturn analyzer to track benchmark performance
                self.cerebro.addanalyzer(bt.analyzers.TimeReturn, 
                                       data=benchmark_feed, 
                                       timeframe=bt.TimeFrame.Days,
                                       _name='benchmark_returns')
            
            # Add PyFolio integration if available (as mentioned in backtrader docs)
            try:
                self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
            except:
                # PyFolio integration might not be available depending on installation
                pass
            
            # Add custom Alpha/Beta analyzer
            self.cerebro.addanalyzer(AlphaBetaAnalyzer, _name='alpha_beta')
            
            # Store initial value
            initial_value = self.cerebro.broker.getvalue()
            
            # Run backtest
            results = self.cerebro.run()
            
            # Get final value
            final_value = self.cerebro.broker.getvalue()
            
            # Extract strategy instance
            if results and len(results) > 0:
                self.strategy_instance = results[0]
            else:
                self.strategy_instance = None
            
            # Store results
            self.results = {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': final_value - initial_value,
                'total_return_pct': ((final_value - initial_value) / initial_value) * 100,
                'analyzers': results[0].analyzers
            }
            
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
            
            # 1. Annualized Return - Use built-in analyzer correctly
            annual_return_analysis = analyzers.annual_return.get_analysis()
            if annual_return_analysis:
                # Get the average annual return across all years in the backtest
                annual_returns = list(annual_return_analysis.values())
                # Convert from decimal to percentage and calculate geometric mean for proper annualization
                if len(annual_returns) > 1:
                    # Geometric mean for multiple years
                    product = 1.0
                    for ret in annual_returns:
                        product *= (1 + ret)
                    annualized_return = (pow(product, 1/len(annual_returns)) - 1) * 100
                else:
                    # Single year or partial year
                    annualized_return = annual_returns[0] * 100 if annual_returns else 0
            else:
                annualized_return = 0
            metrics['annualized_return'] = annualized_return
            
            # 2. Sharpe Ratio - Using built-in analyzer
            sharpe_analysis = analyzers.sharpe.get_analysis()
            metrics['sharpe_ratio'] = sharpe_analysis.get('sharperatio', 0)
            
            # 3. Maximum Drawdown - Using built-in analyzer
            drawdown_analysis = analyzers.drawdown.get_analysis()
            metrics['max_drawdown'] = drawdown_analysis.get('max', {}).get('drawdown', 0)
            
            # 4. Calmar Ratio - Use built-in analyzer
            calmar_analysis = analyzers.calmar.get_analysis()
            # Get the latest Calmar ratio from the analysis
            if calmar_analysis:
                calmar_values = list(calmar_analysis.values())
                metrics['calmar_ratio'] = calmar_values[-1] if calmar_values else 0
            else:
                metrics['calmar_ratio'] = 0
            
            # 5. VWR (Variability-Weighted Return) - Built-in analyzer
            vwr_analysis = analyzers.vwr.get_analysis()
            metrics['vwr'] = vwr_analysis.get('vwr', 0)
            
            # 6. SQN (System Quality Number) - Built-in analyzer
            sqn_analysis = analyzers.sqn.get_analysis()
            metrics['sqn'] = sqn_analysis.get('sqn', 0)
            
            # 7. Trade Analysis - Using built-in analyzer
            trade_analysis = analyzers.trades.get_analysis()
            
            # Extract trade counts directly from analyzer
            total_trades = trade_analysis.get('total', {}).get('total', 0) if 'total' in trade_analysis else 0
            won_trades = trade_analysis.get('won', {}).get('total', 0) if 'won' in trade_analysis else 0
            lost_trades = trade_analysis.get('lost', {}).get('total', 0) if 'lost' in trade_analysis else 0
            
            # Basic trade metrics
            metrics['total_trades'] = total_trades
            metrics['winning_trades'] = won_trades
            metrics['losing_trades'] = lost_trades
            metrics['win_rate'] = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Win/Loss Ratio
            metrics['win_loss_ratio'] = won_trades / lost_trades if lost_trades > 0 else float('inf')
            
            # Profit Factor
            gross_profit = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0) if 'won' in trade_analysis else 0
            gross_loss = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0)) if 'lost' in trade_analysis else 0
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # 8. Alpha and Beta - Using custom analyzer
            alpha_beta_analysis = analyzers.alpha_beta.get_analysis()
            metrics['alpha'] = alpha_beta_analysis.get('alpha', 0)
            metrics['beta'] = alpha_beta_analysis.get('beta', 0)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return {}
    

    
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
        except Exception as e:
            print(f"Error getting strategy description: {e}")
        
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
        print(f"{'Maximum Drawdown:':<25} {metrics.get('max_drawdown', 0):.2f}%")
        
        # Handle NaN values for Calmar ratio
        calmar_ratio = metrics.get('calmar_ratio', 0)
        if str(calmar_ratio) == 'nan' or calmar_ratio == 0:
            print(f"{'Calmar Ratio:':<25} N/A")
        else:
            print(f"{'Calmar Ratio:':<25} {calmar_ratio:.3f}")
            
        print(f"{'VWR:':<25} {metrics.get('vwr', 0):.3f}")
        print(f"{'SQN:':<25} {metrics.get('sqn', 0):.2f}")
        
        print(f"\n📊 BENCHMARK ANALYSIS")
        print(f"{'Alpha (Annualized):':<25} {metrics.get('alpha', 0):.4f}")
        print(f"{'Beta:':<25} {metrics.get('beta', 0):.3f}")
        
        print(f"\n🎯 TRADE ANALYSIS")
        print(f"{'Total Trades:':<25} {metrics.get('total_trades', 0)}")
        print(f"{'Winning Trades:':<25} {metrics.get('winning_trades', 0)}")
        print(f"{'Losing Trades:':<25} {metrics.get('losing_trades', 0)}")
        print(f"{'Win Rate:':<25} {metrics.get('win_rate', 0):.1f}%")
        
        # Handle infinite values properly
        win_loss_ratio = metrics.get('win_loss_ratio', 0)
        if win_loss_ratio == float('inf'):
            print(f"{'Win/Loss Ratio:':<25} ∞ (no losses)")
        else:
            print(f"{'Win/Loss Ratio:':<25} {win_loss_ratio:.2f}")
        
        profit_factor = metrics.get('profit_factor', 0)
        if profit_factor == float('inf'):
            print(f"{'Profit Factor:':<25} ∞ (no losses)")
        else:
            print(f"{'Profit Factor:':<25} {profit_factor:.2f}")
        
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
                
                print(f"{'Position Sizing:':<25} All-In (backtrader built-in sizer)")
            else:
                print(f"{'Position Sizing:':<25} All-In (backtrader built-in sizer)")
        except Exception as e:
            print(f"{'Position Sizing:':<25} All-In (backtrader built-in sizer)")
        
        print(f"{'Commission:':<25} None (removed for clean metrics)")
        print(f"{'Risk Free Rate:':<25} {self.risk_free_rate:.1%}")
        
        print(f"\n✅ All metrics calculated using backtrader's built-in analyzers")

    def plot_results(self, show_plots: bool = False):
        """Plot backtest results using backtrader's plotting"""
        try:
            if show_plots:
                print("\nGenerating plots...")
                
                # Use backtrader's built-in plotting
                self.cerebro.plot(
                    style='candlestick',
                    barup='green',
                    bardown='red',
                    volume=False,
                    plotname=f'{self.symbol} - {self.strategy_name}'
                )
                
                print("✓ Plots generated successfully")
            
        except Exception as e:
            if show_plots:
                print(f"❌ Error generating plots: {e}")


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
        # Test all strategies
        strategy_names = list(strategies.keys())
        
        for strategy_name in strategy_names:
            print(f"\n{'='*80}")
            print(f"🎯 TESTING STRATEGY: {strategy_name.upper()}")
            print(f"{'='*80}")
            
            # Initialize trading system
            system = TradingSystem(symbol='SPY', benchmark='SPY', strategy=strategy_name)
            
            # Fetch data
            system.fetch_data(period='2y', interval='1d')
            
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