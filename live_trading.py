#!/usr/bin/env python3
"""Live Trading Engine - Alpaca integration"""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import pandas_ta as ta

# Alpaca SDK imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Strategy imports
from strategies import get_strategy_class, list_available_strategies

# Load environment variables
load_dotenv()


class LiveTradingEngine:
    """Live Trading Engine for Alpaca Markets"""
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize live trading engine
        
        Args:
            paper_trading: Use paper trading environment (default: True)
        """
        self.paper_trading = paper_trading
        self.trading_client = None
        self.data_client = None
        self.data_stream = None
        self.is_connected = False
        
        # Strategy execution state
        self.running_strategy = None
        self.strategy_thread = None
        self.strategy_running = False
        self.strategy_stop_event = threading.Event()
        
        # Get API credentials from environment (official Alpaca naming convention)
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.secret_key = os.getenv('APCA_SECRET_KEY')
        
        # Note: The alpaca-py SDK automatically handles base URL based on paper parameter
        # Paper: https://paper-api.alpaca.markets
        # Live: https://api.alpaca.markets
        
        # Validate credentials
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "API credentials must be set in environment variables: "
                "APCA_API_KEY_ID and APCA_SECRET_KEY"
            )
        
    def connect(self) -> bool:
        """
        Connect to Alpaca API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper_trading
            )
            
            # Initialize data client for fetching market data
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            # Test connection by getting account info
            account = self.trading_client.get_account()
            self.is_connected = True
            print(f"✓ Connected to Alpaca {'Paper Trading' if self.paper_trading else 'Live Trading'}")
            print(f"  Account Status: {account.status}")
            print(f"  Buying Power: ${float(account.buying_power):,.2f}")
            return True
            
        except APIError as e:
            print(f"❌ Alpaca API Error: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Alpaca API"""
        if self.trading_client:
            # Alpaca SDK doesn't require explicit disconnection, but we'll clean up
            self.trading_client = None
            self.is_connected = False
            print("✓ Disconnected from Alpaca API")
    
    def place_order(self, symbol: str, quantity: float, side: str, order_type: str = 'market', 
                   limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                   time_in_force: str = 'day') -> Optional[Dict]:
        """
        Place a trading order
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            quantity: Number of shares (must be positive)
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', or 'stop'
            limit_price: Required for limit orders
            stop_price: Required for stop orders
            time_in_force: 'day', 'gtc', 'ioc', 'fok' (default: 'day')
            
        Returns:
            Order object as dictionary, or None if order failed
        """
        if not self.is_connected or not self.trading_client:
            print("❌ Not connected to Alpaca API. Call connect() first.")
            return None
        
        try:
            # Convert side string to OrderSide enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Convert time_in_force string to enum
            tif_map = {
                'day': TimeInForce.DAY,
                'gtc': TimeInForce.GTC,
                'ioc': TimeInForce.IOC,
                'fok': TimeInForce.FOK
            }
            tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)
            
            # Create order request based on order type
            if order_type.lower() == 'market':
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=tif
                )
            elif order_type.lower() == 'limit':
                if limit_price is None:
                    print("❌ limit_price is required for limit orders")
                    return None
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    limit_price=limit_price,
                    time_in_force=tif
                )
            elif order_type.lower() == 'stop':
                if stop_price is None:
                    print("❌ stop_price is required for stop orders")
                    return None
                order_request = StopOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    stop_price=stop_price,
                    time_in_force=tif
                )
            else:
                print(f"❌ Invalid order_type: {order_type}. Must be 'market', 'limit', or 'stop'")
                return None
            
            # Submit order
            order = self.trading_client.submit_order(order_data=order_request)
            
            print(f"✓ Order placed: {order.side} {order.qty} shares of {order.symbol} ({order.order_type})")
            print(f"  Order ID: {order.id}")
            print(f"  Status: {order.status}")
            
            # Convert order object to dictionary for return
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side.value,
                'order_type': order.order_type.value,
                'status': order.status.value,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0.0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None
            }
            
        except APIError as e:
            print(f"❌ Alpaca API Error placing order: {e}")
            return None
        except Exception as e:
            print(f"❌ Error placing order: {e}")
            return None
    
    def get_positions(self) -> Dict:
        """
        Get current positions
        
        Returns:
            Dictionary mapping symbols to position information
        """
        if not self.is_connected or not self.trading_client:
            print("❌ Not connected to Alpaca API. Call connect() first.")
            return {}
        
        try:
            positions = self.trading_client.get_all_positions()
            positions_dict = {}
            
            for position in positions:
                positions_dict[position.symbol] = {
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'side': position.side.value if hasattr(position.side, 'value') else str(position.side)
                }
            
            return positions_dict
            
        except APIError as e:
            print(f"❌ Alpaca API Error getting positions: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return {}
    
    def get_account_info(self) -> Dict:
        """
        Get account information
        
        Returns:
            Dictionary with account information
        """
        if not self.is_connected or not self.trading_client:
            print("❌ Not connected to Alpaca API. Call connect() first.")
            return {}
        
        try:
            account = self.trading_client.get_account()
            
            return {
                'account_number': account.account_number,
                'status': account.status.value if hasattr(account.status, 'value') else str(account.status),
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at.isoformat() if account.created_at else None,
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'multiplier': float(account.multiplier),
                'shorting_enabled': account.shorting_enabled,
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value)
            }
            
        except APIError as e:
            print(f"❌ Alpaca API Error getting account info: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error getting account info: {e}")
            return {}
    
    def fetch_historical_data(self, symbol: str, days: int = 100, timeframe: TimeFrame = TimeFrame.Day) -> pd.DataFrame:
        """
        Fetch historical data from Alpaca
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data to fetch
            timeframe: TimeFrame enum (Day, Hour, Minute, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request_params)
            
            # Convert to DataFrame
            data_list = []
            for bar in bars.data.get(symbol, []):
                data_list.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })
            
            if not data_list:
                raise ValueError(f"No data retrieved for {symbol}")
            
            df = pd.DataFrame(data_list)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            raise
    
    def _calculate_indicators(self, df: pd.DataFrame, strategy_name: str, params: Dict) -> pd.DataFrame:
        """
        Calculate technical indicators based on strategy type
        
        Args:
            df: DataFrame with OHLCV data
            strategy_name: Name of the strategy
            params: Strategy parameters
            
        Returns:
            DataFrame with indicators added
        """
        df = df.copy()
        
        # RSI Mean Reversion
        if strategy_name == 'rsi_mean_reversion':
            period = params.get('rsi_period', 14)
            df['rsi'] = ta.rsi(df['close'], length=period)
        
        # Moving Average Crossover
        elif strategy_name == 'ma_crossover':
            short_period = params.get('short_period', 10)
            long_period = params.get('long_period', 50)
            df[f'sma_{short_period}'] = ta.sma(df['close'], length=short_period)
            df[f'sma_{long_period}'] = ta.sma(df['close'], length=long_period)
            df['ma_crossover'] = df[f'sma_{short_period}'] - df[f'sma_{long_period}']
            df['ma_crossover_prev'] = df['ma_crossover'].shift(1)
        
        # MACD Crossover
        elif strategy_name == 'macd_crossover':
            fast = params.get('macd_fast', 12)
            slow = params.get('macd_slow', 26)
            signal = params.get('macd_signal', 9)
            macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
            df['macd'] = macd[f'MACD_{fast}_{slow}_{signal}']
            df['macd_signal'] = macd[f'MACDs_{fast}_{slow}_{signal}']
            df['macd_crossover'] = df['macd'] - df['macd_signal']
            df['macd_crossover_prev'] = df['macd_crossover'].shift(1)
        
        # EMA Crossover
        elif strategy_name == 'ema_crossover':
            fast = params.get('ema_fast', 12)
            slow = params.get('ema_slow', 26)
            df[f'ema_{fast}'] = ta.ema(df['close'], length=fast)
            df[f'ema_{slow}'] = ta.ema(df['close'], length=slow)
            df['ema_crossover'] = df[f'ema_{fast}'] - df[f'ema_{slow}']
            df['ema_crossover_prev'] = df['ema_crossover'].shift(1)
        
        # Bollinger Bands
        elif strategy_name == 'bollinger_bands':
            period = params.get('bb_period', 20)
            std = params.get('bb_std', 2)
            bb = ta.bbands(df['close'], length=period, std=std)
            df['bb_upper'] = bb[f'BBU_{period}_{std}.0']
            df['bb_middle'] = bb[f'BBM_{period}_{std}.0']
            df['bb_lower'] = bb[f'BBL_{period}_{std}.0']
        
        # Stochastic Oscillator
        elif strategy_name == 'stochastic_oscillator':
            k_period = params.get('stoch_k', 14)
            d_period = params.get('stoch_d', 3)
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=k_period, d=d_period)
            df['stoch_k'] = stoch[f'STOCHk_{k_period}_{d_period}_{d_period}']
            df['stoch_d'] = stoch[f'STOCHd_{k_period}_{d_period}_{d_period}']
        
        # Momentum (Williams %R + SMA)
        elif strategy_name == 'momentum':
            williams_period = params.get('williams_period', 40)
            # Williams %R calculation using pandas-ta
            willr = ta.willr(df['high'], df['low'], df['close'], length=williams_period)
            df['williams_r'] = willr if isinstance(willr, pd.Series) else willr.iloc[:, 0]
            df['sma_fast'] = ta.sma(df['close'], length=20)
            df['sma_slow'] = ta.sma(df['close'], length=50)
            df['williams_r_prev'] = df['williams_r'].shift(1)
        
        # Parabolic SAR
        elif strategy_name == 'parabolic_sar':
            af = params.get('sar_af', 0.02)
            afmax = params.get('sar_afmax', 0.20)
            # Parabolic SAR - pandas-ta returns DataFrame with PSAR columns
            psar = ta.psar(df['high'], df['low'], df['close'], af0=af, af=af, afmax=afmax)
            # Extract PSAR value (pandas-ta returns DataFrame with columns like 'PSARl_0.02_0.02_0.2')
            if isinstance(psar, pd.DataFrame):
                # Get the first PSAR column (usually the lower/actual PSAR value)
                psar_cols = [col for col in psar.columns if 'PSAR' in col.upper()]
                if psar_cols:
                    df['sar'] = psar[psar_cols[0]]
                else:
                    # Fallback: use first column
                    df['sar'] = psar.iloc[:, 0]
            elif isinstance(psar, pd.Series):
                df['sar'] = psar
            else:
                # Fallback calculation if pandas-ta doesn't work
                df['sar'] = df['close'] * 0.98  # Placeholder
            # For PSAR, we need to check if price crosses SAR
            df['sar_crossover'] = df['close'] - df['sar']
            df['sar_crossover_prev'] = df['sar_crossover'].shift(1)
        
        # Bollinger Bands + RSI Combined
        elif strategy_name == 'bollinger_rsi':
            bb_period = params.get('bb_period', 20)
            bb_devfactor = params.get('bb_devfactor', 2.0)
            rsi_period = params.get('rsi_period', 14)
            bb = ta.bbands(df['close'], length=bb_period, std=bb_devfactor)
            df['bb_upper'] = bb[f'BBU_{bb_period}_{bb_devfactor}.0']
            df['bb_middle'] = bb[f'BBM_{bb_period}_{bb_devfactor}.0']
            df['bb_lower'] = bb[f'BBL_{bb_period}_{bb_devfactor}.0']
            df['rsi'] = ta.rsi(df['close'], length=rsi_period)
        
        return df
    
    def _check_signal(self, df: pd.DataFrame, strategy_name: str, params: Dict) -> Tuple[bool, bool]:
        """
        Check if strategy generates buy or sell signal
        
        Args:
            df: DataFrame with indicators
            strategy_name: Name of the strategy
            params: Strategy parameters
            
        Returns:
            Tuple of (should_buy, should_sell)
        """
        if len(df) < 2:
            return False, False
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # RSI Mean Reversion
        if strategy_name == 'rsi_mean_reversion':
            oversold = params.get('rsi_oversold', 30)
            overbought = params.get('rsi_overbought', 70)
            buy = latest['rsi'] < oversold if not pd.isna(latest['rsi']) else False
            sell = latest['rsi'] > overbought if not pd.isna(latest['rsi']) else False
        
        # Moving Average Crossover
        elif strategy_name == 'ma_crossover':
            buy = latest['ma_crossover'] > 0 and prev['ma_crossover'] <= 0
            sell = latest['ma_crossover'] < 0 and prev['ma_crossover'] >= 0
        
        # MACD Crossover
        elif strategy_name == 'macd_crossover':
            buy = latest['macd_crossover'] > 0 and prev['macd_crossover'] <= 0
            sell = latest['macd_crossover'] < 0 and prev['macd_crossover'] >= 0
        
        # EMA Crossover
        elif strategy_name == 'ema_crossover':
            buy = latest['ema_crossover'] > 0 and prev['ema_crossover'] <= 0
            sell = latest['ema_crossover'] < 0 and prev['ema_crossover'] >= 0
        
        # Bollinger Bands
        elif strategy_name == 'bollinger_bands':
            buy = latest['close'] < latest['bb_lower'] if not pd.isna(latest['bb_lower']) else False
            sell = latest['close'] > latest['bb_upper'] if not pd.isna(latest['bb_upper']) else False
        
        # Stochastic Oscillator
        elif strategy_name == 'stochastic_oscillator':
            oversold = params.get('stoch_oversold', 20)
            overbought = params.get('stoch_overbought', 80)
            buy = latest['stoch_k'] < oversold if not pd.isna(latest['stoch_k']) else False
            sell = latest['stoch_k'] > overbought if not pd.isna(latest['stoch_k']) else False
        
        # Momentum (Williams %R + SMA)
        elif strategy_name == 'momentum':
            signal_level = params.get('signal_level', -50)
            oversold_level = params.get('oversold_level', -80)
            overbought_level = params.get('overbought_level', -20)
            # Buy: Williams %R was oversold and now rallies above signal level, with uptrend
            buy = (latest['williams_r'] > signal_level and 
                   prev['williams_r'] <= oversold_level and
                   latest['sma_fast'] > latest['sma_slow']) if not pd.isna(latest['williams_r']) else False
            # Sell: Williams %R falls below signal level or goes overbought
            sell = (latest['williams_r'] < signal_level or
                   latest['williams_r'] > overbought_level) if not pd.isna(latest['williams_r']) else False
        
        # Parabolic SAR
        elif strategy_name == 'parabolic_sar':
            # Buy: Price crosses above SAR (bullish reversal)
            buy = latest['sar_crossover'] > 0 and prev['sar_crossover'] <= 0
            # Sell: Price crosses below SAR (bearish reversal)
            sell = latest['sar_crossover'] < 0 and prev['sar_crossover'] >= 0
        
        # Bollinger Bands + RSI Combined
        elif strategy_name == 'bollinger_rsi':
            rsi_oversold = params.get('rsi_oversold', 30)
            rsi_overbought = params.get('rsi_overbought', 70)
            # Buy: Price <= lower BB AND RSI < oversold
            buy = (latest['close'] <= latest['bb_lower'] and 
                   latest['rsi'] < rsi_oversold) if (not pd.isna(latest['bb_lower']) and not pd.isna(latest['rsi'])) else False
            # Sell: Price >= upper BB AND RSI > overbought
            sell = (latest['close'] >= latest['bb_upper'] and 
                   latest['rsi'] > rsi_overbought) if (not pd.isna(latest['bb_upper']) and not pd.isna(latest['rsi'])) else False
        
        else:
            return False, False
        
        return buy, sell
    
    def _strategy_loop(self, symbol: str, strategy_name: str, parameters: Dict, interval_seconds: int = 60):
        """
        Main strategy execution loop
        
        Args:
            symbol: Stock symbol to trade
            strategy_name: Name of the strategy
            parameters: Strategy parameters
            interval_seconds: How often to check for signals (default: 60 seconds)
        """
        print(f"📊 Starting live strategy loop for {symbol} using {strategy_name}")
        print(f"   Checking signals every {interval_seconds} seconds")
        
        # Determine timeframe based on strategy (default to daily)
        timeframe = TimeFrame.Day
        if parameters.get('interval') in ['1m', '5m', '15m', '30m', '1h']:
            timeframe = TimeFrame.Hour if 'h' in parameters.get('interval', '') else TimeFrame.Minute
        
        last_signal_time = None
        
        while not self.strategy_stop_event.is_set():
            try:
                # Fetch latest data
                df = self.fetch_historical_data(symbol, days=100, timeframe=timeframe)
                
                # Calculate indicators
                df = self._calculate_indicators(df, strategy_name, parameters)
                
                # Check for signals
                should_buy, should_sell = self._check_signal(df, strategy_name, parameters)
                
                # Get current positions
                positions = self.get_positions()
                has_position = symbol in positions
                
                # Execute trades based on signals
                if should_buy and not has_position:
                    # Calculate position size (use 95% of buying power)
                    account = self.get_account_info()
                    buying_power = account.get('buying_power', 0)
                    current_price = df['close'].iloc[-1]
                    
                    if buying_power > 0 and current_price > 0:
                        # Use 95% of buying power
                        position_value = buying_power * 0.95
                        quantity = int(position_value / current_price)
                        
                        if quantity > 0:
                            print(f"\n🟢 BUY SIGNAL: {symbol} @ ${current_price:.2f}")
                            print(f"   Quantity: {quantity} shares")
                            order = self.place_order(symbol, quantity, 'buy', 'market')
                            if order:
                                last_signal_time = datetime.now()
                
                elif should_sell and has_position:
                    position = positions[symbol]
                    quantity = int(position['qty'])
                    
                    if quantity > 0:
                        current_price = df['close'].iloc[-1]
                        print(f"\n🔴 SELL SIGNAL: {symbol} @ ${current_price:.2f}")
                        print(f"   Quantity: {quantity} shares")
                        order = self.place_order(symbol, quantity, 'sell', 'market')
                        if order:
                            last_signal_time = datetime.now()
                
                # Wait before next check
                self.strategy_stop_event.wait(interval_seconds)
                
            except Exception as e:
                print(f"❌ Error in strategy loop: {e}")
                time.sleep(interval_seconds)
        
        print(f"⏹️  Strategy loop stopped for {symbol}")
    
    def start_strategy(self, symbol: str, strategy_name: str, parameters: Dict, interval_seconds: int = 60):
        """
        Start live trading with a strategy
        
        Args:
            symbol: Stock symbol to trade (e.g., 'AAPL')
            strategy_name: Name of the strategy (e.g., 'rsi_mean_reversion')
            parameters: Strategy parameters dictionary
            interval_seconds: How often to check for signals (default: 60 seconds)
        """
        if not self.is_connected:
            print("❌ Not connected to Alpaca API. Call connect() first.")
            return
        
        # Validate strategy exists and exclude buy_and_hold (not suitable for live trading)
        available_strategies = list_available_strategies()
        if strategy_name not in available_strategies:
            print(f"❌ Strategy '{strategy_name}' not found.")
            print(f"   Available strategies: {list(available_strategies.keys())}")
            return
        
        if strategy_name == 'buy_and_hold':
            print("❌ 'buy_and_hold' strategy is not available for live trading.")
            print("   It's designed for backtesting/benchmarking only.")
            return
        
        if self.strategy_running:
            print("❌ A strategy is already running. Stop it first.")
            return
        
        print(f"\n🚀 Starting live strategy: {strategy_name}")
        print(f"   Symbol: {symbol}")
        print(f"   Parameters: {parameters}")
        
        # Reset stop event
        self.strategy_stop_event.clear()
        self.strategy_running = True
        self.running_strategy = {
            'symbol': symbol,
            'strategy_name': strategy_name,
            'parameters': parameters
        }
        
        # Start strategy in background thread
        self.strategy_thread = threading.Thread(
            target=self._strategy_loop,
            args=(symbol, strategy_name, parameters, interval_seconds),
            daemon=True
        )
        self.strategy_thread.start()
        
        print(f"✓ Strategy started in background thread")
        print(f"   Press Ctrl+C or call stop_strategy() to stop")
    
    def stop_strategy(self):
        """Stop live trading strategy"""
        if not self.strategy_running:
            print("⚠️  No strategy is currently running")
            return
        
        print("⏹️  Stopping strategy...")
        self.strategy_stop_event.set()
        
        if self.strategy_thread and self.strategy_thread.is_alive():
            self.strategy_thread.join(timeout=5)
        
        self.strategy_running = False
        self.running_strategy = None
        print("✓ Strategy stopped")


def main():
    """Demo of live trading engine"""
    print("=" * 60)
    print("LIVE TRADING ENGINE - ALPACA INTEGRATION")
    print("=" * 60)
    
    try:
        # Initialize engine
        engine = LiveTradingEngine(paper_trading=True)
        
        # Connect to Alpaca
        if not engine.connect():
            print("❌ Failed to connect. Please check your API credentials.")
            return
        
        # Display account information
        print("\n" + "=" * 60)
        print("ACCOUNT INFORMATION")
        print("=" * 60)
        account_info = engine.get_account_info()
        if account_info:
            print(f"Account Number: {account_info.get('account_number', 'N/A')}")
            print(f"Status: {account_info.get('status', 'N/A')}")
            print(f"Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
            print(f"Cash: ${account_info.get('cash', 0):,.2f}")
            print(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            print(f"Equity: ${account_info.get('equity', 0):,.2f}")
        
        # Display current positions
        print("\n" + "=" * 60)
        print("CURRENT POSITIONS")
        print("=" * 60)
        positions = engine.get_positions()
        if positions:
            for symbol, pos in positions.items():
                print(f"{symbol}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}")
                print(f"  Market Value: ${pos['market_value']:,.2f}")
                print(f"  Unrealized P/L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.2f}%)")
        else:
            print("No open positions")
        
        # Disconnect
        print("\n" + "=" * 60)
        engine.disconnect()
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("   Please ensure API credentials are set in your .env file:")
        print("   - APCA_API_KEY_ID and APCA_SECRET_KEY")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 