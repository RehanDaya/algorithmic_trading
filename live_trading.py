#!/usr/bin/env python3
"""Live Trading Engine - Alpaca integration"""

import os
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Alpaca SDK imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

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
        self.data_stream = None
        self.is_connected = False
        
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
    
    def start_strategy(self, strategy_name: str, parameters: Dict):
        """
        Start live trading with a strategy
        
        Args:
            strategy_name: Name of the strategy to run
            parameters: Strategy parameters dictionary
        """
        if not self.is_connected:
            print("❌ Not connected to Alpaca API. Call connect() first.")
            return
        
        print(f"🚀 Starting strategy: {strategy_name}")
        print(f"   Parameters: {parameters}")
        print("   Note: Strategy execution logic needs to be implemented")
        # TODO: Implement strategy execution loop
    
    def stop_strategy(self):
        """Stop live trading strategy"""
        print("⏹️  Stopping strategy")
        # TODO: Implement strategy stop logic


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