# Algorithmic Trading System - Multi-Strategy Framework

## Overview

This is a comprehensive algorithmic trading system implementing multiple trading strategies for backtesting. The system is designed for easy extension to live trading with Alpaca Markets and includes extensive performance metrics and risk analysis.

## Available Strategies

### 1. RSI Mean Reversion Strategy
- **Entry Signal**: RSI < 30 (oversold condition)
- **Exit Signal**: RSI > 70 (overbought condition)
- **Parameters**: 14-period RSI
- **Description**: Buys when oversold, sells when overbought

### 2. Moving Average Crossover Strategy
- **Entry Signal**: 10-period MA crosses above 50-period MA (golden cross)
- **Exit Signal**: 10-period MA crosses below 50-period MA (death cross)
- **Parameters**: 10-day short MA, 50-day long MA
- **Description**: Follows trend momentum using moving average crossovers

## Common Configuration
- **Timeframe**: 15-minute bars (with daily fallback)
- **Position Size**: 1 share (fixed)
- **Target Asset**: AAPL (Apple Inc.)
- **Benchmark**: SPY (S&P 500 ETF)

## Features

### 📈 Comprehensive Performance Metrics
- **Risk-Adjusted Returns**: Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Benchmark Analysis**: Alpha, Beta, Information Ratio
- **Drawdown Analysis**: Maximum Drawdown, Recovery Periods
- **Trade Analysis**: Win/Loss Ratio, Profit Factor, Trade Frequency
- **Fundamental Analysis**: Piotroski F-Score calculation

#### Detailed Metric Descriptions:

**Alpha**: Excess return over the S&P 500 (SPY) benchmark, adjusted for a risk-free rate (assume 2%).

**Beta**: Strategy's sensitivity to SPY returns.

**Sharpe Ratio**: Risk-adjusted return using standard deviation.

**Sortino Ratio**: Risk-adjusted return using downside deviation.

**Maximum Drawdown**: Largest peak-to-trough loss.

**Win/Loss Ratio**: Ratio of winning to losing trades.

**Profit Factor**: Gross profit divided by gross loss.

**Annualized Return**: Compounded return scaled to one year.

**Calmar Ratio**: Annualized return divided by maximum drawdown.

**Information Ratio**: Excess return over SPY divided by tracking error.

**Trade Frequency**: Number of trades over the backtest period.

**Average Trade Duration**: Average time positions are held.

**Piotroski F-Score**: Calculate AAPL's F-Score using yfinance fundamental data (e.g., ROA, debt ratios) to confirm stock quality before backtesting.

### 🔧 Technical Features
- Real-time data fetching using `yfinance`
- Professional backtesting with `backtrader`
- Comprehensive error handling
- Secure API key management
- Extensible architecture for live trading

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the files**:
   ```bash
   # Ensure you have these files:
   # - trading_bot.py
   # - requirements.txt
   # - README.md
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file**:
   Create a `.env` file in the same directory with your Alpaca API credentials:
   ```env
   # Alpaca API Configuration
   ALPACA_API_KEY=PK6HKCPZHA3OB007UNM2
   ALPACA_SECRET_KEY=your_secret_key_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
   
   # Risk-free rate for calculations (2% annual)
   RISK_FREE_RATE=0.02
   ```

## Usage

### Running the Backtest

```bash
python trading_bot.py
```

The system will automatically:
1. Fetch 6 months of 15-minute data for AAPL and SPY
2. Calculate Piotroski F-Score for fundamental analysis
3. Run the RSI mean-reversion backtest
4. Calculate comprehensive performance metrics
5. Display results and generate plots

### Expected Output

The system provides detailed output including:

```
🚀 ALGORITHMIC TRADING SYSTEM STARTING...
============================================================
📋 Checking environment setup...
Fetching 6mo of 15m data for AAPL and SPY...
✓ Retrieved 1,234 bars for AAPL
✓ Retrieved 1,234 bars for SPY
Calculating Piotroski F-Score for AAPL...
✓ Piotroski F-Score calculated: 7/9
Running backtest with $10,000.00 initial cash...
✓ Backtest completed successfully
...
```

### Performance Metrics

The system calculates and displays:

- **Basic Performance**: Total Return, Annualized Return
- **Risk Metrics**: Sharpe Ratio, Sortino Ratio, Maximum Drawdown
- **Benchmark Comparison**: Alpha, Beta, Information Ratio
- **Trade Analysis**: Win Rate, Profit Factor, Trade Frequency
- **Fundamental Analysis**: Piotroski F-Score

## Configuration

### Strategy Selection

You can choose different strategies when initializing the system:

```python
# RSI Mean Reversion Strategy
system = TradingSystem(symbol='AAPL', benchmark='SPY', strategy='rsi_mean_reversion')

# Moving Average Crossover Strategy
system = TradingSystem(symbol='AAPL', benchmark='SPY', strategy='ma_crossover')
```

### Strategy Parameters

**RSI Mean Reversion Strategy:**
```python
params = (
    ('rsi_period', 14),        # RSI calculation period
    ('rsi_oversold', 30),      # Oversold threshold
    ('rsi_overbought', 70),    # Overbought threshold
    ('position_size', 1),      # Fixed position size
    ('printlog', False),       # Enable trade logging
)
```

**Moving Average Crossover Strategy:**
```python
params = (
    ('short_period', 10),      # Short MA period
    ('long_period', 50),       # Long MA period
    ('position_size', 1),      # Fixed position size
    ('printlog', False),       # Enable trade logging
)
```

### System Configuration

Modify the `TradingSystem` initialization:

```python
# Change symbol, benchmark, and strategy
system = TradingSystem(symbol='AAPL', benchmark='SPY', strategy='ma_crossover')

# Modify data period and interval
system.fetch_data(period='6mo', interval='15m')

# Adjust initial capital
system.run_backtest(initial_cash=10000.0)
```

## Extending to Live Trading

The system is designed for easy extension to live trading with Alpaca:

### 1. Data Streaming
```python
# TODO: Replace yfinance with Alpaca real-time data
# from alpaca.data import StockDataStream
```

### 2. Order Execution
```python
# TODO: Implement Alpaca order execution
# from alpaca.trading import TradingClient
```

### 3. Portfolio Management
```python
# TODO: Add position sizing based on account equity
# TODO: Implement risk management rules
```

## Risk Disclaimer

⚠️ **Important**: This system is for educational and research purposes only. Past performance does not guarantee future results. Always:

- Test thoroughly in paper trading before live deployment
- Understand all risks involved in algorithmic trading
- Never risk more than you can afford to lose
- Consider transaction costs and slippage in live trading
- Monitor systems continuously during live operation

## Troubleshooting

### Common Issues

1. **Data Fetching Errors**:
   - Check internet connection
   - Verify symbol names are correct
   - Try reducing the data period if timeouts occur

2. **Missing Dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Environment Variables**:
   - Ensure `.env` file exists in the same directory
   - Check that API keys are correctly formatted

4. **Plotting Issues**:
   - Install additional plotting backends if needed:
   ```bash
   pip install PyQt5  # or pip install tkinter
   ```

### Performance Optimization

For better performance:
- Use SSD storage for data caching
- Increase available RAM for larger datasets
- Consider using multiprocessing for multiple symbols

## License

This project is provided as-is for educational purposes. Use at your own risk.

## Project Structure

```
algorithmic_trading/
├── strategies/                 # Strategy package folder
│   ├── __init__.py            # Package initialization and registry
│   ├── base_strategy.py       # Base strategy class
│   ├── rsi_mean_reversion.py  # RSI mean reversion strategy
│   └── ma_crossover.py        # Moving average crossover strategy
├── trading_bot.py             # Main trading system
├── test_strategies.py         # Strategy testing script
├── requirements.txt           # Python dependencies
├── example.env               # Environment template
└── README.md                 # This file
```

## Adding New Strategies

To add a new trading strategy:

1. **Create a new strategy file** in the `strategies/` folder (e.g., `strategies/my_custom.py`):
```python
from .base_strategy import BaseStrategy
import backtrader as bt

class MyCustomStrategy(BaseStrategy):
    params = (
        ('param1', 10),
        ('param2', 20),
        ('position_size', 1),
        ('printlog', False),
    )
    
    def init_indicators(self):
        # Initialize your indicators here
        self.my_indicator = bt.indicators.SMA(self.data.close, period=self.params.param1)
    
    def get_strategy_name(self) -> str:
        return "My Custom Strategy"
    
    def get_strategy_description(self) -> str:
        return f"Custom strategy with param1={self.params.param1}"
    
    def should_buy(self) -> bool:
        # Implement buy logic
        return self.data.close[0] > self.my_indicator[0]
    
    def should_sell(self) -> bool:
        # Implement sell logic
        return self.data.close[0] < self.my_indicator[0]
```

2. **Register the strategy** in `strategies/__init__.py`:
```python
from .my_custom import MyCustomStrategy

AVAILABLE_STRATEGIES = {
    'rsi_mean_reversion': RSIMeanReversionStrategy,
    'ma_crossover': MovingAverageCrossoverStrategy,
    'my_custom': MyCustomStrategy,  # Add your strategy here
}
```

3. **Use the new strategy**:
```python
system = TradingSystem(symbol='AAPL', benchmark='SPY', strategy='my_custom')
```

## Contributing

Feel free to submit issues and enhancement requests. Key areas for improvement:
- Additional technical indicators
- Multi-symbol portfolio management
- Advanced risk management features
- Real-time alerting systems
- Web-based dashboard

## Support

For questions or support:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Consult the official documentation for dependencies:
   - [Backtrader Documentation](https://www.backtrader.com/docu/)
   - [yfinance Documentation](https://github.com/ranaroussi/yfinance)
   - [Alpaca Documentation](https://docs.alpaca.markets/) 