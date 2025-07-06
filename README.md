# Algorithmic Trading System

## Overview

A backtesting system for trading strategies using backtrader, yfinance, and designed for future Alpaca integration. Analyzes strategy performance with comprehensive metrics and visualizations.

## Available Strategies

### RSI Mean Reversion
- Buy when RSI < 30 (oversold)
- Sell when RSI > 70 (overbought)
- Uses 14-period RSI

### Moving Average Crossover
- Buy when 10-day MA crosses above 50-day MA
- Sell when 10-day MA crosses below 50-day MA

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
RISK_FREE_RATE=0.02
```

## Usage

### Interactive Analysis
```bash
python run_strategies.py
```

Choose single or multiple strategies, customize parameters (symbol, timeframe, period, capital), and view detailed analysis with plots.

### Direct Backtesting
```bash
python backtesting_engine.py
```

Runs automated tests on both strategies with default parameters.

## Configuration

### Strategy Parameters

**RSI Strategy:**
- `rsi_period`: 14
- `rsi_oversold`: 30
- `rsi_overbought`: 70

**Moving Average Strategy:**
- `short_period`: 10
- `long_period`: 50

### System Parameters
- Default symbol: AAPL
- Default benchmark: SPY
- Default period: 6 months
- Default interval: 15 minutes
- Default capital: $10,000
- Commission: 0.1%

## Performance Metrics

The system calculates comprehensive performance metrics including risk-adjusted returns, benchmark analysis, trade analysis, and fundamental analysis. Detailed metric definitions are provided in the source code documentation.

## Project Structure

```
algorithmic_trading/
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── rsi_mean_reversion.py
│   └── ma_crossover.py
├── backtesting_engine.py
├── run_strategies.py
├── live_trading.py
├── requirements.txt
└── README.md
```

## Support

For questions, consult the official documentation:
- [Backtrader Documentation](https://www.backtrader.com/docu/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Alpaca Documentation](https://docs.alpaca.markets/) 