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

### Buy and Hold (Benchmark)
- Buy at the start of the backtest period
- Hold until the end (no selling)
- Perfect benchmark for comparing active strategies

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

Choose single or multiple strategies from 3 available options, customize parameters (symbol, timeframe, period, capital), and view detailed analysis with plots. Supports multi-strategy comparison with buy-and-hold benchmark.

### Direct Backtesting
```bash
python backtesting_engine.py
```

Runs automated tests on all three strategies with default parameters. Includes comprehensive metrics and plotting.

## Configuration

### Strategy Parameters

**RSI Strategy:**
- `rsi_period`: 14
- `rsi_oversold`: 30
- `rsi_overbought`: 70

**Moving Average Strategy:**
- `short_period`: 10
- `long_period`: 50

**Buy and Hold Strategy:**
- No parameters (buys once and holds)

### System Parameters
- Default symbol: AAPL
- Default benchmark: SPY
- Default period: 3 months
- Default interval: 1 day
- Default capital: $10,000
- Commission: 0.1%
- Position sizing: All-In (backtrader built-in sizer)

## Performance Metrics

The system provides comprehensive performance analysis with the following metrics:

### 💰 Performance Metrics

**Initial Capital**
- Starting cash amount for the backtest
- Baseline for calculating returns

**Final Value** 
- Portfolio value at the end of the backtest period
- Includes cash + market value of positions

**Total Return**
- Absolute dollar gain/loss: Final Value - Initial Capital
- Also shown as percentage: (Final - Initial) / Initial × 100%

**Annualized Return**
- Return adjusted for time period to show yearly equivalent
- Formula: (Final/Initial)^(1/years) - 1
- Allows comparison across different time periods

### 📊 Risk Metrics

**Sharpe Ratio**
- Risk-adjusted return measure: (Return - Risk-Free Rate) / Standard Deviation
- Higher is better (typically > 1.0 is good, > 2.0 is excellent)
- Measures return per unit of volatility risk

**Sortino Ratio**
- Like Sharpe ratio but only considers downside deviation
- Better measure for strategies with asymmetric return distributions
- Higher values indicate better risk-adjusted performance

**Maximum Drawdown**
- Largest peak-to-trough decline during the backtest period
- Expressed as percentage from peak value
- Lower is better - shows worst-case loss scenario

**Calmar Ratio**
- Annualized return divided by maximum drawdown
- Higher is better - rewards consistent performance
- Good measure of return relative to worst-case risk

### 📈 Benchmark Comparison

**Beta**
- Measures strategy's sensitivity to benchmark (market) movements
- Beta = 1.0 means moves with market, > 1.0 more volatile, < 1.0 less volatile
- Typical range: 0.5 to 1.5 for equity strategies

**Alpha**
- Excess return over what Beta would predict from benchmark
- Measures value added by strategy beyond market exposure
- Positive alpha indicates outperformance vs risk-adjusted benchmark

**Information Ratio**
- Measures consistency of outperformance vs benchmark
- (Strategy Return - Benchmark Return) / Tracking Error
- Higher values indicate more consistent excess returns

### 🎯 Trade Analysis

**Total Trades**
- Number of completed buy-sell cycles during backtest
- Higher frequency strategies will have more trades

**Winning/Losing Trades**
- Count of profitable vs unprofitable trades
- Used to calculate win rate and other trade statistics

**Win Rate**
- Percentage of trades that were profitable
- Win Rate = Winning Trades / Total Trades × 100%
- High win rate doesn't guarantee profitability (need to consider trade sizes)

**Win/Loss Ratio**
- Ratio of winning trades to losing trades
- Values > 1.0 indicate more wins than losses
- Shows "∞" when there are no losing trades

**Profit Factor**
- Gross profit divided by gross loss
- Values > 1.0 indicate profitable strategy
- Higher values show better profitability relative to losses

### ⏰ Trade Frequency

**Trade Frequency**
- Number of trades per year (annualized)
- Higher frequency strategies have more transaction costs
- Useful for understanding strategy turnover

**Average Trade Duration**
- Average time positions are held (in hours)
- Helps classify strategy type (scalping, swing, position trading)
- Longer duration generally means lower transaction costs

### 📋 Fundamental Analysis

**Piotroski F-Score**
- 9-point financial strength score (0-9 scale)
- Based on profitability, leverage, liquidity, and efficiency metrics
- Score interpretation:
  - 8-9: Strong financial position
  - 5-7: Moderate financial position  
  - 0-4: Weak financial position
- Only available for individual stocks (not ETFs/indices)

### 💡 Interpreting Results

**Good Strategy Characteristics:**
- Positive total return and alpha
- Sharpe ratio > 1.0
- Maximum drawdown < 20%
- Win rate > 50% (though not required if avg win > avg loss)
- Profit factor > 1.5

**Risk Considerations:**
- Higher returns often come with higher risk (drawdown)
- Consistent strategies (high Sharpe/Sortino) may be preferable to volatile high-return strategies
- Consider transaction costs impact on high-frequency strategies

**Benchmark Comparison:**
- Compare strategy metrics to buy-and-hold benchmark
- Strategy should outperform benchmark on risk-adjusted basis
- Consider if additional complexity is worth the performance improvement

## Project Structure

```
algorithmic_trading/
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── rsi_mean_reversion.py
│   ├── ma_crossover.py
│   └── buy_and_hold.py
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