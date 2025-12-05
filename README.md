# stocksimpy

#### Easy Python backtesting for stocks — fast and simple

stocksimpy lets you prototype trading ideas with minimal boilerplate. Everything is explicit — load data, define rules, run a backtest. No configuration sprawl, no hidden state, and no magic methods.. Perfect for beginners and anyone who wants results fast.

---

## 🎯 Quick Features

- Load stock prices from `yfinance` or your own database in seconds.
- Run fixed or dynamic backtests in just a few lines of code.
- Built-in example strategies like SMA/EMA crossover, RSI, and price action.
- Clear and simple design — no hidden magic, everything is easy to read.
- Beginner-friendly: start experimenting immediately.

---

## ⚡ [Get Started](docs/source/getting_started.md)

### 1. Install `stocksimpy` (and `yfinance` for convenient data input)

```bash
pip install stocksimpy yfinance
```

### 2. Imports

```python
from stocksimpy import StockData, Backtester, Visualize, Performance
```

### 3. Load Data

```python
# If you want to load your own data from .csv, .sqlite or similar, use appropriate functions built into `StockData()`
data = StockData().from_yfinance([your_stock_symbol], your_starting_date, your_end_date)
```

### 4. Define Your Strategy

```python
def sma_crossover(prices):
    short = prices["Close"].rolling(20).mean()
    long = prices["Close"].rolling(50).mean()

    # Buy when short crosses above long, sell when it crosses below
    signal = (short > long).astype(int)
    return signal
```

### 5. Run Backtest

```python
# You can use your own strategy or one of the built-in ones for testing (e.g. Strategy.rsi_momentum_fixed())
bt = Backtester('AAPL', data, sma_crossover)

# Depending on your strategy you may need to run bt.run_backtest_dynamic(), see more on the documentation
bt.run_backtest_fixed()
```

### 6. Evaluate the Results

```python
# View performance metrics like max drawdown and sharpe ratio
perf = Performance(bt)
print(perf.generate_risk_report())
```

### 7. Visualize

```python
# Visualize the graph
graph = Visualize(bt)
graph.visualize_backtest().show() # You can use .savefig(...) instead of .show() to record the graph in disk
```

And thats it, you just ran your first backtest 🚀

---

## 🤝 Contributions

Ideas, strategies, or improvements? We welcome all contributions! Check out [CONTRIBUTING.md](CONTRIBUTING.md) for a more detailed information about contributing
