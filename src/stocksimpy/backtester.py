# src/stocksimpy/backtester.py

from stock_data import StockData
from portfolio import Portfolio
import pandas as pd

class Backtester:
    def __init__(self, symbol:str, data:StockData, strategy, initial_cap: float= 100_000, transaction_fee: float = 0.000, trade_amount: float = 10_000):
        self.data =data.to_dataframe()
        self.strategy = strategy
        self.initial_cap = initial_cap
        self.transaction_fee = transaction_fee
        
        self.portfolio = Portfolio(initial_cap)
        self.symbol = symbol
        self.trade_amount = trade_amount
        
    def _process_trade(self, signal: str, shares:int, price:float, date):
        if signal in ['buy', 'sell']:
            self.portfolio.exec_trade(
                symbol=self.symbol,
                trade_type=signal,
                price=price,
                shares=shares,
                date=date,
                transaction_fee=self.transaction_fee
            )
            
            self.portfolio.update_value(date, {self.symbol: price})
            
        
    def run_backtest_fixed(self):
        for i in range(1, len(self.data)):
            current_date = self.data.index[i]
            
            historic_vals = self.data.iloc[:i+1]
            # If MultiIndex (e.g. yfinance data) only get the data for a specific symbol
            if isinstance(self.data.columns, pd.MultiIndex):
                historic_vals = historic_vals.xs(self.symbol, axis=1, level=-1)
                
            
            signal = self.strategy(historic_vals)
            
            price = self.data.loc[current_date, ('Close', self.symbol)]
            
            if price >0:
                shares_to_trade = int(self.trade_amount/price)
            else:
                shares_to_trade = 0
                
            self._process_trade(signal, shares_to_trade, price, current_date)
            
    def run_backtest_dynamic(self):
        for i in range(1, len(self.data)):
            current_date = self.data.index[i]
            historic_vals = self.data.iloc[:i+1]
                        
            try:
                signal, shares_to_trade = self.strategy(historic_vals, self.portfolio.holdings[self.symbol])
            except ValueError:
                raise TypeError(
                    "strategy function should return a tuple (signal, shares) for dynamic sizing."
                )
            
            historic_vals = self.data.iloc[:i+1]
            # If MultiIndex (e.g. yfinance data) only get the data for a specific symbol
            if isinstance(self.data.columns, pd.MultiIndex):
                historic_vals = historic_vals.xs(self.symbol, axis=1, level=-1)
                
            price = self.data.loc[current_date, ('Close', self.symbol)]
            
            self._process_trade(signal, shares_to_trade, price, current_date)
            
    def generate_report(self) -> dict:
        if self.portfolio.value_history.empty:
            return {
                'final_value': self.portfolio.initial_cap,
                'total_return_percent': 0.0,
                'number_of_trades': 0
            }
        
        final_value = self.portfolio.value_history.iloc[-1]
        total_return = (final_value - self.portfolio.initial_cap) / self.portfolio.initial_cap
        
        return {
            'final_value': final_value,
            'total_return_percent': total_return * 100,
            'number_of_trades': len(self.portfolio.trade_log)
        }