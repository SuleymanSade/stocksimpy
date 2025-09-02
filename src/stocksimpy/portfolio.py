# src/stocksimpy/portfolio.py

from collections import defaultdict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Portfolio:
    def __init__(self, initial_cap:float=100_000):
        self.initial_cap =initial_cap
        self.cash = initial_cap
        self.holdings = defaultdict(int) # To prevent code from crashing if a non-initialized symbol is tried to accessed
        
        self.value_history = pd.Series(dtype='float64')
        self.trade_log = pd.DataFrame(
        columns=[
            'Date', 'Symbol', 'Type', 'Price', 'Shares',
            'TransactionFee', 'TotalAmount'
            ]
        ).astype({
            'Date': 'datetime64[ns]', 'Symbol': 'object', 'Type': 'object',
            'Price': 'float64', 'Shares': 'float64',
            'TransactionFee': 'float64', 'TotalAmount': 'float64'
        })
        
    def _log_trade(self, symbol:str, trade_type:str, price:float, shares:float, transaction_fee:float, total_amount:float, date:pd.Timestamp):
        add = pd.DataFrame([{
            'Date': date,
            'Symbol': symbol,
            'Type': trade_type,
            'Price': price,
            'Shares': shares,
            'TransactionFee': transaction_fee,
            'TotalAmount': total_amount
        }])
        self.trade_log = pd.concat([self.trade_log, add], ignore_index=True)
    
    def exec_trade(self, symbol:str, trade_type:str, price:float, shares:float, date:pd.Timestamp, transaction_fee:float=0.000):
        trade_type = trade_type.lower()
        if price <= 0:
            raise ValueError("price cannot be 0 or less than 0")
        if shares < 0:
            raise ValueError("shares cannot be negative")
        if trade_type not in ['buy', 'sell']:
            raise ValueError("trade_type has to be one of the following: 'buy', 'sell'")
        if transaction_fee < 0:
            raise ValueError("transaction_fee cannot be negative")
        
        shares = int(shares)
        
        if trade_type == 'buy':
            total_cost = price * shares + transaction_fee
                
            if self.cash < total_cost:
                logger.warning("Available cash is less than the cost of trade. Buying the max amount of shares possible.")
                max_shares_possible = int((self.cash - transaction_fee) / price)
                
                # Edge case of transaction_fee > self.cash
                if max_shares_possible <= 0:
                    logger.warning("Not enough cash to buy any shares.")
                    return
                
                shares = max_shares_possible
                total_cost = price * shares + transaction_fee
            
            self.cash -= total_cost
            self.holdings[symbol] +=shares
            
            self._log_trade(symbol, trade_type, price, shares, transaction_fee, total_cost, date)
            
            # For debugging purposes:
            # print(f"successfully bought {shares} shares of {symbol} for {price} per stock, total spent: {total_cost}")
            
        else:
            if self.holdings[symbol] < shares:
                logger.warning(f"Not enough shares to execute sell order. \nSelling all the shares for {symbol}")
                shares = self.holdings[symbol]
            
            total_rev = (shares * price) - transaction_fee
            self.cash += total_rev
            self.holdings[symbol] -= shares
            
            self._log_trade(symbol, trade_type, price, shares, transaction_fee, total_rev, date)
            
    def update_value(self, current_date, current_prices: dict):
        """
        Updates the total value of the portfolio and appends it to value_history.
        This is a corrected version that calculates the total value (cash + holdings).
        """
        holdings_value = 0
        for symbol, num_shares in self.holdings.items():
            if symbol in current_prices:
                holdings_value += num_shares * current_prices[symbol]

        total_value = self.cash + holdings_value
        self.value_history.loc[current_date] = total_value