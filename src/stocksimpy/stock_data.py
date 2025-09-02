# src/stocksimpy/data_handler.py
import pandas as pd
import numpy as np
from datetime import date, timedelta

class StockData:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # self._process_and_validate(df)
        
    def _clean(self, df):
        
        df = df.copy()
        
        # We check if the df is already a DatetimeIndex which is seen when loaded from yfinance or formatted this way
        # The below code handles the case when df isn't already DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            else:
                raise ValueError("Input DataFrame must have a 'Date' column or a DatetimeIndex.")
    
        df.columns = [col.strip().capitalize() for col in df.columns]
        
        df.sort_index(inplace=True)
        return df
    
    def _validate(self, df:pd.DataFrame):
        
        if self.df.empty:
            raise ValueError("DataFrame cannot be empty.")

        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
            
        # Check for sorted and unique index
        if not self.df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index is not sorted monotonically.")
        if self.df.index.has_duplicates:
            raise ValueError("DataFrame index contains duplicate dates.")
        
        # Checks if all the required columns are present in the data frame
        required_cols = ['Open', 'Close', 'Volume', 'High', 'Low']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check all data numerical
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(self.df[col].dtype):
                raise TypeError(f"Column '{col}' must be a numerical type. Found: {self.df[col].dtype}")
            
        if (self.df['Volume'] < 0).any():
            raise ValueError("Volume data contains negative values.")
        
        # Basic logic checking (Low <= Open, Low <= Close, High >= Open, High >= Close)
        if (self.df['Low'] > self.df['High']).any():
            raise ValueError("OHLC data inconsistency: Low price is greater than High price.")
        if (self.df['Open'] > self.df['High']).any() or (self.df['Open'] < self.df['Low']).any():
            raise ValueError("OHLC data inconsistency: Open price is outside the High/Low range.")
        if (self.df['Close'] > self.df['High']).any() or (self.df['Close'] < self.df['Low']).any():
            raise ValueError("OHLC data inconsistency: Close price is outside the High/Low range.")
        
        

        
        
    def _process_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = self._clean(df)
        self._validate(df_clean)
        return df_clean
    
    # ------------------------------------------
    # LOAD DATA
    
    @classmethod
    def generate_mock_data(cls, days: int = 100, seed: int = 42):
        np.random.seed(seed)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
        prices = np.cumsum(np.random.randn(days)) + 100
        high = prices + np.random.rand(days) * 2
        low = prices - np.random.rand(days) * 2
        open_ = prices + np.random.randn(days) * 0.5
        close = prices + np.random.randn(days) * 0.5
        volume = np.random.randint(1000, 10000, size=days)

        df = pd.DataFrame({
            'Date': dates,
            'Open': open_,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
        return cls(df)
        
    
    @classmethod
    def from_csv(cls, file_path:str):
        df =pd.read_csv(file_path)
        return cls(df)
    
    @classmethod
    def from_excel(cls, file_path:str):
        df = pd.read_excel(file_path)
        return cls(df)
    
    @classmethod
    def from_sql(cls, query: str, connection):
        df = pd.read_sql(query, connection)
        return cls(df)
    
    @classmethod
    def from_yfinance(cls, ticker: str, start_date: date = None, end_date: date = None, days_before: int = None):
        import yfinance as yf
        try:
            if days_before:
                today = date.today()
                start_date = today - timedelta(days=days_before)
                end_date = today
            data = yf.download(ticker, start=start_date, end=end_date)
            data.reset_index(inplace=True)
            return cls(data)
        except ImportError:
            raise ImportError("yfinance is not installed. Install it to use Yahoo Finance loaders.")
        
    
    @classmethod
    def from_data_frame(cls, df:pd.DataFrame):
        return cls(df)
    
    @classmethod
    def from_dict(cls, data:dict):
        df = pd.DataFrame(data)
        return cls(df)
    
    @classmethod
    def from_json(cls, json_data: dict):
        df = pd.DataFrame(json_data)
        return cls(df)
    
    @classmethod
    def from_sqlite(cls, query: str, db_path: str):
        import sqlite3
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        return cls(df)
    
    @staticmethod
    def auto_loader(source, **kwargs):
        """
        Automatically detects input type and routes to the correct StockData loader.
        Supports: CSV, Excel, SQLite, DataFrame, dict, JSON, and yfinance.
        """
        import os
        from datetime import date

        # DataFrame
        if isinstance(source, pd.DataFrame):
            return StockData.from_dataframe(source)

        # Dict (possibly JSON data)
        elif isinstance(source, dict):
            return StockData.from_dict(source)

        # File path string
        elif isinstance(source, str):
            ext = os.path.splitext(source)[-1].lower()
            if ext == '.csv':
                return StockData.from_csv(source)
            elif ext in ['.xls', '.xlsx']:
                return StockData.from_excel(source)
            elif ext == '.db':
                query = kwargs.get('query', "SELECT * FROM stock_data")
                return StockData.from_sqlite(query, source)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        # Tuple → yfinance loader
        elif isinstance(source, tuple):
            if len(source) == 2 and isinstance(source[1], int):
                ticker, days_before = source
                return StockData.from_yfinance(ticker=ticker, days_before=days_before)
            elif (
                len(source) == 3 and
                isinstance(source[1], date) and
                isinstance(source[2], date)
            ):
                ticker, start_date, end_date = source
                return StockData.from_yfinance(ticker=ticker, start_date=start_date, end_date=end_date)
            else:
                raise ValueError("Tuple input must be (ticker, days_before) or (ticker, start_date, end_date)")

        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
    
    # -----------------------
    # BASIC INFO AND FUNCTIONALITIES
    
    def get(self, column: str):
        return self.df[column]

    def slice(self, start=None, end=None):
        return self.df.loc[start:end]

    def head(self, n=5):
        return self.df.head(n)
    
    def info(self):
        return self.df.info()

    def fill_missing(self, method='ffill'):
        self.df.fillna(method=method, inplace=True)
        return self

    def check_missing(self):
        return self.df.isnull().sum()
    
    # --------------------------
    # EXPORT DATA

    def to_csv(self, file_path: str, **kwargs):
        """Export the DataFrame to a CSV file."""
        self.df.to_csv(file_path, **kwargs)
        return file_path

    def to_excel(self, file_path: str, **kwargs):
        """Export the DataFrame to an Excel file."""
        self.df.to_excel(file_path, **kwargs)
        return file_path

    def to_sql(self, table_name: str, connection, if_exists='replace', **kwargs):
        """Export the DataFrame to a SQL table using an open connection."""
        self.df.to_sql(table_name, connection, if_exists=if_exists, index=True, **kwargs)
        return table_name

    def to_sqlite(self, table_name: str, db_path: str, if_exists='replace', **kwargs):
        """Export the DataFrame to a SQLite database file."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        self.df.to_sql(table_name, conn, if_exists=if_exists, index=True, **kwargs)
        conn.close()
        return db_path

    def to_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self.df.copy()

    def to_dict(self, orient='records'):
        """Export the DataFrame to a Python dictionary."""
        return self.df.to_dict(orient=orient)

    def to_json(self, file_path: str = None, orient='records', **kwargs):
        """Export the DataFrame to a JSON string or file."""
        json_str = self.df.to_json(orient=orient, **kwargs)
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
            return file_path
        return json_str
    
    def to_custom(self, export_func, *args, **kwargs):
        """
        Export the DataFrame using a custom function.

        Parameters
        ----------
        export_func : callable
            A function that takes a DataFrame as its first argument and returns an export result.
        *args, **kwargs :
            Additional arguments to pass to the export_func.

        Returns
        -------
        The result of export_func(self.df, *args, **kwargs)
        """
        return export_func(self.df, *args, **kwargs)