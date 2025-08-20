# src/stocksimpy/indicators.py

"""
This module provides basic functions for calculating various technical indicators:
Ex: SMA, RSI, MACD, DEMA, TEMA, etc.
"""

import pandas as pd
import numpy as np
import math

def _validate_indicator_inputs(data_series: pd.Series, window: int, min_data_length: int = 1) -> None:
    """Validate common inputs for indicator calculations.

    This function checks if data_series is a valid pandas Series,
    if it's not empty, if window is a positive integer, and if
    the data_series has sufficient length for the given window.

    Parameters
    ----------
    data_series : pandas.Series
        The input data series to validate.
    window : int
        The window period to validate.
    min_data_length : int, optional
        The minimum required length of the data_series after accounting for the window.
        Defaults to 1, meaning data_series length must be at least 'window'.
        Set to 0 if the indicator can handle 'window > len(data_series)' by returning NaNs.

    Raises
    ------
    TypeError
        If `data_series` is not a pandas Series.
    ValueError
        If `data_series` is empty, `window` is not a positive integer,
        or if `data_series` is too short for the `window` (and `min_data_length` check).
    """
    if type(window) is not int:
        raise TypeError("Window has to be an integer")
    if not isinstance(data_series, pd.Series):
        raise TypeError("Input 'data_series' must be a pandas Series.")
    if data_series.empty:
        raise ValueError("Input 'data_series' cannot be empty.")
    if not pd.api.types.is_numeric_dtype(data_series.dtype):
        raise TypeError("Input 'data_series' must be a numerical pandas Series")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("Window must be a positive integer.")
    if len(data_series) < (window + (min_data_length -1)): # Adjust min_data_length logic
        raise ValueError(
            f"Input data series length ({len(data_series)}) is too short for the specified window ({window}) "
            f"and required minimum data length (at least {window + min_data_length -1} entries needed). "
            "Please provide more data or a smaller window."
        )
    

# -----------------------------
# DIFFERENT TYPES OF EMA

def calculate_sma(data_series: pd.Series, window:int) -> pd.Series:
    """Calculate the Simple Moving Average (SMA) of a given data series.
    
    The SMA is the average of a selected range of prices,
    usually closing prices, by the number of periods in that range.

    Parameters:
    ----------
        data_series: pandas.Series
            The input data series for which to calculate the SMA.
        window: int
            The window size (number of periods) to use for the SMA calculation.

    Returns:
    ----------
    pandas.Series
        A pandas Series containing the SMA values. The initial 'window - 1' values will be NaN.
    """
    _validate_indicator_inputs(data_series=data_series, window=window)
    
    return data_series.rolling(window=window).mean()

def calculate_wma(data_series: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Weighted Moving Average (WMA) for a pandas Series.

    The WMA assigns a greater weight to more recent data points within the window.
    The weights increase linearly from 1 to the `window` size, with the most
    recent data point receiving the highest weight.

    Parameters
    ----------
    data_series : pd.Series
        The input pandas Series for which to calculate the WMA.
        Expected to contain numerical data.
    window : int
        The size of the moving window. This must be a positive integer.

    Returns
    -------
    pd.Series
        A pandas Series containing the Weighted Moving Average.
        The first `(window - 1)` values will be `NaN`, as a full window
        is required to calculate the WMA. The index of the returned Series
        will match the input `data_series`.
    """
    _validate_indicator_inputs(data_series, window)
    
    weights = pd.Series(np.arange(1, window + 1))
    weights_total = weights.sum()
    
    wma_series = pd.Series(index=data_series.index, dtype=float)
    
    for i in range(window - 1, len(data_series)):
        current_window_data_values = data_series.iloc[i - window + 1 : i + 1].values

        weighted_sum = (current_window_data_values * weights).sum()

        wma = weighted_sum / weights_total
        wma_series.iloc[i] = wma
    
    return wma_series
    
    
    
    
def calculate_ema(data_series: pd.Series, window: int) -> pd.Series:
    """Calculates the Exponential Moving Average (EMA) of a data series.

    Parameters:
    ----------
    data_series : pd.Series
        The input data series (e.g., closing prices).
    window : int
        The period for the EMA calculation.

    Returns:
    -------
    pd.Series
        A Series containing the EMA values.
    """
    
    _validate_indicator_inputs(data_series=data_series, window=window)
    
    return data_series.ewm(span=window, adjust=False, min_periods=window).mean()

def wilders_smoothing(data_series:pd.Series, window:int) -> pd.Series:
    """
    Calculate Wilder's Smoothing for a given data series.

    Wilder's Smoothing is an exponential moving average (EMA) with a specific smoothing factor,
    commonly used in technical indicators such as RSI and ATR. It provides a smoothed average
    that reacts more slowly to price changes than a simple moving average.

    Parameters
    ----------
    data_series : pandas.Series
        The input data series to smooth.
    window : int
        The window size (number of periods) for the smoothing calculation.

    Returns
    -------
    pandas.Series
        A pandas Series containing the smoothed values. The initial 'window - 1' values will be NaN.
    """
    
    _validate_indicator_inputs(data_series, window)
    
    return data_series.ewm(com = window-1, adjust=False, min_periods=window).mean()

def calculate_dema(data_series: pd.Series, window: int) -> pd.Series:
    """Calculate the Double Exponential Moving Average (DEMA) of a data series.

    DEMA attempts to reduce the lag inherent in traditional EMAs by
    using a double-smoothed EMA.

    Parameters
    ----------
    data_series : pandas.Series
        The input data series (e.g., closing prices).
    window : int
        The number of periods to use for the DEMA calculation.

    Returns
    -------
    pandas.Series
        A pandas Series containing the DEMA values. The initial 'window' * 2 - 1 values will be NaN
        (due to the double EMA calculation).

    Notes
    -----
    The formula for DEMA is:
    :math:`DEMA = (2 \\times EMA_{1}) - EMA_{2}`
    where:
    :math:`EMA_{1}` is the Exponential Moving Average of the original data series.
    :math:`EMA_{2}` is the Exponential Moving Average of :math:`EMA_{1}`.
    """
    
    _validate_indicator_inputs(data_series, window)
    
    if(len(data_series)< (2 * window - 1)):
        raise ValueError("Input data series length")
    
    ema1 = calculate_ema(data_series, window)
    ema2 = calculate_ema(ema1, window)
    
    return (2 * ema1) - ema2

def calculate_tema(data_series: pd.Series, window: int) -> pd.Series:
    """Calculate the Triple Exponential Moving Average (TEMA) of a data series.

    TEMA further reduces lag compared to DEMA by using a triple-smoothed EMA.

    Parameters
    ----------
    data_series : pandas.Series
        The input data series (e.g., closing prices).
    window : int
        The number of periods to use for the TEMA calculation.

    Returns
    -------
    pandas.Series
        A pandas Series containing the TEMA values. The initial 'window' * 3 - 2 values will be NaN
        (due to the triple EMA calculation).

    Notes
    -----
    The formula for TEMA is:
    :math:`TEMA = (3 \\times EMA_{1}) - (3 \\times EMA_{2}) + EMA_{3}`
    where:
    :math:`EMA_{1}` is the Exponential Moving Average of the original data series.
    :math:`EMA_{2}` is the Exponential Moving Average of :math:`EMA_{1}`.
    :math:`EMA_{3}` is the Exponential Moving Average of :math:`EMA_{2}`.
    """
    
    _validate_indicator_inputs(data_series, window)
    
    if(len(data_series)< (3 * window - 2)):
        raise ValueError("Input data series length")
    
    ema1 = calculate_ema(data_series, window)
    ema2 = calculate_ema(ema1, window)
    ema3 = calculate_ema(ema2, window)
    
    return (3*ema1) - (3*ema2) + ema3

def calculate_hma(data_series: pd.Series, window: int) -> pd.Series:
    """Calculate the Hull Moving Average (HMA) of a data series.

    The Hull Moving Average (HMA) is a directional trend indicator designed
    to minimize lag while maintaining smoothness. It achieves this by using
    a weighted moving average of price data and then applying a second WMA
    to further reduce lag. This implementation uses EMA as an approximation for WMA.

    Parameters
    ----------
    data_series : pandas.Series
        The input data series (e.g., closing prices).
    window : int
        The number of periods to use for the HMA calculation.

    Returns
    -------
    pandas.Series
        A pandas Series containing the HMA values. The initial values will be NaN
        due to the nested EMA calculations.

    Raises
    ------
    ValueError
        If the `window` is less than 2, as HMA calculation requires at least two periods.

    Notes
    -----
    This implementation uses the following equation to calculate HMA:
        HMA(n) = WMA(2 * WMA(n/2) - WMA(n)), √n)
    For WMA calculation `calculate_ema()` is used as an approximation
    """
    
    _validate_indicator_inputs(data_series, window)
    
    if window<2:
        raise ValueError("Window for HMA calculation cannot be less than 2")
    
    hma = calculate_ema((2 * calculate_ema(data_series, window//2)) - calculate_ema(data_series, window), int(math.sqrt(window)))
    return hma



# ----------------

def calculate_rsi(data_series: pd.Series, window:int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI) of a given data series.
    
    The RSI is a momentum oscillator that measures the speed and change of price movements.
    It ranges from 0 to 100 and is typically used to identify overbought or oversold conditions.

    Parameters:
    ----------
        data_series: pandas.Series
            The input data series for which to calculate the RSI.
        window: int, optional
            The number of periods to use for the RSI calculation (default is 14).

    Returns:
    ----------
    pandas.Series
        A pandas Series containing the RSI values. The initial 'window - 1' values will be NaN.
    """
    
    _validate_indicator_inputs(data_series=data_series, window=window, min_data_length=2)
    
    delta = data_series.diff()
    
    gain = delta.copy()
    loss = delta.copy()
    
    # Set all the NaN values to 0 for proper RSI calc
    gain[gain<0] = 0
    loss[loss>0] = 0
    loss = loss.abs()
    
    # Use Wilder's smoothing
    avg_gain = wilders_smoothing(gain, window)
    avg_loss = wilders_smoothing(loss, window)
        
    rs = avg_gain/avg_loss
    
    return 100 - (100/(1+rs))

# -----------------------------
# DIFFERENT TYPES OF MACD
def _validate_macd_inputs(data_series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, min_data_lenght: int=1):
    _validate_indicator_inputs(data_series, window=max(slow_period, signal_period), min_data_length=min_data_lenght)
    
    if not isinstance(fast_period, int) or fast_period <= 0:
        raise ValueError("fast_period must be a positive integer.")
    if not isinstance(slow_period, int) or slow_period <= 0:
        raise ValueError("slow_period must be a positive integer.")
    if not isinstance(signal_period, int) or signal_period <= 0:
        raise ValueError("signal_period must be a positive integer.")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period.")

def calculate_macd(data_series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD) indicator.

    The MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security’s price.

    Parameters:
    ----------
    data_series : pandas.Series
        The input data series (e.g., closing prices).
    fast_period : int, optional
        The period for the fast EMA (default is 12).
    slow_period : int, optional
        The period for the slow EMA (default is 26).
    signal_period : int, optional
        The period for the signal line EMA (default is 9).

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing three Series:
        - 'MACD': The MACD Line (Fast EMA - Slow EMA).
        - 'Signal': The Signal Line (EMA of the MACD Line).
        - 'Histogram': The MACD Histogram (MACD Line - Signal Line).
    """
    
    _validate_macd_inputs(data_series, fast_period, slow_period, signal_period, min_data_lenght=1)
    
    ema_fast =  calculate_ema(data_series, fast_period)
    ema_slow =  calculate_ema(data_series, slow_period)
    signal_line = calculate_ema(data_series, signal_period)
    
    macd_line = ema_fast-ema_slow
    macd_histogram = macd_line-signal_line
    return pd.DataFrame({"MACD": macd_line,
                         "Signal": signal_line,
                         "Histogram":macd_histogram})
    
def calculate_wilders_macd(data_series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD) indicator,
    using Wilder's Smoothing for the Signal Line.

    The MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price. In this variant, the
    signal line is smoothed using Wilder's Smoothing instead of a standard EMA.

    Parameters:
    ----------
    data_series : pandas.Series
        The input data series (e.g., closing prices).
    fast_period : int, optional
        The period for the fast EMA (default is 12).
    slow_period : int, optional
        The period for the slow EMA (default is 26).
    signal_period : int, optional
        The period for the signal line (Wilder's Smoothing) (default is 9).

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing three Series:
        - 'wilders_MACD': The MACD Line (Fast EMA - Slow EMA).
        - 'wilders_Signal': The Signal Line (Wilder's Smoothing of the MACD Line).
        - 'wilders_Histogram': The MACD Histogram (MACD Line - Signal Line).
    """
    
    _validate_macd_inputs(data_series, fast_period, slow_period, signal_period, min_data_lenght=1)
    
    ema_fast =  calculate_ema(data_series, fast_period)
    ema_slow =  calculate_ema(data_series, slow_period)
    
    macd_line = ema_fast-ema_slow
    signal_line = wilders_smoothing(macd_line, window=signal_period)
    
    macd_histogram = macd_line-signal_line
    return pd.DataFrame({"wilders_MACD": macd_line,
                         "wilders_Signal": signal_line,
                         "wilders_Histogram":macd_histogram})
    
def calculate_tema_macd(data_series: pd.Series, fast_period: pd.Series=12, slow_period: pd.Series=26, signal_period: int= 9) -> pd.DataFrame:
    """Calculate the Triple Exponential Moving Average (TEMA) MACD indicator.

    This variant of the MACD indicator uses a TEMA for smoothing the signal line,
    aiming to provide a more responsive signal compared to the traditional EMA-smoothed MACD.

    Parameters
    ----------
    data_series : pandas.Series
        The input data series (e.g., closing prices).
    fast_period : int, optional
        The period for the fast Exponential Moving Average (EMA) used in the MACD line (default is 12).
    slow_period : int, optional
        The period for the slow Exponential Moving Average (EMA) used in the MACD line (default is 26).
    signal_period : int, optional
        The period for the Triple Exponential Moving Average (TEMA) used to smooth the signal line (default is 9).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing three Series:
        - 'TEMA_MACD': The MACD Line (Fast EMA - Slow EMA).
        - 'TEMA_Signal': The Signal Line (TEMA of the MACD Line).
        - 'TEMA_Histogram': The MACD Histogram (MACD Line - Signal Line).

    Notes
    -----
    The calculation steps are:
    1.  Calculate the Fast EMA of the `data_series` using `fast_period`.
    2.  Calculate the Slow EMA of the `data_series` using `slow_period`.
    3.  The MACD Line is the difference between the Fast EMA and the Slow EMA:
        :math:`MACD_Line = EMA(data, fast_period) - EMA(data, slow_period)`
    4.  The Signal Line is the Triple Exponential Moving Average (TEMA) of the MACD Line:
        :math:`Signal_Line = TEMA(MACD_Line, signal_period)`
    5.  The Histogram is the difference between the MACD Line and the Signal Line:
        :math:`Histogram = MACD_Line - Signal_Line`
    """
    
    _validate_macd_inputs(data_series, fast_period, slow_period, signal_period, min_data_lenght=1)
    ema_fast =  calculate_ema(data_series, fast_period)
    ema_slow =  calculate_ema(data_series, slow_period)
    
    macd_line = ema_fast-ema_slow
    signal_line = calculate_tema(macd_line, window=signal_period)
    
    macd_histogram = macd_line-signal_line
    return pd.DataFrame({"TEMA_MACD": macd_line,
                         "TEMA_Signal": signal_line,
                         "TEMA_Histogram":macd_histogram})
    
def calculate_hma_macd(data_series: pd.Series, fast_period: pd.Series=12, slow_period: pd.Series=26, signal_period: int= 9) -> pd.DataFrame:
    """Calculate the Hull Moving Average (HMA) MACD indicator.

    This variant of the MACD indicator uses a Hull Moving Average (HMA) for smoothing
    the signal line. The HMA aims to minimize lag, making this MACD variant potentially
    more responsive to recent price changes than traditional MACD.

    Parameters
    ----------
    data_series : pandas.Series
        The input data series (e.g., closing prices).
    fast_period : int, optional
        The period for the fast Exponential Moving Average (EMA) used in the MACD line (default is 12).
    slow_period : int, optional
        The period for the slow Exponential Moving Average (EMA) used in the MACD line (default is 26).
    signal_period : int, optional
        The period for the Hull Moving Average (HMA) used to smooth the signal line (default is 9).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing three Series:
        - 'HMA_MACD': The MACD Line (Fast EMA - Slow EMA).
        - 'HMA_Signal': The Signal Line (HMA of the MACD Line).
        - 'HMA_Histogram': The MACD Histogram (MACD Line - Signal Line).

    Notes
    -----
    The calculation steps are:
    1.  Calculate the Fast EMA of the `data_series` using `fast_period`.
    2.  Calculate the Slow EMA of the `data_series` using `slow_period`.
    3.  The MACD Line is the difference between the Fast EMA and the Slow EMA:
        :math:`MACD_Line = EMA(data, fast_period) - EMA(data, slow_period)`
    4.  The Signal Line is the Hull Moving Average (HMA) of the MACD Line:
        :math:`Signal_Line = HMA(MACD_Line, signal_period)`
    5.  The Histogram is the difference between the MACD Line and the Signal Line:
        :math:`Histogram = MACD_Line - Signal_Line`
    """
    
    _validate_macd_inputs(data_series, fast_period, slow_period, signal_period, min_data_lenght=1)
    ema_fast =  calculate_ema(data_series, fast_period)
    ema_slow =  calculate_ema(data_series, slow_period)
    
    macd_line = ema_fast-ema_slow
    signal_line = calculate_hma(macd_line, window=signal_period)
    
    macd_histogram = macd_line-signal_line
    return pd.DataFrame({"HMA_MACD": macd_line,
                         "HMA_Signal": signal_line,
                         "HMA_Histogram":macd_histogram})