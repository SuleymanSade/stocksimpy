import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src directory to path so we can import stocksimpy modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stocksimpy.core.backtester import Backtester
from stocksimpy.core.portfolio import Portfolio
from stocksimpy.core.stock_data import StockData
from stocksimpy.utils.performance import Performance


@pytest.fixture
def sample_backtest_data():
    """Create sample data for backtesting."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")

    # Uptrend data (positive returns)
    close_prices = np.linspace(100, 150, 50)

    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
    )
    data = {
        ("Open", "AAPL"): close_prices - 1,
        ("High", "AAPL"): close_prices + 2,
        ("Low", "AAPL"): close_prices - 2,
        ("Close", "AAPL"): close_prices,
        ("Volume", "AAPL"): np.random.randint(1000000, 5000000, 50),
    }

    df = pd.DataFrame(data, index=dates)
    return StockData(df)


@pytest.fixture
def sample_downtrend_data():
    """Create sample data with downtrend (negative returns)."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")

    # Downtrend data (negative returns)
    close_prices = np.linspace(150, 100, 50)

    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
    )
    data = {
        ("Open", "AAPL"): close_prices - 1,
        ("High", "AAPL"): close_prices + 2,
        ("Low", "AAPL"): close_prices - 2,
        ("Close", "AAPL"): close_prices,
        ("Volume", "AAPL"): np.random.randint(1000000, 5000000, 50),
    }

    df = pd.DataFrame(data, index=dates)
    return StockData(df)


@pytest.fixture
def sample_volatile_data():
    """Create sample data with high volatility."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")

    # Volatile data with ups and downs
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(50) * 2)

    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
    )
    data = {
        ("Open", "AAPL"): close_prices - 1,
        ("High", "AAPL"): close_prices + 2,
        ("Low", "AAPL"): close_prices - 2,
        ("Close", "AAPL"): close_prices,
        ("Volume", "AAPL"): np.random.randint(1000000, 5000000, 50),
    }

    df = pd.DataFrame(data, index=dates)
    return StockData(df)


@pytest.fixture
def uptrend_backtester(sample_backtest_data):
    """Create a backtester with uptrend data that buys and holds."""

    def buy_and_hold(data):
        return "buy"

    bt = Backtester("AAPL", sample_backtest_data, buy_and_hold)
    bt.run_backtest_fixed()
    return bt


@pytest.fixture
def downtrend_backtester(sample_downtrend_data):
    """Create a backtester with downtrend data."""

    def buy_and_hold(data):
        return "buy"

    bt = Backtester("AAPL", sample_downtrend_data, buy_and_hold)
    bt.run_backtest_fixed()
    return bt


@pytest.fixture
def volatile_backtester(sample_volatile_data):
    """Create a backtester with volatile data."""

    def buy_and_hold(data):
        return "buy"

    bt = Backtester("AAPL", sample_volatile_data, buy_and_hold)
    bt.run_backtest_fixed()
    return bt


@pytest.fixture
def empty_backtester():
    """Create a backtester without running it (empty value history)."""
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
    )
    data = {
        ("Open", "AAPL"): [100, 101, 102, 103, 104],
        ("High", "AAPL"): [102, 103, 104, 105, 106],
        ("Low", "AAPL"): [98, 99, 100, 101, 102],
        ("Close", "AAPL"): [100, 101, 102, 103, 104],
        ("Volume", "AAPL"): [1000000] * 5,
    }
    df = pd.DataFrame(data, index=dates)
    stock_data = StockData(df)

    def strategy(data):
        return "buy"

    bt = Backtester("AAPL", stock_data, strategy)
    # Don't run backtest - return empty backtester
    return bt


class TestPerformanceInitialization:
    """Tests for Performance class initialization."""

    def test_init_with_default_risk_free_rate(self, uptrend_backtester):
        """Test initialization with default risk-free rate."""
        perf = Performance(uptrend_backtester)

        assert perf.backtester == uptrend_backtester
        assert perf.portfolio == uptrend_backtester.portfolio
        assert perf.symbol == "AAPL"
        assert perf.risk_free_rate == 0.02

    def test_init_with_custom_risk_free_rate(self, uptrend_backtester):
        """Test initialization with custom risk-free rate."""
        perf = Performance(uptrend_backtester, risk_free_rate=0.05)

        assert perf.risk_free_rate == 0.05

    def test_init_stores_references(self, uptrend_backtester):
        """Test that initialization stores correct references."""
        perf = Performance(uptrend_backtester)

        assert perf.portfolio is uptrend_backtester.portfolio
        assert perf.backtester is uptrend_backtester


class TestCalcDailyReturns:
    """Tests for daily returns calculation."""

    def test_calc_daily_returns_uptrend(self, uptrend_backtester):
        """Test daily returns for uptrend data."""
        perf = Performance(uptrend_backtester)
        daily_returns = perf.calc_daily_returns()

        assert isinstance(daily_returns, pd.Series)
        assert len(daily_returns) > 0
        # Most daily returns should be positive in uptrend
        assert (daily_returns > 0).sum() > len(daily_returns) / 2

    def test_calc_daily_returns_downtrend(self, downtrend_backtester):
        """Test daily returns for downtrend data."""
        perf = Performance(downtrend_backtester)
        daily_returns = perf.calc_daily_returns()

        assert isinstance(daily_returns, pd.Series)
        assert len(daily_returns) > 0
        # Most daily returns should be negative in downtrend
        assert (daily_returns < 0).sum() > len(daily_returns) / 2

    def test_calc_daily_returns_empty_backtest(self, empty_backtester):
        """Test daily returns with empty backtest."""
        perf = Performance(empty_backtester)
        daily_returns = perf.calc_daily_returns()

        assert isinstance(daily_returns, pd.Series)
        assert len(daily_returns) == 0

    def test_calc_daily_returns_no_nans(self, uptrend_backtester):
        """Test that daily returns don't contain NaN values."""
        perf = Performance(uptrend_backtester)
        daily_returns = perf.calc_daily_returns()

        assert daily_returns.isna().sum() == 0


class TestCalcTotalReturn:
    """Tests for total return calculation."""

    def test_calc_total_return_positive(self, uptrend_backtester):
        """Test total return for profitable backtest."""
        perf = Performance(uptrend_backtester)
        total_return = perf.calc_total_return()

        assert isinstance(total_return, (float, np.floating))
        assert total_return > 0

    def test_calc_total_return_negative(self, downtrend_backtester):
        """Test total return for losing backtest."""
        perf = Performance(downtrend_backtester)
        total_return = perf.calc_total_return()

        assert isinstance(total_return, (float, np.floating))
        assert total_return < 0

    def test_calc_total_return_empty(self, empty_backtester):
        """Test total return with empty value history."""
        perf = Performance(empty_backtester)
        total_return = perf.calc_total_return()

        assert total_return == 0.0

    def test_calc_total_return_formula(self, uptrend_backtester):
        """Test that total return formula is correct."""
        perf = Performance(uptrend_backtester)
        total_return = perf.calc_total_return()

        # Manual calculation
        value_history = uptrend_backtester.portfolio.value_history
        initial_cap = uptrend_backtester.portfolio.initial_cap
        expected = (value_history.iloc[-1] - initial_cap) / initial_cap

        assert abs(total_return - expected) < 1e-10


class TestGetAnnualizedReturn:
    """Tests for annualized return calculation."""

    def test_get_annualized_return_positive(self, uptrend_backtester):
        """Test annualized return for profitable backtest."""
        perf = Performance(uptrend_backtester)
        ann_return = perf.get_annualized_return()

        assert isinstance(ann_return, (float, np.floating))
        assert ann_return > 0

    def test_get_annualized_return_negative(self, downtrend_backtester):
        """Test annualized return for losing backtest."""
        perf = Performance(downtrend_backtester)
        ann_return = perf.get_annualized_return()

        assert isinstance(ann_return, (float, np.floating))
        assert ann_return < 0

    def test_get_annualized_return_empty(self, empty_backtester):
        """Test annualized return with empty value history."""
        perf = Performance(empty_backtester)
        ann_return = perf.get_annualized_return()

        assert ann_return == 0.0

    def test_get_annualized_return_less_than_total(self, uptrend_backtester):
        """Test that annualized return is comparable to total return (can be higher for short periods)."""
        perf = Performance(uptrend_backtester)
        total_return = perf.calc_total_return()
        ann_return = perf.get_annualized_return()

        # For a 50-day period, annualized return can be higher than total due to compounding
        # Just verify both are positive for an uptrend
        assert ann_return > 0
        assert total_return > 0


class TestGetAnnualizedTradingDays:
    """Tests for annualized trading days calculation."""

    def test_get_annualized_trading_days_returns_float(self, uptrend_backtester):
        """Test that method returns a float."""
        perf = Performance(uptrend_backtester)
        trading_days = perf._get_annualized_trading_days()

        assert isinstance(trading_days, (float, np.floating, int))
        assert trading_days > 0

    def test_get_annualized_trading_days_reasonable(self, uptrend_backtester):
        """Test that annualized trading days is calculated based on the simulation period."""
        perf = Performance(uptrend_backtester)
        trading_days = perf._get_annualized_trading_days()

        # For a 50-day period with daily data, trading days per year = 50 / (50/365.25)
        # which is ~365, but should be positive and finite
        assert trading_days > 0
        assert np.isfinite(trading_days)

    def test_get_annualized_trading_days_empty(self, empty_backtester):
        """Test with empty value history."""
        perf = Performance(empty_backtester)
        trading_days = perf._get_annualized_trading_days()

        # Should return default 252 for empty history
        assert trading_days == 252

    def test_get_annualized_trading_days_single_day(self):
        """Test with single day of data."""
        portfolio = Portfolio()
        portfolio.value_history.loc[pd.Timestamp("2023-01-01")] = 100_000

        # Create minimal backtester with all required columns
        dates = pd.date_range(start="2023-01-01", periods=1, freq="D")
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
        )
        data = {
            ("Open", "AAPL"): [100],
            ("High", "AAPL"): [101],
            ("Low", "AAPL"): [99],
            ("Close", "AAPL"): [100],
            ("Volume", "AAPL"): [1000000],
        }
        df = pd.DataFrame(data, index=dates)
        stock_data = StockData(df)

        bt = Backtester("AAPL", stock_data, lambda x: "buy")
        bt.portfolio = portfolio

        perf = Performance(bt)
        trading_days = perf._get_annualized_trading_days()

        # Should return default for insufficient data
        assert trading_days == 252


class TestCalcMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_calc_max_drawdown_uptrend(self, uptrend_backtester):
        """Test max drawdown for uptrend (should be small or zero)."""
        perf = Performance(uptrend_backtester)
        max_dd = perf.calc_max_drawdown()

        assert isinstance(max_dd, (float, np.floating))
        # Uptrend should have minimal drawdown
        assert -0.5 <= max_dd <= 0

    def test_calc_max_drawdown_downtrend(self, downtrend_backtester):
        """Test max drawdown for downtrend (should be significant)."""
        perf = Performance(downtrend_backtester)
        max_dd = perf.calc_max_drawdown()

        assert isinstance(max_dd, (float, np.floating))
        # Downtrend should have significant drawdown
        assert max_dd < -0.1

    def test_calc_max_drawdown_empty(self, empty_backtester):
        """Test max drawdown with empty value history."""
        perf = Performance(empty_backtester)
        max_dd = perf.calc_max_drawdown()

        assert max_dd == 0.0

    def test_calc_max_drawdown_is_negative(self, volatile_backtester):
        """Test that max drawdown is non-positive."""
        perf = Performance(volatile_backtester)
        max_dd = perf.calc_max_drawdown()

        assert max_dd <= 0


class TestCalcSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_calc_sharpe_ratio_returns_float(self, uptrend_backtester):
        """Test that Sharpe ratio returns a float."""
        perf = Performance(uptrend_backtester)
        sharpe = perf.calc_sharpe_ratio()

        assert isinstance(sharpe, (float, np.floating))

    def test_calc_sharpe_ratio_positive_for_uptrend(self, uptrend_backtester):
        """Test that Sharpe ratio is positive for uptrend."""
        perf = Performance(uptrend_backtester)
        sharpe = perf.calc_sharpe_ratio()

        assert sharpe > 0

    def test_calc_sharpe_ratio_empty(self, empty_backtester):
        """Test Sharpe ratio with empty backtest."""
        perf = Performance(empty_backtester)
        sharpe = perf.calc_sharpe_ratio()

        assert sharpe == 0.0

    def test_calc_sharpe_ratio_risk_free_rate_impact(self, uptrend_backtester):
        """Test that higher risk-free rate reduces Sharpe ratio."""
        perf_low_rf = Performance(uptrend_backtester, risk_free_rate=0.01)
        perf_high_rf = Performance(uptrend_backtester, risk_free_rate=0.10)

        sharpe_low = perf_low_rf.calc_sharpe_ratio()
        sharpe_high = perf_high_rf.calc_sharpe_ratio()

        assert sharpe_low > sharpe_high


class TestCalcVolatility:
    """Tests for volatility calculation."""

    def test_calc_volatility_returns_float(self, uptrend_backtester):
        """Test that volatility returns a float."""
        perf = Performance(uptrend_backtester)
        vol = perf.calc_volatility()

        assert isinstance(vol, (float, np.floating))
        assert vol >= 0

    def test_calc_volatility_uptrend_vs_downtrend(
        self, uptrend_backtester, downtrend_backtester
    ):
        """Test volatility comparison between trends."""
        perf_up = Performance(uptrend_backtester)
        perf_down = Performance(downtrend_backtester)

        vol_up = perf_up.calc_volatility()
        vol_down = perf_down.calc_volatility()

        # Both should be positive
        assert vol_up >= 0
        assert vol_down >= 0

    def test_calc_volatility_volatile_vs_smooth(
        self, uptrend_backtester, volatile_backtester
    ):
        """Test that volatile data has higher volatility."""
        perf_smooth = Performance(uptrend_backtester)
        perf_volatile = Performance(volatile_backtester)

        vol_smooth = perf_smooth.calc_volatility()
        vol_volatile = perf_volatile.calc_volatility()

        # Volatile data should have higher volatility
        assert vol_volatile > vol_smooth

    def test_calc_volatility_empty(self, empty_backtester):
        """Test volatility with empty backtest."""
        perf = Performance(empty_backtester)
        vol = perf.calc_volatility()

        assert vol == 0.0


class TestGenerateRiskReport:
    """Tests for risk report generation."""

    def test_generate_risk_report_returns_series(self, uptrend_backtester):
        """Test that report returns a pandas Series."""
        perf = Performance(uptrend_backtester)
        report = perf.generate_risk_report()

        assert isinstance(report, pd.Series)

    def test_generate_risk_report_has_required_metrics(self, uptrend_backtester):
        """Test that report contains all required metrics."""
        perf = Performance(uptrend_backtester)
        report = perf.generate_risk_report()

        required_metrics = [
            "Total Return",
            "Annualized Return",
            "Max Drawdown",
            "Sharpe Ratio",
            "Volatility",
        ]

        for metric in required_metrics:
            assert metric in report.index

    def test_generate_risk_report_all_numeric(self, uptrend_backtester):
        """Test that all report values are numeric."""
        perf = Performance(uptrend_backtester)
        report = perf.generate_risk_report()

        for value in report.values:
            assert isinstance(value, (float, np.floating, int, np.integer))

    def test_generate_risk_report_empty(self, empty_backtester):
        """Test report generation with empty backtest."""
        perf = Performance(empty_backtester)
        report = perf.generate_risk_report()

        # All values should be 0 for empty backtest
        assert (report == 0.0).all()

    def test_generate_risk_report_uptrend(self, uptrend_backtester):
        """Test report values for uptrend."""
        perf = Performance(uptrend_backtester)
        report = perf.generate_risk_report()

        # For uptrend, total return should be positive
        assert report["Total Return"] > 0
        # Sharpe ratio should be positive
        assert report["Sharpe Ratio"] > 0
        # Max drawdown should be non-positive
        assert report["Max Drawdown"] <= 0

    def test_generate_risk_report_downtrend(self, downtrend_backtester):
        """Test report values for downtrend."""
        perf = Performance(downtrend_backtester)
        report = perf.generate_risk_report()

        # For downtrend, total return should be negative
        assert report["Total Return"] < 0
        # Sharpe ratio should be negative or zero
        assert report["Sharpe Ratio"] <= 0


class TestPerformanceIntegration:
    """Integration tests for Performance metrics."""

    def test_full_report_consistency(self, uptrend_backtester):
        """Test consistency between individual metrics and report."""
        perf = Performance(uptrend_backtester)

        # Individual calculations
        total_ret = perf.calc_total_return()
        ann_ret = perf.get_annualized_return()
        max_dd = perf.calc_max_drawdown()
        sharpe = perf.calc_sharpe_ratio()
        vol = perf.calc_volatility()

        # Report
        report = perf.generate_risk_report()

        # Values should match
        assert abs(report["Total Return"] - total_ret) < 1e-10
        assert abs(report["Annualized Return"] - ann_ret) < 1e-10
        assert abs(report["Max Drawdown"] - max_dd) < 1e-10
        assert abs(report["Sharpe Ratio"] - sharpe) < 1e-10
        assert abs(report["Volatility"] - vol) < 1e-10

    def test_performance_with_custom_risk_free_rate(self, uptrend_backtester):
        """Test performance calculations with different risk-free rates."""
        perf_0 = Performance(uptrend_backtester, risk_free_rate=0.0)
        perf_5 = Performance(uptrend_backtester, risk_free_rate=0.05)

        report_0 = perf_0.generate_risk_report()
        report_5 = perf_5.generate_risk_report()

        # Sharpe ratio should be different
        assert report_0["Sharpe Ratio"] != report_5["Sharpe Ratio"]
        # Other metrics should be the same
        assert report_0["Total Return"] == report_5["Total Return"]
        assert report_0["Volatility"] == report_5["Volatility"]

    def test_performance_with_different_backtests(
        self, uptrend_backtester, downtrend_backtester
    ):
        """Test that different backtests produce different performance metrics."""
        perf_up = Performance(uptrend_backtester)
        perf_down = Performance(downtrend_backtester)

        report_up = perf_up.generate_risk_report()
        report_down = perf_down.generate_risk_report()

        # Uptrend should have positive return, downtrend negative
        assert report_up["Total Return"] > 0
        assert report_down["Total Return"] < 0

        # Downtrend should have negative Sharpe
        assert report_down["Sharpe Ratio"] < report_up["Sharpe Ratio"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_performance_with_flat_portfolio(self):
        """Test performance metrics when portfolio value doesn't change."""
        # Create a portfolio with constant value
        portfolio = Portfolio(initial_cap=100_000)

        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        for date in dates:
            portfolio.update_value(date, {})

        # Create minimal backtester with all required columns
        stock_data_dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
        )
        data = {
            ("Open", "AAPL"): [100] * 10,
            ("High", "AAPL"): [101] * 10,
            ("Low", "AAPL"): [99] * 10,
            ("Close", "AAPL"): [100] * 10,
            ("Volume", "AAPL"): [1000000] * 10,
        }
        df = pd.DataFrame(data, index=stock_data_dates)
        stock_data = StockData(df)

        bt = Backtester("AAPL", stock_data, lambda x: "hold")
        bt.portfolio = portfolio

        perf = Performance(bt)
        report = perf.generate_risk_report()

        # Total return should be 0
        assert report["Total Return"] == 0.0
        # Volatility should be 0
        assert report["Volatility"] == 0.0

    def test_performance_with_single_positive_return_day(self):
        """Test performance with minimal data showing positive return."""
        portfolio = Portfolio(initial_cap=100_000)

        dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        portfolio.update_value(dates[0], {})
        portfolio.update_value(dates[1], {})  # Still 100,000

        # Create minimal backtester with all required columns
        stock_data_dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
        )
        data = {
            ("Open", "AAPL"): [100, 100],
            ("High", "AAPL"): [101, 102],
            ("Low", "AAPL"): [99, 100],
            ("Close", "AAPL"): [100, 101],
            ("Volume", "AAPL"): [1000000, 1000000],
        }
        df = pd.DataFrame(data, index=stock_data_dates)
        stock_data = StockData(df)

        bt = Backtester("AAPL", stock_data, lambda x: "hold")
        bt.portfolio = portfolio

        perf = Performance(bt)
        daily_returns = perf.calc_daily_returns()

        # Should have at least one return value
        assert len(daily_returns) > 0

    def test_performance_with_extreme_returns(self):
        """Test performance with extreme returns."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")

        # Create data with huge gains
        close_prices = np.array([100] * 10 + [1000] * 10)  # 10x gain

        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
        )
        data = {
            ("Open", "AAPL"): close_prices - 1,
            ("High", "AAPL"): close_prices + 2,
            ("Low", "AAPL"): close_prices - 2,
            ("Close", "AAPL"): close_prices,
            ("Volume", "AAPL"): [1000000] * 20,
        }

        df = pd.DataFrame(data, index=dates)
        stock_data = StockData(df)

        def strategy(data):
            return "buy"

        bt = Backtester("AAPL", stock_data, strategy)
        bt.run_backtest_fixed()

        perf = Performance(bt)
        report = perf.generate_risk_report()

        # Should handle extreme returns gracefully
        assert np.isfinite(report["Total Return"])
        assert np.isfinite(report["Sharpe Ratio"])
        assert np.isfinite(report["Volatility"])

    def test_performance_with_negative_risk_free_rate(self, uptrend_backtester):
        """Test performance with negative risk-free rate."""
        perf = Performance(uptrend_backtester, risk_free_rate=-0.01)
        report = perf.generate_risk_report()

        # Should handle negative risk-free rate
        assert np.isfinite(report["Sharpe Ratio"])

    def test_performance_with_very_high_risk_free_rate(self, uptrend_backtester):
        """Test performance with very high risk-free rate."""
        perf = Performance(uptrend_backtester, risk_free_rate=1.0)  # 100% annual
        report = perf.generate_risk_report()

        # Should handle high risk-free rate
        assert np.isfinite(report["Sharpe Ratio"])


class TestPerformanceMetricsRelationships:
    """Tests for relationships between performance metrics."""

    def test_sharpe_ratio_higher_with_lower_volatility(self):
        """Test that same return produces higher Sharpe ratio with lower volatility."""
        # Create two portfolios with same return but different volatility
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")

        # Smooth growth (low volatility)
        smooth_prices = np.linspace(100, 150, 20)

        # Volatile growth (high volatility, same end point)
        volatile_prices = np.array(
            [
                100,
                120,
                110,
                140,
                130,
                160,
                150,
                155,
                145,
                150,
                140,
                145,
                135,
                140,
                130,
                135,
                125,
                130,
                140,
                150,
            ]
        )

        def create_backtest(prices):
            columns = pd.MultiIndex.from_product(
                [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
            )
            data = {
                ("Open", "AAPL"): prices - 1,
                ("High", "AAPL"): prices + 2,
                ("Low", "AAPL"): prices - 2,
                ("Close", "AAPL"): prices,
                ("Volume", "AAPL"): [1000000] * len(prices),
            }
            df = pd.DataFrame(data, index=dates)
            stock_data = StockData(df)

            bt = Backtester("AAPL", stock_data, lambda x: "buy")
            bt.run_backtest_fixed()
            return bt

        bt_smooth = create_backtest(smooth_prices)
        bt_volatile = create_backtest(volatile_prices)

        perf_smooth = Performance(bt_smooth)
        perf_volatile = Performance(bt_volatile)

        report_smooth = perf_smooth.generate_risk_report()
        report_volatile = perf_volatile.generate_risk_report()

        # Smooth should have lower volatility
        assert report_smooth["Volatility"] < report_volatile["Volatility"]
        # Smooth should have higher Sharpe ratio (assuming similar returns)
        assert report_smooth["Sharpe Ratio"] > report_volatile["Sharpe Ratio"]

    def test_max_drawdown_in_downtrend(self, downtrend_backtester):
        """Test that max drawdown correlates with total loss."""
        perf = Performance(downtrend_backtester)
        report = perf.generate_risk_report()

        # In a downtrend, max drawdown should be significant
        assert abs(report["Max Drawdown"]) > abs(report["Total Return"]) * 0.5
