import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src directory to path so we can import stocksimpy modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stocksimpy.core.portfolio import Portfolio


@pytest.fixture
def portfolio():
    """Create a fresh portfolio with default initial capital."""
    return Portfolio()


@pytest.fixture
def portfolio_custom_capital():
    """Create a portfolio with custom initial capital."""
    return Portfolio(initial_cap=50_000)


@pytest.fixture
def sample_date():
    """Create a sample timestamp."""
    return pd.Timestamp("2023-01-01")


class TestPortfolioInitialization:
    """Tests for Portfolio initialization."""

    def test_init_default_capital(self):
        """Test initialization with default capital."""
        p = Portfolio()
        assert p.initial_cap == 100_000
        assert p.cash == 100_000
        assert len(p.holdings) == 0
        assert len(p.trade_log) == 0
        assert len(p.value_history) == 0

    def test_init_custom_capital(self):
        """Test initialization with custom capital."""
        p = Portfolio(initial_cap=50_000)
        assert p.initial_cap == 50_000
        assert p.cash == 50_000

    def test_init_zero_capital(self):
        """Test initialization with zero capital."""
        p = Portfolio(initial_cap=0)
        assert p.initial_cap == 0
        assert p.cash == 0

    def test_init_negative_capital(self):
        """Test initialization with negative capital (should allow)."""
        p = Portfolio(initial_cap=-10_000)
        assert p.initial_cap == -10_000
        assert p.cash == -10_000

    def test_holdings_defaults_to_zero(self, portfolio):
        """Test that accessing a non-existent holding returns 0."""
        assert portfolio.holdings["AAPL"] == 0


class TestExecTradeBuy:
    """Tests for buying trades via exec_trade."""

    def test_buy_single_share(self, portfolio, sample_date):
        """Test buying a single share."""
        initial_cash = portfolio.cash
        portfolio.exec_trade("AAPL", "buy", price=100, shares=1, date=sample_date)

        assert portfolio.holdings["AAPL"] == 1
        assert portfolio.cash == initial_cash - 100
        assert len(portfolio.trade_log) == 1
        assert portfolio.trade_log.iloc[0]["Type"] == "buy"

    def test_buy_multiple_shares(self, portfolio, sample_date):
        """Test buying multiple shares."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)

        assert portfolio.holdings["AAPL"] == 10
        assert portfolio.cash == 100_000 - 1_000

    def test_buy_with_transaction_fee(self, portfolio, sample_date):
        """Test buying with transaction fee."""
        portfolio.exec_trade(
            "AAPL", "buy", price=100, shares=10, date=sample_date, transaction_fee=5
        )

        expected_cost = 100 * 10 + 5  # 1005
        assert portfolio.holdings["AAPL"] == 10
        assert portfolio.cash == 100_000 - expected_cost

    def test_buy_insufficient_cash(self, portfolio, sample_date):
        """Test buying with insufficient cash reduces shares."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=1000, date=sample_date)

        # Can only afford 1000 shares at 100 each = 100,000
        assert portfolio.holdings["AAPL"] == 1000
        assert portfolio.cash == 0

    def test_buy_insufficient_cash_adjusted_shares(self, portfolio, sample_date):
        """Test that shares are adjusted when cash is insufficient."""
        # With 100,000 capital and 150/share, can only afford 666 shares
        portfolio.exec_trade("AAPL", "buy", price=150, shares=1000, date=sample_date)

        assert portfolio.holdings["AAPL"] == 666  # (100000 / 150) = 666.666... -> 666
        assert portfolio.cash == 100_000 - (666 * 150)

    def test_buy_fee_exceeds_cash(self, portfolio, sample_date):
        """Test that buy is skipped if fee exceeds available cash."""
        portfolio.cash = 5
        portfolio.exec_trade(
            "AAPL", "buy", price=100, shares=1, date=sample_date, transaction_fee=10
        )

        # Trade should be skipped
        assert portfolio.holdings["AAPL"] == 0
        assert portfolio.cash == 5
        assert len(portfolio.trade_log) == 0

    def test_buy_case_insensitive(self, portfolio, sample_date):
        """Test that buy command is case-insensitive."""
        portfolio.exec_trade("AAPL", "BUY", price=100, shares=1, date=sample_date)
        assert portfolio.holdings["AAPL"] == 1

    def test_buy_shares_converted_to_int(self, portfolio, sample_date):
        """Test that fractional shares are converted to int."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=5.9, date=sample_date)

        # 5.9 should be converted to 5
        assert portfolio.holdings["AAPL"] == 5
        assert portfolio.cash == 100_000 - 500

    def test_buy_multiple_different_stocks(self, portfolio, sample_date):
        """Test buying multiple different stocks."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)
        portfolio.exec_trade("MSFT", "buy", price=200, shares=5, date=sample_date)

        assert portfolio.holdings["AAPL"] == 10
        assert portfolio.holdings["MSFT"] == 5
        assert portfolio.cash == 100_000 - (1_000 + 1_000)
        assert len(portfolio.trade_log) == 2


class TestExecTradeSell:
    """Tests for selling trades via exec_trade."""

    def test_sell_all_shares(self, portfolio, sample_date):
        """Test selling all holdings of a stock."""
        # First buy
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)
        initial_cash_after_buy = portfolio.cash

        # Then sell
        portfolio.exec_trade("AAPL", "sell", price=110, shares=10, date=sample_date)

        assert portfolio.holdings["AAPL"] == 0
        assert portfolio.cash == initial_cash_after_buy + (10 * 110)

    def test_sell_partial_shares(self, portfolio, sample_date):
        """Test selling partial holdings."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)
        portfolio.exec_trade("AAPL", "sell", price=110, shares=5, date=sample_date)

        assert portfolio.holdings["AAPL"] == 5

    def test_sell_with_transaction_fee(self, portfolio, sample_date):
        """Test selling with transaction fee."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)
        initial_cash = portfolio.cash

        portfolio.exec_trade(
            "AAPL", "sell", price=110, shares=10, date=sample_date, transaction_fee=5
        )

        expected_revenue = 10 * 110 - 5
        assert portfolio.cash == initial_cash + expected_revenue

    def test_sell_more_than_held(self, portfolio, sample_date):
        """Test selling more shares than held (sells all available)."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)
        initial_cash = portfolio.cash

        portfolio.exec_trade("AAPL", "sell", price=110, shares=100, date=sample_date)

        # Should only sell the 10 shares we own
        assert portfolio.holdings["AAPL"] == 0
        assert portfolio.cash == initial_cash + (10 * 110)

    def test_sell_non_existent_stock(self, portfolio, sample_date):
        """Test selling a stock we don't own."""
        initial_cash = portfolio.cash
        portfolio.exec_trade("AAPL", "sell", price=100, shares=10, date=sample_date)

        assert portfolio.holdings["AAPL"] == 0
        assert portfolio.cash == initial_cash

    def test_sell_zero_shares(self, portfolio, sample_date):
        """Test selling zero shares."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)
        initial_cash = portfolio.cash

        portfolio.exec_trade("AAPL", "sell", price=110, shares=0, date=sample_date)

        assert portfolio.holdings["AAPL"] == 10
        assert portfolio.cash == initial_cash

    def test_sell_case_insensitive(self, portfolio, sample_date):
        """Test that sell command is case-insensitive."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)
        portfolio.exec_trade("AAPL", "SELL", price=110, shares=5, date=sample_date)

        assert portfolio.holdings["AAPL"] == 5


class TestExecTradeValidation:
    """Tests for input validation in exec_trade."""

    def test_invalid_trade_type(self, portfolio, sample_date):
        """Test that invalid trade type raises ValueError."""
        with pytest.raises(
            ValueError, match="trade_type has to be one of the following"
        ):
            portfolio.exec_trade("AAPL", "hold", price=100, shares=1, date=sample_date)

    def test_negative_price(self, portfolio, sample_date):
        """Test that negative price raises ValueError."""
        with pytest.raises(ValueError, match="price cannot be 0 or less than 0"):
            portfolio.exec_trade("AAPL", "buy", price=-100, shares=1, date=sample_date)

    def test_zero_price(self, portfolio, sample_date):
        """Test that zero price raises ValueError."""
        with pytest.raises(ValueError, match="price cannot be 0 or less than 0"):
            portfolio.exec_trade("AAPL", "buy", price=0, shares=1, date=sample_date)

    def test_negative_shares(self, portfolio, sample_date):
        """Test that negative shares raise ValueError."""
        with pytest.raises(ValueError, match="shares cannot be negative"):
            portfolio.exec_trade("AAPL", "buy", price=100, shares=-5, date=sample_date)

    def test_negative_transaction_fee(self, portfolio, sample_date):
        """Test that negative transaction fee raises ValueError."""
        with pytest.raises(ValueError, match="transaction_fee cannot be negative"):
            portfolio.exec_trade(
                "AAPL", "buy", price=100, shares=1, date=sample_date, transaction_fee=-5
            )


class TestTradeLog:
    """Tests for trade logging."""

    def test_trade_log_records_buy(self, portfolio, sample_date):
        """Test that buy trades are recorded in trade_log."""
        portfolio.exec_trade(
            "AAPL", "buy", price=100, shares=10, date=sample_date, transaction_fee=2
        )

        assert len(portfolio.trade_log) == 1
        trade = portfolio.trade_log.iloc[0]
        assert trade["Symbol"] == "AAPL"
        assert trade["Type"] == "buy"
        assert trade["Price"] == 100
        assert trade["Shares"] == 10
        assert trade["TransactionFee"] == 2
        assert trade["TotalAmount"] == 1002

    def test_trade_log_records_sell(self, portfolio, sample_date):
        """Test that sell trades are recorded in trade_log."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)
        portfolio.exec_trade(
            "AAPL", "sell", price=110, shares=5, date=sample_date, transaction_fee=1
        )

        assert len(portfolio.trade_log) == 2
        sell_trade = portfolio.trade_log.iloc[1]
        assert sell_trade["Type"] == "sell"
        assert sell_trade["Shares"] == 5
        assert sell_trade["TotalAmount"] == (5 * 110 - 1)

    def test_trade_log_date_recorded(self, portfolio):
        """Test that trade date is correctly recorded."""
        date1 = pd.Timestamp("2023-01-01")
        date2 = pd.Timestamp("2023-01-02")

        portfolio.exec_trade("AAPL", "buy", price=100, shares=1, date=date1)
        portfolio.exec_trade("MSFT", "buy", price=200, shares=1, date=date2)

        assert portfolio.trade_log.iloc[0]["Date"] == date1
        assert portfolio.trade_log.iloc[1]["Date"] == date2

    def test_trade_log_dtypes(self, portfolio, sample_date):
        """Test that trade_log has correct data types."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)

        assert pd.api.types.is_datetime64_any_dtype(portfolio.trade_log["Date"])
        assert pd.api.types.is_object_dtype(portfolio.trade_log["Symbol"])
        assert pd.api.types.is_object_dtype(portfolio.trade_log["Type"])
        assert pd.api.types.is_numeric_dtype(portfolio.trade_log["Price"])
        assert pd.api.types.is_numeric_dtype(portfolio.trade_log["Shares"])


class TestUpdateValue:
    """Tests for portfolio value tracking."""

    def test_update_value_cash_only(self, portfolio, sample_date):
        """Test updating value with only cash (no holdings)."""
        portfolio.update_value(sample_date, {})

        assert sample_date in portfolio.value_history.index
        assert portfolio.value_history[sample_date] == 100_000

    def test_update_value_with_holdings(self, portfolio, sample_date):
        """Test updating value with cash and holdings."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)

        # Now portfolio has 99,000 cash + 10 AAPL shares
        portfolio.update_value(sample_date, {"AAPL": 120})

        expected_value = 99_000 + (10 * 120)
        assert portfolio.value_history[sample_date] == expected_value

    def test_update_value_multiple_holdings(self, portfolio, sample_date):
        """Test updating value with multiple holdings."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)
        portfolio.exec_trade("MSFT", "buy", price=200, shares=5, date=sample_date)

        # Cash: 100,000 - 1,000 - 1,000 = 98,000
        # Holdings: 10*120 + 5*250 = 1,200 + 1,250 = 2,450
        portfolio.update_value(sample_date, {"AAPL": 120, "MSFT": 250})

        expected_value = 98_000 + 1_200 + 1_250
        assert portfolio.value_history[sample_date] == expected_value

    def test_update_value_missing_price(self, portfolio, sample_date):
        """Test updating value when price for holding is missing."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)

        # If AAPL price is not provided, it shouldn't be counted
        portfolio.update_value(sample_date, {})

        expected_value = 99_000  # Only cash
        assert portfolio.value_history[sample_date] == expected_value

    def test_update_value_multiple_dates(self, portfolio):
        """Test updating value across multiple dates."""
        date1 = pd.Timestamp("2023-01-01")
        date2 = pd.Timestamp("2023-01-02")

        portfolio.update_value(date1, {})
        portfolio.update_value(date2, {})

        assert len(portfolio.value_history) == 2
        assert portfolio.value_history[date1] == 100_000
        assert portfolio.value_history[date2] == 100_000


class TestPortfolioReturns:
    """Tests for portfolio return calculations."""

    def test_returns_no_trades(self, portfolio):
        """Test returns when no trades have occurred."""
        # Value remains at initial capital
        date = pd.Timestamp("2023-01-01")
        portfolio.update_value(date, {})

        # Returns should be 0%
        returns = (
            portfolio.value_history[date] - portfolio.initial_cap
        ) / portfolio.initial_cap
        assert returns == 0

    def test_returns_profitable_trade(self, portfolio):
        """Test returns on profitable trades."""
        date1 = pd.Timestamp("2023-01-01")
        date2 = pd.Timestamp("2023-01-02")

        # Buy at 100
        portfolio.exec_trade("AAPL", "buy", price=100, shares=100, date=date1)
        portfolio.update_value(date1, {"AAPL": 100})

        # Sell at 110
        portfolio.exec_trade("AAPL", "sell", price=110, shares=100, date=date2)
        portfolio.update_value(date2, {})

        # Initial: 100,000, Final: 100,000 - 10,000 + 11,000 = 101,000
        expected_returns = (101_000 - 100_000) / 100_000
        actual_returns = (
            portfolio.value_history[date2] - portfolio.initial_cap
        ) / portfolio.initial_cap
        assert actual_returns == expected_returns

    def test_returns_loss(self, portfolio):
        """Test returns on losing trades."""
        date1 = pd.Timestamp("2023-01-01")
        date2 = pd.Timestamp("2023-01-02")

        # Buy at 100
        portfolio.exec_trade("AAPL", "buy", price=100, shares=100, date=date1)
        portfolio.update_value(date1, {"AAPL": 100})

        # Sell at 90
        portfolio.exec_trade("AAPL", "sell", price=90, shares=100, date=date2)
        portfolio.update_value(date2, {})

        # Initial: 100,000, Final: 100,000 - 10,000 + 9,000 = 99,000
        actual_value = portfolio.value_history[date2]
        assert actual_value == 99_000


class TestPortfolioIntegration:
    """Integration tests for portfolio operations."""

    def test_multiple_buy_sell_cycle(self, portfolio):
        """Test a complete buy-sell cycle with multiple stocks."""
        dates = [pd.Timestamp(f"2023-01-{i:02d}") for i in range(1, 11)]

        # Day 1: Buy AAPL and MSFT
        portfolio.exec_trade("AAPL", "buy", price=100, shares=50, date=dates[0])
        portfolio.exec_trade("MSFT", "buy", price=200, shares=25, date=dates[0])

        assert portfolio.holdings["AAPL"] == 50
        assert portfolio.holdings["MSFT"] == 25

        # Day 5: Sell half of AAPL
        portfolio.exec_trade("AAPL", "sell", price=110, shares=25, date=dates[4])
        assert portfolio.holdings["AAPL"] == 25

        # Day 10: Sell all MSFT
        portfolio.exec_trade("MSFT", "sell", price=210, shares=25, date=dates[9])
        assert portfolio.holdings["MSFT"] == 0

        assert len(portfolio.trade_log) == 4

    def test_portfolio_with_fees(self, portfolio):
        """Test portfolio accounting with transaction fees."""
        date = pd.Timestamp("2023-01-01")

        # Buy with fee
        portfolio.exec_trade(
            "AAPL", "buy", price=100, shares=100, date=date, transaction_fee=50
        )

        # Cash should be: 100,000 - (100*100 + 50) = 89,950
        assert portfolio.cash == 89_950

        # Sell with fee
        portfolio.exec_trade(
            "AAPL", "sell", price=110, shares=100, date=date, transaction_fee=50
        )

        # Cash should be: 89,950 + (100*110 - 50) = 89,950 + 10,950 = 100,900
        assert portfolio.cash == 100_900

    def test_portfolio_state_consistency(self, portfolio):
        """Test that portfolio state remains consistent after operations."""
        date = pd.Timestamp("2023-01-01")

        # Execute several trades
        portfolio.exec_trade("AAPL", "buy", price=100, shares=50, date=date)
        portfolio.exec_trade("MSFT", "buy", price=200, shares=25, date=date)
        portfolio.exec_trade("AAPL", "sell", price=105, shares=25, date=date)

        # Verify final state
        total_cost = (50 * 100) + (25 * 200)
        total_revenue = 25 * 105
        expected_cash = 100_000 - total_cost + total_revenue

        assert portfolio.cash == expected_cash
        assert portfolio.holdings["AAPL"] == 25
        assert portfolio.holdings["MSFT"] == 25
        assert len(portfolio.trade_log) == 3


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_buy_with_exactly_enough_cash(self, portfolio, sample_date):
        """Test buying when cash exactly equals cost."""
        portfolio.cash = 1000
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)

        assert portfolio.holdings["AAPL"] == 10
        assert portfolio.cash == 0

    def test_very_small_share_price(self, portfolio, sample_date):
        """Test with very small share prices."""
        portfolio.exec_trade(
            "PENNY", "buy", price=0.01, shares=1_000_000, date=sample_date
        )

        assert portfolio.holdings["PENNY"] == 1_000_000
        assert portfolio.cash == 90_000

    def test_very_large_share_price(self, portfolio, sample_date):
        """Test with very large share prices."""
        portfolio.exec_trade(
            "EXPENSIVE", "buy", price=10_000, shares=5, date=sample_date
        )

        assert portfolio.holdings["EXPENSIVE"] == 5
        assert portfolio.cash == 50_000

    def test_same_stock_multiple_buys(self, portfolio, sample_date):
        """Test accumulating shares in same stock."""
        portfolio.exec_trade("AAPL", "buy", price=100, shares=10, date=sample_date)
        portfolio.exec_trade("AAPL", "buy", price=105, shares=10, date=sample_date)
        portfolio.exec_trade("AAPL", "buy", price=110, shares=10, date=sample_date)

        assert portfolio.holdings["AAPL"] == 30
        expected_cost = (100 * 10) + (105 * 10) + (110 * 10)
        assert portfolio.cash == 100_000 - expected_cost
