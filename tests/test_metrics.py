import numpy as np
import pandas as pd

from src.backtest.metrics import (
    cagr,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    summarize_performance,
)


def test_max_drawdown_known_series():
    equity = pd.Series([100, 120, 90, 130, 80, 150])
    # running max: 100,120,120,130,130,150 -> drawdowns: 0,0,-0.25,0,-0.3846,0
    mdd = max_drawdown(equity)
    assert np.isclose(mdd, -0.3846, atol=1e-3)


def test_sharpe_zero_std_returns_zero():
    equity = pd.Series([100, 100, 100, 100])
    assert sharpe_ratio(equity.pct_change().fillna(0.0)) == 0.0


def test_cagr_flat_series_is_zero():
    equity = pd.Series([100] * 300)
    assert np.isclose(cagr(equity), 0.0, atol=1e-6)


def test_calmar_ratio_zero_drawdown_returns_zero():
    equity = pd.Series(np.linspace(100, 100, 50))
    assert calmar_ratio(equity) == 0.0


def test_summarize_performance_keys():
    equity = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 100)))
    summary = summarize_performance(equity, label="test")
    expected_keys = {
        "label", "total_return_pct", "cagr_pct", "sharpe_ratio",
        "max_drawdown_pct", "calmar_ratio", "volatility_annualized_pct",
    }
    assert expected_keys.issubset(summary.keys())
