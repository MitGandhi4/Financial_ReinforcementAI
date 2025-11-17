"""Performance metrics: Sharpe ratio, max drawdown, Calmar ratio, CAGR."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().fillna(0.0)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / std)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def cagr(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    n_periods = len(equity_curve)
    if n_periods < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def calmar_ratio(equity_curve: pd.Series) -> float:
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(cagr(equity_curve) / mdd)


def summarize_performance(equity_curve: pd.Series, label: str = "strategy") -> dict:
    returns = compute_returns(equity_curve)
    summary = {
        "label": label,
        "total_return_pct": float((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100),
        "cagr_pct": float(cagr(equity_curve) * 100),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown_pct": float(max_drawdown(equity_curve) * 100),
        "calmar_ratio": calmar_ratio(equity_curve),
        "volatility_annualized_pct": float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100),
    }
    logger.info("Performance summary [%s]: %s", label, summary)
    return summary
