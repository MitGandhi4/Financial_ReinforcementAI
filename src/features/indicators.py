"""Technical indicators: RSI, MACD, Bollinger Bands, moving averages.

Every indicator here is computed using only past and present values at each
row (rolling windows, no `.shift(-1)` or forward-looking operations), so
there is no lookahead bias by construction.
"""
from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


def add_moving_averages(df: pd.DataFrame, windows: tuple[int, ...] = (10, 20, 50)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(window=w, min_periods=w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False, min_periods=w).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False, min_periods=signal).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    mid = df["close"].rolling(window=window, min_periods=window).mean()
    std = df["close"].rolling(window=window, min_periods=window).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + num_std * std
    df["bb_lower"] = mid - num_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid.replace(0.0, 1e-9)
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    returns = df["close"].pct_change()
    df["volatility_20"] = returns.rolling(window=window, min_periods=window).std()
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full indicator set to a single-ticker, date-sorted frame."""
    if df["date"].is_monotonic_increasing is False:
        raise ValueError("DataFrame must be sorted by date before computing indicators")
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volatility(df)
    logger.info("Computed technical indicators (%d rows)", len(df))
    return df


def add_indicators_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Apply indicators independently within each ticker group."""
    out = []
    for ticker, group in df.groupby("ticker", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        out.append(add_all_indicators(group))
    return pd.concat(out, ignore_index=True)
