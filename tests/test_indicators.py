import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    add_all_indicators,
    add_bollinger_bands,
    add_macd,
    add_moving_averages,
    add_rsi,
)


def _make_price_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"date": dates, "close": prices, "open": prices, "high": prices + 1, "low": prices - 1, "volume": 1000})


def test_moving_averages_no_lookahead():
    df = _make_price_df()
    out = add_moving_averages(df, windows=(10,))
    # SMA at row i must equal the mean of close[i-9:i+1], using only past data
    i = 50
    expected = df["close"].iloc[i - 9 : i + 1].mean()
    assert np.isclose(out["sma_10"].iloc[i], expected)
    # warm-up rows should be NaN, not silently filled with future data
    assert out["sma_10"].iloc[:9].isna().all()


def test_rsi_bounds():
    df = _make_price_df()
    out = add_rsi(df)
    valid = out["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd_columns_present():
    df = _make_price_df()
    out = add_macd(df)
    assert {"macd", "macd_signal", "macd_hist"}.issubset(out.columns)


def test_bollinger_band_ordering():
    df = _make_price_df()
    out = add_bollinger_bands(df)
    valid = out.dropna(subset=["bb_upper", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_add_all_indicators_requires_sorted_dates():
    df = _make_price_df()
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    with pytest.raises(ValueError):
        add_all_indicators(shuffled)


def test_indicators_do_not_change_when_future_prices_change():
    df = _make_price_df()
    original = add_all_indicators(df)
    changed = df.copy()
    changed.loc[80:, ["open", "high", "low", "close"]] *= 3
    modified = add_all_indicators(changed)
    for column in ("sma_10", "rsi_14", "macd", "bb_upper", "volatility_20"):
        pd.testing.assert_series_equal(
            original.loc[:79, column], modified.loc[:79, column], check_names=False
        )
