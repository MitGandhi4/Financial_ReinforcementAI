import numpy as np
import pandas as pd
import pytest

from src.env.trading_env import TradingEnv, TradingEnvConfig
from src.features.feature_engineering import FEATURE_COLUMNS


def _make_env_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    prices = 100 + np.cumsum(rng.normal(0, 0.5, n))
    data = {"date": dates, "open": prices, "close": prices}
    for col in FEATURE_COLUMNS:
        data[col] = rng.normal(0, 1, n)
    return pd.DataFrame(data)


def test_env_reset_and_step_shapes():
    df = _make_env_df()
    config = TradingEnvConfig(window_size=5, initial_cash=10_000)
    env = TradingEnv(df, config)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    action = np.array([0.5], dtype=np.float32)
    obs2, reward, terminated, truncated, step_info = env.step(action)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert "net_worth" in step_info


def test_env_position_limits_enforced_by_action_space():
    df = _make_env_df()
    config = TradingEnvConfig(window_size=5, max_position=0.5)
    env = TradingEnv(df, config)
    assert env.action_space.high[0] == 0.5
    assert env.action_space.low[0] == -0.5


def test_env_transaction_costs_reduce_cash():
    df = _make_env_df()
    config = TradingEnvConfig(window_size=5, transaction_cost_pct=0.01, slippage_pct=0.0)
    env = TradingEnv(df, config)
    env.reset()
    cash_before = env.cash
    env.step(np.array([1.0], dtype=np.float32))
    assert env.cash < cash_before  # transaction cost was deducted


def test_env_rejects_missing_columns():
    df = _make_env_df().drop(columns=["rsi_14"] if "rsi_14" in FEATURE_COLUMNS else [])
    if "rsi_14" in FEATURE_COLUMNS:
        with pytest.raises(ValueError):
            TradingEnv(df, TradingEnvConfig(window_size=5))


def test_env_terminates_at_end_of_data():
    df = _make_env_df(n=20)
    config = TradingEnvConfig(window_size=5)
    env = TradingEnv(df, config)
    env.reset()
    terminated = False
    steps = 0
    while not terminated and steps < 100:
        obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        steps += 1
    assert terminated


def test_costs_reduce_net_worth_and_obey_configured_defaults():
    df = _make_env_df(n=20)
    df.loc[:, ["open", "close"]] = 100.0
    config = TradingEnvConfig(window_size=5)
    env = TradingEnv(df, config)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert config.transaction_cost_pct == 0.001
    assert config.slippage_pct == 0.0005
    assert info["net_worth"] < config.initial_cash


def test_step_rejects_non_finite_actions():
    env = TradingEnv(_make_env_df(n=20), TradingEnvConfig(window_size=5))
    with pytest.raises(ValueError, match="finite"):
        env.step(np.array([np.nan], dtype=np.float32))


def test_step_rejects_calls_after_episode_end():
    env = TradingEnv(_make_env_df(n=12), TradingEnvConfig(window_size=5))
    while True:
        _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
        if terminated:
            break
    with pytest.raises(RuntimeError, match="episode ended"):
        env.step(np.array([0.0], dtype=np.float32))


def test_environment_terminates_at_ruin_threshold():
    df = _make_env_df(n=20)
    df.loc[:, "open"] = 100.0
    df.loc[:, "close"] = 10.0
    env = TradingEnv(
        df,
        TradingEnvConfig(window_size=5, transaction_cost_pct=0, slippage_pct=0),
    )
    _, _, terminated, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert terminated
    assert info["net_worth"] <= env.config.initial_cash * env.config.ruin_threshold


def test_existing_position_earns_overnight_gap_before_rebalance():
    df = _make_env_df(n=20)
    df.loc[:, ["open", "close"]] = 100.0
    df.loc[7:, ["open", "close"]] = 110.0
    env = TradingEnv(
        df,
        TradingEnvConfig(window_size=5, transaction_cost_pct=0, slippage_pct=0),
    )
    env.step(np.array([1.0], dtype=np.float32))
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["net_worth"] == pytest.approx(110_000.0)
