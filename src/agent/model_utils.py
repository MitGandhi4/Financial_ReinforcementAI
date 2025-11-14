"""Helpers shared between training and backtesting: build envs, load models."""
from __future__ import annotations

import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.trading_env import TradingEnv, TradingEnvConfig
from src.logging_config import get_logger

logger = get_logger(__name__)


def make_env(df: pd.DataFrame, config: TradingEnvConfig | None = None):
    def _init():
        env = TradingEnv(df, config)
        return Monitor(env)
    return _init


def make_vec_env(df: pd.DataFrame, config: TradingEnvConfig | None = None) -> DummyVecEnv:
    return DummyVecEnv([make_env(df, config)])
