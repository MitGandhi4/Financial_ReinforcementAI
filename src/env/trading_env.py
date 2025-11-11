"""Custom Gymnasium trading environment.

Key design points:
  * Single-asset, continuous position sizing: the action is a target
    portfolio exposure in [-max_position, max_position] (fraction of net
    worth), so the agent learns HOW MUCH to hold, not just buy/sell/hold.
  * Trades execute at the *next* bar's open price (avoids using the same
    bar's close both to decide and to price the trade -- another lookahead
    guard), with slippage and proportional transaction costs applied.
  * Position limits enforce leverage bounds.
  * Portfolio state (cash fraction, position, unrealized pnl) is included
    in the observation so the agent is aware of its own exposure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.features.feature_engineering import FEATURE_COLUMNS
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TradingEnvConfig:
    window_size: int = 10
    initial_cash: float = 100_000.0
    max_position: float = 1.0          # max fraction of net worth deployed (long or short)
    transaction_cost_pct: float = 0.001    # 10 bps per unit traded
    slippage_pct: float = 0.0005           # 5 bps adverse slippage per unit traded
    allow_short: bool = True
    reward_scaling: float = 100.0
    drawdown_penalty: float = 0.5      # penalty coefficient on new drawdown
    ruin_threshold: float = 0.5
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


class TradingEnv(gym.Env):
    """Continuous position-sizing single-asset trading environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, config: TradingEnvConfig | None = None):
        super().__init__()
        required = {"date", "close", "open"} | set((config or TradingEnvConfig()).feature_columns)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"TradingEnv requires columns {missing} missing from input df")

        self.config = config or TradingEnvConfig()
        if self.config.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.config.initial_cash <= 0 or self.config.max_position <= 0:
            raise ValueError("initial_cash and max_position must be positive")
        if not 0 < self.config.ruin_threshold < 1:
            raise ValueError("ruin_threshold must be between 0 and 1")
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        if self.n_steps <= self.config.window_size + 1:
            raise ValueError("DataFrame too short for the requested window_size")

        n_features = len(self.config.feature_columns)
        obs_dim = self.config.window_size * n_features + 3  # +cash_frac, position, unrealized_pnl

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        low = -self.config.max_position if self.config.allow_short else 0.0
        self.action_space = spaces.Box(
            low=low, high=self.config.max_position, shape=(1,), dtype=np.float32
        )

        self._feature_matrix = self.df[self.config.feature_columns].to_numpy(dtype=np.float32)
        self._close = self.df["close"].to_numpy(dtype=np.float64)
        self._open = self.df["open"].to_numpy(dtype=np.float64)

        self.reset()

    def _get_obs(self) -> np.ndarray:
        window = self._feature_matrix[self._t - self.config.window_size : self._t]
        portfolio_state = np.array(
            [self.cash / self.net_worth, self.position, self._unrealized_pnl_pct()],
            dtype=np.float32,
        )
        return np.concatenate([window.flatten(), portfolio_state]).astype(np.float32)

    def _unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        price = self._close[self._t]
        return float((price - self.entry_price) / self.entry_price * np.sign(self.position or 1))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = self.config.window_size
        self.cash = self.config.initial_cash
        self.position = 0.0          # fraction of net worth currently deployed
        self.shares = 0.0
        self.entry_price = 0.0
        self.net_worth = self.config.initial_cash
        self.max_net_worth = self.config.initial_cash
        self.trade_log: list[dict] = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if self._t >= self.n_steps - 1:
            raise RuntimeError("step() called after the episode ended; call reset()")
        target_position = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        if not np.isfinite(target_position):
            raise ValueError("action must be finite")

        exec_price = float(self._open[self._t + 1])
        prev_net_worth = self.net_worth

        # Existing exposure earns the overnight gap; the new target is filled
        # at the next open and only earns the open-to-close move.
        previous_close = float(self._close[self._t])
        overnight_return = (exec_price - previous_close) / previous_close
        overnight_pnl = self.position * overnight_return * prev_net_worth
        equity_at_open = max(prev_net_worth + overnight_pnl, 1e-6)

        trade_delta = target_position - self.position
        traded_notional = abs(trade_delta) * equity_at_open

        slippage_direction = 1 if trade_delta > 0 else -1
        effective_price = exec_price * (1 + slippage_direction * self.config.slippage_pct)
        transaction_cost = traded_notional * self.config.transaction_cost_pct
        slippage_cost = traded_notional * self.config.slippage_pct

        if abs(trade_delta) > 1e-6:
            self.trade_log.append(
                {
                    "step": int(self._t),
                    "date": str(self.df.loc[self._t, "date"]),
                    "prev_position": self.position,
                    "new_position": target_position,
                    "trade_delta": trade_delta,
                    "exec_price": effective_price,
                    "transaction_cost": transaction_cost,
                    "slippage_cost": slippage_cost,
                }
            )
            self.entry_price = effective_price

        self.position = target_position
        execution_cost = transaction_cost + slippage_cost
        equity_after_costs = max(equity_at_open - execution_cost, 1e-6)
        intraday_return = (float(self._close[self._t + 1]) - exec_price) / exec_price
        self.net_worth = max(
            equity_after_costs * (1 + self.position * intraday_return), 1e-6
        )
        self.cash = self.net_worth * (1 - self.position)
        self._t += 1

        previous_drawdown = (self.max_net_worth - prev_net_worth) / self.max_net_worth
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth

        step_return = (self.net_worth - prev_net_worth) / prev_net_worth
        reward = step_return * self.config.reward_scaling
        reward -= self.config.drawdown_penalty * max(0.0, drawdown - previous_drawdown)

        terminated = (
            self._t >= self.n_steps - 1
            or self.net_worth <= self.config.initial_cash * self.config.ruin_threshold
        )
        truncated = False
        info = {
            "net_worth": self.net_worth,
            "position": self.position,
            "drawdown": drawdown,
            "step_return": step_return,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        logger.info(
            "t=%d net_worth=%.2f position=%.3f cash=%.2f",
            self._t, self.net_worth, self.position, self.cash,
        )

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)
