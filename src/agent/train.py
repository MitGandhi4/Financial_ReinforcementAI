"""Train a PPO agent (Stable-Baselines3) on the AlphaRL trading environment,
with every run logged to MLflow (hyperparameters, training curve, artifacts).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.agent.model_utils import make_vec_env
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, settings
from src.env.trading_env import TradingEnvConfig
from src.features.feature_engineering import build_model_ready_splits
from src.logging_config import get_logger

logger = get_logger(__name__)


class MlflowCallback(BaseCallback):
    """Streams SB3's rollout/training metrics into the active MLflow run."""

    def __init__(self, log_every: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            for key, value in self.model.logger.name_to_value.items():
                try:
                    mlflow.log_metric(key.replace("/", "_"), float(value), step=self.num_timesteps)
                except (TypeError, ValueError):
                    continue
        return True


def train(
    ticker: str,
    total_timesteps: int,
    dataset_path: Path | None = None,
    experiment_name: str = "alpharl-ppo",
    run_name: str | None = None,
) -> Path:
    dataset_path = dataset_path or (PROCESSED_DATA_DIR / "unified_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} not found. Run `python -m src.data_pipeline.pipeline` first."
        )

    raw_df = pd.read_csv(dataset_path, parse_dates=["date"])
    raw_df = raw_df[raw_df["ticker"] == ticker].reset_index(drop=True)
    if raw_df.empty:
        raise ValueError(f"No rows for ticker {ticker} in {dataset_path}")

    splits = build_model_ready_splits(raw_df)
    train_df = splits["train"]

    env_config = TradingEnvConfig(initial_cash=settings.initial_cash)
    vec_env = make_vec_env(train_df, env_config)

    hyperparams = dict(
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    model_path = MODELS_DIR / f"ppo_{ticker.lower()}.zip"

    with mlflow.start_run(run_name=run_name or f"ppo_{ticker.lower()}"):
        mlflow.log_params({**hyperparams, "ticker": ticker, "total_timesteps": total_timesteps})
        mlflow.log_params(
            {
                "window_size": env_config.window_size,
                "max_position": env_config.max_position,
                "transaction_cost_pct": env_config.transaction_cost_pct,
                "slippage_pct": env_config.slippage_pct,
                "initial_cash": env_config.initial_cash,
            }
        )

        model = PPO(
            hyperparams.pop("policy"),
            vec_env,
            learning_rate=hyperparams["learning_rate"],
            n_steps=hyperparams["n_steps"],
            batch_size=hyperparams["batch_size"],
            n_epochs=hyperparams["n_epochs"],
            gamma=hyperparams["gamma"],
            gae_lambda=hyperparams["gae_lambda"],
            clip_range=hyperparams["clip_range"],
            ent_coef=hyperparams["ent_coef"],
            verbose=1,
        )

        logger.info("Starting PPO training for %s: %d timesteps", ticker, total_timesteps)
        model.learn(total_timesteps=total_timesteps, callback=MlflowCallback())

        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))
        logger.info("Training complete. Model saved to %s", model_path)

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train AlphaRL PPO agent")
    parser.add_argument("--ticker", type=str, default=settings.ticker_list[0])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    train(args.ticker, args.timesteps, run_name=args.run_name)


if __name__ == "__main__":
    main()
