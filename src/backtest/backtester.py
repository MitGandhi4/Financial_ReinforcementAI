"""Backtesting: single-pass evaluation, walk-forward validation, and
buy-and-hold baseline comparison. Results are logged to MLflow and saved
as JSON so the API/dashboard can serve them without re-running anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.agent.model_utils import make_vec_env
from src.backtest.metrics import summarize_performance
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, settings
from src.env.trading_env import TradingEnv, TradingEnvConfig
from src.features.feature_engineering import build_model_ready_splits
from src.logging_config import get_logger

logger = get_logger(__name__)

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_policy(model: PPO, df: pd.DataFrame, config: TradingEnvConfig | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Roll a trained policy forward through `df` deterministically."""
    env = TradingEnv(df, config)
    obs, _ = env.reset()
    equity = [env.net_worth]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        equity.append(info["net_worth"])
        done = terminated or truncated
    dates = df["date"].iloc[env.config.window_size : env.config.window_size + len(equity)].reset_index(drop=True)
    equity_curve = pd.Series(equity, index=range(len(equity)), name="net_worth")
    equity_curve.index = dates.reindex(range(len(equity))).ffill()
    return equity_curve, env.get_trade_log()


def buy_and_hold_curve(df: pd.DataFrame, initial_cash: float, window_size: int) -> pd.Series:
    prices = df["close"].iloc[window_size:].reset_index(drop=True)
    if prices.empty:
        raise ValueError("Not enough rows to build a buy-and-hold curve")
    shares = initial_cash / prices.iloc[0]
    curve = shares * prices
    curve.index = df["date"].iloc[window_size:].reset_index(drop=True)
    return curve


def backtest_ticker(ticker: str, model_path: Path | None = None) -> dict:
    dataset_path = PROCESSED_DATA_DIR / "unified_dataset.csv"
    raw_df = pd.read_csv(dataset_path, parse_dates=["date"])
    raw_df = raw_df[raw_df["ticker"] == ticker].reset_index(drop=True)

    splits = build_model_ready_splits(raw_df)
    test_df = splits["test"]

    model_path = model_path or (MODELS_DIR / f"ppo_{ticker.lower()}.zip")
    model = PPO.load(str(model_path))

    env_config = TradingEnvConfig(initial_cash=settings.initial_cash)
    equity_curve, trade_log = run_policy(model, test_df, env_config)
    bh_curve = buy_and_hold_curve(test_df, settings.initial_cash, env_config.window_size)

    strategy_summary = summarize_performance(equity_curve, label="ppo_agent")
    bh_summary = summarize_performance(bh_curve, label="buy_and_hold")

    result = {
        "ticker": ticker,
        "strategy": strategy_summary,
        "buy_and_hold": bh_summary,
        "n_trades": int(len(trade_log)),
        "equity_curve": equity_curve.reset_index().rename(
            columns={"index": "date", "net_worth": "net_worth"}
        ).astype(str).to_dict(orient="records"),
        "buy_and_hold_curve": bh_curve.rename("net_worth").reset_index().rename(
            columns={"index": "date"}
        ).astype(str).to_dict(orient="records"),
        "data_source": (
            "synthetic demonstration"
            if "data_source" in raw_df and raw_df["data_source"].eq("synthetic").all()
            else "market data"
        ),
        "trade_log": trade_log.to_dict(orient="records") if not trade_log.empty else [],
    }

    out_path = RESULTS_DIR / f"backtest_{ticker.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved backtest results to %s", out_path)

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("alpharl-backtest")
        with mlflow.start_run(run_name=f"backtest_{ticker.lower()}"):
            mlflow.log_metrics({f"strategy_{k}": v for k, v in strategy_summary.items() if isinstance(v, (int, float))})
            mlflow.log_metrics({f"buyhold_{k}": v for k, v in bh_summary.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(out_path))
    except Exception as exc:  # noqa: BLE001 - MLflow tracking is best-effort
        logger.warning("Could not log backtest to MLflow: %s", exc)

    return result


def walk_forward_validation(
    ticker: str, n_folds: int = 3, timesteps_per_fold: int = 20_000
) -> list[dict]:
    """Rolling-origin walk-forward validation: train on an expanding window,
    evaluate on the immediately following out-of-sample fold, repeat.
    """
    dataset_path = PROCESSED_DATA_DIR / "unified_dataset.csv"
    raw_df = pd.read_csv(dataset_path, parse_dates=["date"])
    raw_df = raw_df[raw_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)

    from src.features.feature_engineering import engineer_features, fit_normalization, FEATURE_COLUMNS

    features_df = engineer_features(raw_df)
    n = len(features_df)
    fold_size = n // (n_folds + 1)

    env_config = TradingEnvConfig(initial_cash=settings.initial_cash)
    fold_results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_end = fold_size * (fold + 2)
        train_slice = features_df.iloc[:train_end].reset_index(drop=True)
        test_slice = features_df.iloc[train_end:test_end].reset_index(drop=True)
        if len(test_slice) <= env_config.window_size + 1:
            continue

        stats = fit_normalization(train_slice, FEATURE_COLUMNS)
        train_norm = stats.apply(train_slice, FEATURE_COLUMNS)
        test_norm = stats.apply(test_slice, FEATURE_COLUMNS)

        vec_env = make_vec_env(train_norm, env_config)
        model = PPO("MlpPolicy", vec_env, verbose=0)
        logger.info("Walk-forward fold %d/%d: training on %d rows", fold + 1, n_folds, len(train_norm))
        model.learn(total_timesteps=timesteps_per_fold)

        equity_curve, _ = run_policy(model, test_norm, env_config)
        bh_curve = buy_and_hold_curve(test_norm, settings.initial_cash, env_config.window_size)

        fold_results.append(
            {
                "fold": fold + 1,
                "train_rows": len(train_norm),
                "test_rows": len(test_norm),
                "strategy": summarize_performance(equity_curve, label=f"fold{fold+1}_strategy"),
                "buy_and_hold": summarize_performance(bh_curve, label=f"fold{fold+1}_buyhold"),
            }
        )

    out_path = RESULTS_DIR / f"walk_forward_{ticker.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(fold_results, f, indent=2, default=str)
    logger.info("Saved walk-forward validation results to %s", out_path)
    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Backtest AlphaRL agent")
    parser.add_argument("--ticker", type=str, default=settings.ticker_list[0])
    parser.add_argument("--mode", choices=["single", "walk-forward"], default="single")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--timesteps-per-fold", type=int, default=20_000)
    args = parser.parse_args()

    if args.mode == "single":
        backtest_ticker(args.ticker)
    else:
        walk_forward_validation(args.ticker, args.folds, args.timesteps_per_fold)


if __name__ == "__main__":
    main()
