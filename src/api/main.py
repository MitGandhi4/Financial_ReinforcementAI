"""FastAPI backend for AlphaRL.

Endpoints:
  GET  /health                       - liveness/config check
  POST /train                        - kick off PPO training for a ticker (background task)
  POST /backtest                     - run (or re-run) a backtest for a ticker
  GET  /backtest/{ticker}            - fetch the most recent saved backtest results
  GET  /trades/{ticker}              - fetch the trade log from the last backtest
  POST /signal                       - get the trained agent's current target position
  POST /reasoning                    - get an LLM explanation for a specific past trade
"""
from __future__ import annotations

import json

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, settings
from src.env.trading_env import TradingEnv, TradingEnvConfig
from src.features.feature_engineering import build_model_ready_splits
from src.logging_config import get_logger

# Stable-Baselines3 (and its torch dependency) are imported lazily inside the
# request handlers below rather than at module load time. This keeps
# `/health` and other lightweight endpoints usable even in slim deployments
# or during local development before the full ML stack is installed.
from src.reasoning_agent.reasoning import TradeContext, explain_trade
from src.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    HealthResponse,
    ReasoningRequest,
    ReasoningResponse,
    TradeSignalRequest,
    TradeSignalResponse,
    TrainRequest,
    TrainResponse,
)

logger = get_logger(__name__)

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

app = FastAPI(
    title="AlphaRL API",
    description="RL trading agent with an LLM reasoning layer.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", mlflow_tracking_uri=settings.mlflow_tracking_uri)


@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest, background_tasks: BackgroundTasks) -> TrainResponse:
    dataset_path = PROCESSED_DATA_DIR / "unified_dataset.csv"
    if not dataset_path.exists():
        raise HTTPException(
            status_code=400,
            detail="No dataset found. Run `python -m src.data_pipeline.pipeline` first.",
        )

    def _run():
        try:
            from src.agent.train import train as train_agent

            train_agent(request.ticker, request.total_timesteps)
        except Exception as exc:  # noqa: BLE001
            logger.error("Background training failed for %s: %s", request.ticker, exc)

    background_tasks.add_task(_run)
    model_path = MODELS_DIR / f"ppo_{request.ticker.lower()}.zip"
    return TrainResponse(ticker=request.ticker, model_path=str(model_path), status="training_started")


@app.post("/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest) -> BacktestResponse:
    model_path = MODELS_DIR / f"ppo_{request.ticker.lower()}.zip"
    if not model_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"No trained model found for {request.ticker}. Call /train first.",
        )
    from src.backtest.backtester import backtest_ticker, walk_forward_validation

    try:
        if request.mode == "walk-forward":
            result = walk_forward_validation(request.ticker)
            return BacktestResponse(
                ticker=request.ticker,
                strategy={"folds": result},
                buy_and_hold={},
                n_trades=0,
            )
        result = backtest_ticker(request.ticker)
        return BacktestResponse(
            ticker=result["ticker"],
            strategy=result["strategy"],
            buy_and_hold=result["buy_and_hold"],
            n_trades=result["n_trades"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Backtest failed for %s: %s", request.ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/backtest/{ticker}")
def get_backtest_results(ticker: str) -> dict:
    result_path = RESULTS_DIR / f"backtest_{ticker.lower()}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"No backtest results for {ticker.upper()}")
    with open(result_path) as f:
        return json.load(f)


@app.get("/trades/{ticker}")
def get_trade_log(ticker: str) -> list[dict]:
    result_path = RESULTS_DIR / f"backtest_{ticker.lower()}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"No backtest results for {ticker.upper()}")
    with open(result_path) as f:
        data = json.load(f)
    return data.get("trade_log", [])


@app.post("/signal", response_model=TradeSignalResponse)
def get_signal(request: TradeSignalRequest) -> TradeSignalResponse:
    """Return the trained agent's current target position for the most recent
    available bar. This is a decision-support signal only -- AlphaRL does not
    place real trades or move money on the user's behalf.
    """
    model_path = MODELS_DIR / f"ppo_{request.ticker.lower()}.zip"
    dataset_path = PROCESSED_DATA_DIR / "unified_dataset.csv"
    if not model_path.exists() or not dataset_path.exists():
        raise HTTPException(status_code=400, detail="Model or dataset not found. Train first.")

    raw_df = pd.read_csv(dataset_path, parse_dates=["date"])
    raw_df = raw_df[raw_df["ticker"] == request.ticker].reset_index(drop=True)
    splits = build_model_ready_splits(raw_df)
    test_df = splits["test"]

    env_config = TradingEnvConfig(initial_cash=settings.initial_cash)
    env = TradingEnv(test_df, env_config)
    obs, _ = env.reset()

    from stable_baselines3 import PPO

    model = PPO.load(str(model_path))
    action, _ = model.predict(obs, deterministic=True)

    latest_row = splits["test_raw"].iloc[env_config.window_size]
    return TradeSignalResponse(
        ticker=request.ticker,
        date=str(latest_row["date"]),
        target_position=float(action[0]),
        price=float(latest_row["close"]),
    )


@app.post("/reasoning", response_model=ReasoningResponse)
def get_reasoning(request: ReasoningRequest) -> ReasoningResponse:
    result_path = RESULTS_DIR / f"backtest_{request.ticker.lower()}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"No backtest results for {request.ticker}")
    with open(result_path) as f:
        data = json.load(f)

    trades = data.get("trade_log", [])
    if request.trade_index >= len(trades):
        raise HTTPException(status_code=404, detail="trade_index out of range")

    trade = trades[request.trade_index]

    dataset_path = PROCESSED_DATA_DIR / "unified_dataset.csv"
    raw_df = pd.read_csv(dataset_path, parse_dates=["date"])
    raw_df = raw_df[raw_df["ticker"] == request.ticker].reset_index(drop=True)
    context_row = raw_df.iloc[(raw_df["date"].astype(str) == trade["date"]).idxmax()] if (raw_df["date"].astype(str) == trade["date"]).any() else raw_df.iloc[0]

    context = TradeContext(
        ticker=request.ticker,
        date=str(trade["date"]),
        action="increased long position" if trade["trade_delta"] > 0 else "reduced/shorted position",
        prev_position=trade["prev_position"],
        new_position=trade["new_position"],
        price=trade["exec_price"],
        sentiment_score=float(context_row.get("sentiment_score", 0.0)),
        vix=float(context_row.get("vix", 18.0)),
        yield_curve_slope=float(context_row.get("yield_curve_slope", 0.0)),
        rsi_14=float(context_row.get("rsi_14", 50.0)) if "rsi_14" in context_row else 50.0,
        macd_hist=float(context_row.get("macd_hist", 0.0)) if "macd_hist" in context_row else 0.0,
    )
    explanation = explain_trade(context)
    return ReasoningResponse(ticker=request.ticker, trade=trade, explanation=explanation)
