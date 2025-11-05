"""Unified data pipeline: OHLCV + FinBERT sentiment + macro indicators,
merged into one per-ticker, per-day dataset ready for feature engineering.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, settings
from src.data_pipeline.macro_data import MacroRequest, fetch_macro
from src.data_pipeline.market_data import fetch_many
from src.data_pipeline.sentiment import build_daily_sentiment
from src.logging_config import get_logger

logger = get_logger(__name__)


def build_dataset(
    tickers: list[str], start_date: str, end_date: str
) -> pd.DataFrame:
    logger.info("Building unified dataset for %s [%s -> %s]", tickers, start_date, end_date)

    ohlcv = fetch_many(tickers, start_date, end_date)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.tz_localize(None)

    macro = fetch_macro(MacroRequest(start_date=start_date, end_date=end_date))
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)

    sentiment_frames = [
        build_daily_sentiment(t, start_date, end_date) for t in tickers
    ]
    sentiment = pd.concat(sentiment_frames, ignore_index=True)
    sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.tz_localize(None)

    merged = ohlcv.merge(sentiment, on=["date", "ticker"], how="left")
    merged = merged.merge(macro, on="date", how="left")
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    n_before = len(merged)
    merged = merged.dropna(subset=["open", "high", "low", "close", "volume"])
    if len(merged) < n_before:
        logger.warning("Dropped %d rows with missing OHLCV data", n_before - len(merged))

    logger.info("Unified dataset built: %d rows, %d columns", *merged.shape)
    return merged


def save_dataset(df: pd.DataFrame, name: str = "unified_dataset.csv") -> Path:
    out_path = PROCESSED_DATA_DIR / name
    df.to_csv(out_path, index=False)
    logger.info("Saved unified dataset to %s", out_path)
    return out_path


def run(tickers: list[str] | None = None, start_date: str | None = None, end_date: str | None = None) -> Path:
    tickers = tickers or settings.ticker_list
    start_date = start_date or settings.start_date
    end_date = end_date or settings.end_date
    df = build_dataset(tickers, start_date, end_date)
    return save_dataset(df)


if __name__ == "__main__":
    run()
