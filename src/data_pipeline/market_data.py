"""OHLCV market data acquisition via yfinance."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, field_validator

from src.config import RAW_DATA_DIR
from src.logging_config import get_logger

logger = get_logger(__name__)


class OHLCVRequest(BaseModel):
    """Validated request for a single ticker's OHLCV history."""

    ticker: str
    start_date: str
    end_date: str
    interval: str = "1d"

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ticker must be non-empty")
        return v.strip().upper()

    @field_validator("interval")
    @classmethod
    def _valid_interval(cls, v: str) -> str:
        allowed = {"1d", "1h", "1wk"}
        if v not in allowed:
            raise ValueError(f"interval must be one of {allowed}")
        return v


def fetch_ohlcv(
    request: OHLCVRequest, max_retries: int = 3, retry_delay_sec: float = 2.0
) -> pd.DataFrame:
    """Download OHLCV data for one ticker with retries and validation.

    Raises RuntimeError if data cannot be retrieved after retries, or if the
    returned frame is empty / missing required columns.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Fetching OHLCV for %s [%s -> %s] (attempt %d/%d)",
                request.ticker,
                request.start_date,
                request.end_date,
                attempt,
                max_retries,
            )
            df = yf.download(
                request.ticker,
                start=request.start_date,
                end=request.end_date,
                interval=request.interval,
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
            )
            if df is None or df.empty:
                raise RuntimeError(f"No data returned for {request.ticker}")

            df = df.rename(columns=str.lower)
            required = {"open", "high", "low", "close", "volume"}
            missing = required - set(df.columns)
            if missing:
                raise RuntimeError(f"Missing columns {missing} for {request.ticker}")

            df.index.name = "date"
            df = df.reset_index()
            df["ticker"] = request.ticker
            logger.info("Fetched %d rows for %s", len(df), request.ticker)
            return df
        except Exception as exc:  # noqa: BLE001 - we deliberately retry any failure
            last_exc = exc
            logger.warning(
                "OHLCV fetch failed for %s on attempt %d/%d: %s",
                request.ticker,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(retry_delay_sec * attempt)

    raise RuntimeError(
        f"Failed to fetch OHLCV for {request.ticker} after {max_retries} attempts"
    ) from last_exc


def fetch_many(
    tickers: list[str], start_date: str, end_date: str, interval: str = "1d"
) -> pd.DataFrame:
    """Fetch OHLCV for multiple tickers and concatenate into one long frame."""
    frames = []
    for ticker in tickers:
        try:
            req = OHLCVRequest(
                ticker=ticker, start_date=start_date, end_date=end_date, interval=interval
            )
            frames.append(fetch_ohlcv(req))
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping %s: %s", ticker, exc)
    if not frames:
        raise RuntimeError("No OHLCV data could be fetched for any requested ticker")
    return pd.concat(frames, ignore_index=True)


def save_raw(df: pd.DataFrame, name: str = "ohlcv.csv") -> Path:
    out_path = RAW_DATA_DIR / name
    df.to_csv(out_path, index=False)
    logger.info("Saved raw OHLCV data to %s (%d rows)", out_path, len(df))
    return out_path


if __name__ == "__main__":
    from src.config import settings

    data = fetch_many(settings.ticker_list, settings.start_date, settings.end_date)
    save_raw(data)
