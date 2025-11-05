"""Macro indicators (VIX, yield curve) via the FRED API."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, field_validator

from src.config import RAW_DATA_DIR, settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# FRED series used:
#  VIXCLS   - CBOE Volatility Index
#  DGS10    - 10-Year Treasury Constant Maturity Rate
#  DGS2     - 2-Year Treasury Constant Maturity Rate
# Yield curve slope = DGS10 - DGS2 (a classic recession-risk indicator).
FRED_SERIES = {"vix": "VIXCLS", "dgs10": "DGS10", "dgs2": "DGS2"}


class MacroRequest(BaseModel):
    start_date: str
    end_date: str

    @field_validator("start_date", "end_date")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("date must not be empty")
        return v


def fetch_macro(request: MacroRequest) -> pd.DataFrame:
    """Fetch macro series from FRED. Falls back to a flat synthetic series
    (clearly logged) if FRED_API_KEY is missing or a request fails, so the
    pipeline never hard-crashes on a missing optional key.
    """
    date_range = pd.date_range(request.start_date, request.end_date, freq="D")
    frame = pd.DataFrame({"date": date_range})

    if not settings.fred_api_key:
        logger.warning("FRED_API_KEY not configured; using synthetic flat macro series")
        frame["vix"] = 18.0
        frame["dgs10"] = 4.0
        frame["dgs2"] = 4.5
        frame["yield_curve_slope"] = frame["dgs10"] - frame["dgs2"]
        return frame

    try:
        from fredapi import Fred

        fred = Fred(api_key=settings.fred_api_key)
        for col, series_id in FRED_SERIES.items():
            try:
                series = fred.get_series(
                    series_id, observation_start=request.start_date,
                    observation_end=request.end_date,
                )
                series = series.rename(col)
                series.index.name = "date"
                frame = frame.merge(
                    series.reset_index(), on="date", how="left"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("FRED fetch failed for series %s: %s", series_id, exc)
                frame[col] = pd.NA

        frame = frame.sort_values("date").ffill()
        frame["vix"] = frame["vix"].fillna(18.0)
        frame["dgs10"] = frame["dgs10"].fillna(4.0)
        frame["dgs2"] = frame["dgs2"].fillna(4.5)
        frame["yield_curve_slope"] = frame["dgs10"] - frame["dgs2"]
        logger.info("Fetched macro data: %d rows", len(frame))
        return frame
    except Exception as exc:  # noqa: BLE001
        logger.error("FRED client initialization failed, using synthetic data: %s", exc)
        frame["vix"] = 18.0
        frame["dgs10"] = 4.0
        frame["dgs2"] = 4.5
        frame["yield_curve_slope"] = frame["dgs10"] - frame["dgs2"]
        return frame


def save_macro(df: pd.DataFrame, name: str = "macro.csv") -> Path:
    out_path = RAW_DATA_DIR / name
    df.to_csv(out_path, index=False)
    logger.info("Saved macro data to %s (%d rows)", out_path, len(df))
    return out_path


if __name__ == "__main__":
    data = fetch_macro(MacroRequest(start_date=settings.start_date, end_date=settings.end_date))
    save_macro(data)
