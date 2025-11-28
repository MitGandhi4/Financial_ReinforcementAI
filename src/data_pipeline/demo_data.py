"""Create a deterministic, clearly labeled dataset for offline demos."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.logging_config import get_logger

logger = get_logger(__name__)


def generate_demo_data(ticker: str = "AAPL", rows: int = 1_500, seed: int = 7) -> pd.DataFrame:
    """Generate reproducible OHLCV and context data without network access."""
    if rows < 250:
        raise ValueError("rows must be at least 250 to cover feature warm-up and evaluation")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2025-01-31", periods=rows)
    returns = rng.normal(0.0003, 0.012, rows)
    returns += 0.0015 * np.sin(np.arange(rows) / 35)
    close = 100 * np.exp(np.cumsum(returns))
    open_prices = np.concatenate(([close[0] * (1 + rng.normal(0, 0.003))],
                                  close[:-1] * (1 + rng.normal(0, 0.003, rows - 1))))
    intraday_range = np.abs(rng.normal(0.008, 0.003, rows))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "high": np.maximum(open_prices, close) * (1 + intraday_range),
            "low": np.minimum(open_prices, close) * (1 - intraday_range),
            "close": close,
            "volume": rng.integers(1_000_000, 8_000_000, rows),
            "ticker": ticker.strip().upper(),
            "sentiment_score": np.clip(
                0.2 * np.sin(np.arange(rows) / 18) + rng.normal(0, 0.15, rows), -1, 1
            ),
            "vix": 18 + 5 * np.sin(np.arange(rows) / 45) + rng.normal(0, 1.2, rows),
            "dgs10": 4 + 0.4 * np.sin(np.arange(rows) / 90),
            "dgs2": 4.2 + 0.35 * np.sin(np.arange(rows) / 75),
            "data_source": "synthetic",
        }
    )
    frame["yield_curve_slope"] = frame["dgs10"] - frame["dgs2"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline AlphaRL demo data")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--rows", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    output = PROCESSED_DATA_DIR / "unified_dataset.csv"
    frame = generate_demo_data(args.ticker, args.rows, args.seed)
    frame.to_csv(output, index=False)
    logger.info("Wrote %d rows of synthetic demo data to %s", len(frame), output)
    logger.info("Next: train with `python -m src.agent.train --ticker %s --timesteps 4096`", args.ticker.upper())


if __name__ == "__main__":
    main()
