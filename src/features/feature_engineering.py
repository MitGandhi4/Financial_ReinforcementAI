"""Feature engineering: assembles the model-ready feature matrix.

Lookahead-bias controls:
  1. All technical indicators (src/features/indicators.py) use only rolling
     windows of past+current data.
  2. Normalization statistics (mean/std) are fit ONLY on the training split
     and then applied to validation/test splits -- never fit on the full
     series, which would leak future information into the scaler.
  3. The observation at time t only ever contains information available at
     the close of day t (or earlier); the environment consumes features at
     t to decide the action executed at t+1's open (see src/env).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.indicators import add_indicators_per_ticker
from src.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "sma_10", "sma_20", "sma_50",
    "ema_10", "ema_20", "ema_50",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_mid", "bb_upper", "bb_lower", "bb_width",
    "volatility_20",
    "sentiment_score",
    "vix", "yield_curve_slope",
]


@dataclass
class NormalizationStats:
    mean: pd.Series
    std: pd.Series

    def apply(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        std_safe = self.std.replace(0.0, 1e-9)
        df[columns] = (df[columns] - self.mean[columns]) / std_safe[columns]
        return df


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators and drop warm-up rows that contain NaNs."""
    df = add_indicators_per_ticker(raw_df)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    logger.info(
        "Dropped %d warm-up rows with incomplete indicator windows (%d -> %d)",
        before - len(df), before, len(df),
    )
    return df


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered split per ticker (no shuffling, no leakage across time)."""
    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby("ticker", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train : n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val :])
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    logger.info(
        "Chronological split -> train=%d val=%d test=%d", len(train), len(val), len(test)
    )
    return train, val, test


def fit_normalization(train_df: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS) -> NormalizationStats:
    """Fit mean/std ONLY on the training split."""
    mean = train_df[columns].mean()
    std = train_df[columns].std()
    return NormalizationStats(mean=mean, std=std)


def build_model_ready_splits(
    raw_df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> dict[str, pd.DataFrame]:
    features_df = engineer_features(raw_df)
    train, val, test = chronological_split(features_df, train_frac, val_frac)
    stats = fit_normalization(train)

    train_norm = stats.apply(train, FEATURE_COLUMNS)
    val_norm = stats.apply(val, FEATURE_COLUMNS)
    test_norm = stats.apply(test, FEATURE_COLUMNS)

    return {
        "train": train_norm,
        "val": val_norm,
        "test": test_norm,
        "train_raw": train,
        "val_raw": val,
        "test_raw": test,
    }
