"""FinBERT sentiment scoring for financial news headlines.

Headlines are pulled from NewsAPI (https://newsapi.org) when NEWSAPI_KEY is
configured. If no headlines are available (missing key, API failure, or an
empty result set) this module falls back to a neutral-sentiment placeholder
so the rest of the pipeline still runs end to end -- this is logged loudly
so it's never silently mistaken for real signal.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel, field_validator

from src.config import RAW_DATA_DIR, settings
from src.logging_config import get_logger

logger = get_logger(__name__)

_FINBERT_MODEL_NAME = "ProsusAI/finbert"


class HeadlineQuery(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    page_size: int = 50

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


@lru_cache(maxsize=1)
def _load_finbert():
    """Lazily load the FinBERT tokenizer/model (expensive, so cached)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading FinBERT model %s ...", _FINBERT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL_NAME)
    model.eval()
    logger.info("FinBERT model loaded")
    return tokenizer, model


def score_headlines(headlines: list[str]) -> pd.DataFrame:
    """Return a DataFrame with columns [headline, positive, negative, neutral, sentiment_score]."""
    if not headlines:
        return pd.DataFrame(
            columns=["headline", "positive", "negative", "neutral", "sentiment_score"]
        )

    import torch

    tokenizer, model = _load_finbert()
    rows = []
    batch_size = 16
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
        # FinBERT label order: positive, negative, neutral
        for headline, (pos, neg, neu) in zip(batch, probs):
            rows.append(
                {
                    "headline": headline,
                    "positive": float(pos),
                    "negative": float(neg),
                    "neutral": float(neu),
                    "sentiment_score": float(pos - neg),
                }
            )
    return pd.DataFrame(rows)


def fetch_headlines(query: HeadlineQuery) -> list[str]:
    """Fetch financial news headlines for a ticker from NewsAPI.

    Returns an empty list (never raises) on any failure so callers can fall
    back gracefully; failures are logged with full context.
    """
    if not settings.newsapi_key:
        logger.warning(
            "NEWSAPI_KEY not configured; skipping headline fetch for %s", query.ticker
        )
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query.ticker,
        "from": query.start_date,
        "to": query.end_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": query.page_size,
        "apiKey": settings.newsapi_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        articles = payload.get("articles", [])
        headlines = [a["title"] for a in articles if a.get("title")]
        logger.info("Fetched %d headlines for %s", len(headlines), query.ticker)
        return headlines
    except requests.RequestException as exc:
        logger.error("NewsAPI request failed for %s: %s", query.ticker, exc)
        return []
    except (KeyError, ValueError) as exc:
        logger.error("NewsAPI response parsing failed for %s: %s", query.ticker, exc)
        return []


def build_daily_sentiment(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Build a per-day sentiment score series for one ticker.

    Falls back to a flat neutral score (0.0) for the whole window if no
    headlines could be retrieved, clearly logged as synthetic.
    """
    headlines = fetch_headlines(
        HeadlineQuery(ticker=ticker, start_date=start_date, end_date=end_date)
    )
    date_range = pd.date_range(start_date, end_date, freq="D")

    if not headlines:
        logger.warning(
            "Using neutral placeholder sentiment for %s (no headlines available)",
            ticker,
        )
        return pd.DataFrame(
            {"date": date_range, "ticker": ticker, "sentiment_score": 0.0}
        )

    scored = score_headlines(headlines)
    avg_score = scored["sentiment_score"].mean() if not scored.empty else 0.0
    # NewsAPI free tier does not reliably give per-article dates aligned to
    # trading days in this simplified pipeline, so we apply the aggregate
    # score across the requested window. Swap in per-article publishedAt
    # timestamps here for a production-grade, date-aligned signal.
    return pd.DataFrame(
        {"date": date_range, "ticker": ticker, "sentiment_score": avg_score}
    )


def save_sentiment(df: pd.DataFrame, name: str = "sentiment.csv") -> Path:
    out_path = RAW_DATA_DIR / name
    df.to_csv(out_path, index=False)
    logger.info("Saved sentiment data to %s (%d rows)", out_path, len(df))
    return out_path


if __name__ == "__main__":
    frames = [
        build_daily_sentiment(t, settings.start_date, settings.end_date)
        for t in settings.ticker_list
    ]
    save_sentiment(pd.concat(frames, ignore_index=True))
