"""Centralized, validated configuration for AlphaRL.

All environment variables are loaded and validated here via pydantic-settings.
Every other module should import `settings` from this file rather than
calling os.environ directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application settings, sourced from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # External API keys (optional at import time; validated lazily by callers
    # that actually need them, so the rest of the app can run without every
    # key configured).
    fred_api_key: str | None = Field(default=None, alias="FRED_API_KEY")
    newsapi_key: str | None = Field(default=None, alias="NEWSAPI_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")

    # LLM reasoning layer. Defaults to a free, local Ollama model so the
    # project runs at zero cost out of the box. Set REASONING_PROVIDER=claude
    # (and ANTHROPIC_API_KEY) only if you want to pay for hosted inference.
    reasoning_provider: str = Field(default="ollama", alias="REASONING_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1", alias="OLLAMA_MODEL")

    alpaca_api_key: str | None = Field(default=None, alias="ALPACA_API_KEY")
    alpaca_secret_key: str | None = Field(default=None, alias="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL"
    )

    tickers: str = Field(default="AAPL,MSFT,SPY", alias="TICKERS")
    start_date: str = Field(default="2018-01-01", alias="START_DATE")
    end_date: str = Field(default="2024-12-31", alias="END_DATE")
    initial_cash: float = Field(default=100_000.0, alias="INITIAL_CASH")

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("initial_cash")
    @classmethod
    def _positive_cash(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("initial_cash must be positive")
        return v

    @property
    def ticker_list(self) -> List[str]:
        return [t.strip().upper() for t in self.tickers.split(",") if t.strip()]


settings = Settings()
