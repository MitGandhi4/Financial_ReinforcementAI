"""Pydantic request/response schemas for the FastAPI backend."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrainRequest(BaseModel):
    ticker: str
    total_timesteps: int = Field(default=50_000, gt=0, le=5_000_000)

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    model_path: str
    status: str


class BacktestRequest(BaseModel):
    ticker: str
    mode: str = Field(default="single", pattern="^(single|walk-forward)$")

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class BacktestResponse(BaseModel):
    ticker: str
    strategy: dict[str, Any]
    buy_and_hold: dict[str, Any]
    n_trades: int


class TradeSignalRequest(BaseModel):
    """Ask the trained agent for a target position given the latest features."""

    ticker: str

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class TradeSignalResponse(BaseModel):
    ticker: str
    date: str
    target_position: float
    price: float


class ReasoningRequest(BaseModel):
    ticker: str
    trade_index: int = Field(ge=0)

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class ReasoningResponse(BaseModel):
    ticker: str
    trade: dict[str, Any]
    explanation: str


class HealthResponse(BaseModel):
    status: str
    mlflow_tracking_uri: str
