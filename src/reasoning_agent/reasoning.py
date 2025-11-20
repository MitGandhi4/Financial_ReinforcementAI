"""LLM reasoning layer: after each trade, generate a natural-language
explanation grounded in the sentiment/macro/technical context at that time.

Two providers are supported:
  * "ollama" (default) -- free, runs entirely locally via a running Ollama
    server (https://ollama.com). No API key, no per-call cost.
  * "claude" -- optional, paid, hosted inference via the Anthropic API.
    Only used if REASONING_PROVIDER=claude and ANTHROPIC_API_KEY is set.

The project defaults to Ollama so it can be run and demoed at zero cost.
"""
from __future__ import annotations

import json

import requests
from pydantic import BaseModel

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class TradeContext(BaseModel):
    """Everything the reasoning layer needs to explain one trade."""

    ticker: str
    date: str
    action: str            # e.g. "increased long position", "reduced short exposure"
    prev_position: float
    new_position: float
    price: float
    sentiment_score: float
    vix: float
    yield_curve_slope: float
    rsi_14: float
    macd_hist: float


PROMPT_TEMPLATE = """You are a trading assistant explaining a single trade made by an \
automated reinforcement-learning trading agent. Be concise (3-5 sentences), \
factual, and reference the specific numbers given. Do not give financial advice \
or recommend future action -- only explain the trade that already happened.

Ticker: {ticker}
Date: {date}
Action: {action} (position moved from {prev_position:.2f} to {new_position:.2f} \
of net worth, on a -1 to +1 scale)
Price at execution: {price:.2f}

Context available to the agent at the time of the trade:
- News sentiment score (FinBERT, -1 very negative to +1 very positive): {sentiment_score:.3f}
- VIX (market volatility index): {vix:.2f}
- Yield curve slope (10y - 2y Treasury): {yield_curve_slope:.3f}
- RSI (14-day): {rsi_14:.1f}
- MACD histogram: {macd_hist:.4f}

Explain, in plain English, the most plausible reasons the agent made this trade \
given this context.
"""


def _call_ollama(prompt: str) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    try:
        resp = requests.post(
            url,
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("response", "").strip()
    except requests.RequestException as exc:
        logger.error("Ollama call failed (%s): %s", url, exc)
        raise RuntimeError(
            "Could not reach local Ollama server. Install it from "
            "https://ollama.com, run `ollama pull llama3.1`, and make sure "
            "`ollama serve` is running."
        ) from exc


def _call_claude(prompt: str) -> str:
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "REASONING_PROVIDER=claude but ANTHROPIC_API_KEY is not set. "
            "Either set the key (this will incur API costs) or switch back "
            "to REASONING_PROVIDER=ollama for a free local model."
        )
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()
    except anthropic.APIError as exc:
        logger.error("Anthropic API call failed: %s", exc)
        raise RuntimeError(f"Claude API call failed: {exc}") from exc


def explain_trade(context: TradeContext) -> str:
    """Generate a natural-language explanation for one trade.

    Uses whichever provider is configured (default: free local Ollama).
    Returns a deterministic rule-based fallback explanation if the LLM
    provider is unreachable, so the pipeline never breaks a demo.
    """
    prompt = PROMPT_TEMPLATE.format(**context.model_dump())

    try:
        if settings.reasoning_provider == "claude":
            text = _call_claude(prompt)
        else:
            text = _call_ollama(prompt)
        if text:
            return text
        raise RuntimeError("Empty response from LLM provider")
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM reasoning unavailable (%s); using rule-based fallback", exc)
        return _fallback_explanation(context)


def _fallback_explanation(context: TradeContext) -> str:
    direction = "increased" if context.new_position > context.prev_position else "decreased"
    drivers = []
    if context.sentiment_score > 0.15:
        drivers.append("positive news sentiment")
    elif context.sentiment_score < -0.15:
        drivers.append("negative news sentiment")
    if context.rsi_14 > 70:
        drivers.append("overbought RSI conditions")
    elif context.rsi_14 < 30:
        drivers.append("oversold RSI conditions")
    if context.macd_hist > 0:
        drivers.append("positive MACD momentum")
    elif context.macd_hist < 0:
        drivers.append("negative MACD momentum")
    if context.vix > 25:
        drivers.append("elevated market volatility (VIX)")

    driver_text = ", ".join(drivers) if drivers else "the current mix of technical and macro signals"
    return (
        f"[rule-based fallback] On {context.date}, the agent {direction} its position in "
        f"{context.ticker} from {context.prev_position:.2f} to {context.new_position:.2f} "
        f"at a price of {context.price:.2f}, most consistent with {driver_text}."
    )


if __name__ == "__main__":
    demo_context = TradeContext(
        ticker="AAPL", date="2024-01-15", action="increased long position",
        prev_position=0.1, new_position=0.4, price=185.5,
        sentiment_score=0.22, vix=14.2, yield_curve_slope=-0.3,
        rsi_14=62.0, macd_hist=0.8,
    )
    print(explain_trade(demo_context))
