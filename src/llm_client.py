"""
llm_client.py — LLM API client for the Contract Intelligence Engine.
Wraps OpenAI (gpt-4o-mini / ChatGPT Nano 5) with:
- Retries with exponential backoff (tenacity)
- Response caching (diskcache)
- Strict JSON parsing
- Async support
"""

from __future__ import annotations
import json
import asyncio
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_MAX_RETRIES,
    LLM_RETRY_WAIT,
)
from .cache import cache_get, cache_set
from .logger import get_logger

logger = get_logger(__name__)

# Module-level async client singleton
_async_client: AsyncOpenAI | None = None


def _get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        _async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _async_client


async def llm_call(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    expect_json: bool = True,
) -> Optional[dict]:
    """
    Core LLM call function. Returns parsed JSON dict or None on failure.

    Args:
        prompt: The full prompt string.
        model: LLM model to use.
        temperature: Sampling temperature (0 for deterministic).
        max_tokens: Maximum tokens in response.
        expect_json: If True, attempt JSON parsing of response.

    Returns:
        Parsed dict if expect_json=True and parsing succeeds.
        Raw string if expect_json=False.
        None on complete failure.
    """
    # Check cache first
    cached = cache_get(prompt, model)
    if cached is not None:
        logger.debug("Returning cached LLM response")
        return cached

    logger.debug(f"LLM call → model={model}, prompt_len={len(prompt)}")

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            client = _get_async_client()

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise legal contract analysis assistant. "
                        "You extract information with high accuracy. "
                        "You ALWAYS respond with valid JSON only. "
                        "You NEVER hallucinate or invent contract text."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if expect_json else None,
            )

            raw_text = response.choices[0].message.content.strip()

            logger.debug(f"LLM raw response (first 200 chars): {raw_text[:200]}")

            if expect_json:
                result = _parse_json_safe(raw_text)
                if result is not None:
                    cache_set(prompt, model, result)
                    return result
                else:
                    logger.warning(f"JSON parse failed on attempt {attempt}, raw: {raw_text[:200]}")
            else:
                cache_set(prompt, model, {"text": raw_text})
                return {"text": raw_text}

        except OpenAIError as e:
            logger.error(f"OpenAI API error (attempt {attempt}/{LLM_MAX_RETRIES}): {e}")
            if attempt < LLM_MAX_RETRIES:
                wait = LLM_RETRY_WAIT * (2 ** (attempt - 1))
                logger.info(f"Retrying in {wait}s...")
                await asyncio.sleep(wait)
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            break

    logger.error(f"LLM call failed after {LLM_MAX_RETRIES} attempts")
    return None


def _parse_json_safe(text: str) -> Optional[dict]:
    """
    Safely parse JSON from LLM output.
    Handles common LLM formatting issues:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace
    - Extra text before/after JSON
    """
    if not text:
        return None

    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last ``` if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object from text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None
