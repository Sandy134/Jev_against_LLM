import logging
import os
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import APIStatusError, InternalServerError, RateLimitError

import config

load_dotenv()
logger = logging.getLogger(__name__)
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            reasoning_effort=config.OPENAI_REASONING_EFFORT,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=0,
        )
    return _llm


def _is_retryable(error: Exception) -> bool:
    return isinstance(error, (RateLimitError, InternalServerError)) or (
        isinstance(error, APIStatusError) and error.status_code >= 500
    )


def _empty_result(
    *, latency_ms: float, raw_response: str
) -> dict[str, Any]:
    return {
        "predicted_answer": None,
        "parsed_ok": False,
        "latency_ms": latency_ms,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "confidence": None,
        "probabilities": None,
        "raw_response": raw_response,
    }


def classify(inputs: dict[str, Any]) -> dict[str, Any]:
    question = inputs["question"]
    options = inputs["options"]
    nonce = uuid.uuid4()
    prompt = (
        f"Question: {question}\n"
        f"Options:\n{chr(10).join(f'- {option}' for option in options)}\n\n"
        "Classify the question by choosing exactly one option. "
        "Respond with ONLY the exact category string from the options, nothing else.\n"
        f"Request nonce: {nonce}"
    )

    started_at = time.perf_counter()
    for attempt in range(config.MAX_ATTEMPTS):
        try:
            response = _get_llm().invoke(prompt)
            usage = response.usage_metadata or {}
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            input_details = usage.get("input_token_details", {})
            cached_tokens = min(
                input_tokens, int(input_details.get("cache_read", 0))
            )
            uncached_tokens = input_tokens - cached_tokens
            cost_usd = (
                uncached_tokens
                / 1_000_000
                * config.OPENAI_INPUT_PRICE_PER_MILLION
                + cached_tokens
                / 1_000_000
                * config.OPENAI_CACHED_INPUT_PRICE_PER_MILLION
                + output_tokens
                / 1_000_000
                * config.OPENAI_OUTPUT_PRICE_PER_MILLION
            )
            raw_content = response.content
            raw_response = (
                raw_content if isinstance(raw_content, str) else str(raw_content)
            )
            predicted_answer = raw_response.strip().strip("\"'").strip()
            parsed_ok = predicted_answer in options
            return {
                "predicted_answer": predicted_answer,
                "parsed_ok": parsed_ok,
                "latency_ms": (time.perf_counter() - started_at) * 1000,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "confidence": None,
                "probabilities": None,
                "raw_response": raw_response,
            }
        except Exception as error:
            if not _is_retryable(error) or attempt == config.MAX_ATTEMPTS - 1:
                logger.exception("OpenAI classification failed")
                return _empty_result(
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    raw_response=f"{type(error).__name__}: {error}",
                )
            delay = config.RETRY_DELAYS_SECONDS[attempt]
            logger.warning("Retryable OpenAI error; retrying in %s seconds", delay)
            time.sleep(delay)

    raise RuntimeError("OpenAI retry loop exited unexpectedly")


openai_runnable = RunnableLambda(classify)