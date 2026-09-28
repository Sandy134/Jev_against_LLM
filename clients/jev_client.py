import json
import logging
import os
import time
import uuid
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

import config

load_dotenv()
logger = logging.getLogger(__name__)


def _empty_result(*, latency_ms: float, raw_response: str) -> dict[str, Any]:
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
    body = {
        "model": config.JEV_MODEL,
        "state": {"ticket": question, "_nonce": str(uuid.uuid4())},
        "questions": {
            "category": {
                "type": "choice",
                "instructions": (
                    "Classify this support ticket into the correct category."
                ),
                "criteria": {option: option for option in options},
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
        "Content-Type": "application/json",
    }

    started_at = time.perf_counter()
    for attempt in range(config.MAX_ATTEMPTS):
        try:
            response = requests.post(
                config.JEV_ENDPOINT,
                headers=headers,
                json=body,
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                error = requests.HTTPError(
                    f"HTTP {response.status_code}: {response.text}",
                    response=response,
                )
                retryable = response.status_code == 429 or response.status_code >= 500
                if retryable and attempt < config.MAX_ATTEMPTS - 1:
                    delay = config.RETRY_DELAYS_SECONDS[attempt]
                    logger.warning("Retryable Jev response; retrying in %s seconds", delay)
                    time.sleep(delay)
                    continue
                logger.error("Jev request failed: %s", error)
                return _empty_result(
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    raw_response=str(error),
                )

            data = response.json()
            answer = data.get("answers", {}).get("category", {})
            usage = data.get("usage", {})
            predicted_answer = answer.get("choice")
            probabilities = answer.get("probabilities")
            raw_response = json.dumps(data)
            return {
                "predicted_answer": predicted_answer,
                "parsed_ok": predicted_answer in options,
                "latency_ms": (time.perf_counter() - started_at) * 1000,
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "cost_usd": float(usage.get("cost", 0.0)),
                "confidence": answer.get("confidence"),
                "probabilities": probabilities,
                "raw_response": raw_response,
            }
        except requests.RequestException as error:
            if attempt == config.MAX_ATTEMPTS - 1:
                logger.exception("Jev request failed")
                return _empty_result(
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    raw_response=f"{type(error).__name__}: {error}",
                )
            delay = config.RETRY_DELAYS_SECONDS[attempt]
            logger.warning("Retryable Jev network error; retrying in %s seconds", delay)
            time.sleep(delay)
        except (ValueError, TypeError, KeyError) as error:
            logger.exception("Could not parse Jev response")
            return _empty_result(
                latency_ms=(time.perf_counter() - started_at) * 1000,
                raw_response=f"{type(error).__name__}: {error}",
            )

    raise RuntimeError("Jev retry loop exited unexpectedly")


jev_runnable = RunnableLambda(classify)