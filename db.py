import json
import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any
from collections.abc import Iterator

import config


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(config.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                model TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                predicted_answer TEXT,
                correct_answer TEXT NOT NULL,
                is_correct BOOLEAN NOT NULL,
                parsed_ok BOOLEAN NOT NULL,
                latency_ms REAL NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                confidence REAL,
                probabilities_json TEXT,
                timestamp TEXT NOT NULL,
                raw_response TEXT,
                UNIQUE (question_id, model, iteration)
            )
            """
        )
        connection.commit()


def get_completed_runs() -> set[tuple[str, str, int]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT question_id, model, iteration
            FROM results
            WHERE parsed_ok = 1
               OR predicted_answer IS NOT NULL
               OR input_tokens > 0
               OR output_tokens > 0
            """
        ).fetchall()
    return {(row[0], row[1], row[2]) for row in rows}


def insert_result(
    *,
    question_id: str,
    difficulty: str,
    model: str,
    iteration: int,
    predicted_answer: str | None,
    correct_answer: str,
    is_correct: bool,
    parsed_ok: bool,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    confidence: float | None,
    probabilities: Mapping[str, Any] | None,
    raw_response: str,
) -> None:
    probabilities_json = (
        json.dumps(probabilities) if probabilities is not None else None
    )
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO results (
                question_id, difficulty, model, iteration, predicted_answer,
                correct_answer, is_correct, parsed_ok, latency_ms, input_tokens,
                output_tokens, cost_usd, confidence, probabilities_json, timestamp,
                raw_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            ON CONFLICT (question_id, model, iteration) DO UPDATE SET
                difficulty = excluded.difficulty,
                predicted_answer = excluded.predicted_answer,
                correct_answer = excluded.correct_answer,
                is_correct = excluded.is_correct,
                parsed_ok = excluded.parsed_ok,
                latency_ms = excluded.latency_ms,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                cost_usd = excluded.cost_usd,
                confidence = excluded.confidence,
                probabilities_json = excluded.probabilities_json,
                timestamp = CURRENT_TIMESTAMP,
                raw_response = excluded.raw_response
            """,
            (
                question_id,
                difficulty,
                model,
                iteration,
                predicted_answer,
                correct_answer,
                is_correct,
                parsed_ok,
                latency_ms,
                input_tokens,
                output_tokens,
                cost_usd,
                confidence,
                probabilities_json,
                raw_response,
            ),
        )
        connection.commit()