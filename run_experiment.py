import json
import logging
import time

from dotenv import load_dotenv

load_dotenv()

import config
import db
from clients.jev_client import jev_runnable
from clients.openai_client import openai_runnable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    with config.QUESTIONS_PATH.open(encoding="utf-8") as questions_file:
        questions = json.load(questions_file)

    db.initialize_database()
    completed = db.get_completed_runs()
    clients = (("openai", openai_runnable), ("jev", jev_runnable))
    pending = [
        (question, model, runnable, iteration)
        for question in questions
        for model, runnable in clients
        for iteration in range(1, config.N_ITERATIONS + 1)
        if (question["id"], model, iteration) not in completed
    ]

    total_calls = len(pending)
    for call_number, (question, model, runnable, iteration) in enumerate(
        pending, start=1
    ):
        result = runnable.invoke(
            {"question": question["question"], "options": question["options"]}
        )
        is_correct = bool(
            result["parsed_ok"]
            and result["predicted_answer"] == question["correct_answer"]
        )
        db.insert_result(
            question_id=question["id"],
            difficulty=question["difficulty"],
            model=model,
            iteration=iteration,
            predicted_answer=result["predicted_answer"],
            correct_answer=question["correct_answer"],
            is_correct=is_correct,
            parsed_ok=result["parsed_ok"],
            latency_ms=result["latency_ms"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=result["cost_usd"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            raw_response=result["raw_response"],
        )
        status = "correct" if is_correct else "incorrect"
        answer = result["predicted_answer"] or "unparsed"
        print(
            f"[{call_number}/{total_calls}] {model} {question['id']} "
            f"iter{iteration} -> {answer} ({status}) "
            f"{result['latency_ms']:.0f}ms",
            flush=True,
        )
        if call_number < total_calls:
            time.sleep(0.2)

    if total_calls == 0:
        print("All question/model/iteration combinations are already in results.db.")


if __name__ == "__main__":
    main()