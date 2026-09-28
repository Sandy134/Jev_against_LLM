import json
import logging
import math
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from statistics import fmean
from urllib.parse import parse_qs, urlsplit

from dotenv import load_dotenv

load_dotenv()

import config
import db

HOST = "127.0.0.1"
PORT = 8765
MODEL_LABELS = {"jev": "Jev 1.13", "openai": "GPT-6 Sol"}
DIFFICULTIES = ("easy", "medium", "hard")
ITERATIONS = (1, 2, 3)
logger = logging.getLogger("dashboard")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def get_dashboard_data(
    *, difficulty: str = "all", iteration: int | None = None, daily_volume: int = 10_000
) -> dict[str, object]:
    where = ["parsed_ok = 1", "predicted_answer IS NOT NULL"]
    parameters: list[object] = []
    if difficulty in DIFFICULTIES:
        where.append("difficulty = ?")
        parameters.append(difficulty)
    if iteration in ITERATIONS:
        where.append("iteration = ?")
        parameters.append(iteration)

    with db.get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT model, difficulty, iteration, is_correct, latency_ms,
                   cost_usd, confidence, timestamp
            FROM results
            WHERE {' AND '.join(where)}
            ORDER BY timestamp, id
            """,
            parameters,
        ).fetchall()

    by_model: dict[str, list[dict[str, object]]] = {
        model: [] for model in MODEL_LABELS
    }
    for row in rows:
        if row["model"] in by_model:
            by_model[row["model"]].append(dict(row))

    summaries: dict[str, dict[str, object]] = {}
    for model, model_rows in by_model.items():
        count = len(model_rows)
        correct = sum(bool(row["is_correct"]) for row in model_rows)
        latencies = [float(row["latency_ms"]) for row in model_rows]
        total_cost = sum(float(row["cost_usd"]) for row in model_rows)
        mean_cost = total_cost / count if count else None
        summaries[model] = {
            "key": model,
            "label": MODEL_LABELS[model],
            "calls": count,
            "correct": correct,
            "accuracy": correct / count if count else None,
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "mean_latency_ms": _mean(latencies),
            "total_cost_usd": total_cost,
            "mean_cost_usd": mean_cost,
            "cost_per_correct_usd": total_cost / correct if correct else None,
            "projected_daily_cost_usd": (
                mean_cost * daily_volume if mean_cost is not None else None
            ),
            "projected_monthly_cost_usd": (
                mean_cost * daily_volume * 30 if mean_cost is not None else None
            ),
        }

    by_difficulty = []
    for level in DIFFICULTIES:
        item: dict[str, object] = {"difficulty": level}
        for model, model_rows in by_model.items():
            difficulty_rows = [row for row in model_rows if row["difficulty"] == level]
            item[f"{model}_calls"] = len(difficulty_rows)
            item[f"{model}_accuracy"] = (
                sum(bool(row["is_correct"]) for row in difficulty_rows)
                / len(difficulty_rows)
                if difficulty_rows
                else None
            )
        by_difficulty.append(item)

    by_iteration = []
    for iteration_number in ITERATIONS:
        item = {"iteration": iteration_number}
        for model, model_rows in by_model.items():
            iteration_rows = [
                row for row in model_rows if row["iteration"] == iteration_number
            ]
            item[f"{model}_latency_ms"] = _mean(
                [float(row["latency_ms"]) for row in iteration_rows]
            )
        by_iteration.append(item)

    jev_rows = by_model["jev"]
    confidence_correct = [
        float(row["confidence"])
        for row in jev_rows
        if row["confidence"] is not None and bool(row["is_correct"])
    ]
    confidence_incorrect = [
        float(row["confidence"])
        for row in jev_rows
        if row["confidence"] is not None and not bool(row["is_correct"])
    ]

    timestamps = [str(row["timestamp"]) for row in rows if row["timestamp"]]
    return {
        "models": summaries,
        "by_difficulty": by_difficulty,
        "by_iteration": by_iteration,
        "confidence": {
            "correct": _mean(confidence_correct),
            "incorrect": _mean(confidence_incorrect),
        },
        "filters": {
            "difficulty": difficulty if difficulty in DIFFICULTIES else "all",
            "iteration": iteration if iteration in ITERATIONS else None,
            "daily_volume": daily_volume,
        },
        "successful_rows": len(rows),
        "updated_at": max(timestamps) if timestamps else None,
        "source": "parsed_ok = 1 and predicted_answer is not null",
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(config.PROJECT_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed_url = urlsplit(self.path)
        if parsed_url.path == "/api/dashboard":
            self._serve_dashboard_data(parse_qs(parsed_url.query))
            return
        if parsed_url.path == "/":
            self.path = "/dashboard.html"
        super().do_GET()

    def _serve_dashboard_data(self, query: dict[str, list[str]]) -> None:
        difficulty = query.get("difficulty", ["all"])[0]
        iteration_value = query.get("iteration", ["all"])[0]
        volume_value = query.get("daily_volume", ["10000"])[0]
        try:
            iteration = int(iteration_value) if iteration_value != "all" else None
            daily_volume = min(max(int(volume_value), 0), 1_000_000)
        except ValueError:
            self.send_error(400, "iteration and daily_volume must be integers")
            return

        payload = json.dumps(
            get_dashboard_data(
                difficulty=difficulty,
                iteration=iteration,
                daily_volume=daily_volume,
            )
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    db.initialize_database()
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Benchmark dashboard: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()