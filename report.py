import logging

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import config
import db

MONTHLY_DAILY_VOLUME = 10_000
DAYS_PER_MONTH = 30
logger = logging.getLogger(__name__)


def main() -> None:
    db.initialize_database()
    with db.get_connection() as connection:
        results = pd.read_sql_query("SELECT * FROM results", connection)

    if results.empty:
        print("No results found. Run run_experiment.py first.")
        return

    print("Accuracy by model")
    print(results.groupby("model")["is_correct"].mean().rename("accuracy").to_string())

    print("\nAccuracy by model and difficulty")
    accuracy_by_difficulty = results.pivot_table(
        index="difficulty", columns="model", values="is_correct", aggfunc="mean"
    )
    print(accuracy_by_difficulty.to_string())

    parsed_mask = results["parsed_ok"].astype(bool)
    parsed_results = results[parsed_mask]
    latency_summary = parsed_results.groupby("model")["latency_ms"].quantile(
        [0.5, 0.95]
    ).unstack()
    latency_summary.columns = ["p50_ms", "p95_ms"]
    print("\nLatency percentiles for parsed responses")
    print(latency_summary.to_string())

    print("\nTotal measured cost and projected monthly cost")
    cost_summary = results.groupby("model").agg(
        total_cost_usd=("cost_usd", "sum"), calls=("id", "count")
    )
    cost_summary["projected_monthly_usd"] = (
        cost_summary["total_cost_usd"]
        / cost_summary["calls"].clip(lower=1)
        * MONTHLY_DAILY_VOLUME
        * DAYS_PER_MONTH
    )
    print(cost_summary.to_string())
    print(f"Projection volume: {MONTHLY_DAILY_VOLUME:,} requests/day")

    print("\nMean latency by model and iteration")
    latency_by_iteration = results.pivot_table(
        index="iteration", columns="model", values="latency_ms", aggfunc="mean"
    )
    print(latency_by_iteration.to_string())

    print("\nUnparsed responses for manual review")
    unparsed = results.loc[
        ~parsed_mask,
        ["question_id", "difficulty", "model", "iteration", "raw_response"],
    ]
    print(unparsed.to_string(index=False) if not unparsed.empty else "None")

    print("\nJev confidence by correctness")
    jev_results = results[
        (results["model"] == "jev") & results["confidence"].notna()
    ]
    confidence_summary = jev_results.groupby("is_correct")["confidence"].mean()
    confidence_summary.index = confidence_summary.index.map(
        {1: "correct", 0: "incorrect"}
    )
    print(confidence_summary.to_string() if not confidence_summary.empty else "No Jev confidence data")

    accuracy_by_difficulty.plot(kind="bar", rot=0)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy by Difficulty")
    plt.tight_layout()
    plt.savefig(config.PROJECT_DIR / "accuracy_by_difficulty.png", dpi=160)
    plt.close()

    if not parsed_results.empty:
        parsed_results.boxplot(column="latency_ms", by="model", grid=False)
        plt.suptitle("")
        plt.title("Latency by Model (Parsed Responses)")
        plt.ylabel("Latency (ms)")
        plt.tight_layout()
        plt.savefig(config.PROJECT_DIR / "latency_by_model.png", dpi=160)
        plt.close()
    else:
        logger.warning("No parsed responses available for the latency plot")

    print("\nSaved accuracy_by_difficulty.png and latency_by_model.png")


if __name__ == "__main__":
    main()