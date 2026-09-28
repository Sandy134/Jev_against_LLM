# test_jev.py
import os
import json
import time
import uuid
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
print("Key loaded:", bool(api_key), api_key[:8] + "..." if api_key else None)

JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"

body = {
    "model": "typesafe/jev-1.13",
    "state": {
        "ticket": "How do I reset my password?",
        "_nonce": str(uuid.uuid4()),
    },
    "questions": {
        "category": {
            "type": "choice",
            "instructions": "Classify this support ticket into the correct category.",
            "criteria": {
                "account_management": "account_management",
                "billing": "billing",
                "technical_support": "technical_support",
            },
        }
    },
}

start = time.perf_counter()
resp = requests.post(
    JEV_ENDPOINT,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json=body,
    timeout=30,
)
latency_ms = (time.perf_counter() - start) * 1000

print("\nStatus:", resp.status_code)
print("Latency:", round(latency_ms, 1), "ms")

try:
    data = resp.json()
    print("\nFull response:")
    print(json.dumps(data, indent=2))

    answer = data.get("answers", {}).get("category", {})
    print("\nChoice:", answer.get("choice"))
    print("Confidence:", answer.get("confidence"))
    print("Probabilities:", answer.get("probabilities"))
    print("Usage:", data.get("usage"))
except Exception as e:
    print("\nCould not parse JSON. Raw text:")
    print(resp.text)
    print("Parse error:", e)