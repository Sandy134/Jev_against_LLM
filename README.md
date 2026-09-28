# Jev vs. OpenAI Classification Benchmark

This project compares Jev through OpenRouter's Decisions API with GPT-6 Sol through LangChain on the questions in `classification_questions.json`.

## Setup

Use Python 3.11 or newer, create/activate a virtual environment, and install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a `.env` file in the project root with both keys:

```text
OPENAI_API_KEY=your-openai-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

## Run

Run the experiment and then generate the report:

```powershell
.\.venv\Scripts\python.exe .\run_experiment.py
.\.venv\Scripts\python.exe .\report.py
```

Results are committed to `results.db` after each model call, so rerunning the experiment skips completed question/model/iteration combinations. The report prints accuracy, latency, cost, parse failures, and Jev confidence summaries, and saves `accuracy_by_difficulty.png` and `latency_by_model.png` in the project root.

## Live Dashboard

Start the local, read-only dashboard server:

```powershell
.\.venv\Scripts\python.exe .\dashboard_server.py
```

Open `http://127.0.0.1:8765`. The page refreshes every 30 seconds and lets you filter by difficulty, iteration, and projected request volume. All comparative KPIs and charts use successful parsed decisions only; API failures and unparsed responses are excluded.