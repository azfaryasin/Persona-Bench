# AGENTS.md

Context for AI coding agents working on this repository.

## What this project is

Persona Bench is an AI agent evaluation harness. It simulates realistic user personas talking to a target chatbot/agent, then scores the resulting conversations using a multi-judge LLM ensemble. The goal is to catch hallucinations, safety issues, and quality problems before a real user does.

## Architecture

Single FastAPI backend serving a single-file vanilla JS frontend (`static/index.html`) — no separate frontend framework, no build step. Keep it this way; a prior attempt to split this into a Next.js frontend broke the deployment and introduced a command-injection vulnerability. Do not reintroduce a separate frontend service.

**Core flow**: `simulator.py` runs a persona against a target agent (`target_agent.py` for built-in demo configs, `custom_agent.py` for user-supplied external endpoints) → produces a transcript → `judge.py` scores it via a multi-judge ensemble → `report.py` aggregates results across personas into a human-readable report.

## Conventions

- **LLM calls**: always go through `llm_client.py`'s `call_with_retry()` — never call `client.chat.completions.create()` directly from another module. This wrapper handles rate-limit retries with exponential backoff.
- **Content extraction**: always use `extract_content()` in `llm_client.py` to pull text from an LLM response — never call `.content.strip()` directly. Some providers (notably NVIDIA NIM) return `None` content on `finish_reason == "content_filter"`, which crashes naive `.strip()` calls.
- **Provider portability**: this app is built to be provider-agnostic via `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` environment variables. Don't hardcode a model name or base URL anywhere in application code — read from environment only.
- **Score parsing**: judge/report output field types can vary between providers (a score field may come back as a raw int in some cases, a nested dict in others). Any code reading judge output fields should type-check before calling `.get()` on it.

## Known past bugs (avoid reintroducing)

1. **Dockerfile CMD must use shell form** — `CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]`. Exec-form JSON array CMD does not expand `${PORT}`, which breaks deployment on Railway.
2. **Command injection**: a prior version had a Next.js API route that shelled out to a Python CLI using `execSync` with an unsanitized query parameter. Never shell out to subprocess calls with user-supplied input.
3. **NaN fields in reports**: judge output must always return every required field (Adaptation, Brand Risk, etc.), even if the value is a conservative default — never omit a field, as this produces NaN in aggregated report averages.
4. **Confidence must reflect completion rate**: if fewer than ~75% of persona tests complete successfully, the report's confidence level and overall verdict must reflect that — never present a high-confidence numeric score based on a majority-failed run.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv/bin/activate.fish for fish shell
pip install -r requirements.txt
export OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=...
uvicorn app:app --reload
```

## Testing changes

There is no formal test suite yet. When making changes:
1. Run the app locally and manually exercise the affected endpoint(s) via the UI or `curl`
2. For judge/scoring changes, run a full evaluation and read the actual report output — don't just confirm the code runs without error, confirm the scores make sense
3. For provider/model changes, test across at least two different niches, including one that touches sensitive topics (e.g. healthcare), since content-filtering behavior varies by provider

## Things to be cautious about

- Multi-Judge Ensemble and Consistency Check modes multiply API call volume (up to ~6-8x baseline) — be mindful of rate limits and token budgets when testing these paths repeatedly
- The Bot Builder feature (`custom_agent.py` + a standalone bot server) is separate from the core evaluation flow — if it's not working, it does not block the core weak/improved evaluation and report generation
