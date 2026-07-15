# Persona Bench — Agent Guide

## What This Is
Persona Bench evaluates AI customer-support chatbots by simulating adversarial user personas, running multi-turn conversations, and scoring each transcript with a separate LLM judge. It's designed to catch hallucinations, task failures, and topic drift that human QA would miss.

## Architecture (5 files that matter)

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  personas.py │────▶│ simulator.py │────▶│ target_agent.py│
│  (who to     │     │ (runs multi- │     │ (the bot under │
│   simulate)  │     │  turn convo) │     │   test)        │
└─────────────┘     └──────┬───────┘     └────────────────┘
                           │ transcript
                           ▼
                    ┌──────────────┐
                    │  judge.py    │
                    │ (LLM-as-judge│
                    │  scoring)    │
                    └──────────────┘
```

- **`llm_client.py`** — Shared AsyncOpenAI client. Reads `OPENAI_API_KEY` from env; falls back to `/etc/.z-ai-config` for local dev. All three consumer modules import `client` and `MODEL` from here.
- **`personas.py`** — Dict of 4 personas, each with a `system_prompt` and `opening_message`. Edit these to target different bot categories. No code changes needed elsewhere.
- **`target_agent.py`** — Two configs (`weak` and `improved`) representing a BoxLoop subscription-box support bot. The `call_target_agent()` function takes `conversation_history` + `target_config`. Swap or add configs here.
- **`simulator.py`** — Orchestrates conversation: persona sends message → target agent replies → persona responds → repeat. Maintains dual message views (target's POV vs persona's POV). Imports `call_target_agent` from target_agent.
- **`judge.py`** — Takes a completed transcript and returns structured JSON: hallucination_detected, task_completed, stayed_on_topic, overall_verdict (pass/fail), notes.
- **`app.py`** — FastAPI server. `POST /run-eval` starts an async eval (accepts `target_config`, `persona_keys`, `num_turns`). `GET /results/{run_id}` polls status. `GET /runs` lists all runs. `GET /personas` lists available personas.
- **`static/index.html`** — Vanilla JS frontend. Two buttons ("Run Weak Bot" / "Run Improved Bot") store run IDs per config and render side-by-side comparison panels with pass/fail/hallucination counts. Expandable transcripts per persona.

## Key Decisions
- **OpenAI SDK** (AsyncOpenAI) — not Anthropic. Uses `gpt-4o-mini` by default (cheap, fast).
- **System prompt in messages array** — OpenAI format, not Anthropic's separate `system` param.
- **Error handling** — Each persona eval is try/except'd. One failure doesn't crash the run; it shows as "error" verdict.
- **In-memory storage** — `RESULTS` dict in app.py. Pruned to last 20 runs. Fine for hackathon; swap for SQLite/Postgres later.
- **No framework** — Plain Python/FastAPI backend, vanilla JS frontend. Single `index.html`.

## How to Run
```bash
export OPENAI_API_KEY=sk-...
uvicorn app:app --host 0.0.0.0 --port $PORT --reload
```

## How to Demo (the 3-click story)
1. Click **"Run Weak Bot"** — the hallucination-prone support bot gets tested by all 4 personas. Watch the hallucination count climb.
2. Click **"Run Improved Bot"** — the safe-fallback bot gets tested. Watch hallucinations drop to zero.
3. The side-by-side panels make the improvement obvious at a glance. Expand any card to read the full transcript and see exactly what the weak bot fabricated.

## Adding a New Persona
Add an entry to `PERSONAS` in `personas.py`:
```python
"my_persona": {
    "name": "Display Name",
    "system_prompt": "Instructions for how to role-play...",
    "opening_message": "First thing this user would say...",
},
```

## Adding a New Target Config
Add an entry to `TARGET_CONFIGS` in `target_agent.py`:
```python
"my_config": {
    "name": "My Config Label",
    "system_prompt": "System prompt for this bot version...",
},
```