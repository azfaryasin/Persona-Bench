# Persona Bench — Agent Guide

## What This Is
Persona Bench evaluates AI customer-support chatbots by simulating adversarial user personas, running multi-turn conversations, and scoring each transcript with a separate LLM judge. It's designed to catch hallucinations, task failures, and topic drift that human QA would miss.

## Architecture (6 files that matter)

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  personas.py │────▶│ simulator.py │────▶│ target_agent.py  │
│  (who to     │     │ (runs multi- │     │ (built-in bots OR│
│   simulate)  │     │  turn convo) │     │  custom_agent)   │
└─────────────┘     └──────┬───────┘     └──────────────────┘
                           │ transcript           ▲
                           ▼                      │
                    ┌──────────────┐       ┌──────────────┐
                    │  judge.py    │       │ custom_agent │
                    │ (LLM-as-judge│       │  .py (BYO)   │
                    │  scoring)    │       └──────────────┘
                    └──────────────┘
```

- **`llm_client.py`** — Shared AsyncOpenAI client + `call_with_retry()` with exponential backoff (5s/10s/20s, catches `RateLimitError`). Reads `OPENAI_API_KEY` from env; falls back to `/etc/.z-ai-config` for local dev.
- **`personas.py`** — Dict of 4 personas, each with a `system_prompt` and `opening_message`. Edit these to target different bot categories. No code changes needed elsewhere.
- **`target_agent.py`** — Two built-in configs (`weak` and `improved`). When `target_config="__custom__"`, routes through `custom_agent.call_custom_target()`. Exports `_custom_target_config` dict for app.py to inject.
- **`custom_agent.py`** — "Bring Your Own Agent" support. Calls any HTTP chatbot API using a configurable body template with `{{message}}`/`{{history}}` placeholders and a dot-notation `response_path` to extract the reply. 15s timeout, defensive error handling, returns error strings instead of crashing.
- **`simulator.py`** — Orchestrates conversation: persona sends message → target agent replies → persona responds → repeat. Maintains dual message views (target's POV vs persona's POV).
- **`judge.py`** — Takes a completed transcript and returns structured JSON: hallucination_detected, task_completed, stayed_on_topic, overall_verdict (pass/fail), notes.
- **`app.py`** — FastAPI server. `POST /run-eval` accepts `target_config` (weak/improved) or `custom_target` dict (url, headers, body_template, response_path). Validates URLs and routes accordingly.
- **`static/index.html`** — Vanilla JS frontend. Built-in buttons + collapsible "Test Your Own Agent" section with form fields and inline validation. Side-by-side comparison panels with pass/fail/hallucination counts.

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

## Testing Your Own Agent (BYO)
Persona Bench can test any HTTP chatbot API, not just the built-in configs.

### The Contract
Your agent endpoint must:
1. Accept a JSON POST with the user's message
2. Return a JSON response with the reply text somewhere in it

### How to Configure
Use the "Test Your Own Agent" panel in the UI, or POST directly:

```bash
curl -X POST /run-eval -H "Content-Type: application/json" -d '{
  "persona_keys": ["adversarial"],
  "num_turns": 3,
  "target_config": "weak",
  "custom_target": {
    "url": "https://your-agent-api.com/chat",
    "headers": {"Authorization": "Bearer YOUR_KEY"},
    "body_template": "{\"message\": \"{{message}}\"}",
    "response_path": "reply"
  }
}'
```

### Template Placeholders
- `{{message}}` — the latest user message (string). Put this **inside** JSON string quotes.
- `{{history}}` — the full conversation as a JSON array of `{role, content}` objects. Put this **outside** quotes.

Examples:
| API shape | body_template | response_path |
|-----------|--------------|---------------|
| `{"message": "hi", "reply": "hello"}` | `{"message": "{{message}}"}` | `reply` |
| OpenAI-compatible | `{"messages": {{history}}, "model": "gpt-4"}` | `choices.0.message.content` |
| Simple wrapper | `{"prompt": "{{message}}", "n": 1}` | `completions.0.text` |

### Error Handling
If your agent is unreachable, times out (15s limit), returns non-JSON, or the `response_path` doesn't exist in the response, the transcript will contain the error message as the agent's "reply" and the judge will score accordingly. The eval run continues — one broken agent response doesn't crash the whole evaluation.