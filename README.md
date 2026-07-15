# Persona Bench

Simulate realistic user personas against an AI chatbot/agent and catch
hallucinations, off-topic drift, and task failures before real users do.

## Why this matters
Teams shipping AI agents rarely test beyond "does it work when I chat
with it nicely." Persona Bench runs a battery of adversarial, confused,
distracted, and impatient users against your agent automatically, then
has a second LLM judge each conversation against a rubric.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app:app --reload
```

Open http://localhost:8000 and click "Run Evaluation."

## Project structure
- `personas.py` — the simulated user types (edit these to fit your demo)
- `target_agent.py` — the chatbot being tested (swap this to test ANY agent)
- `simulator.py` — runs multi-turn async conversations between persona and agent
- `judge.py` — LLM-as-judge scoring against a rubric (hallucination, task success, tone)
- `app.py` — FastAPI endpoints (`POST /run-eval`, `GET /results/{id}`)
- `static/index.html` — minimal frontend, plain fetch calls, no framework

## 5-day build plan
- **Day 1**: Get `simulator.py` working end-to-end for one persona, print transcript to console
- **Day 2**: Wire up `judge.py`, confirm structured JSON scoring works reliably
- **Day 3**: Wrap in FastAPI (`app.py`), get `/run-eval` + `/results` working, deploy to Railway/Render
- **Day 4**: Run against 2-3 target agent configs (one deliberately bad, one better) to show a before/after in the demo. Polish frontend.
- **Day 5**: Record 3-min demo (show it CATCHING a real hallucination live), build slide deck, submit

## Demo framing (per hackathon judging criteria)
Don't pitch this as "an eval framework." Pitch it as:
**"Before you ship your AI agent, know how it breaks."**
Show a concrete failure Persona Bench caught that a human tester would
have missed — that's your innovation + impact story in one shot.

## Deploying
Railway or Render both support this directly — push this repo, set
`ANTHROPIC_API_KEY` as an env var, and set the start command to:
```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Extending later (post-hackathon / toward Dec 15 agent harness)
- Swap in-memory `RESULTS` dict for SQLite/Postgres
- Add more personas, custom rubrics per business type
- Support pointing at arbitrary target agent URLs, not just the built-in one
- Add LangSmith/PromptFoo integration for eval tracking over time
