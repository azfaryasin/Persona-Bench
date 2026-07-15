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
export OPENAI_API_KEY=sk-...
uvicorn app:app --reload
```

Open http://localhost:8000 and click "Run Weak Bot" or "Run Improved Bot."

## Project structure
- `llm_client.py` — shared OpenAI client with retry/backoff
- `personas.py` — the simulated user types (edit these to fit your demo)
- `target_agent.py` — built-in chatbot configs (weak/improved) + custom agent routing
- `custom_agent.py` — "Bring Your Own Agent" HTTP caller with body templates
- `simulator.py` — runs multi-turn async conversations between persona and agent
- `judge.py` — LLM-as-judge scoring against a rubric (hallucination, task success, tone)
- `app.py` — FastAPI endpoints (`POST /run-eval`, `GET /results/{id}`, `GET /runs`)
- `static/index.html` — frontend with side-by-side comparison + BYO agent panel

## Testing your own agent

Point Persona Bench at any HTTP chatbot API using the "Test Your Own Agent"
panel in the UI. You need three things:

1. **Endpoint URL** — your agent's HTTP endpoint
2. **Body Template** — a JSON string with `{{message}}` (latest user msg) and/or `{{history}}` (full conversation array) placeholders
3. **Response Path** — dot-notation to extract the reply (e.g. `reply`, `choices.0.message.content`)

Example for a simple agent:
- URL: `https://my-bot.example.com/chat`
- Body Template: `{"message": "{{message}}"}`
- Response Path: `reply`

## Deploying
Railway or Render both support this directly — push this repo, set
`OPENAI_API_KEY` as an env var. The Dockerfile handles the rest.

## Demo framing (per hackathon judging criteria)
Don't pitch this as "an eval framework." Pitch it as:
**"Before you ship your AI agent, know how it breaks."**
Show a concrete failure Persona Bench caught that a human tester would
have missed — that's your innovation + impact story in one shot.