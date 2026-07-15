"""
FastAPI app: the endpoints the frontend calls.

Run locally:
    export OPENAI_API_KEY=sk-...
    uvicorn app:app --reload

Then open http://localhost:8000
"""

import uuid
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from personas import PERSONAS
from simulator import run_persona_conversation
from judge import judge_transcript
from target_agent import TARGET_CONFIGS

app = FastAPI(title="Persona Bench")

# In-memory store — fine for a hackathon demo
RESULTS: dict[str, dict] = {}
ALL_RUN_IDS: list[str] = []  # ordered list for frontend to list runs


class RunEvalRequest(BaseModel):
    persona_keys: list[str] = list(PERSONAS.keys())  # default: run all
    num_turns: int = 5
    target_config: str = "weak"
    custom_target: dict | None = None  # BYO agent: {url, method, headers, body_template, response_path}


def _validate_url(url: str) -> str | None:
    """Return an error message if the URL is invalid, else None."""
    if not url:
        return "custom_target.url is required."
    if not url.startswith(("http://", "https://")):
        return "custom_target.url must start with http:// or https://"
    return None


@app.post("/run-eval")
async def run_eval(req: RunEvalRequest):
    # If custom_target is provided, it takes priority over target_config
    use_custom = req.custom_target is not None

    if use_custom:
        # Validate custom_target
        ct = req.custom_target
        url_err = _validate_url(ct.get("url", ""))
        if url_err:
            return {"error": url_err}

        run_label = f"Custom: {ct['url'][:50]}"
        if ct.get("response_path"):
            run_label += f" (→ {ct['response_path']})"
        config_key = "custom"
        config_label = run_label
    else:
        # Validate target_config (built-in weak/improved)
        if req.target_config not in TARGET_CONFIGS:
            return {"error": f"Unknown target_config. Must be one of: {list(TARGET_CONFIGS.keys())}"}
        config_key = req.target_config
        config_label = TARGET_CONFIGS[req.target_config]["name"]

    run_id = str(uuid.uuid4())
    RESULTS[run_id] = {
        "status": "running",
        "target_config": config_key,
        "target_config_label": config_label,
        "persona_keys": req.persona_keys,
        "num_turns": req.num_turns,
        "results": [],
        "error": None,
    }
    ALL_RUN_IDS.append(run_id)
    # Keep only the last 20 runs to avoid unbounded memory growth
    while len(ALL_RUN_IDS) > 20:
        old_id = ALL_RUN_IDS.pop(0)
        RESULTS.pop(old_id, None)

    async def _execute():
        from custom_agent import call_custom_target

        scored = []
        for pk in req.persona_keys:
            try:
                if use_custom:
                    # Patch the simulator to use custom agent
                    convo = await run_persona_conversation(pk, "__custom__", req.num_turns)
                else:
                    convo = await run_persona_conversation(pk, req.target_config, req.num_turns)
                verdict = await judge_transcript(convo["persona_name"], convo["transcript"])
                scored.append({**convo, "verdict": verdict})
            except Exception as e:
                # Don't let one bad persona crash the whole eval
                scored.append({
                    "persona_key": pk,
                    "persona_name": PERSONAS.get(pk, {}).get("name", pk),
                    "transcript": [],
                    "verdict": {
                        "hallucination_detected": None,
                        "task_completed": None,
                        "stayed_on_topic": None,
                        "tone_appropriate": None,
                        "overall_verdict": "error",
                        "hallucination_evidence": "",
                        "notes": f"Error during evaluation: {str(e)}",
                    },
                })
        RESULTS[run_id]["status"] = "done"
        RESULTS[run_id]["results"] = scored

    # If custom, inject the custom_target config into target_agent so
    # the simulator's call_target_agent uses it
    if use_custom:
        from target_agent import _custom_target_config
        _custom_target_config["config"] = req.custom_target

    asyncio.create_task(_execute())
    return {"run_id": run_id, "status": "started", "target_config": config_key, "target_config_label": config_label}


@app.get("/results/{run_id}")
async def get_results(run_id: str):
    return RESULTS.get(run_id, {"status": "not_found"})


@app.get("/runs")
async def list_runs():
    """Return metadata for all runs (without full results) for the comparison UI."""
    runs = []
    for rid in ALL_RUN_IDS:
        r = RESULTS.get(rid)
        if r:
            runs.append({
                "run_id": rid,
                "status": r["status"],
                "target_config": r["target_config"],
                "target_config_label": r.get("target_config_label", ""),
            })
    return runs


@app.get("/personas")
async def list_personas():
    return {k: v["name"] for k, v in PERSONAS.items()}


# Serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")