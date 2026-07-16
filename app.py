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

from personas import PERSONAS, NICHE_LABELS, ALL_NICHES, get_personas_for_niche
from simulator import run_persona_conversation
from judge import judge_transcript, JUDGE_DIMENSIONS
from target_agent import TARGET_CONFIGS
from report import generate_report, optimize_system_prompt

app = FastAPI(title="Persona Bench")

# In-memory store — fine for a hackathon demo
RESULTS: dict[str, dict] = {}
ALL_RUN_IDS: list[str] = []  # ordered list for frontend to list runs


class RunEvalRequest(BaseModel):
    persona_keys: list[str] = []  # default: all personas for the selected niche
    num_turns: int = 5
    target_config: str = "weak"
    niche: str = "general"
    ensemble: bool = True  # multi-judge ensemble; set False for legacy single judge
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
    # Resolve personas for the niche
    try:
        niche_personas = await get_personas_for_niche(req.niche)
    except Exception as e:
        return {"error": f"Failed to load personas for niche '{req.niche}': {e}"}

    if not niche_personas:
        return {"error": f"No personas available for niche '{req.niche}'. Try a built-in niche."}

    # If persona_keys not specified, run all for this niche
    persona_keys = req.persona_keys or list(niche_personas.keys())

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
    niche_label = NICHE_LABELS.get(req.niche, req.niche)
    RESULTS[run_id] = {
        "status": "running",
        "target_config": config_key,
        "target_config_label": config_label,
        "niche": req.niche,
        "niche_label": niche_label,
        "persona_keys": persona_keys,
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
        for pk in persona_keys:
            try:
                if use_custom:
                    convo = await run_persona_conversation(pk, "__custom__", req.num_turns, personas_dict=niche_personas)
                else:
                    convo = await run_persona_conversation(pk, req.target_config, req.num_turns, personas_dict=niche_personas)
                verdict = await judge_transcript(convo["persona_name"], convo["transcript"], niche=req.niche, ensemble=req.ensemble)
                scored.append({**convo, "verdict": verdict})
            except Exception as e:
                # Don't let one bad persona crash the whole eval
                scored.append({
                    "persona_key": pk,
                    "persona_name": niche_personas.get(pk, {}).get("name", pk),
                    "transcript": [],
                    "verdict": {
                        "hallucination_detected": None,
                        "overall_verdict": "error",
                        "hallucination_evidence": "",
                        "notes": f"Error during evaluation: {str(e)}",
                        "dimensions": {},
                        "niche": req.niche,
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
                "niche": r.get("niche", "general"),
                "niche_label": r.get("niche_label", ""),
            })
    return runs


@app.get("/niches")
async def list_niches():
    """Return all available niches with labels and whether they have built-in personas."""
    return {
        niche: {
            "label": label,
            "has_builtin": niche in ("general", "customer_support") or niche in {"healthcare", "e_commerce", "legal"},
        }
        for niche, label in NICHE_LABELS.items()
    }


@app.get("/personas")
async def list_personas(niche: str = "general"):
    personas = await get_personas_for_niche(niche)
    return {k: v["name"] for k, v in personas.items()}


# ---------------------------------------------------------------------------
# Report endpoints
# ---------------------------------------------------------------------------

# (report page is now served by the SPA in static/index.html)


@app.get("/report/{run_id}/data")
async def get_report_data(run_id: str):
    """Return the evaluation data for a run (used by the report page JS)."""
    data = RESULTS.get(run_id)
    if not data:
        return {"error": "Run not found"}
    if data["status"] != "done":
        return {"error": "Run is still in progress", "status": data["status"]}
    return data


@app.post("/report/{run_id}/generate")
async def generate_report_endpoint(run_id: str):
    """Generate the LLM-powered report for a completed run."""
    data = RESULTS.get(run_id)
    if not data:
        return {"error": "Run not found"}
    if data["status"] != "done":
        return {"error": "Run is still in progress", "status": data["status"]}

    try:
        report = await generate_report(data)
        return report
    except Exception as e:
        return {"error": f"Report generation failed: {e}"}


@app.post("/report/{run_id}/optimize-prompt")
async def optimize_prompt_endpoint(run_id: str, body: dict | None = None):
    """Generate an optimized system prompt based on failures."""
    data = RESULTS.get(run_id)
    if not data:
        return {"error": "Run not found"}
    if data["status"] != "done":
        return {"error": "Run is still in progress", "status": data["status"]}

    # Extract failures
    failures = [r for r in data["results"] if r.get("verdict", {}).get("overall_verdict") == "fail"]
    if not failures:
        failures = data["results"]  # if no hard fails, use all results for improvement

    current_prompt = None
    if body and body.get("current_prompt"):
        current_prompt = body["current_prompt"]

    try:
        improved = await optimize_system_prompt(
            niche=data.get("niche", "general"),
            failures=failures,
            current_prompt=current_prompt,
        )
        return {"optimized_prompt": improved}
    except Exception as e:
        return {"error": f"Prompt optimization failed: {e}"}


# ---------------------------------------------------------------------------
# Serve the single-page HTML frontend
# ---------------------------------------------------------------------------
import os

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
