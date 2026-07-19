"""
Report generator: uses LLM to produce executive-level analysis
of evaluation results and optimize system prompts.
"""

import json
from llm_client import call_with_retry, MODEL, extract_content


async def generate_report(evaluation_data: dict) -> dict:
    """
    Given a full evaluation run result (with results[].verdict),
    call the LLM to produce a rich analysis report.

    Returns a dict with:
      - executive_summary
      - key_strengths (list)
      - critical_gaps (list, each with {issue, severity, evidence})
      - action_items (list, each with {priority, action, rationale})
      - risk_assessment (list, each with {risk, level})
      - deployment_recommendation (string)
      - one_liner_pitch (string)
    """
    niche = evaluation_data.get("niche", "general")
    niche_label = evaluation_data.get("niche_label", "General")
    target_label = evaluation_data.get("target_config_label", "Target Agent")
    results = evaluation_data.get("results", [])

    # Detect if ensemble data is available
    has_ensemble = any(r.get("verdict", {}).get("ensemble") for r in results)

    # Build a concise summary of each persona test for the LLM
    test_summaries = []
    ensemble_averages = {"quality": [], "safety": [], "persona": [], "business": []}
    confidence_dist = {"high": 0, "medium": 0, "low": 0}
    safety_flags_collected = []

    for r in results:
        v = r.get("verdict", {})
        dims = v.get("dimensions", {})
        failed_dims = [k.replace("_", " ") for k, dv in dims.items() if dv and dv.get("pass") is False]
        line = (
            f"- {r.get('persona_name', 'Unknown')}: "
            f"verdict={v.get('overall_verdict', '?')}, "
            f"hallucination={v.get('hallucination_detected')}, "
            f"notes={v.get('notes', '')}"
            + (f", failed_dimensions={failed_dims}" if failed_dims else "")
        )

        # Append ensemble data if available
        ens = v.get("ensemble")
        if ens:
            js = ens.get("judge_scores", {})
            line += f"\n  Ensemble: overall={ens.get('overall_score', '?')}/100, confidence={ens.get('confidence', '?')}"
            for jname, jval in js.items():
                line += f", {jname}={jval}"
                if isinstance(jval, (int, float)):
                    ensemble_averages.setdefault(jname, []).append(jval)
            line += f"\n  Priority fix: {ens.get('priority_fix', 'N/A')}"
            conf = ens.get("confidence", "medium")
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

            # Collect safety flags
            safety_judge = ens.get("judges", {}).get("safety", {})
            for flag in safety_judge.get("flags", []):
                if flag.get("type") and flag.get("type") != "none":
                    safety_flags_collected.append(f"    - [{flag.get('severity', '?')}] {flag.get('type')}: {flag.get('explanation', '')[:80]}")

        test_summaries.append(line)

    summary_text = "\n".join(test_summaries)

    # Compute numeric stats
    total = len(results)
    passes = sum(1 for r in results if r.get("verdict", {}).get("overall_verdict") == "pass")
    fails = sum(1 for r in results if r.get("verdict", {}).get("overall_verdict") == "fail")
    hallucinations = sum(1 for r in results if r.get("verdict", {}).get("hallucination_detected") is True)
    pass_rate = round((passes / total * 100) if total > 0 else 0, 1)

    # Compute dimension-level pass rates
    dim_scores = {}
    for r in results:
        dims = r.get("verdict", {}).get("dimensions", {})
        for dk, dv in dims.items():
            if dk not in dim_scores:
                dim_scores[dk] = {"pass": 0, "fail": 0, "total": 0}
            dim_scores[dk]["total"] += 1
            if dv and dv.get("pass") is True:
                dim_scores[dk]["pass"] += 1
            elif dv and dv.get("pass") is False:
                dim_scores[dk]["fail"] += 1

    dim_summary = "\n".join(
        f"  - {k.replace('_', ' ')}: {v['pass']}/{v['total']} passed ({round(v['pass']/v['total']*100,1) if v['total'] else 0}%)"
        for k, v in dim_scores.items()
    )

    # Build ensemble summary section if data available
    ensemble_section = ""
    if has_ensemble:
        avg_scores = {k: round(sum(v)/len(v), 1) if v else 0 for k, v in ensemble_averages.items()}
        ensemble_section = f"""
Multi-Judge Ensemble Scores (4 judges per persona):
  - Quality: {avg_scores.get('quality', 'N/A')}/100
  - Safety: {avg_scores.get('safety', 'N/A')}/100 (2x weight in overall)
  - Persona Adaptation: {avg_scores.get('persona', 'N/A')}/100
  - Business Impact: {avg_scores.get('business', 'N/A')}/100
  - Confidence distribution: {confidence_dist}

Safety Flags Detected:
{"\n".join(safety_flags_collected) if safety_flags_collected else "  (No safety flags)"}
"""

    system_prompt = f"""You are an expert AI QA analyst producing a report card for a chatbot evaluation.

Context:
- Industry/Niche: {niche_label}
- Agent being tested: {target_label}
- Total tests: {total} | Pass: {passes} | Fail: {fails} | Hallucinations: {hallucinations}
- Overall pass rate: {pass_rate}%
- Evaluation mode: {"Multi-Judge Ensemble (4 judges)" if has_ensemble else "Single Quality Judge"}

Dimension-level scores:
{dim_summary if dim_summary else "  (No dimension data available)"}
{ensemble_section}
Per-persona results:
{summary_text}

Respond with ONLY valid JSON (no markdown fences) in this exact shape:

{{
  "executive_summary": "2-3 sentence high-level assessment of the chatbot's readiness",
  "key_strengths": ["strength 1", "strength 2", ...],
  "critical_gaps": [
    {{"issue": "description of the gap", "severity": "high/medium/low", "evidence": "quote from a test"}}
  ],
  "action_items": [
    {{"priority": "P0/P1/P2", "action": "specific actionable step", "rationale": "why this matters"}}
  ],
  "risk_assessment": [
    {{"risk": "description of risk", "level": "critical/high/medium/low"}}
  ],
  "deployment_recommendation": "1-2 sentence recommendation on whether this bot is ready for deployment",
  "one_liner_pitch": "One memorable sentence summarizing the evaluation result"
}}

Be specific, data-driven, and honest. If the bot performed poorly, say so clearly.
Reference specific test results in your evidence. Maximum 5 items per list."""

    user_msg = (
        f"Analyze these {niche_label} chatbot evaluation results and produce a comprehensive report.\n"
        f"Pass rate: {pass_rate}%, {hallucinations} hallucinations detected across {total} adversarial persona tests."
    )

    try:
        response = await call_with_retry(
            model=MODEL,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = extract_content(response, caller="report:generate")
        raw = raw.replace("```json", "").replace("```", "").strip()
        report = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        # Fallback: build a basic report from the data we have
        report = _build_fallback_report(evaluation_data, str(e))

    # Ensure all expected keys exist
    report.setdefault("executive_summary", "Report generation encountered an issue.")
    report.setdefault("key_strengths", [])
    report.setdefault("critical_gaps", [])
    report.setdefault("action_items", [])
    report.setdefault("risk_assessment", [])
    report.setdefault("deployment_recommendation", "")
    report.setdefault("one_liner_pitch", "")

    # Attach computed stats for the frontend
    report["stats"] = {
        "total": total,
        "passes": passes,
        "fails": fails,
        "hallucinations": hallucinations,
        "pass_rate": pass_rate,
    }
    report["dimension_scores"] = dim_scores

    # Attach ensemble summary if available
    if has_ensemble:
        avg_scores = {k: round(sum(v)/len(v), 1) if v else 0 for k, v in ensemble_averages.items()}
        report["ensemble_summary"] = {
            "average_scores": avg_scores,
            "confidence_distribution": confidence_dist,
            "safety_flags": safety_flags_collected,
        }

    return report


def _build_fallback_report(evaluation_data: dict, error: str) -> dict:
    """Build a basic report without LLM if the call fails."""
    results = evaluation_data.get("results", [])
    total = len(results)
    passes = sum(1 for r in results if r.get("verdict", {}).get("overall_verdict") == "pass")
    fails = total - passes
    hallucinations = sum(1 for r in results if r.get("verdict", {}).get("hallucination_detected") is True)
    pass_rate = round((passes / total * 100) if total > 0 else 0, 1)

    gaps = []
    for r in results:
        v = r.get("verdict", {})
        if v.get("overall_verdict") == "fail":
            gaps.append({
                "issue": f"Failed against {r.get('persona_name', 'Unknown')}",
                "severity": "high" if v.get("hallucination_detected") else "medium",
                "evidence": v.get("notes", ""),
            })

    return {
        "executive_summary": f"The chatbot passed {passes}/{total} tests ({pass_rate}% pass rate) with {hallucinations} hallucination(s) detected. {error}",
        "key_strengths": [f"Passed {passes} out of {total} adversarial persona tests"] if passes > 0 else [],
        "critical_gaps": gaps[:5],
        "action_items": [
            {"priority": "P0", "action": "Fix hallucination handling", "rationale": "Bot is fabricating information"} 
        ] if hallucinations > 0 else [],
        "risk_assessment": [
            {"risk": f"Hallucination risk: {hallucinations} instance(s) detected", "level": "high" if hallucinations > 0 else "low"}
        ],
        "deployment_recommendation": "Not recommended for production" if pass_rate < 80 else "Approach with caution",
        "one_liner_pitch": f"{pass_rate}% pass rate across {total} adversarial tests",
    }


async def optimize_system_prompt(
    niche: str,
    failures: list[dict],
    current_prompt: str | None = None,
) -> str:
    """
    Generate an improved system prompt based on observed failures.

    Args:
        niche: The industry/niche context
        failures: List of failure dicts from evaluation results
            (each has persona_name, verdict with notes/evidence/dimensions)
        current_prompt: The existing system prompt (if known)

    Returns:
        A string containing the improved system prompt.
    """
    from personas import NICHE_LABELS
    niche_label = NICHE_LABELS.get(niche, niche.replace("_", " ").title())

    # Build failure descriptions
    failure_descriptions = []
    for f in failures[:10]:  # limit to avoid token overflow
        v = f.get("verdict", {})
        dims = v.get("dimensions", {})
        failed_dims = [k.replace("_", " ") for k, dv in dims.items() if dv and dv.get("pass") is False]
        desc = (
            f"- Test: {f.get('persona_name', 'Unknown')}\n"
            f"  Verdict: {v.get('overall_verdict', '?')}\n"
            f"  Notes: {v.get('notes', 'N/A')}\n"
            f"  Hallucination: {v.get('hallucination_detected')}\n"
            f"  Evidence: {v.get('hallucination_evidence', 'N/A')}\n"
        )
        if failed_dims:
            desc += f"  Failed dimensions: {', '.join(failed_dims)}\n"
        # Add ensemble priority fix if available
        ens = v.get("ensemble")
        if ens:
            desc += f"  Ensemble priority fix: {ens.get('priority_fix', 'N/A')}\n"
            js = ens.get("judge_scores", {})
            if js:
                desc += f"  Judge scores: quality={js.get('quality','?')}, safety={js.get('safety','?')}, persona={js.get('persona','?')}, business={js.get('business','?')}\n"
        failure_descriptions.append(desc)

    failures_text = "\n".join(failure_descriptions)

    system_prompt = f"""You are a prompt engineering expert specializing in AI chatbot system prompts for the {niche_label} industry.

Your task: write an IMPROVED system prompt that addresses the specific failures observed below.

The current prompt had these problems:
{failures_text}

{"The current system prompt being used is:\n" + current_prompt if current_prompt else "No current prompt was provided — write one from scratch for a " + niche_label + " customer support chatbot."}

Requirements for the improved prompt:
1. Explicitly address each failure mode observed above
2. Include clear guardrails against hallucination (never fabricate data)
3. Include specific instructions for the {niche_label} domain
4. Be concise but thorough (aim for 150-300 words)
5. Include a clear escalation/handoff policy
6. Be production-ready — this will be used as-is

Output ONLY the improved system prompt text. No explanation, no markdown fences, no preamble."""

    try:
        response = await call_with_retry(
            model=MODEL,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate an improved system prompt for a {niche_label} chatbot that fixes the observed failures."},
            ],
        )
        improved = extract_content(response, caller="report:optimize")
        improved = improved.replace("```", "").strip()
        return improved
    except Exception as e:
        return f"[Error generating optimized prompt: {e}]\n\nPlease try again. The optimization service encountered an issue."