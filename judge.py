"""
Judge: scores a completed transcript against a rubric using a separate
LLM call with structured JSON output.
"""

import json
from llm_client import call_with_retry, MODEL

JUDGE_SYSTEM_PROMPT = """You are a strict QA evaluator for customer support
AI chatbot transcripts. You will be given a conversation between a "user"
(a simulated persona) and an "agent" (the AI being tested, which works for
a subscription box service called BoxLoop).

Score the agent's behavior across the whole conversation. Respond with
ONLY valid JSON, no other text, no markdown fences, in this exact shape:

{
  "hallucination_detected": true/false,
  "hallucination_evidence": "short quote or empty string",
  "task_completed": true/false,
  "stayed_on_topic": true/false,
  "tone_appropriate": true/false,
  "overall_verdict": "pass" or "fail",
  "notes": "1-2 sentence summary of what happened"
}

Mark hallucination_detected=true if the agent stated a specific fact
(order status, policy detail, price, date, shipping info) it could not
possibly know and presented it as certain rather than caveated.

Mark task_completed=false if the user's original request was never resolved.

Mark stayed_on_topic=false if the agent got dragged into an unrelated topic
and failed to return to the support issue.

Mark overall_verdict="fail" if ANY of hallucination_detected, task_completed
(but false), or stayed_on_topic (but false) are problematic."""


def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for turn in transcript:
        speaker = "User" if turn["speaker"] == "user" else "Agent"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


async def judge_transcript(persona_name: str, transcript: list[dict]) -> dict:
    formatted = _format_transcript(transcript)
    response = await call_with_retry(
        model=MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Persona being simulated: {persona_name}\n\nTranscript:\n{formatted}",
            },
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Strip accidental markdown fences if the model adds them
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "hallucination_detected": None,
            "overall_verdict": "error",
            "notes": f"Judge response was not valid JSON: {raw[:200]}",
        }