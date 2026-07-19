"""
Simulator: runs a persona (simulated user) in conversation with the target
agent for N turns, producing a transcript. Fully async.
"""

from llm_client import call_with_retry, MODEL, extract_content
from personas import PERSONAS
from target_agent import call_target_agent


async def get_persona_reply(
    persona_system_prompt: str,
    transcript_from_persona_pov: list[dict],
) -> str:
    """Ask the simulator LLM to generate the persona's next message."""
    messages = [
        {"role": "system", "content": persona_system_prompt},
    ] + transcript_from_persona_pov

    response = await call_with_retry(
        model=MODEL,
        max_tokens=200,
        messages=messages,
    )
    return extract_content(response, caller="simulator:persona")


async def run_persona_conversation(
    persona_key: str,
    target_config: str = "weak",
    num_turns: int = 5,
    personas_dict: dict | None = None,
) -> dict:
    """
    Runs one persona through a multi-turn conversation with the target agent.
    Returns a transcript dict ready for the judge to score.
    """
    personas = personas_dict or PERSONAS
    persona = personas[persona_key]

    target_view: list[dict] = []       # what the target agent sees
    persona_view: list[dict] = []      # what the persona simulator sees

    opening = persona["opening_message"]
    target_view.append({"role": "user", "content": opening})
    persona_view.append({"role": "assistant", "content": opening})

    transcript = [{"speaker": "user", "text": opening}]

    for _ in range(num_turns):
        agent_reply = await call_target_agent(target_view, target_config)
        transcript.append({"speaker": "agent", "text": agent_reply})
        target_view.append({"role": "assistant", "content": agent_reply})
        persona_view.append({"role": "user", "content": agent_reply})

        persona_reply = await get_persona_reply(persona["system_prompt"], persona_view)
        transcript.append({"speaker": "user", "text": persona_reply})
        target_view.append({"role": "user", "content": persona_reply})
        persona_view.append({"role": "assistant", "content": persona_reply})

    return {
        "persona_key": persona_key,
        "persona_name": persona["name"],
        "transcript": transcript,
    }