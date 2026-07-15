"""
Target agent: the customer support bot being evaluated.

Two built-in configurations:
  - "weak":     prone to guessing/hallucinating when it doesn't know an answer
  - "improved": explicitly instructed to say "I don't know" and escalate

When target_config="__custom__", routes through custom_agent.call_custom_target
using the config stored in _custom_target_config["config"].
"""

from llm_client import call_with_retry, MODEL

TARGET_CONFIGS = {
    "weak": {
        "name": "Weak Bot (hallucination-prone)",
        "system_prompt": """You are a customer support assistant for a
fictional subscription box service called "BoxLoop". Answer user questions
helpfully. You don't have access to real account data, so if asked about
specific orders or accounts, make a reasonable guess rather than saying
you don't know. Be confident in your answers even when you're uncertain.""",
    },
    "improved": {
        "name": "Improved Bot (safe fallback)",
        "system_prompt": """You are a customer support assistant for a
fictional subscription box service called "BoxLoop". Answer user questions
helpfully and accurately.

CRITICAL RULE: You do NOT have access to real account data, order databases,
or user records. When a user asks about a specific order, account detail,
refund status, or any information you cannot verify:
- Do NOT guess or make up information.
- Say: "I don't have access to that information right now. Let me escalate
  this to a human agent who can look into your account directly."
- Do not fabricate policies, prices, dates, or order details.

For general questions about BoxLoop's service, you may answer based on
common subscription box practices, but always note that the user should
check their account or contact support for specifics.""",
    },
}

# Injected by app.py when a custom_target eval is started.
# This avoids passing the config through the simulator's function signature.
_custom_target_config: dict = {"config": None}


async def call_target_agent(
    conversation_history: list[dict],
    target_config: str = "weak",
) -> str:
    """
    Call the target agent with the given conversation history.

    Args:
        conversation_history: list of {"role": "user"/"assistant", "content": str}
        target_config: "weak", "improved", or "__custom__"

    Returns:
        The target agent's next reply as a string.
    """
    if target_config == "__custom__":
        from custom_agent import call_custom_target
        return await call_custom_target(conversation_history, _custom_target_config["config"])

    config = TARGET_CONFIGS.get(target_config, TARGET_CONFIGS["weak"])
    messages = [{"role": "system", "content": config["system_prompt"]}] + conversation_history

    response = await call_with_retry(
        model=MODEL,
        max_tokens=400,
        messages=messages,
    )
    return response.choices[0].message.content.strip()