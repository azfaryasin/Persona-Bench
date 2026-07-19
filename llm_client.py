"""
Shared OpenAI client configuration.

Reads OPENAI_API_KEY from the environment.
Optionally reads OPENAI_BASE_URL to point at a compatible proxy.
Falls back to z-ai config for local testing.
On Railway/production: set OPENAI_API_KEY and optionally OPENAI_BASE_URL.
"""

import os
import json
import asyncio
from openai import AsyncOpenAI, RateLimitError


def _load_zai_config():
    """Try to load z-ai config as a fallback for local testing."""
    for path in [os.path.join(os.getcwd(), ".z-ai-config"),
                 os.path.join(os.path.expanduser("~"), ".z-ai-config"),
                 "/etc/.z-ai-config"]:
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return None


def create_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    zai_config = None

    # Fallback: try z-ai config if no OPENAI_API_KEY is set
    if not api_key:
        zai_config = _load_zai_config()
        if zai_config:
            api_key = zai_config.get("apiKey")
            base_url = base_url or zai_config.get("baseUrl")

    if not api_key:
        raise RuntimeError(
            "No API key found. Set OPENAI_API_KEY environment variable, "
            "or ensure a .z-ai-config file exists."
        )

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
        # z-ai proxy requires these headers for authentication
        headers = {"X-Z-AI-From": "Z"}
        if zai_config and zai_config.get("token"):
            headers["X-Token"] = zai_config["token"]
        if zai_config and zai_config.get("chatId"):
            headers["X-Chat-Id"] = zai_config["chatId"]
        if zai_config and zai_config.get("userId"):
            headers["X-User-Id"] = zai_config["userId"]
        kwargs["default_headers"] = headers

    return AsyncOpenAI(**kwargs)


# Shared client instance used by all modules
client = create_client()

# Model to use for all calls (override with OPENAI_MODEL env var)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Retry backoff delays in seconds: 5s, 10s, 20s
_RETRY_DELAYS = [5, 10, 20]


def extract_content(response, caller: str = "unknown") -> str:
    """
    Safely extract text content from an OpenAI API response.

    Handles the case where content is None (common with NVIDIA, Azure, and other
    providers that return None when finish_reason is "content_filter" or the
    model refuses to answer).  Logs the finish_reason so the operator can see
    *why* the response was empty.

    Args:
        response: The ChatCompletion object from the API.
        caller:   A short label (e.g. "simulator", "judge") for log messages.

    Returns:
        The content string (stripped), or a descriptive placeholder if None/empty.
    """
    # Guard: no choices at all
    if not response or not response.choices:
        print(f"[{caller}] WARNING: API response has no choices. Full response: {response}")
        return "[No response from API]"

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    content = choice.message.content if choice.message else None

    if content is None:
        # Log the raw finish_reason — this is the diagnostic the user needs
        print(f"[{caller}] WARNING: content is None. finish_reason={finish_reason!r}")
        if finish_reason == "content_filter":
            return "[Response blocked by content filter]"
        return "[No response — possible content filter or model refusal]"
    elif not content.strip():
        print(f"[{caller}] WARNING: content is empty string. finish_reason={finish_reason!r}")
        if finish_reason == "content_filter":
            return "[Response blocked by content filter]"
        return "[Empty response from model]"

    return content.strip()


async def call_with_retry(**kwargs):
    """
    Wrap client.chat.completions.create() with retry logic for rate limits.

    Catches openai.RateLimitError specifically.
    Exponential backoff: 5s, 10s, 20s. Max 3 retries.
    Only raises after all retries are exhausted.
    """
    last_error = None
    for attempt, delay in enumerate(_RETRY_DELAYS):
        try:
            return await client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_error = e
            print(f"Rate limited, retrying in {delay}s... (attempt {attempt + 1}/3)")
            await asyncio.sleep(delay)
    # All retries exhausted — raise the last error
    raise last_error