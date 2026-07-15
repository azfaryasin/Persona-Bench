"""
Custom agent caller — "Bring Your Own Agent" support.

Lets users point Persona Bench at any HTTP chatbot API instead of only
the built-in weak/improved configs. Follows the same pattern as
Promptfoo's HTTP provider.

config dict expects:
  - url:            (str) HTTP(S) endpoint URL
  - method:         (str) HTTP method, default "POST"
  - headers:        (dict) request headers (e.g. Authorization)
  - body_template:  (str) JSON string with {{message}} and/or {{history}} placeholders
  - response_path:  (str) dot-notation path into the response JSON to extract reply text
"""

import json
import re
import httpx


# Regex to find {{message}} and {{history}} placeholders
_PLACEHOLDER_RE = re.compile(r"\{\{(message|history)\}\}")


def _resolve_path(obj, path: str):
    """
    Walk a dot-notation path into a nested dict/list.
    E.g. "choices.0.message.content" -> obj["choices"][0]["message"]["content"]
    Returns None if any step fails.
    """
    parts = path.strip().split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _build_body(template: str, message: str, history: list[dict]) -> dict | list:
    """
    Substitute {{message}} and {{history}} into the body template.

    - {{message}} is replaced with the latest user message string.
    - {{history}} is replaced with the full conversation history as a
      JSON array of {"role": "user"/"assistant", "content": str} objects.
    """
    history_json = json.dumps(history)
    rendered = _PLACEHOLDER_RE.sub(
        lambda m: message if m.group(1) == "message" else history_json,
        template,
    )
    return json.loads(rendered)


async def call_custom_target(
    conversation_history: list[dict],
    config: dict,
) -> str:
    """
    Call an external chatbot API and return the reply text.

    Args:
        conversation_history: list of {"role": "user"/"assistant", "content": str}
        config: dict with url, method, headers, body_template, response_path

    Returns:
        The extracted reply string, or an error message string on failure.
    """
    url = config["url"]
    method = config.get("method", "POST").upper()
    headers = config.get("headers", {})
    body_template = config.get("body_template", "")
    response_path = config.get("response_path", "reply")

    # Extract latest user message
    latest_message = ""
    for msg in reversed(conversation_history):
        if msg.get("role") == "user":
            latest_message = msg["content"]
            break

    # Build request body from template
    try:
        body = _build_body(body_template, latest_message, conversation_history)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"[custom agent error] Failed to render body_template: {e}"

    # Send request with 15s timeout
    try:
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            resp = await http_client.request(
                method=method,
                url=url,
                json=body,
                headers=headers,
            )
    except httpx.TimeoutException:
        return f"[custom agent error] Request to {url} timed out after 15s"
    except httpx.ConnectError:
        return f"[custom agent error] Could not connect to {url}"
    except httpx.HTTPError as e:
        return f"[custom agent error] HTTP request failed: {e}"

    # Check status code
    if resp.status_code >= 400:
        return f"[custom agent error] HTTP {resp.status_code}: {resp.text[:200]}"

    # Parse response JSON and walk response_path
    try:
        resp_json = resp.json()
    except (json.JSONDecodeError, ValueError):
        return f"[custom agent error] Response was not valid JSON: {resp.text[:200]}"

    reply = _resolve_path(resp_json, response_path)
    if reply is None:
        return (
            f"[custom agent error] Could not find '{response_path}' in response. "
            f"Response keys: {list(_iter_keys(resp_json))}"
        )

    if not isinstance(reply, str):
        reply = str(reply)

    return reply.strip()


def _iter_keys(obj, prefix=""):
    """Recursively collect key paths from a nested dict/list for error messages."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            yield full
            yield from _iter_keys(v, full)
    elif isinstance(obj, list) and len(obj) > 0:
        yield f"{prefix}[0]"
        yield from _iter_keys(obj[0], f"{prefix}[0]")