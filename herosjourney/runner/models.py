"""
herosjourney/runner/models.py
Generic model adapter.

The episode loop is model-agnostic: it only needs a callable
    model_fn(prompt, max_tokens) -> (response_text, thinking, token_counts)
This module provides a default such callable backed by OpenAI-compatible
chat-completions endpoints (OpenAI, vLLM, LM Studio, Ollama, TGI, ...) and
Anthropic's Messages API.

Configuration (environment variables):
    HEROSJOURNEY_BASE_URL  or  OPENAI_BASE_URL   (default: https://api.openai.com/v1)
    HEROSJOURNEY_API_KEY   or  OPENAI_API_KEY    (default: "EMPTY", for local servers)
    HEROSJOURNEY_EXTRA_BODY                         optional JSON object merged into extra_body
    HEROSJOURNEY_CHAT_TEMPLATE_KWARGS               optional JSON chat_template_kwargs
    HEROSJOURNEY_TOP_P / HEROSJOURNEY_PRESENCE_PENALTY optional sampling fields
    HEROSJOURNEY_REASONING_EFFORT                   optional OpenAI reasoning_effort field
    HEROSJOURNEY_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY or JUDGE_API_KEY
    HEROSJOURNEY_ANTHROPIC_BASE_URL or ANTHROPIC_BASE_URL or JUDGE_BASE_URL

For provider-specific backends (Azure, Gemini, Bedrock) or bespoke decoding
parameters, write your own model_fn and pass it to run_single_episode(model_fn=...)
— you do not need to edit this file. The paper's exact provider adapters live in
experiments/models.py (not shipped with the package).

Requires the `openai` package:  pip install "herosjourney[runner]"
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple


def _default_base_url() -> str:
    return os.environ.get("HEROSJOURNEY_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )


def _api_key() -> str:
    return os.environ.get("HEROSJOURNEY_API_KEY") or os.environ.get(
        "OPENAI_API_KEY", "EMPTY"
    )


def _client(base_url: Optional[str] = None):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "The generic model adapter needs the 'openai' package. "
            'Install it with:  pip install "herosjourney[runner]"\n'
            "Or supply your own model_fn to run_single_episode(model_fn=...)."
        ) from e
    return OpenAI(base_url=base_url or _default_base_url(), api_key=_api_key(), timeout=180)


def _model_and_base_url(model: str) -> Tuple[str, Optional[str]]:
    """Support per-call endpoints via MODEL@PORT or MODEL@http://host:port/v1."""
    at_idx = model.rfind("@")
    if at_idx == -1:
        return model, None

    suffix = model[at_idx + 1:]
    model_name = model[:at_idx]
    if suffix.isdigit():
        host = os.environ.get(f"VLLM_HOST_{suffix}", "localhost")
        return model_name, f"http://{host}:{suffix}/v1"
    if suffix.startswith("http://") or suffix.startswith("https://"):
        return model_name, suffix.rstrip("/")
    return model, None


def _provider_and_model(model: str) -> Tuple[str, str]:
    """Return (provider, model_name). Use anthropic:<id> for Anthropic."""
    if model.startswith("anthropic:"):
        return "anthropic", model[len("anthropic:"):]
    provider = os.environ.get("HEROSJOURNEY_PROVIDER", "").strip().lower()
    if provider == "anthropic":
        return "anthropic", model
    return "openai", model


def _anthropic_api_key() -> str:
    key = (
        os.environ.get("HEROSJOURNEY_ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("JUDGE_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "Anthropic model requested but no API key found. Set "
            "HEROSJOURNEY_ANTHROPIC_API_KEY, ANTHROPIC_API_KEY, or JUDGE_API_KEY."
        )
    return key


def _anthropic_base_url() -> str:
    return (
        os.environ.get("HEROSJOURNEY_ANTHROPIC_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or os.environ.get("JUDGE_BASE_URL")
        or "https://api.anthropic.com"
    ).rstrip("/")


def _anthropic_version() -> str:
    return os.environ.get("HEROSJOURNEY_ANTHROPIC_VERSION", "2023-06-01")


def _anthropic_response(
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: Optional[float] = None,
) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    """Call Anthropic Messages API without requiring an extra SDK dependency."""
    import urllib.error
    import urllib.request

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    include_temperature = os.environ.get(
        "HEROSJOURNEY_ANTHROPIC_INCLUDE_TEMPERATURE", ""
    ).lower() in {"1", "true", "yes", "on"}
    if include_temperature:
        payload["temperature"] = (
            float(os.environ.get("HEROSJOURNEY_TEMPERATURE", "1.0"))
            if temperature is None else temperature
        )
    system_prompt = os.environ.get("HEROSJOURNEY_ANTHROPIC_SYSTEM_PROMPT", "").strip()
    if system_prompt:
        payload["system"] = system_prompt
    req = urllib.request.Request(
        f"{_anthropic_base_url()}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": _anthropic_api_key(),
            "anthropic-version": _anthropic_version(),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API error {e.code}: {detail[:1000]}") from e

    content_parts = data.get("content", [])
    text = "".join(
        part.get("text", "")
        for part in content_parts
        if isinstance(part, dict) and part.get("type") == "text"
    )
    usage = data.get("usage") or {}
    token_counts = {
        "prompt_tokens":     int(usage.get("input_tokens", 0) or 0),
        "candidates_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens":      int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0),
    }
    return text.strip(), None, token_counts


def _extra_body() -> Optional[dict]:
    """Optional provider-specific OpenAI-compatible request fields.

    HEROSJOURNEY_DISABLE_REASONING=1 is useful for reasoning-parser-backed vLLM
    servers such as GLM, where otherwise the final answer may be returned as
    null content and the token budget consumed by a separate reasoning field.
    HEROSJOURNEY_CHAT_TEMPLATE_KWARGS can provide a full JSON object override.
    HEROSJOURNEY_EXTRA_BODY can provide other vLLM fields such as top_k/min_p.
    """
    extra_body = {}
    raw_extra = os.environ.get("HEROSJOURNEY_EXTRA_BODY")
    if raw_extra:
        try:
            parsed = json.loads(raw_extra)
        except json.JSONDecodeError as e:
            raise ValueError("HEROSJOURNEY_EXTRA_BODY must be valid JSON") from e
        if not isinstance(parsed, dict):
            raise ValueError("HEROSJOURNEY_EXTRA_BODY must be a JSON object")
        extra_body.update(parsed)

    raw_kwargs = os.environ.get("HEROSJOURNEY_CHAT_TEMPLATE_KWARGS")
    if raw_kwargs:
        try:
            parsed_kwargs = json.loads(raw_kwargs)
        except json.JSONDecodeError as e:
            raise ValueError(
                "HEROSJOURNEY_CHAT_TEMPLATE_KWARGS must be valid JSON"
            ) from e
        if not isinstance(parsed_kwargs, dict):
            raise ValueError("HEROSJOURNEY_CHAT_TEMPLATE_KWARGS must be a JSON object")
        extra_body["chat_template_kwargs"] = parsed_kwargs

    disable_reasoning = os.environ.get("HEROSJOURNEY_DISABLE_REASONING", "").lower()
    if "chat_template_kwargs" not in extra_body and disable_reasoning in {"1", "true", "yes", "on"}:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    return extra_body or None


def _optional_float(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return float(raw)


def _reasoning_effort() -> Optional[str]:
    value = os.environ.get("HEROSJOURNEY_REASONING_EFFORT", "").strip()
    return value or None


def _add_sampling_options(kwargs: dict) -> None:
    top_p = _optional_float("HEROSJOURNEY_TOP_P")
    if top_p is not None:
        kwargs["top_p"] = top_p
    presence_penalty = _optional_float("HEROSJOURNEY_PRESENCE_PENALTY")
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty


def _extract_json_object(text: str) -> str:
    """Return a parseable JSON object when a model wraps it in markdown/prose."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1].strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return text


def agent_response(
    model: str,
    prompt: str,
    max_tokens: int = 512,
) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    """Call an OpenAI-compatible chat endpoint.

    Returns (response_text, thinking, token_counts). `thinking` is always None
    for this generic adapter; token_counts uses the key names the episode loop
    expects ('prompt_tokens', 'candidates_tokens', 'total_tokens').
    """
    provider, provider_model = _provider_and_model(model)
    if provider == "anthropic":
        return _anthropic_response(provider_model, prompt, max_tokens=max_tokens)

    model_name, base_url = _model_and_base_url(provider_model)
    client = _client(base_url)
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": float(os.environ.get("HEROSJOURNEY_TEMPERATURE", "1.0")),
    }
    extra_body = _extra_body()
    if extra_body:
        kwargs["extra_body"] = extra_body
    _add_sampling_options(kwargs)
    reasoning_effort = _reasoning_effort()
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    token_counts = None
    if usage is not None:
        token_counts = {
            "prompt_tokens":     getattr(usage, "prompt_tokens", 0),
            "candidates_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens":      getattr(usage, "total_tokens", 0),
        }
    return (content.strip() if content else ""), None, token_counts


_JSON_CONVERTER_INSTRUCTION = (
    "You extract the single action a game agent wants to take from its text and "
    "return it as JSON matching this schema: {json_schema}. "
    "Action/argument formats: go [location], get [object], buy [object], "
    "perform [ritual], drink [potion], defeat [enemy], rescue [npc], "
    "check_inventory, check_location. Output ONLY the JSON object."
)

_GENERIC_JSON_CONVERTER_INSTRUCTION = (
    "Extract or repair the JSON object from the model generation and return it "
    "as JSON matching this schema: {json_schema}. Preserve the semantic content "
    "of the generation; do not reinterpret it as a game action unless the schema "
    "explicitly asks for an action. Output ONLY the JSON object."
)


def _schema_required_fields(json_schema: dict) -> set[str]:
    fields = json_schema.get("required", [])
    return {str(field) for field in fields if isinstance(field, str)}


def _matches_required_fields(text: str, json_schema: dict) -> bool:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    return _schema_required_fields(json_schema).issubset(data.keys())


def _converter_instruction(json_schema: dict) -> str:
    required = _schema_required_fields(json_schema)
    if {"action", "argument"}.issubset(required):
        return _JSON_CONVERTER_INSTRUCTION.format(json_schema=json_schema)
    return _GENERIC_JSON_CONVERTER_INSTRUCTION.format(json_schema=json_schema)


def json_converter(response_content: str, json_schema: dict, model: Optional[str] = None) -> str:
    """Repair/extract a JSON action from free-form model text via the same endpoint.

    `model` defaults to env HEROSJOURNEY_CONVERTER_MODEL, else HEROSJOURNEY_MODEL.
    """
    extracted = _extract_json_object(response_content)
    if _matches_required_fields(extracted, json_schema):
        return extracted

    conv_model = (
        model
        or os.environ.get("HEROSJOURNEY_CONVERTER_MODEL")
        or os.environ.get("HEROSJOURNEY_MODEL", "gpt-4o-mini")
    )
    instruction = _converter_instruction(json_schema)
    provider, provider_model = _provider_and_model(conv_model)
    if provider == "anthropic":
        raw, _, _ = _anthropic_response(
            provider_model,
            f"{instruction}\n\nModel generation:\n{response_content}",
            max_tokens=512,
            temperature=0,
        )
        return _extract_json_object(raw or "")

    conv_model_name, base_url = _model_and_base_url(provider_model)
    client = _client(base_url)
    kwargs = {
        "model": conv_model_name,
        "messages": [{
            "role": "user",
            "content": f"{instruction}\n\nModel generation:\n{response_content}",
        }],
        "temperature": 0,
        "max_tokens": 512,
    }
    extra_body = _extra_body()
    if extra_body:
        kwargs["extra_body"] = extra_body
    reasoning_effort = _reasoning_effort()
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = client.chat.completions.create(**kwargs)
    return _extract_json_object(resp.choices[0].message.content or "")


# --- Backward-compatible aliases used by qa_episode.py / teacher.py ---
# In this generic adapter there is a single configured endpoint, so the
# "small" / "gemini" variants all route to the generic converter.

def json_converter_small(response_content: str, json_schema: dict) -> str:
    return json_converter(response_content, json_schema)


def json_converter_gemini(response_content: str, json_schema: dict) -> str:
    return json_converter(response_content, json_schema)


def teacher_json_repair_small(text: str, json_schema: dict) -> str:
    return json_converter(text, json_schema)


def teacher_json_repair_gemini(text: str, json_schema: dict) -> str:
    return json_converter(text, json_schema)
