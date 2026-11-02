"""Narrow direct-call adapter for TaskSmith's OpenAI-compatible providers.

This module is intentionally narrow. Only callers that genuinely cannot use
the unified ``scenesmith.agent_utils.sdk_vlm_runner`` funnel should reach for
it — currently:

* ``scenesmith.agent_utils.turn_trimming_session`` — async text-only context
  compression on the stateful-agent hot path; no Pydantic schema.
* ``scripts/test_image_token_accounting.py`` — token-accounting smoke script.

Everything else (task analysis, reward spec, morphology task, RL artifact
exporter, asset router analysis, domain randomization editor,
provider startup probe) now goes through ``run_text_vlm_call`` /
``run_structured_vlm_call_messages`` / ``run_structured_vlm_call``.

Strict rule (unchanged): TaskSmith may consume only the model's final answer
text and must never treat reasoning traces as fallback output.
"""

from __future__ import annotations

import asyncio
import logging
import os

from typing import Any

from openai import OpenAI

from scenesmith.llm_retry import retry_sync
from scenesmith.provider import get_provider_connection_settings
from scenesmith.provider_validation import (
    apply_chat_completion_capabilities,
    get_active_provider_config,
)

console_logger = logging.getLogger(__name__)


def _default_llm_timeout_s() -> float:
    raw = os.environ.get("TASKSMITH_LLM_TIMEOUT_S", "360").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 360.0
    if value <= 0:
        value = 360.0
    return value


def _default_llm_max_retries() -> int:
    raw = os.environ.get("TASKSMITH_LLM_MAX_RETRIES", "5").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 5
    if value < 0:
        value = 0
    return value


def _resolve_timeout_s(timeout: float | None) -> float:
    return _default_llm_timeout_s() if timeout is None else float(timeout)


def _resolve_max_retries(max_retries: int | None) -> int:
    return _default_llm_max_retries() if max_retries is None else int(max_retries)


class ProviderCompatibilityError(RuntimeError):
    """Raised when a provider/model cannot supply a final answer safely."""

    def __init__(
        self,
        *,
        category: str,
        message: str,
        model: str,
        provider_label: str,
        api: str,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.model = model
        self.provider_label = provider_label
        self.api = api


def get_selected_api(
    provider_config: dict[str, Any] | None = None,
    *,
    use_reasoning_model_path: bool | None = None,
) -> str:
    """Return the configured API path for the active model."""
    provider = provider_config or get_active_provider_config()
    api_cfg = provider["api"]
    model_traits = provider["model_traits"]

    reasoning_native = (
        model_traits["reasoning_native"]
        if use_reasoning_model_path is None
        else use_reasoning_model_path
    )
    return (
        api_cfg["reasoning_model_api"]
        if reasoning_native
        else api_cfg["standard_model_api"]
    )


def call_text_completion_sync_with_metadata(
    *,
    model: str,
    messages: list[dict[str, Any]],
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    service_tier: str | None = None,
    response_format: dict[str, Any] | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    provider_config: dict[str, Any] | None = None,
    subsystem: str,
    use_reasoning_model_path: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call the provider and return final assistant text plus usage metadata."""
    provider = provider_config or get_active_provider_config()
    api = get_selected_api(provider, use_reasoning_model_path=use_reasoning_model_path)
    response = _call_openai_sync(
        model=model,
        messages=messages,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        service_tier=service_tier,
        response_format=response_format,
        timeout=timeout,
        max_retries=max_retries,
        provider_config=provider,
        subsystem=subsystem,
    )
    text = extract_final_answer_text(
        response=response,
        model=model,
        provider_config=provider,
        subsystem=subsystem,
        api=api,
    )
    return text, extract_usage_metadata(response)


async def call_text_completion_async_with_metadata(
    *,
    model: str,
    messages: list[dict[str, Any]],
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    service_tier: str | None = None,
    response_format: dict[str, Any] | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    provider_config: dict[str, Any] | None = None,
    subsystem: str,
    use_reasoning_model_path: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Async variant of call_text_completion_sync_with_metadata."""
    return await asyncio.to_thread(
        call_text_completion_sync_with_metadata,
        model=model,
        messages=messages,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        service_tier=service_tier,
        response_format=response_format,
        timeout=timeout,
        max_retries=max_retries,
        provider_config=provider_config,
        subsystem=subsystem,
        use_reasoning_model_path=use_reasoning_model_path,
    )


def _call_openai_sync(
    *,
    model: str,
    messages: list[dict[str, Any]],
    reasoning_effort: str | None,
    verbosity: str | None,
    service_tier: str | None,
    response_format: dict[str, Any] | None,
    timeout: float | None,
    max_retries: int | None,
    provider_config: dict[str, Any],
    subsystem: str,
) -> Any:
    connection = get_provider_connection_settings(provider_config)
    timeout_s = _resolve_timeout_s(timeout)
    max_retries_resolved = _resolve_max_retries(max_retries)

    client = OpenAI(
        base_url=connection["base_url"],
        api_key=connection["api_key"],
        timeout=timeout_s,
        max_retries=0,  # retries handled by retry_sync below
    )

    kwargs: dict[str, Any] = {"model": model, "messages": messages}

    kwargs, extra_body = apply_chat_completion_capabilities(
        kwargs=kwargs,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        service_tier=service_tier,
        provider_config=provider_config,
        subsystem=subsystem,
    )

    if response_format is not None:
        kwargs["response_format"] = response_format

    if extra_body:
        kwargs["extra_body"] = extra_body

    return retry_sync(
        lambda: client.chat.completions.create(**kwargs),
        op_name=f"{subsystem}_chat_completions[{model}]",
        max_attempts=1 + max_retries_resolved,
        logger=console_logger,
    )


def extract_final_answer_text(
    *,
    response: Any,
    model: str,
    provider_config: dict[str, Any] | None = None,
    subsystem: str,
    api: str,
) -> str:
    """Extract final assistant text without ever reading reasoning traces."""
    provider = provider_config or get_active_provider_config()
    provider_label = provider.get("base_url") or "default OpenAI endpoint"

    message = _get_primary_message(response)
    content = _extract_text_content(message)
    if content and content.strip():
        return content.strip()

    top_level_text = _extract_top_level_output_text(response)
    if top_level_text and top_level_text.strip():
        return top_level_text.strip()

    if _message_contains_reasoning_trace(message):
        raise ProviderCompatibilityError(
            category="reasoning_only_no_final_answer",
            model=model,
            provider_label=provider_label,
            api=api,
            message=(
                f"{subsystem} received a reasoning-only response from model {model} "
                f"via {provider_label} using {api}. TaskSmith refuses to consume "
                "reasoning traces, and no final answer text was present."
            ),
        )

    raise RuntimeError(
        f"{subsystem} returned no final answer text for model {model} via "
        f"{provider_label} using {api}."
    )


def _get_primary_message(response: Any) -> Any:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    choice = choices[0]
    return getattr(choice, "message", None)


def extract_usage_metadata(response: Any) -> dict[str, Any]:
    """Extract usage information from an OpenAI response object."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {}

    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    reasoning_tokens = 0
    cached_tokens = 0
    prompt_details = _usage_value(usage, "prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached_tokens = int(prompt_details.get("cached_tokens") or 0)
    else:
        cached_tokens = _usage_value(prompt_details, "cached_tokens")
    completion_details = _usage_value(usage, "completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
    else:
        reasoning_tokens = _usage_value(completion_details, "reasoning_tokens")

    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "requests": 1,
    }


def _usage_value(obj: Any, field_name: str) -> int | dict[str, Any]:
    if obj is None:
        return 0
    if isinstance(obj, dict):
        value = obj.get(field_name, 0)
    else:
        value = getattr(obj, field_name, 0)
    return value or 0


def _extract_top_level_output_text(response: Any) -> str | None:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    return None


def _extract_text_content(message: Any) -> str | None:
    if message is None:
        return None

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            text = _extract_text_from_content_item(item)
            if text:
                text_parts.append(text)
        joined = "".join(text_parts).strip()
        return joined or None

    return None


def _extract_text_from_content_item(item: Any) -> str | None:
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        item_type = item.get("type")
        if item_type in {"reasoning", "reasoning_content", "reasoning.text"}:
            return None
        for key in ("text", "content"):
            value = item.get(key)
            if isinstance(value, str):
                return value
        return None

    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text

    item_type = getattr(item, "type", None)
    if item_type in {"reasoning", "reasoning_content", "reasoning.text"}:
        return None

    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    return None


def _message_contains_reasoning_trace(message: Any) -> bool:
    if message is None:
        return False

    for field_name in ("reasoning", "reasoning_content", "reasoning_details"):
        value = getattr(message, field_name, None)
        if value is None and isinstance(message, dict):
            value = message.get(field_name)
        if value:
            return True

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") in {
                "reasoning",
                "reasoning_content",
                "reasoning.text",
            }:
                return True
            if getattr(item, "type", None) in {
                "reasoning",
                "reasoning_content",
                "reasoning.text",
            }:
                return True

    return False
