"""Tracked wrapper for Agents SDK Model objects.

Wrapping ``OpenAIChatCompletionsModel.get_response`` is the only way to
record **exact per-API-call** token + image counts for Agents SDK flows.
Without wrapping, the SDK only exposes aggregated ``Usage`` across all
internal turns of a ``Runner.run()`` call.

The wrapper:
1. counts image content blocks in the outgoing ``input``;
2. delegates to the inner model unchanged;
3. extracts per-call usage from the returned ``ModelResponse``;
4. extracts the names of any tool calls the response decided to invoke;
5. emits one ``LLMCallRecord`` via the gated writer;
6. returns the inner response untouched.

All extra work is inside a try/except — a tracking bug here can never
break an in-flight agent run.

The active ``TrackingContext`` (set by the pipeline before each agent
``Runner.run`` call) supplies stage / agent_label, so per-run attribution
just works without threading kwargs through the SDK.
"""

from __future__ import annotations

import logging
import time

from typing import Any

from agents.models.interface import Model

from .writer import emit_llm_call_record

console_logger = logging.getLogger(__name__)


def _count_images_in_input(input_obj: Any) -> int:
    """Count image content blocks in an Agents SDK ``input`` value.

    The SDK accepts either a plain string or a list of
    ``ResponseInputItemParam`` dicts. Each item may itself have a
    ``content`` list of blocks; image blocks have ``type`` in
    ``{"input_image", "image_url"}``. The Chat Completions converter
    also passes inline ``image_url`` blocks straight through.
    """

    if not isinstance(input_obj, list):
        return 0
    count = 0
    for item in input_obj:
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("input_image", "image_url"):
            count += 1
            continue
        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "input_image",
                    "image_url",
                ):
                    count += 1
    return count


def _extract_tool_names(response: Any) -> list[str]:
    """Return tool names found in a ``ModelResponse.output`` list."""

    names: list[str] = []
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return names
    for item in output:
        # Both ResponseFunctionToolCall and SDK ToolCallItem expose `name`.
        name = getattr(item, "name", None)
        if isinstance(name, str) and name:
            # Skip plain text messages (those have no `name`).
            names.append(name)
    return names


def _extract_usage(response: Any) -> dict[str, int]:
    """Pull token counts off an SDK ``Usage`` object."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        }
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    cached = 0
    reasoning = 0
    in_details = getattr(usage, "input_tokens_details", None)
    if in_details is not None:
        cached = int(getattr(in_details, "cached_tokens", 0) or 0)
    out_details = getattr(usage, "output_tokens_details", None)
    if out_details is not None:
        reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
    }


class TrackedChatCompletionsModel(Model):
    """Transparent proxy around an Agents SDK Model that emits per-call records.

    Only ``get_response`` is intercepted. Every other attribute access
    delegates to the inner model via ``__getattr__``, so the proxy
    satisfies whatever subset of the SDK ``Model`` protocol the runner
    happens to read (``model``, ``stream_response``, etc.).
    """

    def __init__(self, inner: Any) -> None:
        # Use object.__setattr__ to avoid recursing through __getattr__.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_request_counter", 0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        # The SDK calls get_response with keyword args (system_instructions,
        # input, model_settings, tools, output_schema, handoffs, tracing,
        # previous_response_id, conversation_id, prompt). We don't rely on
        # the full signature staying stable — we only peek at `input`.
        input_obj = kwargs.get("input")
        if input_obj is None and args:
            # Defensive fallback if someone calls positionally.
            input_obj = args[1] if len(args) > 1 else None

        num_images = 0
        try:
            num_images = _count_images_in_input(input_obj)
        except Exception as exc:  # noqa: BLE001
            console_logger.warning("image count failed in wrapper: %s", exc)

        start = time.perf_counter()
        response = await self._inner.get_response(*args, **kwargs)
        duration_s = time.perf_counter() - start

        try:
            usage = _extract_usage(response)
            tool_names = _extract_tool_names(response)
            request_index = self._request_counter
            object.__setattr__(self, "_request_counter", request_index + 1)
            emit_llm_call_record(
                source="agents_sdk",
                provider_path="agents_sdk",
                model=getattr(self._inner, "model", None),
                tool_calls_in_response=tool_names,
                request_index=request_index,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                cached_tokens=usage["cached_tokens"],
                total_tokens=usage["total_tokens"],
                num_images=num_images,
                duration_s=duration_s,
            )
        except Exception as exc:  # noqa: BLE001
            console_logger.warning("record emission failed in wrapper: %s", exc)

        return response

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.stream_response(*args, **kwargs)


def wrap_model(inner: Any) -> Any:
    """Return ``TrackedChatCompletionsModel(inner)``.

    Kept as a tiny factory so ``provider.py`` can call it without
    importing the class symbol directly.
    """

    return TrackedChatCompletionsModel(inner)
