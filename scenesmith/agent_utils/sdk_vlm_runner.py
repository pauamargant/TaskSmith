"""Single-shot LLM/VLM calls via the Agents SDK.

This module is the canonical entry point for every non-stateful (single-shot)
LLM call in TaskSmith. The three helpers it exposes all funnel through the
same ``make_chat_completions_model`` + ``Runner.run`` path that the stateful
planner/designer/critic agents use, so request shaping (``extra_body``,
thinking-on/off), tracking, and patches stay unified:

- ``run_structured_vlm_call`` — multimodal (text + base64 images) with a
  required Pydantic ``output_type``.  Originally replaced the failing
  ``VLMService.create_completion(json_object)`` path which would occasionally
  return reasoning-only (empty ``message.content``) on Qwen3+vLLM.
- ``run_structured_vlm_call_messages`` — system+user (or full conversation)
  message list with a required Pydantic ``output_type``.  Used by callers
  that previously built a ``[{"role":"system",...},{"role":"user",...}]``
  list and called ``VLMService`` or ``structured_output_call``.
- ``run_text_vlm_call`` — text-only with NO ``output_type``; returns the
  model's raw text answer.  Used by the provider startup probe (and any
  future caller that genuinely needs plain text).

See ``base_stateful_agent.py:_get_model_settings`` for the canonical
ModelSettings construction these mirror.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agents import Agent, ModelSettings, Runner
from agents.agent_output import AgentOutputSchemaBase
from openai.types.shared import Reasoning
from pydantic import BaseModel

from scenesmith.experiment_tracking.context import extend_tracking_context
from scenesmith.provider import make_chat_completions_model

console_logger = logging.getLogger(__name__)

# Thinking on/off is routed through the standard ``build_thinking_extra_body``
# path. See ``.docs/canonicalization_vlm_investigation.md`` for the empirical
# basis (only ``enable_thinking`` is honored by the RCP vLLM endpoint).


def _build_model_settings(
    reasoning_effort: str | None,
    verbosity: str | None,
    provider_config: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> ModelSettings:
    """Build ModelSettings for a one-shot multimodal call.

    Dispatches like ``apply_chat_completion_capabilities`` in
    ``provider_validation.py``: when the provider's ``thinking_control_mode``
    is ``"binary"`` we route through ``build_thinking_extra_body`` to set
    ``chat_template_kwargs.enable_thinking`` (never ``Reasoning``, which vLLM
    rejects with HTTP 400 for ``minimal``). For ``"openai"`` mode we forward
    ``reasoning_effort`` directly via the SDK's ``Reasoning`` object.
    ``"disabled"`` sends nothing.
    """
    from scenesmith.provider_validation import (
        build_thinking_extra_body,
        get_active_provider_config,
        get_thinking_control_mode,
        merge_chat_template_kwargs_into_extra_body,
        split_provider_sampling,
    )

    provider = provider_config or get_active_provider_config()
    mode = get_thinking_control_mode(provider)

    kwargs: dict[str, Any] = {}
    if reasoning_effort:
        if mode == "binary":
            extra = build_thinking_extra_body(reasoning_effort, provider)
            if extra:
                kwargs["extra_body"] = extra
        elif mode == "openai":
            kwargs["reasoning"] = Reasoning(effort=reasoning_effort)
        # mode == "disabled": send nothing.
    if verbosity:
        kwargs["verbosity"] = verbosity

    # Safety cap against unbounded thinking on single-shot VLM/structured-output
    # calls. Observed on 2026-05-20 against Qwen3.6-27B with enable_thinking=true
    # and presence_penalty=1.5: physics_analysis calls generated 6443 and 7790
    # output tokens for prompts that should fit in <2k, hitting per-attempt
    # request timeouts and triggering exponential-backoff retry loops. Half the
    # stateful agent cap (32768) — high enough that thinking + structured JSON
    # fit comfortably, low enough to bound wall-clock per attempt.
    kwargs["max_tokens"] = 16384

    # Provider-level sampling + chat_template_kwargs from config.
    native_sampling, extra_sampling = split_provider_sampling(provider)
    for key, value in native_sampling.items():
        kwargs.setdefault(key, value)
    if extra_sampling:
        extra_body = dict(kwargs.get("extra_body") or {})
        for key, value in extra_sampling.items():
            extra_body.setdefault(key, value)
        kwargs["extra_body"] = extra_body
    extra_body = dict(kwargs.get("extra_body") or {})
    merge_chat_template_kwargs_into_extra_body(extra_body, provider)
    if extra_body:
        kwargs["extra_body"] = extra_body

    if timeout_s is not None:
        # Forward a per-call HTTP timeout to the underlying openai client.
        # OpenAIChatCompletionsModel reads ModelSettings.extra_args and passes
        # them verbatim to ``client.chat.completions.create(**extra_args)``.
        from openai import Timeout

        kwargs["extra_args"] = {
            "timeout": Timeout(
                connect=timeout_s, read=timeout_s, write=timeout_s, pool=timeout_s
            )
        }
    return ModelSettings(**kwargs)


def _build_user_input(user_text: str, images_b64: list[str]) -> list[dict[str, Any]]:
    """Construct an Agents-SDK input items list with text and optional images.

    Image-before-text ordering is gated on the provider's
    ``multimodal_image_before_text`` capability. Gemma 4 recommends placing
    image content before text; Qwen3 has no documented preference and keeps
    the historical text-first ordering. See ``provider_validation.py``.
    """
    from scenesmith.provider_validation import get_active_provider_config

    image_first = bool(
        get_active_provider_config()["capabilities"].get(
            "multimodal_image_before_text", False
        )
    )
    text_item = {"type": "input_text", "text": user_text}
    image_items = [
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{img_b64}",
        }
        for img_b64 in images_b64
    ]
    user_content = (
        image_items + [text_item] if image_first else [text_item] + image_items
    )
    return [{"role": "user", "content": user_content}]


def _run_agent_sync(agent: Agent, input_items: list[dict[str, Any]]) -> Any:
    """Invoke Runner.run synchronously, handling running event loops.

    Runner.run_sync uses asyncio.run() which fails when the calling thread
    already has a running event loop. This happens when the helper is
    invoked from inside an Agents-SDK tool callback (the SDK's outer
    Runner.run is awaiting on this same thread). Detect that case and
    run the coroutine on a worker thread that owns its own loop.

    max_turns is set to 5 — these agents have no tools, so a "turn" here
    is effectively a schema-validation retry rather than a tool round-trip.
    Allowing a handful of retries lets the SDK recover from transient
    structured-output parse failures without making the call multi-minute
    on a slow vLLM endpoint.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return Runner.run_sync(starting_agent=agent, input=input_items, max_turns=5)

    # Running loop detected — execute on a worker thread. Copy the current
    # contextvars (TrackingContext lives in a contextvar) so the tracking
    # subsystem set by extend_tracking_context propagates to the worker.
    parent_ctx = contextvars.copy_context()

    def _runner_in_worker() -> Any:
        return asyncio.run(
            Runner.run(starting_agent=agent, input=input_items, max_turns=5)
        )

    def _runner_with_ctx() -> Any:
        return parent_ctx.run(_runner_in_worker)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner_with_ctx).result()


def run_structured_vlm_call(
    *,
    model: str,
    instructions: str,
    user_text: str = "",
    images_b64: list[str] | None = None,
    user_content: list[dict[str, Any]] | None = None,
    output_type: type[BaseModel] | AgentOutputSchemaBase,
    reasoning_effort: str,
    verbosity: str,
    subsystem: str,
    agent_name: str = "vlm_oneshot",
    timeout_s: float | None = None,
) -> BaseModel:
    """Run a one-shot multimodal call and return a typed Pydantic output.

    Args:
        model: Model id passed to make_chat_completions_model (e.g.
            ``"Qwen/Qwen3.6-35B-A3B-FP8"``).
        instructions: Agent system instructions. Pass ``""`` to put all
            content in the user message.
        user_text: Text portion of the user message.
        images_b64: PNG image base64 strings (no ``data:`` prefix). Each
            becomes an ``input_image`` item in the user message.
        output_type: Pydantic BaseModel class the SDK enforces via
            ``response_format=json_schema``.
        reasoning_effort: One of "minimal" / "low" / "medium" / "high".
        verbosity: One of "low" / "medium" / "high".
        subsystem: Tracking subsystem label written onto the LLMCallRecord
            for this call (e.g. ``"physics_analysis"``).
        agent_name: Logical name for the synthetic single-shot agent.

    Returns:
        Instance of ``output_type`` parsed from the model's final output.
    """
    agent = Agent(
        name=agent_name,
        model=make_chat_completions_model(model),
        tools=[],
        instructions=instructions,
        output_type=output_type,
        model_settings=_build_model_settings(
            reasoning_effort, verbosity, timeout_s=timeout_s
        ),
    )

    if user_content is not None:
        input_items = [{"role": "user", "content": user_content}]
    else:
        input_items = _build_user_input(user_text, images_b64 or [])

    with extend_tracking_context(subsystem=subsystem):
        try:
            result = _run_agent_sync(agent, input_items)
        except Exception as exc:
            console_logger.error(
                "sdk_vlm_call FAILED subsystem=%s agent=%s images=%d "
                "exc_type=%s exc=%s\nmodel_settings.extra_body=%s "
                "model_settings.max_tokens=%s model_settings.verbosity=%s "
                "model_settings.reasoning=%s",
                subsystem,
                agent_name,
                sum(
                    1
                    for it in input_items[0].get("content", [])
                    if isinstance(it, dict) and it.get("type") == "input_image"
                ),
                type(exc).__name__,
                exc,
                agent.model_settings.extra_body,
                agent.model_settings.max_tokens,
                agent.model_settings.verbosity,
                agent.model_settings.reasoning,
            )
            # If we have access to the exception's run_data (MaxTurnsExceeded
            # in 0.6.x carries the raw responses), dump them so we can see
            # what the model actually returned.
            for attr in ("run_data", "result", "raw_responses"):
                val = getattr(exc, attr, None)
                if val is not None:
                    console_logger.error("sdk_vlm_call FAILED exc.%s=%r", attr, val)
            raise

    # final_output_as expects the underlying BaseModel type; when wrapped in
    # AgentOutputSchema we resolve to its output_type for parsing.
    target_type = (
        output_type.output_type
        if isinstance(output_type, AgentOutputSchemaBase)
        else output_type
    )
    try:
        return result.final_output_as(target_type)
    except Exception as exc:
        console_logger.error(
            "sdk_vlm_call parse FAILED subsystem=%s agent=%s exc_type=%s exc=%s "
            "raw_final_output=%r",
            subsystem,
            agent_name,
            type(exc).__name__,
            exc,
            getattr(result, "final_output", None),
        )
        raise


def _split_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split a ``[{role, content}, ...]`` list into (instructions, input items).

    A leading ``role=="system"`` message is hoisted into the Agent's
    ``instructions`` (the canonical SDK system-prompt slot). All remaining
    messages are forwarded as ``Runner.run`` input items. Additional
    ``role=="system"`` messages later in the list are kept in-place as input
    items; callers that need that should know what they're doing.
    """
    instructions = ""
    rest = messages
    if messages and messages[0].get("role") == "system":
        first_content = messages[0].get("content", "")
        if isinstance(first_content, str):
            instructions = first_content
        else:
            # Defensive: a list-shaped system content (rare) is flattened
            # by concatenating any text parts. Non-text parts are dropped
            # since instructions must be a string.
            text_parts = [
                p.get("text", "")
                for p in first_content
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            ]
            instructions = "\n".join(t for t in text_parts if t)
        rest = messages[1:]
    return instructions, rest


def run_structured_vlm_call_messages(
    *,
    model: str,
    messages: list[dict[str, Any]],
    output_type: type[BaseModel] | AgentOutputSchemaBase,
    reasoning_effort: str | None,
    verbosity: str | None,
    subsystem: str,
    agent_name: str = "vlm_structured",
    timeout_s: float | None = None,
) -> BaseModel:
    """Run a single-shot call with a full ``messages`` list and structured output.

    Mirrors ``run_structured_vlm_call`` but accepts the chat-completions-style
    ``messages`` list that direct-path callers (task_analysis_agent,
    morphology_task_agent, reward_spec_agent, llm_editor) were
    passing to ``VLMService.create_completion`` / ``structured_output_call``.

    A leading ``role=="system"`` message becomes the Agent's ``instructions``;
    all other entries are forwarded as ``Runner.run`` input items. Messages
    with list-shaped multimodal content are forwarded as-is.
    """
    instructions, input_items = _split_messages(messages)

    agent = Agent(
        name=agent_name,
        model=make_chat_completions_model(model),
        tools=[],
        instructions=instructions,
        output_type=output_type,
        model_settings=_build_model_settings(
            reasoning_effort, verbosity, timeout_s=timeout_s
        ),
    )

    with extend_tracking_context(subsystem=subsystem):
        try:
            result = _run_agent_sync(agent, input_items)
        except Exception as exc:
            console_logger.error(
                "sdk_vlm_call_messages FAILED subsystem=%s agent=%s "
                "n_messages=%d exc_type=%s exc=%s",
                subsystem,
                agent_name,
                len(messages),
                type(exc).__name__,
                exc,
            )
            raise

    target_type = (
        output_type.output_type
        if isinstance(output_type, AgentOutputSchemaBase)
        else output_type
    )
    try:
        return result.final_output_as(target_type)
    except Exception as exc:
        console_logger.error(
            "sdk_vlm_call_messages parse FAILED subsystem=%s agent=%s "
            "exc_type=%s exc=%s raw_final_output=%r",
            subsystem,
            agent_name,
            type(exc).__name__,
            exc,
            getattr(result, "final_output", None),
        )
        raise


def run_text_vlm_call(
    *,
    model: str,
    instructions: str,
    user_text: str,
    reasoning_effort: str | None,
    verbosity: str | None,
    subsystem: str,
    agent_name: str = "vlm_text",
    timeout_s: float | None = None,
) -> str:
    """Run a single-shot text-only call and return the raw model answer.

    Unlike the structured variants this returns a plain string (no Pydantic
    parsing). Intended for callers that truly need free-form text — primarily
    the provider startup probe. New schema-shaped callers should prefer
    ``run_structured_vlm_call_messages`` so the output is typed.
    """
    agent = Agent(
        name=agent_name,
        model=make_chat_completions_model(model),
        tools=[],
        instructions=instructions,
        model_settings=_build_model_settings(
            reasoning_effort, verbosity, timeout_s=timeout_s
        ),
    )

    input_items: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "input_text", "text": user_text}]}
    ]

    with extend_tracking_context(subsystem=subsystem):
        try:
            result = _run_agent_sync(agent, input_items)
        except Exception as exc:
            console_logger.error(
                "sdk_vlm_text_call FAILED subsystem=%s agent=%s exc_type=%s exc=%s",
                subsystem,
                agent_name,
                type(exc).__name__,
                exc,
            )
            raise

    final = getattr(result, "final_output", None)
    if final is None:
        return ""
    return final if isinstance(final, str) else str(final)
