"""Provider capability validation and request shaping helpers.

This module makes provider capability expectations explicit for TaskSmith's
OpenAI-compatible integrations. The configured provider capabilities are the
source of truth; an optional startup probe can verify that an endpoint matches
the declaration.
"""

from __future__ import annotations

import copy
import logging
import os
import time

from typing import Any

from omegaconf import DictConfig, OmegaConf

from scenesmith.utils.omegaconf import register_resolvers

console_logger = logging.getLogger(__name__)

DEFAULT_PROVIDER_MODEL = "openai/gpt-oss-120b"

DEFAULT_PROVIDER_CONFIG: dict[str, Any] = {
    "base_url": None,
    "model": DEFAULT_PROVIDER_MODEL,
    "api_key_env_var": "PROVIDER_API_KEY",
    "fallback_api_key_env_var": "OPENAI_API_KEY",
    "api": {
        "standard_model_api": "chat_completions",
        "reasoning_model_api": "responses",
    },
    "model_traits": {
        "reasoning_native": False,
    },
    "capabilities": {
        "supports_chat_completions": True,
        "supports_responses_api": False,
        "supports_reasoning_models": False,
        "supports_final_answer_only_extraction": True,
        "supports_reasoning_effort": True,
        "supports_verbosity": True,
        "supports_service_tier": True,
        "supports_summarization_thinking": False,
        # How reasoning_effort is delivered to the provider.
        #   "openai"   — forward reasoning_effort kwarg natively (OpenAI / Azure).
        #   "binary"   — vLLM chat_template_kwargs.enable_thinking on/off
        #                (works for Qwen3 and Gemma 4 on the RCP endpoint; the
        #                historical per-effort thinking_budget map was removed
        #                after RCP probing showed it was silently ignored).
        #   "disabled" — never send thinking-related fields.
        "thinking_control_mode": "openai",
        # Re-inject prior assistant reasoning_content into the next turn's
        # message history (Agents-SDK Converter patch in scenesmith/provider.py).
        # Required by Qwen3 to avoid recomputing CoT every tool turn
        # (30-70% completion-token saving). Must be False for Gemma 4
        # ("No Thinking Content in History" per the HF model card).
        "reinject_prior_reasoning": False,
        # Place input_image items before input_text in multimodal user
        # messages (Gemma 4 recommendation). False matches today's
        # text-first ordering used with Qwen3.
        "multimodal_image_before_text": False,
        # Move image content out of `tool`-role messages into a follow-up
        # `user` message. Works around vLLM Gemma 4 31B's chat-template gap:
        # it returns 500 `Failed to apply prompt replacement for
        # mm_items['image'][0]` whenever a `tool` message carries an image
        # item. The Converter post-pass in scenesmith/provider.py splits
        # `tool([text,image])` into `tool(text)` + `user([image, hint])`
        # while keeping the `tool_call_id` linkage intact. Off by default;
        # enable for any deployment that rejects images-in-tool-messages.
        "tool_image_via_user_message": False,
        # Rewrite malformed ``assistant.tool_calls[i].function.arguments`` to
        # ``"{}"`` before sending. Works around a Gemma 4 + enable_thinking
        # output bug: the model occasionally emits ``{`` + tens of thousands
        # of whitespace chars (no properties, no closing ``}``) as args.
        # Replaying that as history makes vLLM 400 with
        # ``Hosted_vllmException - Expecting property name enclosed in
        # double quotes``. Replacing the unparseable string with an empty
        # object lets the request through; the tool gets re-invoked with
        # no args, and the model self-corrects next turn. Off by default;
        # enable for Gemma 4 with thinking on.
        "sanitize_tool_call_arguments": False,
    },
    "validation": {
        "strict": True,
        "probe": False,
        "probe_timeout_s": 30.0,
        # After an initial probe failure, retry this many times, sleeping
        # probe_retry_interval_s between attempts (cold GPU / slow load).
        "probe_retry_attempts": 5,  # default; overridden by provider.validation.probe_retry_attempts in config
        "probe_retry_interval_s": 60.0,
    },
    # Sampling parameters forwarded to every LLM request. Native OpenAI
    # kwargs (temperature, top_p, presence_penalty, frequency_penalty) are
    # passed to ``client.chat.completions.create`` directly; vLLM-only
    # kwargs (top_k, min_p, repetition_penalty) ride in ``extra_body``.
    # Leave the dict empty to use the provider's defaults. Per-key null
    # values are also treated as "not set".
    "sampling": {},
    # Extra ``chat_template_kwargs`` merged into every request's
    # ``extra_body.chat_template_kwargs`` alongside the ``enable_thinking``
    # flag set by ``build_thinking_extra_body``. Use for vLLM template
    # toggles like ``preserve_thinking: true``. Empty by default.
    "chat_template_kwargs": {},
}

# OpenAI Chat Completions API kwargs forwarded as-is to ``create()``.
# Everything else listed in ``_SAMPLING_KEYS`` goes into ``extra_body``.
_OPENAI_NATIVE_SAMPLING_KEYS: frozenset[str] = frozenset(
    {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
)
_SAMPLING_KEYS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "top_k",
        "min_p",
        "repetition_penalty",
    }
)

STATEFUL_AGENT_KEYS = [
    "floor_plan_agent",
    "furniture_agent",
    "wall_agent",
    "ceiling_agent",
    "manipuland_agent",
]
DIRECT_AGENT_KEYS = [
    "task_analysis_agent",
    "reward_spec_agent",
    "morphology_task_agent",
]

_active_provider_config: dict[str, Any] = copy.deepcopy(DEFAULT_PROVIDER_CONFIG)


def reset_provider_config_for_tests() -> None:
    """Reset active provider config to defaults."""
    global _active_provider_config
    _active_provider_config = copy.deepcopy(DEFAULT_PROVIDER_CONFIG)


def get_active_provider_config() -> dict[str, Any]:
    """Return the active provider config used by runtime request shaping."""
    return copy.deepcopy(_active_provider_config)


def resolve_runtime_config(
    cfg: DictConfig | dict[str, Any],
) -> DictConfig | dict[str, Any]:
    """Resolve OmegaConf interpolations before validation/use."""
    if isinstance(cfg, DictConfig):
        register_resolvers()
        OmegaConf.resolve(cfg)
    return cfg


def initialize_provider_runtime(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Resolve provider config, validate it, optionally probe, and cache it."""
    resolve_runtime_config(cfg)
    provider_config = _merge_provider_config(cfg)
    violations = _collect_validation_violations(cfg, provider_config)
    if violations:
        message = _format_validation_error(violations, provider_config)
        if provider_config["validation"]["strict"]:
            raise ValueError(message)
        console_logger.warning(message)

    if provider_config["validation"]["probe"]:
        _probe_provider_capabilities(cfg, provider_config)

    global _active_provider_config
    _active_provider_config = provider_config
    return copy.deepcopy(provider_config)


def get_thinking_control_mode(
    provider_config: dict[str, Any] | None = None,
) -> str:
    """Return the active provider's ``thinking_control_mode``.

    Values: ``"openai"`` (forward reasoning_effort kwarg natively),
    ``"binary"`` (vLLM ``chat_template_kwargs.enable_thinking``), or
    ``"disabled"`` (never send thinking-related fields).
    """
    provider = provider_config or get_active_provider_config()
    return provider["capabilities"].get("thinking_control_mode", "openai")


def provider_uses_chat_template_thinking(
    provider_config: dict[str, Any] | None = None,
) -> bool:
    """True when the provider expects thinking control via vLLM chat_template_kwargs."""
    return get_thinking_control_mode(provider_config) == "binary"


def build_thinking_extra_body(
    reasoning_effort: str | None,
    provider_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate reasoning_effort into a binary vLLM thinking-on/off extra_body.

    Mapping when ``thinking_control_mode == "binary"``:
      - disabled (None / "" / "none" / False) -> {}
      - "minimal"                              -> {"chat_template_kwargs": {"enable_thinking": False}}
      - any other truthy effort (low/medium/high/...) -> {"chat_template_kwargs": {"enable_thinking": True}}

    Returns {} for any other ``thinking_control_mode`` (``"openai"`` /
    ``"disabled"``). Empirical probing of the RCP vLLM endpoint on
    2026-05-20 showed that ``thinking_budget`` (and every other budget-style
    kwarg) is silently ignored on both Qwen3 and Gemma 4 deployments —
    only ``enable_thinking`` is honored. See
    ``.docs/canonicalization_vlm_investigation.md`` §9 and
    ``scripts/debug_qwen_thinking_kwargs.py`` for evidence.
    """
    if not _is_enabled(reasoning_effort):
        return {}
    if get_thinking_control_mode(provider_config) != "binary":
        return {}
    if reasoning_effort == "minimal":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {"chat_template_kwargs": {"enable_thinking": True}}


def split_provider_sampling(
    provider_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (openai_native_kwargs, extra_body_sampling) from provider.sampling.

    Splits the configured sampling dict into:
      - ``openai_native_kwargs``: forwarded directly to ``client.chat.completions.create``
        (and to ``ModelSettings`` field names that match: ``temperature``,
        ``top_p``, ``presence_penalty``, ``frequency_penalty``).
      - ``extra_body_sampling``: vLLM-only sampling kwargs (``top_k``, ``min_p``,
        ``repetition_penalty``) that must ride in ``extra_body``.

    Keys with ``None`` values are dropped (treated as "not set"). Unknown
    keys are ignored with a debug log so typos don't silently affect runs.
    """
    provider = provider_config or get_active_provider_config()
    sampling = provider.get("sampling") or {}
    native: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in sampling.items():
        if value is None:
            continue
        if key in _OPENAI_NATIVE_SAMPLING_KEYS:
            native[key] = value
        elif key in _SAMPLING_KEYS:
            extra[key] = value
        else:
            console_logger.debug(
                "Ignoring unknown provider.sampling key %r (allowed: %s)",
                key,
                sorted(_SAMPLING_KEYS),
            )
    return native, extra


def get_provider_chat_template_kwargs(
    provider_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return user-configured ``chat_template_kwargs`` (without enable_thinking).

    The returned dict is meant to be merged into
    ``extra_body["chat_template_kwargs"]`` alongside the ``enable_thinking``
    flag set by ``build_thinking_extra_body``. Caller is responsible for the
    merge so the thinking flag stays authoritative when both define the same
    key.
    """
    provider = provider_config or get_active_provider_config()
    return dict(provider.get("chat_template_kwargs") or {})


def merge_chat_template_kwargs_into_extra_body(
    extra_body: dict[str, Any],
    provider_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge provider.chat_template_kwargs into extra_body.chat_template_kwargs.

    Existing keys (e.g. ``enable_thinking`` set earlier from
    ``reasoning_effort``) win over user-configured values to keep the
    thinking-control invariant intact. Returns the same ``extra_body`` for
    chaining.
    """
    user_kwargs = get_provider_chat_template_kwargs(provider_config)
    if not user_kwargs:
        return extra_body
    existing = dict(extra_body.get("chat_template_kwargs") or {})
    for key, value in user_kwargs.items():
        existing.setdefault(key, value)
    extra_body["chat_template_kwargs"] = existing
    return extra_body


def apply_chat_completion_capabilities(
    *,
    kwargs: dict[str, Any],
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    service_tier: str | None = None,
    provider_config: dict[str, Any] | None = None,
    subsystem: str = "chat completion request",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply provider-gated chat-completion options.

    Returns (kwargs, extra_body). extra_body is non-empty when the provider
    uses vLLM-style chat_template_kwargs (e.g. to disable thinking).
    """
    provider = provider_config or get_active_provider_config()
    capabilities = provider["capabilities"]
    extra_body: dict[str, Any] = {}

    if _is_enabled(reasoning_effort):
        thinking_extra = build_thinking_extra_body(reasoning_effort, provider)
        if thinking_extra:
            # Binary chat_template_kwargs path takes precedence over the OpenAI
            # reasoning_effort kwarg (which vLLM-style providers ignore).
            extra_body.update(thinking_extra)
        elif (
            get_thinking_control_mode(provider) == "openai"
            and reasoning_effort != "minimal"
            and capabilities["supports_reasoning_effort"]
        ):
            # OpenAI-native path: forward reasoning_effort directly. "minimal"
            # is not a valid OpenAI value and is dropped silently. Other modes
            # ("disabled", or "binary" with effort missing from the budget map)
            # intentionally send nothing.
            kwargs["reasoning_effort"] = reasoning_effort

    if _is_enabled(verbosity):
        if capabilities["supports_verbosity"]:
            kwargs["verbosity"] = verbosity

    if _is_enabled(service_tier):
        if capabilities["supports_service_tier"]:
            kwargs["service_tier"] = service_tier

    native_sampling, extra_sampling = split_provider_sampling(provider)
    for key, value in native_sampling.items():
        kwargs.setdefault(key, value)
    for key, value in extra_sampling.items():
        extra_body.setdefault(key, value)

    merge_chat_template_kwargs_into_extra_body(extra_body, provider)

    return kwargs, extra_body


def provider_supports_summarization_thinking(
    provider_config: dict[str, Any] | None = None,
) -> bool:
    provider = provider_config or get_active_provider_config()
    return bool(provider["capabilities"]["supports_summarization_thinking"])


def _merge_provider_config(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_PROVIDER_CONFIG)
    settings["base_url"] = os.environ.get("PROVIDER_BASE_URL")
    settings["model"] = os.environ.get("PROVIDER_MODEL", DEFAULT_PROVIDER_MODEL)

    provider_cfg = _cfg_get(cfg, "provider")
    if provider_cfg is None:
        return settings

    provider_dict = _to_plain_dict(provider_cfg)
    for key in (
        "base_url",
        "model",
        "api_key_env_var",
        "fallback_api_key_env_var",
    ):
        value = provider_dict.get(key)
        if value is not None:
            settings[key] = value

    settings["api"].update(provider_dict.get("api", {}))
    settings["model_traits"].update(provider_dict.get("model_traits", {}))
    settings["capabilities"].update(provider_dict.get("capabilities", {}))
    settings["validation"].update(provider_dict.get("validation", {}))
    settings["sampling"].update(provider_dict.get("sampling") or {})
    settings["chat_template_kwargs"].update(
        provider_dict.get("chat_template_kwargs") or {}
    )
    return settings


def _collect_validation_violations(
    cfg: DictConfig | dict[str, Any], provider_config: dict[str, Any]
) -> list[str]:
    violations: list[str] = []
    capabilities = provider_config["capabilities"]
    violations.extend(_collect_api_path_violations(cfg, provider_config))

    if not capabilities["supports_reasoning_effort"]:
        violations.extend(_collect_reasoning_violations(cfg))
    if not capabilities["supports_verbosity"]:
        violations.extend(_collect_verbosity_violations(cfg))
    if not capabilities["supports_service_tier"]:
        violations.extend(_collect_service_tier_violations(cfg))
    if not capabilities["supports_summarization_thinking"]:
        violations.extend(_collect_summarization_thinking_violations(cfg))

    return violations


def _collect_api_path_violations(
    cfg: DictConfig | dict[str, Any], provider_config: dict[str, Any]
) -> list[str]:
    violations: list[str] = []
    capabilities = provider_config["capabilities"]
    api_cfg = provider_config["api"]
    model_traits = provider_config["model_traits"]

    for api_name, api_value in api_cfg.items():
        if api_value not in {"chat_completions", "responses"}:
            violations.append(
                f"provider.api.{api_name} must be 'chat_completions' or 'responses', "
                f"got {api_value!r}"
            )

    if not capabilities["supports_chat_completions"]:
        violations.extend(_collect_chat_completions_dependency_violations(cfg))

    if model_traits["reasoning_native"]:
        selected_api = api_cfg["reasoning_model_api"]
        if not capabilities["supports_reasoning_models"]:
            violations.append(
                "provider.model_traits.reasoning_native is true but "
                "provider.capabilities.supports_reasoning_models is false"
            )
        if not capabilities["supports_final_answer_only_extraction"]:
            violations.append(
                "provider.model_traits.reasoning_native is true but "
                "provider.capabilities.supports_final_answer_only_extraction is false"
            )
        if (
            selected_api == "chat_completions"
            and not capabilities["supports_chat_completions"]
        ):
            violations.append(
                "provider.model_traits.reasoning_native is true, "
                "provider.api.reasoning_model_api is chat_completions, but "
                "provider.capabilities.supports_chat_completions is false"
            )
        if selected_api == "responses" and not capabilities["supports_responses_api"]:
            violations.append(
                "provider.model_traits.reasoning_native is true, "
                "provider.api.reasoning_model_api is responses, but "
                "provider.capabilities.supports_responses_api is false"
            )
        if selected_api != "chat_completions":
            for agent_key in STATEFUL_AGENT_KEYS:
                if _cfg_get(cfg, agent_key) is not None:
                    violations.append(
                        f"{agent_key} uses the Agents SDK chat_completions path but "
                        "provider.model_traits.reasoning_native is true and "
                        f"provider.api.reasoning_model_api is {selected_api}"
                    )

    return violations


def _collect_chat_completions_dependency_violations(
    cfg: DictConfig | dict[str, Any],
) -> list[str]:
    violations: list[str] = []
    for agent_key in STATEFUL_AGENT_KEYS + DIRECT_AGENT_KEYS:
        if _cfg_get(cfg, agent_key) is not None:
            violations.append(
                f"{agent_key} requires chat_completions support but "
                "provider.capabilities.supports_chat_completions is false"
            )

    if _cfg_get(cfg, "domain_randomization") is not None:
        violations.append(
            "domain_randomization requires chat_completions support but "
            "provider.capabilities.supports_chat_completions is false"
        )
    return violations


def _collect_reasoning_violations(cfg: DictConfig | dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for agent_key in STATEFUL_AGENT_KEYS:
        reasoning_cfg = _cfg_get(cfg, f"{agent_key}.openai.reasoning_effort")
        if _has_enabled_nested_value(reasoning_cfg):
            violations.append(
                f"{agent_key} uses openai.reasoning_effort but "
                "provider.capabilities.supports_reasoning_effort is false"
            )

    for agent_key in DIRECT_AGENT_KEYS:
        reasoning_cfg = _cfg_get(cfg, f"{agent_key}.openai.reasoning_effort")
        if _has_enabled_nested_value(reasoning_cfg):
            violations.append(
                f"{agent_key} uses VLM reasoning_effort but "
                "provider.capabilities.supports_reasoning_effort is false"
            )
    return violations


def _collect_verbosity_violations(cfg: DictConfig | dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for agent_key in STATEFUL_AGENT_KEYS:
        verbosity_cfg = _cfg_get(cfg, f"{agent_key}.openai.verbosity")
        if _has_enabled_nested_value(verbosity_cfg):
            violations.append(
                f"{agent_key} uses openai.verbosity but "
                "provider.capabilities.supports_verbosity is false"
            )

    for agent_key in DIRECT_AGENT_KEYS:
        verbosity_cfg = _cfg_get(cfg, f"{agent_key}.openai.verbosity")
        if _has_enabled_nested_value(verbosity_cfg):
            violations.append(
                f"{agent_key} uses VLM verbosity but "
                "provider.capabilities.supports_verbosity is false"
            )
    return violations


def _collect_service_tier_violations(cfg: DictConfig | dict[str, Any]) -> list[str]:
    violations: list[str] = []
    service_tier = _cfg_get(cfg, "openai.service_tier")
    if _is_enabled(service_tier):
        violations.append(
            "openai.service_tier is set but "
            "provider.capabilities.supports_service_tier is false"
        )

    for agent_key in DIRECT_AGENT_KEYS:
        agent_service_tier = _cfg_get(cfg, f"{agent_key}.openai.service_tier")
        if _is_enabled(agent_service_tier):
            violations.append(
                f"{agent_key}.openai.service_tier is set but "
                "provider.capabilities.supports_service_tier is false"
            )
    return violations


def _collect_summarization_thinking_violations(
    cfg: DictConfig | dict[str, Any],
) -> list[str]:
    violations: list[str] = []
    for agent_key in STATEFUL_AGENT_KEYS:
        value = _cfg_get(cfg, f"{agent_key}.session_memory.summarization_thinking")
        if _is_enabled(value):
            violations.append(
                f"{agent_key}.session_memory.summarization_thinking is set but "
                "provider.capabilities.supports_summarization_thinking is false"
            )
    return violations


def _probe_provider_capabilities(
    cfg: DictConfig | dict[str, Any], provider_config: dict[str, Any]
) -> None:
    """Optionally confirm declared provider capabilities with a tiny request."""
    # Imported locally to defer the agents-SDK import — this module is
    # imported very early during Hydra startup and sdk_vlm_runner depends
    # back on get_active_provider_config which lives here.
    from scenesmith.agent_utils.sdk_vlm_runner import run_text_vlm_call

    probe_model = provider_config["model"] or _cfg_get(cfg, "openai.model")
    if not probe_model:
        raise ValueError("Provider probe requires a configured model name.")

    provider_label = _provider_label(provider_config)
    validation = provider_config["validation"]
    timeout_s = validation["probe_timeout_s"]
    retry_attempts = max(0, int(validation.get("probe_retry_attempts", 0)))
    interval_s = float(validation.get("probe_retry_interval_s", 60.0))
    total_attempts = 1 + retry_attempts
    selected_api = _get_selected_probe_api(provider_config)

    console_logger.info(
        "Running provider startup probe against %s with model %s via %s "
        "(timeout=%ss, max_attempts=%s, retry_interval=%ss)",
        provider_label,
        probe_model,
        selected_api,
        timeout_s,
        total_attempts,
        interval_s,
    )

    last_exc: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        try:
            print(
                f"[provider probe] status check attempt {attempt}/{total_attempts} — "
                f"{provider_label} model={probe_model} api={selected_api} "
                f"timeout={timeout_s}s",
                flush=True,
            )
            capabilities = provider_config["capabilities"]
            text = run_text_vlm_call(
                model=probe_model,
                instructions="",
                user_text="Reply with exactly OK.",
                reasoning_effort=(
                    "low" if capabilities["supports_reasoning_effort"] else None
                ),
                verbosity="low" if capabilities["supports_verbosity"] else None,
                subsystem="provider_startup_probe",
                agent_name="provider_probe",
                timeout_s=timeout_s,
            )
            print("[provider probe] probe request finished", flush=True)
            if "OK" not in text:
                raise ValueError(
                    "Provider startup probe completed but did not return the expected "
                    f"final answer text. Received: {text!r}"
                )
            print("[provider probe] probe request succeeded", flush=True)
            return
        except Exception as exc:
            last_exc = exc
            print(
                f"[provider probe] attempt {attempt}/{total_attempts} failed: {exc}",
                flush=True,
            )
            if attempt >= total_attempts:
                break
            console_logger.warning(
                "Provider probe sleeping %.0fs before retry (%s/%s attempts used)",
                interval_s,
                attempt,
                total_attempts,
            )
            print(
                f"[provider probe] waiting {interval_s}s before next attempt...",
                flush=True,
            )
            time.sleep(interval_s)

    assert last_exc is not None
    exc = last_exc
    error_text = str(exc).lower()
    if "timed out" in error_text or "timeout" in error_text:
        raise ValueError(
            "Provider startup probe timed out before it could verify the declared "
            f"capabilities for endpoint {provider_label}. This does not prove the "
            "capabilities are wrong; the endpoint may simply be slow. "
            "Retry with a larger probe timeout, more probe retries, or disable probing "
            "for this run."
        ) from exc
    raise ValueError(
        "Provider startup probe failed. The declared provider capabilities do not "
        f"match endpoint {provider_label}. Error: {exc}"
    ) from exc


def _format_validation_error(
    violations: list[str], provider_config: dict[str, Any]
) -> str:
    provider_label = _provider_label(provider_config)
    lines = [
        "Provider capability validation failed.",
        f"Provider: {provider_label}",
        "Fix the provider capability declarations or the incompatible knob settings:",
    ]
    lines.extend(f"- {violation}" for violation in violations)
    return "\n".join(lines)


def _provider_label(provider_config: dict[str, Any]) -> str:
    return provider_config["base_url"] or "default OpenAI base URL"


def _get_selected_probe_api(provider_config: dict[str, Any]) -> str:
    if provider_config["model_traits"]["reasoning_native"]:
        return provider_config["api"]["reasoning_model_api"]
    return provider_config["api"]["standard_model_api"]


def _cfg_get(cfg: DictConfig | dict[str, Any], dotted_path: str) -> Any:
    current: Any = cfg
    for part in dotted_path.split("."):
        if isinstance(current, DictConfig):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        else:
            return None
    return current


def _has_enabled_nested_value(value: Any) -> bool:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return any(_has_enabled_nested_value(v) for v in value.values())
    return _is_enabled(value)


def _is_enabled(value: Any) -> bool:
    if value in (None, "", False):
        return False
    if isinstance(value, str) and value.lower() == "none":
        return False
    return True


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    return dict(value)
