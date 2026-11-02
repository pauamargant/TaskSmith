"""Centralized factory for provider-configured OpenAI clients.

Reads PROVIDER_BASE_URL, PROVIDER_API_KEY, and PROVIDER_MODEL from the
environment (or from a .env file in the project root) so all modules point
at the same inference endpoint. Falls back to standard OPENAI_API_KEY /
default base URL when env vars are absent, keeping compatibility with vanilla
OpenAI.
"""

import logging
import os

from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI

console_logger = logging.getLogger(__name__)

# Environment variable names.
_ENV_BASE_URL = "PROVIDER_BASE_URL"
_ENV_API_KEY = "PROVIDER_API_KEY"
_ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
_ENV_MODEL = "PROVIDER_MODEL"
_DEFAULT_MODEL = "openai/gpt-oss-120b"


def resolve_model_name(
    cfg: Any,
    agent_section: str | None = None,
) -> str:
    """Resolve the model name a caller should use, with a clear fallback chain.

    Order tried (first non-empty wins):

      1. ``cfg.<agent_section>.openai.model`` (when ``agent_section`` is given;
         the canonical Hydra path for per-agent overrides — interpolates
         ``${provider.model}`` when the full config is composed).
      2. ``cfg.openai.model`` (top-level openai block — uncommon but supported).
      3. ``cfg.provider.model`` (env-backed ``PROVIDER_MODEL``; works even for
         callers that build ``cfg`` from a partial dict).
      4. ``os.environ.get("PROVIDER_MODEL")`` (in case the caller didn't
         thread the env-backed cfg through at all).
      5. ``_DEFAULT_MODEL`` last-resort sentinel — but logged at WARNING so
         the issue is visible in stdout (RCP's LiteLLM proxy used to accept
         this as a routing alias, hiding the misconfiguration; strict vLLM
         endpoints 404 on it).

    ``cfg`` is duck-typed: anything supporting ``.get()`` on a section name
    (Hydra ``DictConfig``, plain dict, or a ``dict``-like) works.

    Returns:
        The resolved model name string.
    """

    def _section(node: Any, name: str) -> Any:
        if node is None:
            return None
        # Hydra DictConfig and plain dict both support .get on string keys.
        try:
            value = node.get(name, None)
        except Exception:  # noqa: BLE001 — some attrs raise on missing keys
            value = None
        return value

    def _model_of(section: Any) -> str | None:
        sub = _section(section, "openai")
        if sub is None:
            return None
        value = _section(sub, "model")
        if isinstance(value, str) and value:
            return value
        return None

    candidate: str | None = None

    if agent_section:
        candidate = _model_of(_section(cfg, agent_section))
    if not candidate:
        candidate = _model_of(cfg)
    if not candidate:
        provider_section = _section(cfg, "provider")
        if provider_section is not None:
            value = _section(provider_section, "model")
            if isinstance(value, str) and value:
                candidate = value
    if not candidate:
        env_value = os.environ.get(_ENV_MODEL)
        if env_value:
            candidate = env_value
    if not candidate:
        console_logger.warning(
            "resolve_model_name: no model configured for agent_section=%r — "
            "falling back to sentinel %r. Set PROVIDER_MODEL (or "
            "cfg.provider.model) to silence this warning. Strict vLLM "
            "endpoints will 404 on the sentinel.",
            agent_section,
            _DEFAULT_MODEL,
        )
        candidate = _DEFAULT_MODEL
    return candidate


# Provider client robustness knobs (used for Agents SDK + direct OpenAI clients).
_ENV_TIMEOUT_S = "PROVIDER_TIMEOUT_S"
_ENV_MAX_RETRIES = "PROVIDER_MAX_RETRIES"
_DEFAULT_TIMEOUT_S = float(os.environ.get(_ENV_TIMEOUT_S, "360"))
_DEFAULT_MAX_RETRIES = int(os.environ.get(_ENV_MAX_RETRIES, "5"))

_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv(path: Path) -> None:
    """Parse a simple KEY="VALUE" .env file and populate os.environ for any
    variable that is not already set.  Handles optional ``export`` prefix."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # Strip optional "export " prefix.
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env once at import time so env vars defined there are visible.
_load_dotenv(_DOTENV_PATH)


def _patch_agents_converter_for_reasoning() -> None:
    """Preserve ``reasoning_content`` *and* tool-output images across Agents-SDK turns.

    Two upstream behaviours we override here:

    1. ``reasoning_content`` preservation (Qwen-specific, opt-in via
       ``provider.capabilities.reinject_prior_reasoning``). vLLM with
       ``--reasoning-parser qwen3`` returns ``reasoning_content`` on the
       response message. The Agents SDK captures it in
       ``ResponseReasoningItem.summary`` via ``message_to_output_items``, but
       ``items_to_messages`` reads ``content`` (the Anthropic
       ``thinking_blocks`` shape), not ``summary``, so the reasoning is
       silently dropped when reconstructing the next turn's messages. Qwen
       explicitly recommends passing reasoning back in history; without it
       the model re-derives its chain of thought on every tool turn,
       costing 30-70% extra completion tokens per turn (measured empirically
       against a local vLLM pod serving Qwen3.6-35B-A3B-FP8). Gemma 4
       requires the *opposite* ("No Thinking Content in History" per the HF
       model card), so this pass is gated on the capability flag and only
       runs when the active provider config opts in.

    2. Tool-output image preservation (always-on, general SDK fix). The
       chat-completions branch of the SDK's Converter defaults
       ``preserve_tool_output_all_content=False``, which silently strips
       images returned via ``ToolOutputImage`` from ``function_call_output``
       items (only the text parts survive). This is the long-standing OpenAI
       Chat Completions limitation, but vLLM / LiteLLM-fronted providers
       accept images in tool messages just fine. The upstream LiteLLM model
       wrapper already opts in to the new flag (LiteLLM's model wrapper);
       ``OpenAIChatCompletionsModel`` does not, so direct chat-completions
       traffic against vLLM gets text-only tool payloads even when images
       were returned. The model is then told "here are N rendered views"
       without any actual image data, which on vLLM can trigger long
       phantom-image reasoning that runs to the model context limit. We
       default the flag to True so images survive — applied unconditionally
       because it is not provider-specific.

    The patch wraps ``Converter.items_to_messages`` (signature changed in
    agents SDK (>=0.6.5) to include ``model`` and ``preserve_tool_output_all_content``
    parameters). We accept both new and legacy callsites, forward ``model``
    and ``preserve_thinking_blocks`` unchanged, and flip the default of
    ``preserve_tool_output_all_content`` to True. Reasoning re-injection runs
    after the upstream conversion, gated on the capability flag.
    """
    try:
        from agents.models.openai_chatcompletions import Converter
    except Exception:  # noqa: BLE001 — agents may be absent in lint envs
        return

    _original = Converter.items_to_messages.__func__

    _NON_ASST = "non_asst"
    _ASST = "asst"
    _REASON = "reason"
    _SKIP = "skip"

    def _classify(item: object) -> str:
        if not isinstance(item, dict):
            return _SKIP
        t = item.get("type")
        if t == "reasoning":
            return _REASON
        if t == "message":
            return _ASST if item.get("role") == "assistant" else _NON_ASST
        if t in ("function_call", "function_tool_call", "file_search_call"):
            return _ASST
        if t in ("function_call_output", "tool_call_output"):
            return _NON_ASST
        # Bare {role, content} input messages (no type key)
        if "role" in item and "content" in item:
            return _ASST if item["role"] == "assistant" else _NON_ASST
        return _SKIP

    _IMAGE_CONTENT_TYPES = ("image_url", "input_image", "image")

    def _split_tool_images(msgs: list) -> list:
        """Move image items out of ``tool``-role messages into a follow-up ``user`` message.

        Works around vLLM Gemma 4's chat template gap: it returns 500
        ``Failed to apply prompt replacement for mm_items['image'][0]`` for
        any ``tool`` message that carries an image content item. We rewrite

            tool(content=[{text}, {image_url}, ...])
        into
            tool(content="<joined text>")
            user(content=[{image_url}, ..., {text: "^ image result from tool call ..."}])

        keeping the ``tool_call_id`` linkage intact (the assistant's tool call
        is still answered by the immediately-following tool message). Tool
        messages with string content, or no image items, pass through.
        """
        out: list = []
        for m in msgs:
            if not (isinstance(m, dict) and m.get("role") == "tool"):
                out.append(m)
                continue
            content = m.get("content")
            if not isinstance(content, list):
                out.append(m)
                continue
            images = [
                c
                for c in content
                if isinstance(c, dict) and c.get("type") in _IMAGE_CONTENT_TYPES
            ]
            if not images:
                out.append(m)
                continue
            texts = [
                c.get("text", "")
                for c in content
                if isinstance(c, dict)
                and c.get("type") not in _IMAGE_CONTENT_TYPES
                and isinstance(c.get("text"), str)
            ]
            tool_msg = dict(m)
            tool_msg["content"] = "\n".join(t for t in texts if t) or (
                "(image result returned in the next message)"
            )
            out.append(tool_msg)
            tcid = m.get("tool_call_id", "")
            hint = {
                "type": "text",
                "text": f"^ image result from tool call (tool_call_id={tcid}).",
            }
            out.append({"role": "user", "content": [*images, hint]})
        return out

    def _sanitize_tool_call_arguments(msgs: list) -> list:
        """Replace malformed ``tool_calls[i].function.arguments`` with ``"{}"``.

        Works around a Gemma 4 + ``enable_thinking=True`` output-quality bug:
        Gemma occasionally emits ``{`` followed by tens of thousands of
        whitespace chars (no properties, no closing ``}``) as a tool-call
        ``arguments`` value. When that response is replayed as assistant
        history on the next turn, vLLM's gemma4 tool-call-parser calls
        ``json.loads`` on the string and the server 400s with
        ``Hosted_vllmException - Expecting property name enclosed in double
        quotes``. Replacing the unparseable string with an empty-object JSON
        is the smallest deterministic fix: the request goes through, the tool
        gets re-invoked with no args, and the model can self-correct on the
        next turn.

        Only assistant messages are inspected; tool/user/system messages pass
        through unchanged.
        """
        import json as _json

        out: list = []
        for m in msgs:
            if not (isinstance(m, dict) and m.get("role") == "assistant"):
                out.append(m)
                continue
            tcs = m.get("tool_calls")
            if not tcs:
                out.append(m)
                continue
            changed = False
            new_tcs = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    new_tcs.append(tc)
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    new_tcs.append(tc)
                    continue
                args = fn.get("arguments")
                if not isinstance(args, str) or args == "":
                    new_tcs.append(tc)
                    continue
                try:
                    _json.loads(args)
                    new_tcs.append(tc)
                    continue
                except Exception:  # noqa: BLE001 — any parse failure → sanitize
                    pass
                new_fn = dict(fn)
                new_fn["arguments"] = "{}"
                new_tc = dict(tc)
                new_tc["function"] = new_fn
                new_tcs.append(new_tc)
                changed = True
                try:
                    console_logger.warning(
                        "sanitized malformed tool_call.arguments: "
                        "name=%s len=%d (replaced with '{}')",
                        fn.get("name"),
                        len(args),
                    )
                except Exception:  # noqa: BLE001
                    pass
            if changed:
                new_m = dict(m)
                new_m["tool_calls"] = new_tcs
                out.append(new_m)
            else:
                out.append(m)
        return out

    def _patched(  # type: ignore[no-untyped-def]
        cls,
        items,
        model=None,
        preserve_thinking_blocks: bool = False,
        preserve_tool_output_all_content: bool = True,
    ):
        # Forward to upstream — only pass kwargs the installed signature accepts
        # so we tolerate both legacy (0.6.4) and current (0.6.5+) signatures.
        kwargs = {"preserve_thinking_blocks": preserve_thinking_blocks}
        try:
            import inspect

            sig = inspect.signature(_original)
            if "model" in sig.parameters:
                kwargs["model"] = model
            if "preserve_tool_output_all_content" in sig.parameters:
                kwargs["preserve_tool_output_all_content"] = (
                    preserve_tool_output_all_content
                )
        except Exception:  # noqa: BLE001 — fall back to legacy signature
            pass
        msgs = _original(cls, items, **kwargs)

        # Tool-image → user-message split (post-pass). Gated on the active
        # provider's ``tool_image_via_user_message`` capability. Required for
        # vLLM Gemma 4 31B, harmless elsewhere. Runs BEFORE reasoning
        # re-injection so the latter sees the rewritten message list.
        try:
            from scenesmith.provider_validation import get_active_provider_config

            _caps = get_active_provider_config()["capabilities"]
        except Exception:  # noqa: BLE001 — never break the Converter on a config read
            _caps = {}
        if _caps.get("tool_image_via_user_message", False):
            msgs = _split_tool_images(msgs)

        # Malformed tool_call.arguments sanitization (post-pass). Gated on
        # the active provider's ``sanitize_tool_call_arguments`` capability.
        # Works around a Gemma 4 + enable_thinking output bug where the model
        # emits ``{`` + tens of thousands of whitespace chars instead of valid
        # JSON args; replaying that as history makes vLLM 400.
        if _caps.get("sanitize_tool_call_arguments", False):
            msgs = _sanitize_tool_call_arguments(msgs)

        # Reasoning re-injection (post-pass). Gated at request-build time on
        # the active provider's ``reinject_prior_reasoning`` capability — must
        # be off for Gemma 4, on for Qwen3. Read lazily here (rather than at
        # patch-install time) because the patch is installed at import while
        # ``initialize_provider_runtime`` runs later in the bootstrap.
        if not isinstance(items, list):
            return msgs
        if not _caps.get("reinject_prior_reasoning", False):
            return msgs
        asst_msg_idxs = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        asst_block_idx = -1
        prev_was_asst = False
        pending: str | None = None
        for item in items:
            c = _classify(item)
            if c == _REASON:
                summary = item.get("summary") or []
                texts = [
                    s.get("text", "")
                    for s in summary
                    if isinstance(s, dict) and s.get("type") == "summary_text"
                ]
                if texts:
                    pending = "\n".join(texts)
                continue
            if c == _ASST:
                if not prev_was_asst:
                    asst_block_idx += 1
                    if pending is not None and asst_block_idx < len(asst_msg_idxs):
                        msgs[asst_msg_idxs[asst_block_idx]][
                            "reasoning_content"
                        ] = pending
                    pending = None
                prev_was_asst = True
            elif c == _NON_ASST:
                prev_was_asst = False
                pending = None
        return msgs

    Converter.items_to_messages = classmethod(_patched)


_patch_agents_converter_for_reasoning()


# Convenience exports — but factories below read env at call time to avoid
# stale values when env is set after import (e.g. by Hydra).
PROVIDER_BASE_URL: str | None = os.environ.get(_ENV_BASE_URL)
PROVIDER_API_KEY: str | None = os.environ.get(_ENV_API_KEY) or os.environ.get(
    _ENV_OPENAI_API_KEY
)
PROVIDER_MODEL: str = os.environ.get(_ENV_MODEL, _DEFAULT_MODEL)

__all__ = [
    "PROVIDER_BASE_URL",
    "PROVIDER_API_KEY",
    "PROVIDER_MODEL",
    "get_provider_connection_settings",
    "load_project_dotenv",
    "make_sync_client",
    "make_async_client",
    "make_chat_completions_model",
]


def _read_base_url() -> str | None:
    """Read base URL from env at call time."""
    return os.environ.get(_ENV_BASE_URL)


def _read_api_key() -> str | None:
    """Read API key from env at call time."""
    return os.environ.get(_ENV_API_KEY) or os.environ.get(_ENV_OPENAI_API_KEY)


def load_project_dotenv() -> None:
    """Load the repo-root `.env` file into the current process environment."""
    _load_dotenv(_DOTENV_PATH)


def get_provider_connection_settings(
    provider_config: dict[str, object] | None = None,
) -> dict[str, str | None]:
    """Return provider connection settings shared by the Agents SDK client and the direct openai-client adapter."""
    settings = provider_config or {}
    api_key_env_var = settings.get("api_key_env_var", _ENV_API_KEY)
    fallback_api_key_env_var = settings.get(
        "fallback_api_key_env_var", _ENV_OPENAI_API_KEY
    )

    base_url = settings.get("base_url")
    if base_url is None:
        base_url = _read_base_url()

    model = settings.get("model")
    if model is None:
        model = os.environ.get(_ENV_MODEL, _DEFAULT_MODEL)

    api_key = os.environ.get(str(api_key_env_var))
    if api_key is None and fallback_api_key_env_var:
        api_key = os.environ.get(str(fallback_api_key_env_var))

    return {
        "base_url": str(base_url) if base_url is not None else None,
        "api_key": api_key,
        "model": str(model),
    }


def make_sync_client(**kwargs) -> OpenAI:
    """Return a synchronous OpenAI client pointed at the configured provider."""
    return OpenAI(
        base_url=_read_base_url(),
        api_key=_read_api_key(),
        timeout=_DEFAULT_TIMEOUT_S,
        max_retries=_DEFAULT_MAX_RETRIES,
        **kwargs,
    )


def make_async_client(**kwargs) -> AsyncOpenAI:
    """Return an async OpenAI client pointed at the configured provider."""
    return AsyncOpenAI(
        base_url=_read_base_url(),
        api_key=_read_api_key(),
        timeout=_DEFAULT_TIMEOUT_S,
        max_retries=_DEFAULT_MAX_RETRIES,
        **kwargs,
    )


# Cached async client for make_chat_completions_model (one per process).
_cached_async_client: AsyncOpenAI | None = None


def make_chat_completions_model(model_name: str):
    """Return an OpenAIChatCompletionsModel wrapping *model_name* + a cached client.

    The Agents SDK strips org prefixes from plain model-name strings
    (e.g. "openai/gpt-oss-120b" -> "gpt-oss-120b").  Wrapping in
    OpenAIChatCompletionsModel bypasses that parsing and sends the full name.

    When ``experiment.tracking.enabled`` and
    ``experiment.tracking.agents_sdk_wrapping`` are both true (the
    defaults), the returned model is further wrapped by
    ``TrackedChatCompletionsModel`` so every per-API-call's input
    images and tokens are recorded. When either flag is false the raw
    inner model is returned — behaviour is then identical to the
    pre-refactor code on the Agents SDK hot path.
    """
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

    global _cached_async_client
    if _cached_async_client is None:
        _cached_async_client = make_async_client()

    inner = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=_cached_async_client,
    )

    # Local import keeps provider.py importable even if the tracking
    # package somehow fails to import — provider is a critical path.
    try:
        from scenesmith.experiment_tracking import agents_sdk_wrapping_enabled
        from scenesmith.experiment_tracking.model_wrapper import wrap_model
    except Exception:  # noqa: BLE001
        return inner

    if not agents_sdk_wrapping_enabled():
        return inner

    return wrap_model(inner)
