# Providers and models

LLM access is configured via `.env` at the project root:

```
PROVIDER_BASE_URL="https://inference.rcp.epfl.ch/v1"
PROVIDER_API_KEY="<your-key>"
PROVIDER_MODEL="Qwen/Qwen3.6-27B-FP8"
```

`scenesmith/provider.py` loads `.env` automatically at import; shell variables take precedence. The model name propagates to all agent configs via `${oc.env:PROVIDER_MODEL,...}` in the Hydra config.

---

## Presets

Select a preset via the Hydra CLI (`provider=qwen`) or the `defaults` list in `config.yaml`:

| Preset | `thinking_control_mode` | `reinject_prior_reasoning` | `multimodal_image_before_text` | When to use |
|---|---|---|---|---|
| `neutral` | `"openai"` | `false` | `false` | OpenAI / Azure OpenAI — `reasoning_effort` forwarded natively. |
| `qwen` *(default)* | `"binary"` | `true` | `false` | Qwen3 on vLLM (EPFL/RCP). Re-injects prior `reasoning_content` to save 30–70% completion tokens per tool turn. |
| `gemma` | `"binary"` | `false` | `true` | Gemma 4 on vLLM (`--reasoning-parser gemma4 --tool-call-parser gemma4`). |

Key capability fields (full definitions in `configurations/provider/{neutral,qwen,gemma}.yaml`):

- `thinking_control_mode`: `"openai"` (forward `reasoning_effort` natively), `"binary"` (vLLM `chat_template_kwargs.enable_thinking`), `"disabled"`.
- `reinject_prior_reasoning`: re-inject prior `reasoning_content` into history (Qwen3 only; must be `false` for Gemma 4).
- `multimodal_image_before_text`: image-before-text ordering in multimodal user messages (Gemma 4 requirement).
- `tool_image_via_user_message`: rewrite tool-role image messages into user messages (required for vLLM Gemma 4 31B; off for all others).
- `sanitize_tool_call_arguments`: replace unparseable `tool_calls[i].function.arguments` with `"{}"` (Gemma 4 + thinking-on workaround).
- `supports_final_answer_only_extraction`: declare whether the model produces usable final answer text without reading reasoning traces.
- `provider.sampling`: sampling kwargs forwarded to every request; `top_k`/`min_p`/`repetition_penalty` go in `extra_body`.
- `provider.chat_template_kwargs`: extra vLLM kwargs merged alongside `enable_thinking` (thinking flag wins on key collision).

Sampling defaults per preset:

| Preset | `provider.sampling` |
|---|---|
| `neutral` | `{}` (provider defaults) |
| `qwen` | `temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, presence_penalty: 1.5, repetition_penalty: 1.0` |
| `gemma` | `temperature: 1.0, top_p: 0.95, top_k: 64` |

---

## Run reliability

Hard timeouts across three layers:

| Layer | Default | Env var(s) |
|---|---|---|
| Transport retry (per `Runner.run()`) | 3 attempts, 2 s back-off | `TASKSMITH_PROVIDER_MAX_ATTEMPTS`, `TASKSMITH_PROVIDER_RETRY_BACKOFF_S` |
| SDK client (Agents SDK path) | 360 s, 5 retries | `PROVIDER_TIMEOUT_S`, `PROVIDER_MAX_RETRIES` |
| Direct openai-client calls | 360 s, 5 retries | `TASKSMITH_LLM_TIMEOUT_S`, `TASKSMITH_LLM_MAX_RETRIES` |
| Mesh floater cleanup | 180 s | `TASKSMITH_FLOATER_CLEAN_TIMEOUT_S` |

> `PROVIDER_MAX_RETRIES` runs inside each `TASKSMITH_PROVIDER_MAX_ATTEMPTS` attempt — worst case 3 × 6 = 18 total (3 transport attempts, each allowing 5 retries + the initial try). Reduce `PROVIDER_MAX_RETRIES` if retry storms occur.

Queue runs include an inactivity watchdog (`scripts/run_experiment_queue.py`): `--stuck-min` (default 12 min) terminates a run that produces no output and moves the queue on.

**Quick diagnosis:**
- Last log line shows an in-flight chat-completion with no matching `Wrapper: Completed Call` → run is wedged in a provider call.
- Last log line is `Removing mesh floaters` → run is wedged in mesh sanitization.

Use `bash scripts/kill_tasksmith_zombies.sh` to clear wedged processes before retrying.

---

## Tracing

Disabled by default when running against a private provider (the Agents SDK would otherwise attempt to send traces to `platform.openai.com`). Enable with:

```bash
export OPENAI_TRACING_KEY="sk-..."
```

---

## Vision token accounting

TaskSmith records token usage as returned by the provider (OpenAI-style `usage` fields). Image cost is typically folded into `usage.prompt_tokens`. The `image_url.detail` hint may or may not affect token accounting on a given provider — on the EPFL RCP endpoint, `prompt_tokens` scaled with image resolution regardless of `detail`.

---

## Testing

```bash
uv run python basic_tests/test_provider.py
```

Covers direct call, basic agent, tool calling, structured output, multi-turn, handoff, and streaming.
