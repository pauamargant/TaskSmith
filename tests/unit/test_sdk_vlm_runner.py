"""Tests for the single-shot Agents-SDK VLM call helper."""

from __future__ import annotations

import unittest

from typing import Any
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from scenesmith.agent_utils.sdk_vlm_runner import (
    _build_model_settings,
    _build_user_input,
    run_structured_vlm_call,
)


class _Stub(BaseModel):
    answer: str
    score: int


def _make_run_result(output_obj: BaseModel) -> MagicMock:
    """Construct a Runner.run_sync return value that satisfies final_output_as."""
    result = MagicMock()
    result.final_output_as.return_value = output_obj
    return result


class TestBuildUserInput(unittest.TestCase):
    def test_text_only(self):
        items = _build_user_input("hello", [])
        self.assertEqual(len(items), 1)
        msg = items[0]
        self.assertEqual(msg["role"], "user")
        content = msg["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0], {"type": "input_text", "text": "hello"})

    def test_with_images_uses_input_image_type(self):
        items = _build_user_input("look", ["AAA", "BBB"])
        content = items[0]["content"]
        self.assertEqual(len(content), 3)
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_image")
        self.assertEqual(content[1]["image_url"], "data:image/png;base64,AAA")
        self.assertEqual(content[2]["image_url"], "data:image/png;base64,BBB")


class TestBuildModelSettings(unittest.TestCase):
    def test_no_effort_returns_bare_settings(self):
        settings = _build_model_settings(None, None)
        self.assertIsNone(settings.reasoning)
        self.assertIsNone(settings.verbosity)

    def test_reasoning_and_verbosity_set_when_provided(self):
        settings = _build_model_settings("low", "medium")
        self.assertIsNotNone(settings.reasoning)
        self.assertEqual(settings.reasoning.effort, "low")
        self.assertEqual(settings.verbosity, "medium")

    def test_binary_mode_non_minimal_effort_emits_enable_thinking_true(self):
        provider_cfg = {
            "capabilities": {
                "thinking_control_mode": "binary",
                "supports_reasoning_effort": True,
            }
        }
        settings = _build_model_settings("medium", "low", provider_config=provider_cfg)
        self.assertEqual(
            settings.extra_body,
            {"chat_template_kwargs": {"enable_thinking": True}},
        )
        # Binary path does not set the SDK's native Reasoning object.
        self.assertIsNone(settings.reasoning)

    def test_binary_mode_minimal_effort_emits_enable_thinking_false(self):
        provider_cfg = {
            "capabilities": {
                "thinking_control_mode": "binary",
                "supports_reasoning_effort": True,
            }
        }
        settings = _build_model_settings("minimal", "low", provider_config=provider_cfg)
        self.assertEqual(
            settings.extra_body,
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertIsNone(settings.reasoning)


class TestRunStructuredVlmCall(unittest.TestCase):
    @patch("scenesmith.agent_utils.sdk_vlm_runner._run_agent_sync")
    @patch("scenesmith.agent_utils.sdk_vlm_runner.make_chat_completions_model")
    def test_runs_agent_with_expected_input_items(
        self, mock_make_model: MagicMock, mock_run_agent: MagicMock
    ):
        mock_make_model.return_value = "fake_model"
        mock_run_agent.return_value = _make_run_result(_Stub(answer="ok", score=1))

        out = run_structured_vlm_call(
            model="m1",
            instructions="be helpful",
            user_text="prompt text",
            images_b64=["IMG1"],
            output_type=_Stub,
            reasoning_effort="low",
            verbosity="low",
            subsystem="physics_analysis",
            agent_name="test_oneshot",
        )

        self.assertIsInstance(out, _Stub)
        # _run_agent_sync(agent, input_items)
        agent_arg, input_items_arg = mock_run_agent.call_args.args
        self.assertEqual(agent_arg.name, "test_oneshot")
        self.assertEqual(agent_arg.instructions, "be helpful")
        self.assertEqual(agent_arg.output_type, _Stub)
        self.assertEqual(input_items_arg[0]["role"], "user")
        content = input_items_arg[0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_image")

    @patch("scenesmith.agent_utils.sdk_vlm_runner._run_agent_sync")
    @patch("scenesmith.agent_utils.sdk_vlm_runner.make_chat_completions_model")
    def test_explicit_user_content_overrides_text_and_images(
        self, mock_make_model: MagicMock, mock_run_agent: MagicMock
    ):
        mock_make_model.return_value = "fake_model"
        mock_run_agent.return_value = _make_run_result(_Stub(answer="x", score=0))

        custom_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": "a"},
            {"type": "input_image", "image_url": "data:image/png;base64,Q"},
            {"type": "input_text", "text": "b"},
        ]

        run_structured_vlm_call(
            model="m1",
            instructions="",
            user_content=custom_content,
            output_type=_Stub,
            reasoning_effort="minimal",
            verbosity="low",
            subsystem="physics_analysis",
        )

        _agent, input_items = mock_run_agent.call_args.args
        self.assertEqual(input_items[0]["content"], custom_content)

    @patch("scenesmith.agent_utils.sdk_vlm_runner.extend_tracking_context")
    @patch("scenesmith.agent_utils.sdk_vlm_runner._run_agent_sync")
    @patch("scenesmith.agent_utils.sdk_vlm_runner.make_chat_completions_model")
    def test_subsystem_is_set_on_tracking_context(
        self,
        mock_make_model: MagicMock,
        mock_run_agent: MagicMock,
        mock_extend_ctx: MagicMock,
    ):
        mock_make_model.return_value = "fake_model"
        mock_run_agent.return_value = _make_run_result(_Stub(answer="x", score=0))
        mock_extend_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_extend_ctx.return_value.__exit__ = MagicMock(return_value=False)

        run_structured_vlm_call(
            model="m1",
            instructions="",
            user_text="t",
            images_b64=[],
            output_type=_Stub,
            reasoning_effort="low",
            verbosity="low",
            subsystem="router_validation",
        )

        mock_extend_ctx.assert_called_once_with(subsystem="router_validation")


if __name__ == "__main__":
    unittest.main()
