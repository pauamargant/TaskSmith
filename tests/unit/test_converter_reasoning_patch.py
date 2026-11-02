"""Tests for the agents.Converter monkey-patch in scenesmith.provider.

The patch ensures that reasoning items (ResponseReasoningItem.summary) are
re-injected as `reasoning_content` into the next assistant message when
the SDK reconstructs the message list for the next API turn.
"""

import unittest

from agents.models.openai_chatcompletions import Converter

# Importing scenesmith.provider applies the monkey-patch as a side effect.
import scenesmith.provider  # noqa: F401 — imported for its patch side effect


class TestConverterReasoningPatch(unittest.TestCase):
    def _reasoning(self, text: str, item_id: str = "r1") -> dict:
        return {
            "type": "reasoning",
            "id": item_id,
            "summary": [{"type": "summary_text", "text": text}],
        }

    def _function_call(self, call_id: str = "c1") -> dict:
        return {
            "type": "function_call",
            "id": f"fc_{call_id}",
            "call_id": call_id,
            "name": "f",
            "arguments": "{}",
        }

    def _tool_output(self, call_id: str = "c1", output: str = "ok") -> dict:
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }

    def test_reasoning_then_function_call_attaches_reasoning_content(self):
        items = [
            {"role": "user", "content": "hi"},
            self._reasoning("thought-1"),
            self._function_call(),
        ]
        msgs = Converter.items_to_messages(items)
        asst = [m for m in msgs if m.get("role") == "assistant"]
        self.assertEqual(len(asst), 1)
        self.assertEqual(asst[0].get("reasoning_content"), "thought-1")

    def test_no_reasoning_item_leaves_assistant_clean(self):
        items = [
            {"role": "user", "content": "hi"},
            self._function_call(),
        ]
        msgs = Converter.items_to_messages(items)
        for m in msgs:
            self.assertNotIn("reasoning_content", m)

    def test_empty_summary_does_not_inject_reasoning_content(self):
        items = [
            {"role": "user", "content": "hi"},
            {"type": "reasoning", "id": "r1", "summary": []},
            self._function_call(),
        ]
        msgs = Converter.items_to_messages(items)
        for m in msgs:
            self.assertNotIn("reasoning_content", m)

    def test_two_reasoning_two_assistant_turns_each_paired_correctly(self):
        items = [
            {"role": "user", "content": "hi"},
            self._reasoning("thought-A", "rA"),
            self._function_call("c1"),
            self._tool_output("c1"),
            {"role": "user", "content": "next"},
            self._reasoning("thought-B", "rB"),
            self._function_call("c2"),
        ]
        msgs = Converter.items_to_messages(items)
        asst = [m for m in msgs if m.get("role") == "assistant"]
        self.assertEqual(len(asst), 2)
        self.assertEqual(asst[0].get("reasoning_content"), "thought-A")
        self.assertEqual(asst[1].get("reasoning_content"), "thought-B")

    def test_orphan_trailing_reasoning_is_dropped_silently(self):
        items = [
            {"role": "user", "content": "hi"},
            self._function_call(),
            self._tool_output(),
            {
                "type": "reasoning",
                "id": "rO",
                "summary": [{"type": "summary_text", "text": "orphan"}],
            },
        ]
        msgs = Converter.items_to_messages(items)
        self.assertFalse(
            any(m.get("reasoning_content") == "orphan" for m in msgs),
            "orphan reasoning with no following assistant must not leak",
        )

    def test_multi_line_summary_is_joined_with_newline(self):
        items = [
            {"role": "user", "content": "hi"},
            {
                "type": "reasoning",
                "id": "r1",
                "summary": [
                    {"type": "summary_text", "text": "line1"},
                    {"type": "summary_text", "text": "line2"},
                ],
            },
            self._function_call(),
        ]
        msgs = Converter.items_to_messages(items)
        asst = [m for m in msgs if m.get("role") == "assistant"]
        self.assertEqual(asst[0].get("reasoning_content"), "line1\nline2")

    def test_non_list_items_input_returns_unchanged(self):
        """When items is a plain string, the patch must not interfere."""
        msgs = Converter.items_to_messages("just a string")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertNotIn("reasoning_content", msgs[0])
