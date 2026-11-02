import unittest

from unittest.mock import patch

from omegaconf import OmegaConf

from scenesmith.provider_validation import (
    apply_chat_completion_capabilities,
    build_thinking_extra_body,
    get_active_provider_config,
    initialize_provider_runtime,
    reset_provider_config_for_tests,
)


class TestProviderValidation(unittest.TestCase):
    def setUp(self):
        reset_provider_config_for_tests()

    def tearDown(self):
        reset_provider_config_for_tests()

    def test_declared_supported_knobs_pass_validation(self):
        cfg = OmegaConf.create(
            {
                "provider": {
                    "api": {
                        "standard_model_api": "chat_completions",
                        "reasoning_model_api": "responses",
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
                    },
                },
                "openai": {"service_tier": None},
                "furniture_agent": {
                    "openai": {
                        "reasoning_effort": {"planner": "low"},
                        "verbosity": {"planner": "medium"},
                    },
                    "session_memory": {
                        "summarization_thinking": None,
                    },
                },
            }
        )

        provider_cfg = initialize_provider_runtime(cfg)

        self.assertTrue(provider_cfg["capabilities"]["supports_reasoning_effort"])
        self.assertTrue(provider_cfg["capabilities"]["supports_verbosity"])

    def test_reasoning_and_verbosity_validation_fail_cleanly(self):
        cfg = OmegaConf.create(
            {
                "provider": {
                    "base_url": "https://example.test/v1",
                    "capabilities": {
                        "supports_chat_completions": True,
                        "supports_reasoning_effort": False,
                        "supports_verbosity": False,
                    },
                },
                "furniture_agent": {
                    "openai": {
                        "reasoning_effort": {"planner": "low"},
                        "verbosity": {"planner": "medium"},
                    }
                },
                "task_analysis_agent": {"openai": {"model": "provider/model"}},
            }
        )

        with self.assertRaises(ValueError) as ctx:
            initialize_provider_runtime(cfg)

        message = str(ctx.exception)
        self.assertIn("Provider capability validation failed", message)
        self.assertIn("https://example.test/v1", message)
        self.assertIn("furniture_agent uses openai.reasoning_effort", message)
        self.assertIn("furniture_agent uses openai.verbosity", message)

    def test_service_tier_null_passes_when_unsupported(self):
        cfg = OmegaConf.create(
            {
                "provider": {
                    "capabilities": {
                        "supports_chat_completions": True,
                        "supports_service_tier": False,
                    }
                },
                "openai": {"service_tier": None},
                "reward_spec_agent": {
                    "openai": {"model": "provider/model", "service_tier": None}
                },
            }
        )

        provider_cfg = initialize_provider_runtime(cfg)

        self.assertFalse(provider_cfg["capabilities"]["supports_service_tier"])

    def test_summarization_thinking_fails_when_unsupported(self):
        cfg = OmegaConf.create(
            {
                "provider": {
                    "capabilities": {
                        "supports_chat_completions": True,
                        "supports_summarization_thinking": False,
                    }
                },
                "floor_plan_agent": {
                    "session_memory": {
                        "summarization_thinking": "low",
                    }
                },
            }
        )

        with self.assertRaises(ValueError) as ctx:
            initialize_provider_runtime(cfg)

        self.assertIn("summarization_thinking", str(ctx.exception))

    def test_request_shaping_includes_supported_knobs(self):
        provider_cfg = initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "capabilities": {
                            "supports_chat_completions": True,
                            "supports_reasoning_effort": True,
                            "supports_verbosity": True,
                            "supports_service_tier": True,
                        }
                    }
                }
            )
        )

        kwargs, extra_body = apply_chat_completion_capabilities(
            kwargs={"model": "provider/model", "messages": []},
            reasoning_effort="medium",
            verbosity="low",
            service_tier="flex",
            provider_config=provider_cfg,
        )

        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertEqual(kwargs["verbosity"], "low")
        self.assertEqual(kwargs["service_tier"], "flex")
        self.assertEqual(extra_body, {})

    def test_request_shaping_drops_unsupported_knob(self):
        provider_cfg = initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "base_url": "https://example.test/v1",
                        "capabilities": {
                            "supports_chat_completions": True,
                            "supports_reasoning_effort": False,
                        },
                    }
                }
            )
        )

        kwargs, extra_body = apply_chat_completion_capabilities(
            kwargs={"model": "provider/model", "messages": []},
            reasoning_effort="medium",
            provider_config=provider_cfg,
            subsystem="unit test request",
        )

        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(extra_body, {})

    @patch("scenesmith.provider.make_sync_client")
    def test_probe_disabled_by_default(self, mock_make_sync_client):
        initialize_provider_runtime(OmegaConf.create({}))
        mock_make_sync_client.assert_not_called()

    @patch("scenesmith.agent_utils.sdk_vlm_runner.run_text_vlm_call")
    def test_probe_enabled_success(self, mock_run_text_vlm_call):
        mock_run_text_vlm_call.return_value = "OK"

        provider_cfg = initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "model": "provider/model",
                        "validation": {"probe": True},
                    }
                }
            )
        )

        self.assertEqual(provider_cfg["model"], "provider/model")
        mock_run_text_vlm_call.assert_called_once()

    @patch("scenesmith.provider_validation.time.sleep", autospec=True)
    @patch("scenesmith.agent_utils.sdk_vlm_runner.run_text_vlm_call")
    def test_probe_enabled_mismatch_fails(self, mock_run_text_vlm_call, mock_sleep):
        mock_run_text_vlm_call.side_effect = RuntimeError("unsupported")

        with self.assertRaises(ValueError) as ctx:
            initialize_provider_runtime(
                OmegaConf.create(
                    {
                        "provider": {
                            "base_url": "https://example.test/v1",
                            "model": "provider/model",
                            "validation": {
                                "probe": True,
                                "probe_retry_attempts": 0,
                            },
                        }
                    }
                )
            )

        self.assertIn("Provider startup probe failed", str(ctx.exception))
        self.assertIn("https://example.test/v1", str(ctx.exception))

    @patch("scenesmith.provider_validation.time.sleep", autospec=True)
    @patch("scenesmith.agent_utils.sdk_vlm_runner.run_text_vlm_call")
    def test_probe_retries_until_success(self, mock_call, mock_sleep):
        mock_call.side_effect = [TimeoutError("slow"), "OK"]

        initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "model": "provider/model",
                        "validation": {
                            "probe": True,
                            "probe_retry_attempts": 2,
                            "probe_retry_interval_s": 60.0,
                        },
                    }
                }
            )
        )

        self.assertEqual(mock_call.call_count, 2)
        mock_sleep.assert_called_once_with(60.0)

    @patch("scenesmith.provider_validation.time.sleep", autospec=True)
    @patch("scenesmith.agent_utils.sdk_vlm_runner.run_text_vlm_call")
    def test_probe_retries_exhausted(self, mock_call, mock_sleep):
        mock_call.side_effect = RuntimeError("bad")

        with self.assertRaises(ValueError) as ctx:
            initialize_provider_runtime(
                OmegaConf.create(
                    {
                        "provider": {
                            "base_url": "https://example.test/v1",
                            "model": "provider/model",
                            "validation": {
                                "probe": True,
                                "probe_retry_attempts": 2,
                                "probe_retry_interval_s": 60.0,
                            },
                        }
                    }
                )
            )

        self.assertIn("Provider startup probe failed", str(ctx.exception))
        self.assertEqual(mock_call.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    def test_reasoning_native_requires_supported_api_path(self):
        cfg = OmegaConf.create(
            {
                "provider": {
                    "api": {
                        "standard_model_api": "chat_completions",
                        "reasoning_model_api": "responses",
                    },
                    "model_traits": {"reasoning_native": True},
                    "capabilities": {
                        "supports_chat_completions": True,
                        "supports_responses_api": False,
                        "supports_reasoning_models": True,
                        "supports_final_answer_only_extraction": True,
                    },
                },
                "task_analysis_agent": {"openai": {"model": "provider/model"}},
            }
        )

        with self.assertRaises(ValueError) as ctx:
            initialize_provider_runtime(cfg)

        self.assertIn("supports_responses_api is false", str(ctx.exception))

    def test_reasoning_native_responses_path_blocks_stateful_agents(self):
        cfg = OmegaConf.create(
            {
                "provider": {
                    "api": {"reasoning_model_api": "responses"},
                    "model_traits": {"reasoning_native": True},
                    "capabilities": {
                        "supports_chat_completions": True,
                        "supports_responses_api": True,
                        "supports_reasoning_models": True,
                        "supports_final_answer_only_extraction": True,
                    },
                },
                "furniture_agent": {"openai": {"model": "provider/model"}},
            }
        )

        with self.assertRaises(ValueError) as ctx:
            initialize_provider_runtime(cfg)

        self.assertIn("Agents SDK chat_completions path", str(ctx.exception))

    def test_initialize_sets_active_provider_config(self):
        initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "base_url": "https://example.test/v1",
                        "model": "provider/model",
                    }
                }
            )
        )

        active = get_active_provider_config()
        self.assertEqual(active["base_url"], "https://example.test/v1")
        self.assertEqual(active["model"], "provider/model")

    def test_minimal_effort_maps_to_extra_body_when_thinking_control_supported(self):
        provider_cfg = initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "capabilities": {
                            "supports_chat_completions": True,
                            "supports_reasoning_effort": True,
                            "thinking_control_mode": "binary",
                        }
                    }
                }
            )
        )

        kwargs, extra_body = apply_chat_completion_capabilities(
            kwargs={"model": "provider/model", "messages": []},
            reasoning_effort="minimal",
            provider_config=provider_cfg,
        )

        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(
            extra_body, {"chat_template_kwargs": {"enable_thinking": False}}
        )

    def test_minimal_effort_dropped_when_thinking_control_unsupported(self):
        provider_cfg = initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "capabilities": {
                            "supports_chat_completions": True,
                            "supports_reasoning_effort": True,
                            "thinking_control_mode": "openai",
                        }
                    }
                }
            )
        )

        kwargs, extra_body = apply_chat_completion_capabilities(
            kwargs={"model": "provider/model", "messages": []},
            reasoning_effort="minimal",
            provider_config=provider_cfg,
        )

        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(extra_body, {})


class TestBuildThinkingExtraBody(unittest.TestCase):
    """Tests for build_thinking_extra_body — pure config translation."""

    def setUp(self):
        reset_provider_config_for_tests()

    def tearDown(self):
        reset_provider_config_for_tests()

    def _vllm_cfg(self):
        return {
            "capabilities": {
                "thinking_control_mode": "binary",
                "supports_reasoning_effort": True,
            }
        }

    def test_minimal_maps_to_disable_thinking_when_supported(self):
        result = build_thinking_extra_body("minimal", self._vllm_cfg())
        self.assertEqual(result, {"chat_template_kwargs": {"enable_thinking": False}})

    def test_non_minimal_efforts_map_to_enable_thinking_true(self):
        for effort in ("low", "medium", "high", "xtreme"):
            with self.subTest(effort=effort):
                result = build_thinking_extra_body(effort, self._vllm_cfg())
                self.assertEqual(
                    result, {"chat_template_kwargs": {"enable_thinking": True}}
                )

    def test_unsupported_thinking_control_returns_empty_for_all_efforts(self):
        for mode in ("openai", "disabled"):
            with self.subTest(mode=mode):
                cfg = {"capabilities": {"thinking_control_mode": mode}}
                for effort in ("minimal", "low", "medium", "high"):
                    self.assertEqual(build_thinking_extra_body(effort, cfg), {})

    def test_missing_thinking_control_mode_defaults_to_openai_path(self):
        # When the capability key is absent we default to "openai" mode, so
        # build_thinking_extra_body emits no chat_template_kwargs.
        cfg = {"capabilities": {}}
        for effort in ("minimal", "low", "medium", "high"):
            with self.subTest(effort=effort):
                self.assertEqual(build_thinking_extra_body(effort, cfg), {})

    def test_disabled_effort_values_return_empty(self):
        cfg = self._vllm_cfg()
        for effort in (None, "", "none", "None", False):
            with self.subTest(effort=effort):
                self.assertEqual(build_thinking_extra_body(effort, cfg), {})


class TestCapabilityDefaultsAreNeutral(unittest.TestCase):
    """Defaults must keep TaskSmith provider-agnostic out of the box.

    Qwen-specific behaviors (reasoning re-injection, binary chat_template_kwargs,
    image-first multimodal) only activate when the chosen `provider` preset
    opts in. Without a preset, behavior is OpenAI-friendly.
    """

    def setUp(self):
        reset_provider_config_for_tests()

    def tearDown(self):
        reset_provider_config_for_tests()

    def test_defaults_are_neutral(self):
        provider_cfg = initialize_provider_runtime(OmegaConf.create({}))
        caps = provider_cfg["capabilities"]
        self.assertEqual(caps["thinking_control_mode"], "openai")
        self.assertFalse(caps["reinject_prior_reasoning"])
        self.assertFalse(caps["multimodal_image_before_text"])

    def test_qwen_preset_shape(self):
        # Mirrors configurations/provider/qwen.yaml — kept in sync manually.
        provider_cfg = initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "capabilities": {
                            "thinking_control_mode": "binary",
                            "reinject_prior_reasoning": True,
                            "multimodal_image_before_text": False,
                        }
                    }
                }
            )
        )
        caps = provider_cfg["capabilities"]
        self.assertEqual(caps["thinking_control_mode"], "binary")
        self.assertTrue(caps["reinject_prior_reasoning"])
        self.assertFalse(caps["multimodal_image_before_text"])

    def test_gemma_preset_shape(self):
        # Mirrors configurations/provider/gemma.yaml — kept in sync manually.
        provider_cfg = initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "capabilities": {
                            "thinking_control_mode": "binary",
                            "reinject_prior_reasoning": False,
                            "multimodal_image_before_text": True,
                        }
                    }
                }
            )
        )
        caps = provider_cfg["capabilities"]
        self.assertEqual(caps["thinking_control_mode"], "binary")
        self.assertFalse(caps["reinject_prior_reasoning"])
        self.assertTrue(caps["multimodal_image_before_text"])


class TestApplyChatCompletionThinkingControl(unittest.TestCase):
    """Tests for the thinking-control branch in apply_chat_completion_capabilities."""

    def setUp(self):
        reset_provider_config_for_tests()

    def tearDown(self):
        reset_provider_config_for_tests()

    def test_high_effort_maps_to_enable_thinking_true_on_vllm_provider(self):
        provider_cfg = initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "capabilities": {
                            "supports_chat_completions": True,
                            "supports_reasoning_effort": True,
                            "thinking_control_mode": "binary",
                        }
                    }
                }
            )
        )

        kwargs, extra_body = apply_chat_completion_capabilities(
            kwargs={"model": "provider/model", "messages": []},
            reasoning_effort="high",
            provider_config=provider_cfg,
        )

        # vLLM-style provider: chat_template_kwargs takes precedence over the
        # reasoning_effort kwarg (vLLM ignores it).
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual(
            extra_body, {"chat_template_kwargs": {"enable_thinking": True}}
        )

    def test_minimal_maps_to_enable_thinking_false_on_vllm_provider(self):
        provider_cfg = initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "capabilities": {
                            "supports_chat_completions": True,
                            "supports_reasoning_effort": True,
                            "thinking_control_mode": "binary",
                        }
                    }
                }
            )
        )

        kwargs, extra_body = apply_chat_completion_capabilities(
            kwargs={"model": "provider/model", "messages": []},
            reasoning_effort="minimal",
            provider_config=provider_cfg,
        )

        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual(
            extra_body, {"chat_template_kwargs": {"enable_thinking": False}}
        )

    def test_high_effort_falls_back_to_reasoning_effort_kwarg_on_openai_provider(self):
        provider_cfg = initialize_provider_runtime(
            OmegaConf.create(
                {
                    "provider": {
                        "capabilities": {
                            "supports_chat_completions": True,
                            "supports_reasoning_effort": True,
                            "thinking_control_mode": "openai",
                        }
                    }
                }
            )
        )

        kwargs, extra_body = apply_chat_completion_capabilities(
            kwargs={"model": "provider/model", "messages": []},
            reasoning_effort="high",
            provider_config=provider_cfg,
        )

        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(extra_body, {})
