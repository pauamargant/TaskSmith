import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from scenesmith.agent_utils.room import ObjectType, RoomScene, UniqueID
from scenesmith.agent_utils.scene_analyzer import (
    FurnitureSelectionResponseSchema,
    SceneAnalyzer,
)
from scenesmith.prompts.registry import ManipulandAgentPrompts


class TestSceneAnalyzer(unittest.TestCase):
    """Test SceneAnalyzer class contracts."""

    # Test configuration constants.
    TEST_MODEL = "gpt-4o-mini"
    TEST_REASONING_EFFORT = "low"

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_rendering_manager = Mock()
        self.mock_scene = Mock(spec=RoomScene)

        # Create mock BlenderServer.
        self.mock_blender_server = Mock()
        self.mock_blender_server.is_running.return_value = True

        # Create test config (only OpenAI settings needed).
        test_config_dict = {
            "openai": {
                "model": self.TEST_MODEL,
                "vision_detail": "low",
                "reasoning_effort": {"scene_critique": self.TEST_REASONING_EFFORT},
                "verbosity": {"scene_critique": "low"},
            },
        }
        # Convert to OmegaConf to match expected structure.
        self.test_config = OmegaConf.create(test_config_dict)

        self.scene_analyzer = SceneAnalyzer(
            rendering_manager=self.mock_rendering_manager,
            cfg=self.test_config,
            blender_server=self.mock_blender_server,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scene_analyzer_initialization(self):
        """Test that SceneAnalyzer initializes properly."""
        self.assertIsNotNone(self.scene_analyzer)
        self.assertEqual(
            self.scene_analyzer.rendering_manager, self.mock_rendering_manager
        )
        self.assertEqual(self.scene_analyzer.cfg, self.test_config)
        self.assertEqual(self.scene_analyzer.blender_server, self.mock_blender_server)

    def test_configuration_access(self):
        """Test that SceneAnalyzer can access configuration values."""
        # Verify configuration was stored and accessible.
        self.assertEqual(self.scene_analyzer.cfg["openai"]["model"], self.TEST_MODEL)
        self.assertEqual(
            self.scene_analyzer.cfg["openai"]["reasoning_effort"]["scene_critique"],
            self.TEST_REASONING_EFFORT,
        )


class TestSceneAnalyzerSDKPath(unittest.TestCase):
    """Cover the Agents-SDK structured-output path that replaced analyze_scene."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_rendering_manager = Mock()
        # render_scene returns an empty dir; glob("*.png") will be empty so no
        # actual image encoding runs. The SDK call still receives images_b64=[]
        # which is valid for a unit test that only verifies the wiring.
        self.mock_rendering_manager.render_scene.return_value = self.temp_dir
        self.mock_blender_server = Mock()

        cfg_dict = {
            "openai": {
                "model": "test-model",
                "vision_detail": "low",
                "furniture_analysis_max_retries": 3,
                "reasoning_effort": {"furniture_analysis": "low"},
                "verbosity": {"furniture_analysis": "low"},
            },
        }
        self.cfg = OmegaConf.create(cfg_dict)
        self.analyzer = SceneAnalyzer(
            rendering_manager=self.mock_rendering_manager,
            cfg=self.cfg,
            blender_server=self.mock_blender_server,
        )

        # Minimal RoomScene-shaped mock.
        self.fid = UniqueID("furn_1")
        furniture_obj = Mock()
        furniture_obj.object_id = self.fid
        furniture_obj.object_type = ObjectType.FURNITURE
        furniture_obj.immutable = False
        furniture_obj.bbox_min = (0.0, 0.0, 0.0)
        furniture_obj.bbox_max = (1.0, 1.0, 1.0)
        furniture_obj.name = "test_table"
        furniture_obj.description = "A test dining table"

        self.scene = Mock(spec=RoomScene)
        self.scene.objects = {self.fid: furniture_obj}
        self.scene.room_geometry = None
        self.scene.text_description = "test room"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scenesmith.agent_utils.scene_analyzer.prompt_manager.get_prompt")
    @patch("scenesmith.agent_utils.scene_analyzer.run_structured_vlm_call")
    def test_happy_path_returns_furniture_selection(
        self, mock_run_sdk, mock_get_prompt
    ):
        """SDK call result is mapped to a FurnitureSelection on the valid id."""
        mock_get_prompt.return_value = "system prompt"
        mock_run_sdk.return_value = FurnitureSelectionResponseSchema(
            furniture_selections=[
                {
                    "furniture_id": str(self.fid),
                    "suggested_items": "REQUIRED: plate",
                    "prompt_constraints": "kitchen",
                    "style_notes": "modern",
                }
            ],
            scene_style="modern",
            reasoning="",
        )

        result = self.analyzer.analyze_furniture_for_manipulands(
            scene=self.scene,
            prompt_enum=ManipulandAgentPrompts.ANALYZE_FURNITURE_FOR_PLACEMENT,
        )

        self.assertEqual(mock_run_sdk.call_count, 1)
        kwargs = mock_run_sdk.call_args.kwargs
        self.assertEqual(kwargs["output_type"], FurnitureSelectionResponseSchema)
        self.assertEqual(kwargs["subsystem"], "scene_analysis")
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].furniture_id, self.fid)
        self.assertEqual(result[0].suggested_items, "REQUIRED: plate")

    @patch("scenesmith.agent_utils.scene_analyzer.prompt_manager.get_prompt")
    @patch("scenesmith.agent_utils.scene_analyzer.run_structured_vlm_call")
    def test_retries_then_raises_after_max_attempts(
        self, mock_run_sdk, mock_get_prompt
    ):
        """SDK call failures consume the configured retry budget, then raise."""
        mock_get_prompt.return_value = "system prompt"
        mock_run_sdk.side_effect = ValueError("simulated schema failure")

        with self.assertRaises(RuntimeError) as ctx:
            self.analyzer.analyze_furniture_for_manipulands(
                scene=self.scene,
                prompt_enum=ManipulandAgentPrompts.ANALYZE_FURNITURE_FOR_PLACEMENT,
            )

        self.assertIn("simulated schema failure", str(ctx.exception))
        self.assertEqual(
            mock_run_sdk.call_count,
            self.cfg.openai.furniture_analysis_max_retries,
        )
        # Render is expensive — must run exactly once, outside the retry loop.
        self.assertEqual(self.mock_rendering_manager.render_scene.call_count, 1)

    @patch("scenesmith.agent_utils.scene_analyzer.prompt_manager.get_prompt")
    @patch("scenesmith.agent_utils.scene_analyzer.run_structured_vlm_call")
    def test_invalid_furniture_id_in_response_is_dropped(
        self, mock_run_sdk, mock_get_prompt
    ):
        """Selections referencing unknown furniture ids are filtered out."""
        mock_get_prompt.return_value = "system prompt"
        mock_run_sdk.return_value = FurnitureSelectionResponseSchema(
            furniture_selections=[
                {"furniture_id": "ghost_id", "suggested_items": "x"},
                {"furniture_id": str(self.fid), "suggested_items": "ok"},
            ],
        )

        result = self.analyzer.analyze_furniture_for_manipulands(
            scene=self.scene,
            prompt_enum=ManipulandAgentPrompts.ANALYZE_FURNITURE_FOR_PLACEMENT,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].furniture_id, self.fid)


if __name__ == "__main__":
    unittest.main()
