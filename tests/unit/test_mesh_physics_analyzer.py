import json
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import trimesh

from omegaconf import OmegaConf
from PIL import Image

from scenesmith.agent_utils.mesh_physics_analyzer import (
    MeshPhysicsAnalysis,
    analyze_mesh_orientation_and_material,
    get_front_axis_from_image_number,
    get_view_direction_from_image_number,
)


def create_mock_cfg():
    """Create mock configuration for mesh physics analyzer tests."""
    return OmegaConf.create(
        {
            "openai": {
                "model": "gpt-5",
                "vision_detail": "high",
                "reasoning_effort": {"mesh_analysis": "low"},
                "verbosity": {"mesh_analysis": "low"},
            },
            "asset_manager": {
                "validation_taa_samples": 16,
            },
        }
    )


class TestViewDirectionMapping(unittest.TestCase):
    """Test view direction mapping from image numbers."""

    def test_get_view_direction_top(self):
        """Test view direction mapping for top view."""
        self.assertEqual(get_view_direction_from_image_number(0), "top")

    def test_get_view_direction_bottom(self):
        """Test view direction mapping for bottom view."""
        self.assertEqual(get_view_direction_from_image_number(1), "bottom")

    def test_get_view_direction_side_views_8_views(self):
        """Test view direction mapping for 8 side views."""
        # With 8 side views evenly distributed, test key angles.
        # View 2 (0°) → +x direction.
        self.assertEqual(get_view_direction_from_image_number(2, num_side_views=8), "x")

        # View 4 (90°) → +y direction.
        self.assertEqual(get_view_direction_from_image_number(4, num_side_views=8), "y")

        # View 6 (180°) → -x direction.
        self.assertEqual(
            get_view_direction_from_image_number(6, num_side_views=8), "-x"
        )

        # View 8 (270°) → -y direction.
        self.assertEqual(
            get_view_direction_from_image_number(8, num_side_views=8), "-y"
        )

    def test_get_view_direction_side_views_4_views(self):
        """Test view direction mapping for 4 side views."""
        # With 4 side views.
        self.assertEqual(get_view_direction_from_image_number(2, num_side_views=4), "x")
        self.assertEqual(get_view_direction_from_image_number(3, num_side_views=4), "y")
        self.assertEqual(
            get_view_direction_from_image_number(4, num_side_views=4), "-x"
        )
        self.assertEqual(
            get_view_direction_from_image_number(5, num_side_views=4), "-y"
        )

    def test_get_view_direction_diagonal_views(self):
        """Test view direction mapping with diagonal views enabled."""
        # Top diagonal.
        self.assertEqual(
            get_view_direction_from_image_number(
                10, num_side_views=8, include_diagonal_views=True
            ),
            "top_diagonal",
        )

        # Bottom diagonal.
        self.assertEqual(
            get_view_direction_from_image_number(
                11, num_side_views=8, include_diagonal_views=True
            ),
            "bottom_diagonal",
        )

    def test_get_view_direction_invalid_image_number(self):
        """Test error handling for invalid image number."""
        with self.assertRaises(ValueError) as context:
            get_view_direction_from_image_number(100, num_side_views=8)

        self.assertIn("Invalid image number", str(context.exception))

    def test_get_view_direction_without_vertical_views(self):
        """Test view direction mapping when vertical views are excluded."""
        # With include_vertical_views=False, numbering starts at 0 for side views.
        self.assertEqual(
            get_view_direction_from_image_number(
                0, num_side_views=4, include_vertical_views=False
            ),
            "x",
        )
        self.assertEqual(
            get_view_direction_from_image_number(
                1, num_side_views=4, include_vertical_views=False
            ),
            "y",
        )
        self.assertEqual(
            get_view_direction_from_image_number(
                2, num_side_views=4, include_vertical_views=False
            ),
            "-x",
        )
        self.assertEqual(
            get_view_direction_from_image_number(
                3, num_side_views=4, include_vertical_views=False
            ),
            "-y",
        )


class TestFrontAxisMapping(unittest.TestCase):
    """Test front axis calculation from image numbers."""

    def test_get_front_axis_top_view(self):
        """Test front axis for top view."""
        self.assertEqual(get_front_axis_from_image_number(0), "z")

    def test_get_front_axis_bottom_view(self):
        """Test front axis for bottom view."""
        self.assertEqual(get_front_axis_from_image_number(1), "-z")

    def test_get_front_axis_side_views(self):
        """Test front axis for side views."""
        # View 2 (0°) → x.
        self.assertEqual(get_front_axis_from_image_number(2, num_side_views=8), "x")

        # View 4 (90°) → y.
        self.assertEqual(get_front_axis_from_image_number(4, num_side_views=8), "y")

        # View 6 (180°) → -x.
        self.assertEqual(get_front_axis_from_image_number(6, num_side_views=8), "-x")

        # View 8 (270°) → -y.
        self.assertEqual(get_front_axis_from_image_number(8, num_side_views=8), "-y")

    def test_get_front_axis_diagonal_views(self):
        """Test front axis for diagonal views."""
        # Top diagonal → z.
        self.assertEqual(
            get_front_axis_from_image_number(
                10, num_side_views=8, include_diagonal_views=True
            ),
            "z",
        )

        # Bottom diagonal → -z.
        self.assertEqual(
            get_front_axis_from_image_number(
                11, num_side_views=8, include_diagonal_views=True
            ),
            "-z",
        )

    def test_get_front_axis_without_vertical_views(self):
        """Test front axis mapping when vertical views are excluded."""
        # With include_vertical_views=False, image indices map directly to side views.
        # Image 0 (0°) → x.
        self.assertEqual(
            get_front_axis_from_image_number(
                0, num_side_views=4, include_vertical_views=False
            ),
            "x",
        )
        # Image 1 (90°) → y.
        self.assertEqual(
            get_front_axis_from_image_number(
                1, num_side_views=4, include_vertical_views=False
            ),
            "y",
        )
        # Image 2 (180°) → -x.
        self.assertEqual(
            get_front_axis_from_image_number(
                2, num_side_views=4, include_vertical_views=False
            ),
            "-x",
        )
        # Image 3 (270°) → -y.
        self.assertEqual(
            get_front_axis_from_image_number(
                3, num_side_views=4, include_vertical_views=False
            ),
            "-y",
        )


class TestMeshPhysicsAnalysis(unittest.TestCase):
    """Test MeshPhysicsAnalysis dataclass."""

    def test_mesh_physics_analysis_creation(self):
        """Test creating MeshPhysicsAnalysis instance."""
        analysis = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.5,
            mass_range_kg=(8.0, 12.0),
        )

        self.assertEqual(analysis.up_axis, "+Z")
        self.assertEqual(analysis.front_axis, "+Y")
        self.assertEqual(analysis.material, "wood")
        self.assertEqual(analysis.mass_kg, 10.5)
        self.assertEqual(analysis.mass_range_kg, (8.0, 12.0))


class TestVLMResponseParsing(unittest.TestCase):
    """Test VLM response parsing logic."""

    def test_parse_vlm_response_valid_json(self):
        """Test VLM response parsing with valid JSON."""
        # Mock VLM response.
        valid_response = {
            "material": "wood",
            "mass_kg": 10.5,
            "mass_range_kg": [8.0, 12.0],
            "canonical_orientation": {"up_axis": "z", "front_view_image_index": 2},
        }

        # Mock VLM service.
        mock_vlm_instance = MagicMock()
        mock_vlm_instance.create_completion.return_value = json.dumps(valid_response)

        # Create temporary image files.
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        image_paths = [
            temp_path / "view_0.png",
            temp_path / "view_1.png",
            temp_path / "view_2.png",
        ]
        for img_path in image_paths:
            # Create a simple 10x10 black image.
            img = Image.new("RGB", (10, 10), color="black")
            img.save(img_path)

        # Mock BlenderServer.
        mock_blender_server = MagicMock()
        mock_blender_server.is_running.return_value = True
        mock_blender_server.render_multiview_for_analysis.return_value = image_paths

        # Create a dummy mesh file.
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            mesh_path = Path(f.name)
            cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
            cube.export(mesh_path)

        try:
            # Analyze mesh.
            result = analyze_mesh_orientation_and_material(
                mesh_path=mesh_path,
                cfg=create_mock_cfg(),
                elevation_degrees=20.0,
                blender_server=mock_blender_server,
            )

            # Verify parsing.
            self.assertEqual(result.up_axis, "+Z")
            self.assertEqual(result.front_axis, "+X")  # Image 2 maps to x.
            self.assertEqual(result.material, "wood")
            self.assertEqual(result.mass_kg, 10.5)
            self.assertEqual(result.mass_range_kg, (8.0, 12.0))

        finally:
            # Cleanup.
            if mesh_path.exists():
                mesh_path.unlink()
            if temp_path.exists():
                shutil.rmtree(temp_path)

    def test_parse_vlm_response_missing_fields(self):
        """Test error handling for malformed VLM response."""
        # Mock VLM response with missing fields.
        invalid_response = {"material": "wood"}  # Missing mass and orientation.

        # Mock VLM service.
        mock_vlm_instance = MagicMock()
        mock_vlm_instance.create_completion.return_value = json.dumps(invalid_response)

        # Create temporary image files.
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        image_path = temp_path / "view_0.png"
        img = Image.new("RGB", (10, 10), color="black")
        img.save(image_path)

        # Mock BlenderServer.
        mock_blender_server = MagicMock()
        mock_blender_server.is_running.return_value = True
        mock_blender_server.render_multiview_for_analysis.return_value = [image_path]

        # Create a dummy mesh file.
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            mesh_path = Path(f.name)
            cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
            cube.export(mesh_path)

        try:
            # Should raise ValueError.
            with self.assertRaises(ValueError) as context:
                analyze_mesh_orientation_and_material(
                    mesh_path=mesh_path,
                    cfg=create_mock_cfg(),
                    elevation_degrees=20.0,
                    blender_server=mock_blender_server,
                )

            self.assertIn("Failed to parse VLM response", str(context.exception))

        finally:
            # Cleanup.
            if mesh_path.exists():
                mesh_path.unlink()
            if temp_path.exists():
                shutil.rmtree(temp_path)

    def test_parse_vlm_response_negative_axis(self):
        """Test VLM response parsing with negative axis."""
        # Mock VLM response with negative axis.
        response = {
            "material": "metal",
            "mass_kg": 5.0,
            "mass_range_kg": [4.0, 6.0],
            "canonical_orientation": {"up_axis": "-y", "front_view_image_index": 1},
        }

        # Mock VLM service.
        mock_vlm_instance = MagicMock()
        mock_vlm_instance.create_completion.return_value = json.dumps(response)

        # Create temporary image files.
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        image_paths = [temp_path / "view_0.png", temp_path / "view_1.png"]
        for img_path in image_paths:
            img = Image.new("RGB", (10, 10), color="black")
            img.save(img_path)

        # Mock BlenderServer.
        mock_blender_server = MagicMock()
        mock_blender_server.is_running.return_value = True
        mock_blender_server.render_multiview_for_analysis.return_value = image_paths

        # Create a dummy mesh file.
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            mesh_path = Path(f.name)
            cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
            cube.export(mesh_path)

        try:
            # Analyze mesh.
            result = analyze_mesh_orientation_and_material(
                mesh_path=mesh_path,
                cfg=create_mock_cfg(),
                elevation_degrees=20.0,
                blender_server=mock_blender_server,
            )

            # Verify negative axis is preserved.
            self.assertEqual(result.up_axis, "-Y")
            self.assertEqual(result.front_axis, "-Z")  # Image 1 = bottom = -z.
            self.assertEqual(result.material, "metal")

        finally:
            # Cleanup.
            if mesh_path.exists():
                mesh_path.unlink()
            if temp_path.exists():
                shutil.rmtree(temp_path)

    def test_hssd_prompt_validation_success(self):
        """Test HSSD validation passes when up_axis is 'z'."""
        # Mock VLM response with up_axis="z" (valid for HSSD).
        response = {
            "material": "wood",
            "mass_kg": 5.0,
            "mass_range_kg": [4.0, 6.0],
            "canonical_orientation": {"up_axis": "z", "front_view_image_index": 1},
        }

        # Mock VLM service.
        mock_vlm_instance = MagicMock()
        mock_vlm_instance.create_completion.return_value = json.dumps(response)

        # Create temporary image files.
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        image_paths = [temp_path / f"view_{i}.png" for i in range(4)]
        for img_path in image_paths:
            img = Image.new("RGB", (10, 10), color="black")
            img.save(img_path)

        # Mock BlenderServer.
        mock_blender_server = MagicMock()
        mock_blender_server.is_running.return_value = True
        mock_blender_server.render_multiview_for_analysis.return_value = image_paths

        # Create a dummy mesh file.
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            mesh_path = Path(f.name)
            cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
            cube.export(mesh_path)

        try:
            # Analyze mesh with HSSD prompt type.
            result = analyze_mesh_orientation_and_material(
                mesh_path=mesh_path,
                cfg=create_mock_cfg(),
                elevation_degrees=20.0,
                blender_server=mock_blender_server,
                prompt_type="hssd",
                include_vertical_views=False,
            )

            # Verify up_axis is correctly parsed as "+Z".
            self.assertEqual(result.up_axis, "+Z")
            self.assertEqual(result.front_axis, "+Y")

        finally:
            # Cleanup.
            if mesh_path.exists():
                mesh_path.unlink()
            if temp_path.exists():
                shutil.rmtree(temp_path)

    def test_hssd_prompt_validation_failure(self):
        """Test HSSD validation fails when up_axis is not 'z'."""
        # Mock VLM response with up_axis="x" (invalid for HSSD).
        response = {
            "material": "wood",
            "mass_kg": 5.0,
            "mass_range_kg": [4.0, 6.0],
            "canonical_orientation": {"up_axis": "x", "front_view_image_index": 1},
        }

        # Mock VLM service.
        mock_vlm_instance = MagicMock()
        mock_vlm_instance.create_completion.return_value = json.dumps(response)

        # Create temporary image files.
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        image_paths = [temp_path / f"view_{i}.png" for i in range(4)]
        for img_path in image_paths:
            img = Image.new("RGB", (10, 10), color="black")
            img.save(img_path)

        # Mock BlenderServer.
        mock_blender_server = MagicMock()
        mock_blender_server.is_running.return_value = True
        mock_blender_server.render_multiview_for_analysis.return_value = image_paths

        # Create a dummy mesh file.
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            mesh_path = Path(f.name)
            cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
            cube.export(mesh_path)

        try:
            # Should raise ValueError due to HSSD constraint violation.
            with self.assertRaises(ValueError) as context:
                analyze_mesh_orientation_and_material(
                    mesh_path=mesh_path,
                    cfg=create_mock_cfg(),
                    elevation_degrees=20.0,
                    blender_server=mock_blender_server,
                    prompt_type="hssd",
                    include_vertical_views=False,
                )

            # Verify error message contains HSSD constraint info.
            self.assertIn("HSSD object must have up_axis='z'", str(context.exception))
            self.assertIn("canonically upright", str(context.exception))

        finally:
            # Cleanup.
            if mesh_path.exists():
                mesh_path.unlink()
            if temp_path.exists():
                shutil.rmtree(temp_path)


class TestSplitPhysicsFunctions(unittest.TestCase):
    """Test render_and_encode / analyze_from_encoded split functions for parallel VLM."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cfg = create_mock_cfg()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scenesmith.agent_utils.mesh_physics_analyzer.encode_image_to_base64")
    def test_render_and_encode_returns_base64_list(self, mock_encode):
        """render_and_encode_multiview_images_for_mesh_physics returns one base64 str per rendered image."""
        from scenesmith.agent_utils.mesh_physics_analyzer import (
            render_and_encode_multiview_images_for_mesh_physics,
        )

        n_views = 4
        fake_paths = [self.temp_dir / f"view_{i}.png" for i in range(n_views)]
        for p in fake_paths:
            img = Image.new("RGB", (10, 10), color="black")
            img.save(p)

        mock_blender = MagicMock()
        mock_blender.render_multiview_for_analysis.return_value = fake_paths
        mock_encode.side_effect = lambda p: f"b64_{p.stem}"

        result = render_and_encode_multiview_images_for_mesh_physics(
            mesh_path=self.temp_dir / "mesh.gltf",
            cfg=self.cfg,
            elevation_degrees=20.0,
            blender_server=mock_blender,
            num_side_views=n_views,
            include_vertical_views=False,
        )

        self.assertEqual(len(result), n_views)
        self.assertTrue(all(isinstance(s, str) for s in result))
        mock_blender.render_multiview_for_analysis.assert_called_once()

    def test_analyze_from_encoded_images_parses_vlm_response(self):
        """analyze_mesh_physics_from_encoded_images parses a valid JSON response correctly."""
        from scenesmith.agent_utils.mesh_physics_analyzer import (
            analyze_mesh_physics_from_encoded_images,
        )

        valid_response = json.dumps(
            {
                "material": "metal",
                "mass_kg": 8.0,
                "mass_range_kg": [6.0, 10.0],
                "canonical_orientation": {
                    "up_axis": "z",
                    "front_view_image_index": 0,
                },
            }
        )
        mock_vlm = MagicMock()
        mock_vlm.create_completion.return_value = valid_response

        result = analyze_mesh_physics_from_encoded_images(
            encoded_images=["b64_img1", "b64_img2", "b64_img3", "b64_img4"],
            mesh_stem="test_object",
            cfg=self.cfg,
            num_side_views=4,
            prompt_type="hssd",
            include_vertical_views=False,
        )

        self.assertIsInstance(result, MeshPhysicsAnalysis)
        self.assertEqual(result.material, "metal")
        self.assertEqual(result.mass_kg, 8.0)
        self.assertEqual(result.up_axis, "+Z")
        mock_vlm.create_completion.assert_called_once()

    def test_analyze_from_encoded_images_invalid_json(self):
        """analyze_mesh_physics_from_encoded_images raises RuntimeError for non-JSON VLM output."""
        from scenesmith.agent_utils.mesh_physics_analyzer import (
            analyze_mesh_physics_from_encoded_images,
        )

        mock_vlm = MagicMock()
        mock_vlm.create_completion.return_value = "not valid json {"

        with self.assertRaises(RuntimeError) as ctx:
            analyze_mesh_physics_from_encoded_images(
                encoded_images=["b64_img"],
                mesh_stem="bad_object",
                cfg=self.cfg,
                num_side_views=4,
                prompt_type="hssd",
                include_vertical_views=False,
            )

        self.assertIn("VLM analysis failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
