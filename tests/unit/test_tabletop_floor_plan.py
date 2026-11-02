"""Unit tests for scenesmith.tabletop.floor_plan."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.agent_utils.room import ObjectType, RoomScene
from scenesmith.tabletop.floor_plan import (
    _build_floor_object,
    _write_floor_sdf,
    build_tabletop_house_layout,
)


class TestWriteFloorSdf(unittest.TestCase):
    def test_sdf_is_valid_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "floor.sdf"
            _write_floor_sdf(path, 4.0, 3.0)
            tree = ET.parse(path)
            root = tree.getroot()
            self.assertEqual(root.tag, "sdf")

    def test_sdf_floor_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "floor.sdf"
            _write_floor_sdf(path, 5.0, 6.0)
            content = path.read_text()
            self.assertIn("5.0000", content)
            self.assertIn("6.0000", content)

    def test_sdf_has_collision_and_visual(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "floor.sdf"
            _write_floor_sdf(path, 4.0, 4.0)
            content = path.read_text()
            self.assertIn("floor_collision", content)
            self.assertIn("floor_visual", content)


class TestBuildFloorObject(unittest.TestCase):
    def test_object_type_is_floor(self):
        obj = _build_floor_object(4.0, 4.0)
        self.assertEqual(obj.object_type, ObjectType.FLOOR)

    def test_bbox_spans_full_floor(self):
        obj = _build_floor_object(4.0, 6.0)
        np.testing.assert_allclose(obj.bbox_min[:2], [-2.0, -3.0])
        np.testing.assert_allclose(obj.bbox_max[:2], [2.0, 3.0])

    def test_floor_is_immutable(self):
        obj = _build_floor_object(4.0, 4.0)
        self.assertTrue(obj.immutable)

    def test_no_geometry_paths(self):
        obj = _build_floor_object(4.0, 4.0)
        self.assertIsNone(obj.geometry_path)
        self.assertIsNone(obj.sdf_path)


class TestBuildTabletopHouseLayout(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.scene_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, prompt="A mug and a book on the table.", w=4.0, l=4.0):
        return build_tabletop_house_layout(
            prompt=prompt, scene_dir=self.scene_dir, room_width=w, room_length=l
        )

    def test_returns_house_layout(self):
        layout = self._build()
        self.assertIsInstance(layout, HouseLayout)

    def test_single_room_named_tabletop(self):
        layout = self._build()
        self.assertEqual(layout.room_ids, ["tabletop"])

    def test_room_dimensions(self):
        layout = self._build(w=3.0, l=5.0)
        geom = layout.get_room_geometry("tabletop")
        self.assertEqual(geom.width, 3.0)
        self.assertEqual(geom.length, 5.0)

    def test_no_walls(self):
        layout = self._build()
        geom = layout.get_room_geometry("tabletop")
        self.assertEqual(geom.walls, [])
        self.assertEqual(geom.wall_normals, {})

    def test_floor_object_present(self):
        layout = self._build()
        geom = layout.get_room_geometry("tabletop")
        self.assertIsNotNone(geom.floor)
        self.assertEqual(geom.floor.object_type, ObjectType.FLOOR)

    def test_room_prompt_contains_constraint(self):
        layout = self._build(prompt="A mug on the table.")
        spec = layout.get_room_spec("tabletop")
        self.assertIn("A mug on the table.", spec.prompt)
        self.assertIn("TABLETOP SCENE CONSTRAINT", spec.prompt)
        self.assertIn("exactly ONE table", spec.prompt)

    def test_sdf_file_written_to_disk(self):
        self._build()
        sdf_path = self.scene_dir / "floor_plans" / "tabletop" / "room_geometry.sdf"
        self.assertTrue(sdf_path.exists())

    def test_placement_and_connectivity_valid(self):
        layout = self._build()
        self.assertTrue(layout.placement_valid)
        self.assertTrue(layout.connectivity_valid)

    def test_wall_height_sentinel(self):
        layout = self._build()
        geom = layout.get_room_geometry("tabletop")
        self.assertGreater(geom.wall_height, 5.0)

    def test_json_roundtrip(self):
        """HouseLayout must survive to_dict → JSON → from_dict → RoomScene."""
        layout = self._build()
        layout_dict = layout.to_dict(scene_dir=self.scene_dir)
        serialized = json.dumps(layout_dict)
        restored = HouseLayout.from_dict(
            json.loads(serialized), house_dir=self.scene_dir
        )
        self.assertEqual(restored.room_ids, ["tabletop"])
        geom = restored.get_room_geometry("tabletop")
        self.assertIsNotNone(geom)
        self.assertEqual(geom.walls, [])

    def test_room_scene_construction(self):
        """RoomScene must be constructable from the tabletop geometry."""
        layout = self._build()
        geom = layout.get_room_geometry("tabletop")
        room_dir = self.scene_dir / "room_tabletop"
        room_dir.mkdir()
        scene = RoomScene(
            room_geometry=geom,
            scene_dir=room_dir,
            room_id="tabletop",
        )
        # No walls added (walls list is empty).
        for wall in geom.walls:
            scene.add_object(wall)
        self.assertEqual(len(scene.objects), 0)
        self.assertIsNotNone(scene.room_geometry.floor)


if __name__ == "__main__":
    unittest.main()
