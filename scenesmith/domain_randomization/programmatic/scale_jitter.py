"""Scale jitter: vary scale_factor per object within a configurable range."""

from __future__ import annotations

import logging
import re

from pathlib import Path

import numpy as np

from scenesmith.agent_utils.room import RoomScene
from scenesmith.domain_randomization.programmatic.base import BaseRandomizationOp
from scenesmith.domain_randomization.programmatic.collision_checker import (
    check_collisions,
    check_room_bounds,
)
from scenesmith.domain_randomization.programmatic.ops_utils import _resolve_types

console_logger = logging.getLogger(__name__)

_MESH_BLOCK_RE = re.compile(r"(<mesh\b[^>]*>)(.*?)(</mesh>)", re.DOTALL)
_URI_INNER_RE = re.compile(r"(?P<indent>[ \t]*)<uri>([^<]+)</uri>")
_SCALE_RE = re.compile(r"<scale>\s*([^<]+?)\s*</scale>")


def _rescale_sdf_in_place(sdf_path: Path, ratio: float) -> int:
    """Apply ``ratio`` to every ``<mesh>`` block's effective scale.

    For each ``<mesh>...</mesh>``: if a ``<scale>`` is present, multiply
    its components by ``ratio``; if missing, **insert** one with value
    ``ratio ratio ratio`` after the ``<uri>`` (Drake's default is 1, so
    omission cannot be assumed safe). Returns the number of mesh
    blocks rewritten.

    Inertial blocks are deliberately untouched; DR ScaleJitter treats
    mass/inertia as fixed and only visual + collision mesh sizes scale.
    """
    if not sdf_path.exists():
        return 0
    text = sdf_path.read_text()
    count = 0

    def _rewrite_block(block_match: re.Match) -> str:
        nonlocal count
        open_tag, body, close_tag = (
            block_match.group(1),
            block_match.group(2),
            block_match.group(3),
        )
        if _SCALE_RE.search(body):

            def _scale_repl(m: re.Match) -> str:
                parts = [float(x) for x in m.group(1).split()]
                if len(parts) == 1:
                    parts = parts * 3
                scaled = " ".join(f"{p * ratio:.8f}" for p in parts)
                return f"<scale>{scaled}</scale>"

            body_new = _SCALE_RE.sub(_scale_repl, body)
        else:
            uri_match = _URI_INNER_RE.search(body)
            if uri_match is None:
                return block_match.group(0)
            indent = uri_match.group("indent")
            scale_line = f"\n{indent}<scale>{ratio:.8f} {ratio:.8f} {ratio:.8f}</scale>"
            body_new = body[: uri_match.end()] + scale_line + body[uri_match.end() :]
        count += 1
        return f"{open_tag}{body_new}{close_tag}"

    new_text = _MESH_BLOCK_RE.sub(_rewrite_block, text)
    if count and new_text != text:
        sdf_path.write_text(new_text)
    return count


class ScaleJitter(BaseRandomizationOp):
    """Randomly vary the scale_factor of objects within a range.

    Recomputes bbox_min/max proportionally so downstream checks remain
    valid AND multiplies the per-mesh ``<scale>`` elements in the
    object's SDF so the renderer + simulator see the new size. The
    variant scene is scaffolded via ``scaffold_variation_v2`` (flat
    copytree), so each variant already owns its own SDF copies and
    rewriting them in place is safe.

    Params:
        scale_range (tuple[float, float]): [min_scale, max_scale]. Default [0.85, 1.15].
        target_object_types (list[str]): Object type strings to jitter.
        seed (int | None): Random seed for reproducibility.
    """

    def apply(self, scene: RoomScene, params: dict, locked_ids: set[str]) -> RoomScene:
        rng = np.random.default_rng(params.get("seed"))
        scale_min, scale_max = params.get("scale_range", [0.85, 1.15])
        target_types = _resolve_types(params.get("target_object_types", []))

        for obj in scene.objects.values():
            if obj.object_id in locked_ids:
                continue
            if target_types and obj.object_type not in target_types:
                continue

            factor = float(rng.uniform(scale_min, scale_max))
            current = obj.scale_factor if obj.scale_factor else 1.0
            ratio = factor / current

            obj.scale_factor = factor
            # Preserve numpy array type — SceneObject.to_dict calls
            # .tolist() on bbox_min/max, so plain Python lists crash
            # serialization later (observed on E2 v1 N=3, 2/3 variants
            # using ScaleJitter failed to save scene_state.json).
            obj.bbox_min = np.array([v * ratio for v in obj.bbox_min])
            obj.bbox_max = np.array([v * ratio for v in obj.bbox_max])

            if obj.sdf_path:
                rewrote = _rescale_sdf_in_place(Path(obj.sdf_path), ratio)
                if rewrote:
                    console_logger.debug(
                        "ScaleJitter: %s factor=%.3f ratio=%.3f (%d <scale> tags)",
                        obj.object_id,
                        factor,
                        ratio,
                        rewrote,
                    )
                else:
                    console_logger.warning(
                        "ScaleJitter: %s sdf_path=%s — no <scale> elements found, "
                        "render size will not change",
                        obj.object_id,
                        obj.sdf_path,
                    )

        return scene

    def validate(self, scene: RoomScene) -> list[str]:
        return check_collisions(scene) + check_room_bounds(scene)
