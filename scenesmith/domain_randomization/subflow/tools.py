"""DR v2 Designer tool wraps + Critic deterministic checks."""

from __future__ import annotations

import hashlib
import json
import logging

from dataclasses import dataclass
from typing import Any

from agents import function_tool
from agents.tool import FunctionTool

console_logger = logging.getLogger(__name__)

from scenesmith.domain_randomization.programmatic.asset_substitution import (
    AssetSubstitution,
)
from scenesmith.domain_randomization.programmatic.object_dropout import ObjectDropout
from scenesmith.domain_randomization.programmatic.pose_perturbation import (
    PosePerturbation,
)
from scenesmith.domain_randomization.programmatic.scale_jitter import ScaleJitter
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools

_OP_REGISTRY = {
    "pose_perturbation": PosePerturbation(),
    "scale_jitter": ScaleJitter(),
    "object_dropout": ObjectDropout(),
    "asset_substitution": AssetSubstitution(),
}

_IDENTITY_FIELDS = ("geometry_path", "sdf_path", "name", "object_type")
_TRANSFORM_TOLERANCE = 1e-6


class LockedIDError(RuntimeError):
    """Designer tool refused to act on a locked (task-critical) object ID."""


class PoseBoundsError(RuntimeError):
    """Designer tool refused a pose change that exceeded the task-object bounds."""


@dataclass
class CheckResult:
    passed: bool
    detail: str
    score_hint: float | None = None


def assert_id_not_locked(object_id: str, locked_ids: set[str]) -> None:
    if object_id in locked_ids:
        raise LockedIDError(
            f"Refused: object_id={object_id!r} is task-critical (locked). "
            f"Use a non-locked manipuland id, or pick a different operation."
        )


def _assert_pose_within_bounds(
    object_id: str,
    new_pos_xz: tuple[float, float],
    pose_bounds: dict[str, dict],
) -> None:
    """Per the DR v2 design (`dr_subflow/director.yaml` hard rules), task-
    critical object POSE may change within their on-surface position range
    while identity must not. This enforces the position-range half."""
    spec = pose_bounds.get(object_id)
    if not spec:
        # No bounds declared → identity-lock is still authoritative; the
        # caller's locked-id check should refuse before reaching here.
        return
    nominal = spec.get("nominal_position_2d") or [0.0, 0.0]
    dxdz = spec.get("max_dxdz") or [0.0, 0.0]
    dx_allowed, dz_allowed = float(dxdz[0]), float(dxdz[1])
    nx, nz = float(nominal[0]), float(nominal[1])
    px, pz = float(new_pos_xz[0]), float(new_pos_xz[1])
    if abs(px - nx) > dx_allowed or abs(pz - nz) > dz_allowed:
        raise PoseBoundsError(
            f"Refused move of task-critical {object_id!r}: requested "
            f"({px:.3f}, {pz:.3f}) is outside position_range_xy_on_surface "
            f"±({dx_allowed:.3f}, {dz_allowed:.3f}) around nominal "
            f"({nx:.3f}, {nz:.3f}). Pick a position within the allowed box."
        )


class DRDesignerTools:
    """Thin wrap over ManipulandTools + programmatic ops with two-tier guards.

    ``locked_ids`` are *identity-locked* (task-critical): the object MUST
    remain present and keep its asset, but its POSE may change within the
    declared on-surface bounds (`pose_bounds`). Non-locked ids are fully
    movable / removable.

    ``pose_bounds`` is a per-id dict ``{nominal_position_2d, max_dxdz}``
    derived from ``rl_artifacts.task_object_variants``. Empty for non-
    locked ids (their pose is unconstrained); required for locked ids
    that the designer is allowed to nudge.
    """

    def __init__(
        self,
        manipuland_tools: ManipulandTools,
        locked_ids: set[str],
        pose_bounds: dict[str, dict] | None = None,
        collision_check_enabled: bool = False,
        collision_margin_m: float = 0.0,
        collision_ignore_within_task_critical: bool = False,
    ):
        self._mt = manipuland_tools
        self._locked = set(locked_ids)
        self._pose_bounds = dict(pose_bounds or {})
        self._cc_enabled = bool(collision_check_enabled)
        self._cc_margin = float(collision_margin_m)
        self._cc_ignore_within_locked = bool(collision_ignore_within_task_critical)

    # ------------------------------------------------------------------
    # Collision feedback helpers (AABB; rotation-ignorant; cheap).
    # ------------------------------------------------------------------

    def _collision_pairs(self) -> set[frozenset[str]]:
        """Return the current set of colliding {id_a, id_b} pairs.

        Only manipuland-vs-manipuland overlaps are reported. Pairs that
        involve walls / floor / furniture are excluded because the AABB
        check is rotation-ignorant and treats a manipuland resting on a
        table as "colliding" with the table — that's design intent, not
        a bug. The Designer needs the signal for things like two plates
        placed at the same surface coordinates.
        """
        if not self._cc_enabled:
            return set()
        # Local import to keep the module light when collision_check is off.
        from scenesmith.agent_utils.room import ObjectType
        from scenesmith.domain_randomization.programmatic.collision_checker import (
            aabb_overlap,
            get_world_aabb,
        )

        scene = self._mt.scene
        objs = [
            obj
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.MANIPULAND
        ]

        pairs: set[frozenset[str]] = set()
        margin = self._cc_margin
        for i, a in enumerate(objs):
            min_a, max_a = get_world_aabb(a)
            for b in objs[i + 1 :]:
                if (
                    self._cc_ignore_within_locked
                    and a.object_id in self._locked
                    and b.object_id in self._locked
                ):
                    continue
                min_b, max_b = get_world_aabb(b)
                if aabb_overlap(min_a, max_a, min_b, max_b, margin=margin):
                    pairs.add(frozenset({str(a.object_id), str(b.object_id)}))
        return pairs

    def _wrap_result_with_collisions(
        self, result_json: str, pre_pairs: set[frozenset[str]]
    ) -> str:
        """Add `new_collisions` / `remaining_collisions` to a tool result JSON.

        ``new_collisions`` lists pairs introduced by this mutation that
        weren't present before (the Designer's signal to retry).
        ``remaining_collisions`` is the total pair list after the
        mutation — surfaced so the Designer can also see leftovers it
        could clean up next.
        """
        if not self._cc_enabled:
            return result_json
        post_pairs = self._collision_pairs()
        new_pairs = post_pairs - pre_pairs
        try:
            parsed = json.loads(result_json) if result_json else {}
            if not isinstance(parsed, dict):
                parsed = {"raw": parsed}
        except json.JSONDecodeError:
            parsed = {"raw": result_json}
        if new_pairs:
            parsed["new_collisions"] = sorted([sorted(list(p)) for p in new_pairs])
        if post_pairs:
            parsed["remaining_collisions"] = sorted(
                [sorted(list(p)) for p in post_pairs]
            )
        return json.dumps(parsed)

    @property
    def room(self):
        return self._mt.scene

    def move_manipuland(self, object_id: str, **kwargs) -> str:
        # Identity-locked items may move within their bounds; pose check
        # applies to locked AND unlocked equally when bounds are declared.
        if object_id in self._locked and object_id not in self._pose_bounds:
            raise LockedIDError(
                f"Refused: task-critical {object_id!r} has no declared "
                f"position_range_xy_on_surface — its pose is implicitly locked."
            )
        pos_x = kwargs.get("position_x")
        pos_z = kwargs.get("position_z")
        if pos_x is not None and pos_z is not None and object_id in self._pose_bounds:
            _assert_pose_within_bounds(object_id, (pos_x, pos_z), self._pose_bounds)
        return self._mt._move_manipuland_impl(object_id, **kwargs)

    def remove_manipuland(self, object_id: str, **kwargs) -> str:
        # Identity-locked: MUST remain present. Always refuse.
        assert_id_not_locked(object_id, self._locked)
        return self._mt._remove_manipuland_impl(object_id, **kwargs)

    def rescale_manipuland(self, object_id: str, scale_factor: float, **kwargs) -> str:
        assert_id_not_locked(object_id, self._locked)
        return self._mt._rescale_manipuland_impl(object_id, scale_factor, **kwargs)

    def place_manipuland_on_surface(self, **kwargs) -> str:
        new_id = kwargs.get("object_id")
        if new_id is not None:
            assert_id_not_locked(new_id, self._locked)
        return self._mt._place_manipuland_on_surface_impl(**kwargs)

    def swap_asset(self, object_id: str, **kwargs) -> str:
        """Swap one object's HSSD mesh while keeping its identity / pose.

        Refuses identity-locked ids (task-critical objects must keep
        their original asset). For non-locked ids the AssetSubstitution
        op runs in HSSD mode, scoped to just the requested ``object_id``
        via ``target_object_ids=[object_id]``. SDF regeneration is off
        by default — pass ``regenerate_sdf=True`` plus ``blender_server``
        and ``collision_client`` for the full collision-aware path.
        """
        assert_id_not_locked(object_id, self._locked)
        params = dict(kwargs)
        params.setdefault("source", "hssd")
        params.setdefault("target_object_types", ["manipuland"])
        params.setdefault("target_object_ids", [object_id])
        params.setdefault("seed", 0)
        if params["source"] == "catalog":
            params.setdefault(
                "catalog_dir",
                str(self._mt.scene.scene_dir / "generated_assets"),
            )
        before_geom = None
        scene_obj = (
            self._mt.scene.objects.get(object_id)
            if hasattr(self._mt.scene, "objects")
            else None
        )
        if scene_obj is not None:
            before_geom = scene_obj.geometry_path
        scene = _OP_REGISTRY["asset_substitution"].apply(
            self._mt.scene, params, self._locked
        )
        self._mt.scene = scene
        after_obj = (
            self._mt.scene.objects.get(object_id)
            if hasattr(self._mt.scene, "objects")
            else None
        )
        swapped = after_obj is not None and after_obj.geometry_path != before_geom
        return json.dumps(
            {
                "success": True,
                "swapped": swapped,
                "object_id": object_id,
                "new_geometry_path": (
                    after_obj.geometry_path if after_obj is not None else None
                ),
                "message": (
                    f"swap_asset applied to {object_id}"
                    if swapped
                    else f"swap_asset found no alternatives for {object_id}"
                ),
            }
        )

    def apply_programmatic_op(self, op_name: str, **kwargs) -> str:
        oid = kwargs.get("object_id") or kwargs.get("target_id")
        if oid is not None:
            assert_id_not_locked(str(oid), self._locked)
        op = _OP_REGISTRY[op_name]
        params = {
            k: v for k, v in kwargs.items() if k not in ("object_id", "target_id")
        }
        scene = op.apply(self._mt.scene, params, self._locked)
        self._mt.scene = scene
        return json.dumps({"success": True, "message": f"{op_name} applied"})

    def llm_edit_scene(self, prompt: str, **kwargs) -> str:
        return "<llm_edit_scene stub: would have edited scene>"

    def as_function_tools(
        self,
        *,
        enable_swap_asset: bool = True,
        swap_asset_blender_server: Any = None,
        swap_asset_collision_client: Any = None,
    ) -> list[FunctionTool]:
        """Expose the locked-ID-guarded mutations as Agents-SDK FunctionTools.

        ``move_manipuland`` and ``remove_manipuland`` are always exposed.
        ``swap_asset`` is exposed by default and uses the HSSD-backed
        AssetSubstitution under the hood (HSSD server is lazy-started on
        first call). When ``swap_asset_blender_server`` and
        ``swap_asset_collision_client`` are both provided, the tool will
        also regenerate the Drake SDF for the swapped asset so downstream
        physics stays consistent — otherwise it does a mesh-only swap
        (visual change only; existing collision shape preserved).

        Args:
            enable_swap_asset: Set False to drop the swap tool entirely
                (useful when the caller doesn't want any asset-identity
                changes in this subflow run).
            swap_asset_blender_server: BlenderServer instance for
                GLB→GLTF + full-SDF-regen mode.
            swap_asset_collision_client: ConvexDecompositionClient
                instance for CoACD-based SDF regeneration.
        """
        designer = self

        @function_tool
        async def move_manipuland(
            object_id: str,
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Move a manipuland to a new (x, z) position on a surface.

            Task-critical (locked) ids MAY be moved provided the new
            (position_x, position_z) stays within their declared
            position_range_xy_on_surface bounds. Non-locked ids are
            unbounded.

            Args:
                object_id: Manipuland id to move.
                surface_id: Target support surface id (e.g. "S_3").
                position_x: New x position on the surface (surface-local frame).
                position_z: New z position on the surface (surface-local frame).
                rotation_degrees: Rotation about the surface normal, degrees.

            Returns:
                JSON string describing success and updated transform. When
                collision_check is enabled, includes ``new_collisions``
                (pairs introduced by THIS move) and ``remaining_collisions``
                (pairs still present after the move).
            """
            pre_pairs = designer._collision_pairs()
            try:
                result_json = designer.move_manipuland(
                    object_id=object_id,
                    surface_id=surface_id,
                    position_x=position_x,
                    position_z=position_z,
                    rotation_degrees=rotation_degrees,
                )
            except LockedIDError as exc:
                console_logger.info(
                    "[DR designer] move_manipuland REFUSED locked id=%s",
                    object_id,
                )
                return json.dumps({"success": False, "error": str(exc)})
            except PoseBoundsError as exc:
                console_logger.info(
                    "[DR designer] move_manipuland REFUSED out-of-bounds id=%s",
                    object_id,
                )
                return json.dumps({"success": False, "error": str(exc)})
            try:
                parsed = json.loads(result_json)
                ok = bool(parsed.get("success", False))
            except json.JSONDecodeError:
                ok = "success" in result_json.lower()
            console_logger.info(
                "[DR designer] move_manipuland id=%s surface=%s x=%.3f z=%.3f rot=%.1f -> success=%s",
                object_id,
                surface_id,
                position_x,
                position_z,
                rotation_degrees,
                ok,
            )
            return designer._wrap_result_with_collisions(result_json, pre_pairs)

        @function_tool
        async def remove_manipuland(object_id: str) -> str:
            """Remove a non-locked manipuland from the scene.

            Args:
                object_id: Manipuland id to remove (must not be task-critical / locked).

            Returns:
                JSON string describing success. When collision_check is
                enabled, the response also reports ``remaining_collisions``.
            """
            pre_pairs = designer._collision_pairs()
            try:
                result_json = designer.remove_manipuland(object_id=object_id)
            except LockedIDError as exc:
                console_logger.info(
                    "[DR designer] remove_manipuland REFUSED locked id=%s",
                    object_id,
                )
                return json.dumps({"success": False, "error": str(exc)})
            try:
                parsed = json.loads(result_json)
                ok = bool(parsed.get("success", False))
            except json.JSONDecodeError:
                ok = "success" in result_json.lower()
            console_logger.info(
                "[DR designer] remove_manipuland id=%s -> success=%s",
                object_id,
                ok,
            )
            return designer._wrap_result_with_collisions(result_json, pre_pairs)

        tools: list[FunctionTool] = [move_manipuland, remove_manipuland]

        if enable_swap_asset:
            blender_server_ref = swap_asset_blender_server
            collision_client_ref = swap_asset_collision_client
            regenerate_sdf = (
                blender_server_ref is not None and collision_client_ref is not None
            )

            @function_tool
            async def swap_asset(object_id: str) -> str:
                """Swap one manipuland's HSSD-backed mesh.

                Refuses identity-locked (task-critical) ids — those must
                keep their original asset. For other manipulands the
                tool queries HSSD for a same-semantic-category candidate
                whose bbox is close to the current object's dimensions,
                downloads the GLB, and updates the scene to point at the
                new mesh. Returns JSON with a ``swapped`` flag (false
                means HSSD had no usable alternative).

                When the subflow was constructed with both a Blender
                server and a collision client wired through to this tool,
                the SDF is regenerated (GLB → GLTF + CoACD + Drake SDF)
                so downstream physics also uses the new collision shape.
                Otherwise the existing SDF is reused (mesh-only swap).

                Args:
                    object_id: Manipuland id to swap.

                Returns:
                    JSON string ``{"success", "swapped", "object_id",
                    "new_geometry_path", "message"}``.
                """
                pre_pairs = designer._collision_pairs()
                try:
                    swap_params: dict[str, Any] = {
                        "source": "hssd",
                        "target_object_types": ["manipuland"],
                        "target_object_ids": [object_id],
                    }
                    if regenerate_sdf:
                        swap_params["regenerate_sdf"] = True
                        swap_params["blender_server"] = blender_server_ref
                        swap_params["collision_client"] = collision_client_ref
                    result_json = designer.swap_asset(object_id, **swap_params)
                except LockedIDError as exc:
                    console_logger.info(
                        "[DR designer] swap_asset REFUSED locked id=%s",
                        object_id,
                    )
                    return json.dumps({"success": False, "error": str(exc)})
                except Exception as exc:  # noqa: BLE001 — surface as a tool failure
                    console_logger.warning(
                        "[DR designer] swap_asset id=%s raised: %s", object_id, exc
                    )
                    return json.dumps({"success": False, "error": str(exc)})

                try:
                    parsed = json.loads(result_json)
                except json.JSONDecodeError:
                    parsed = {"success": False, "swapped": False}
                console_logger.info(
                    "[DR designer] swap_asset id=%s swapped=%s regen_sdf=%s",
                    object_id,
                    parsed.get("swapped"),
                    regenerate_sdf,
                )
                return designer._wrap_result_with_collisions(result_json, pre_pairs)

            tools.append(swap_asset)

        return tools


def _scene_objects(scene: dict) -> dict[str, dict]:
    objects = scene.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("scene.objects must be a dict keyed by object id")
    return objects


def _transform_equal(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return a == b

    def _close(x: float, y: float) -> bool:
        return abs(x - y) <= _TRANSFORM_TOLERANCE

    trans_a = a.get("translation") or []
    trans_b = b.get("translation") or []
    if len(trans_a) != len(trans_b):
        return False
    if not all(_close(x, y) for x, y in zip(trans_a, trans_b)):
        return False

    rot_a = a.get("rotation_wxyz") or []
    rot_b = b.get("rotation_wxyz") or []
    if len(rot_a) != len(rot_b):
        return False
    return all(_close(x, y) for x, y in zip(rot_a, rot_b))


def check_locked_ids_untouched(
    base_scene: dict,
    current_scene: dict,
    locked_ids: set[str],
) -> CheckResult:
    base_objects = _scene_objects(base_scene)
    current_objects = _scene_objects(current_scene)

    for object_id in sorted(locked_ids):
        if object_id not in base_objects:
            return CheckResult(
                False, f"locked id {object_id!r} missing from base scene"
            )
        if object_id not in current_objects:
            return CheckResult(
                False, f"locked id {object_id!r} missing from current scene"
            )

        base_obj = base_objects[object_id]
        current_obj = current_objects[object_id]
        if not _transform_equal(
            base_obj.get("transform"), current_obj.get("transform")
        ):
            return CheckResult(False, f"locked id {object_id!r}: transform changed")

        for field in _IDENTITY_FIELDS:
            if base_obj.get(field) != current_obj.get(field):
                return CheckResult(False, f"locked id {object_id!r}: {field} changed")

    return CheckResult(True, "all locked ids untouched")


def check_position_bounds(
    base_scene: dict,
    current_scene: dict,
    rl_artifacts: dict,
) -> CheckResult:
    base_objects = _scene_objects(base_scene)
    current_objects = _scene_objects(current_scene)
    variants = rl_artifacts.get("domain_randomization", {}).get(
        "task_object_variants", []
    )

    for variant in variants:
        object_id = variant.get("object_id")
        bounds = variant.get("position_range_xy_on_surface") or []
        if not object_id or len(bounds) < 2:
            continue

        base_obj = base_objects.get(object_id)
        current_obj = current_objects.get(object_id)
        if base_obj is None or current_obj is None:
            return CheckResult(False, f"task object {object_id!r} missing from scene")

        base_pos = (base_obj.get("placement_info") or {}).get("position_2d")
        current_pos = (current_obj.get("placement_info") or {}).get("position_2d")
        if base_pos is None or current_pos is None:
            return CheckResult(
                False,
                f"task object {object_id!r}: missing placement_info.position_2d",
            )

        dx, dy = float(bounds[0]), float(bounds[1])
        if abs(float(current_pos[0]) - float(base_pos[0])) > dx:
            return CheckResult(
                False,
                f"task object {object_id!r} out of bounds on x: "
                f"actual={current_pos}, nominal={base_pos}, allowed=±{dx}",
            )
        if abs(float(current_pos[1]) - float(base_pos[1])) > dy:
            return CheckResult(
                False,
                f"task object {object_id!r} out of bounds on y: "
                f"actual={current_pos}, nominal={base_pos}, allowed=±{dy}",
            )

    return CheckResult(True, "all task objects within position bounds")


def check_manipuland_invariance(base_scene: dict, current_scene: dict) -> CheckResult:
    base_objects = _scene_objects(base_scene)
    current_objects = _scene_objects(current_scene)

    for object_id, base_obj in sorted(base_objects.items()):
        if base_obj.get("object_type") == "manipuland":
            continue
        current_obj = current_objects.get(object_id)
        if current_obj != base_obj:
            return CheckResult(
                False,
                f"non-manipuland {object_id!r} changed between base and current",
            )

    return CheckResult(True, "non-manipuland objects unchanged")


def _diversity_features(scene: dict) -> tuple[int, str, float, float]:
    objects = _scene_objects(scene)
    manip_ids = sorted(
        oid for oid, obj in objects.items() if obj.get("object_type") == "manipuland"
    )
    xs: list[float] = []
    ys: list[float] = []
    for oid in manip_ids:
        placement = objects[oid].get("placement_info") or {}
        pos = placement.get("position_2d") or [0.0, 0.0]
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
    mean_x = sum(xs) / len(xs) if xs else 0.0
    mean_y = sum(ys) / len(ys) if ys else 0.0
    digest = hashlib.sha256(",".join(manip_ids).encode()).hexdigest()
    return len(manip_ids), digest, mean_x, mean_y


def _diversity_distance(
    a: tuple[int, str, float, float], b: tuple[int, str, float, float]
) -> float:
    count_dist = abs(a[0] - b[0])
    hash_dist = 0.0 if a[1] == b[1] else 1.0
    mean_dist = abs(a[2] - b[2]) + abs(a[3] - b[3])
    return count_dist + hash_dist + mean_dist


def check_diversity_vs_siblings(
    current_scene: dict,
    sibling_snapshots: list[dict],
    *,
    eps: float = 0.05,
) -> CheckResult:
    if not sibling_snapshots:
        return CheckResult(True, "no siblings to compare")

    current_features = _diversity_features(current_scene)
    distances = [
        _diversity_distance(current_features, _diversity_features(sibling))
        for sibling in sibling_snapshots
    ]
    min_distance = min(distances)
    if min_distance <= eps:
        return CheckResult(
            False,
            f"insufficient diversity vs siblings (min_distance={min_distance:.4f}, eps={eps})",
        )
    return CheckResult(True, f"diverse vs siblings (min_distance={min_distance:.4f})")
