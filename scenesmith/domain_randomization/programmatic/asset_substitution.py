"""Asset substitution op for v1 domain randomization.

Two sources are supported via the ``source`` param:

* ``"catalog"`` (default, backward compatible): index ``asset_registry.json``
  files under ``catalog_dir`` and swap to any same-``name`` alternative.
  Inert against a single-scene catalog because each scene only ever
  generates one asset per semantic name, so this path is mostly useful
  with an aggregated multi-scene corpus.

* ``"hssd"``: query the HSSD retrieval server (started on demand) for
  alternatives matching each object's ``description`` and bbox
  dimensions, then swap the mesh path. Mirrors how
  ``AssetManager._retrieve_hssd_assets`` builds requests in the normal
  generation pipeline.

  Two flavors of the HSSD path:

  * ``regenerate_sdf=True`` runs the **full** swap path: GLB->GLTF +
    convex decomposition + fresh Drake SDF so downstream physics uses
    the new collision geometry. Requires ``blender_server`` and
    ``collision_client`` to be passed through ``params`` — the
    orchestrator's normal-generation pipeline already runs both
    servers and can hand the same instances to the op.

  * ``regenerate_sdf=False`` (the lite default) writes a per-object
    *visual-only* SDF copy whose ``<visual>`` block points at the new
    HSSD mesh with scale ``1 1 1`` (HSSD pre-scales to
    ``desired_dimensions``). Collision geometry keeps the **original**
    object's collision meshes — physics is approximate but renders
    pick up the swap. Cheap enough to run from the v1 deterministic
    orchestrator and the v2 ``swap_asset`` tool without needing
    blender / CoACD servers.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import random
import re
import shutil
import threading

from pathlib import Path
from typing import Any

from scenesmith.agent_utils.hssd_retrieval_server import (
    HssdRetrievalClient,
    HssdRetrievalServerRequest,
)
from scenesmith.agent_utils.room import ObjectType, RoomScene
from scenesmith.domain_randomization.programmatic.base import BaseRandomizationOp

console_logger = logging.getLogger(__name__)


# Process-wide singleton: one HSSD server per (host, port) regardless of
# how many AssetSubstitution invocations happen. Lifecycle:
#   - first hssd-mode apply() health-checks the address; if a server is
#     already running we reuse it and never own its lifecycle
#   - otherwise we start a new HssdRetrievalServer and register an atexit
#     hook to stop it on process exit (matches what standalone_server.py
#     does for orphaned-process protection on Linux)
_SERVER_LOCK = threading.Lock()
_SERVER_REGISTRY: dict[tuple[str, int], object] = {}


def _ensure_hssd_server(
    host: str,
    port: int,
    *,
    hssd_data_path: str | None,
    hssd_preprocessed_path: str | None,
    preload: bool = True,
    wait_timeout_s: float = 180.0,
) -> HssdRetrievalClient:
    """Return a healthy client. Start a server in-process if none is reachable."""
    client = HssdRetrievalClient(host=host, port=port)
    if client.health_check():
        console_logger.info(
            "AssetSubstitution: HSSD server already healthy at %s:%d", host, port
        )
        return client

    with _SERVER_LOCK:
        # Re-check inside the lock — another caller may have started one.
        if client.health_check():
            return client
        if (host, port) in _SERVER_REGISTRY:
            return client  # registered + still booting; client.retrieve_objects retries

        # Import lazily so the op module doesn't pull in CLIP / Flask at import time.
        from scenesmith.agent_utils.hssd_retrieval_server.server_manager import (
            HssdRetrievalServer,
        )

        console_logger.info(
            "AssetSubstitution: starting HSSD server on %s:%d "
            "(hssd_data_path=%s, hssd_preprocessed_path=%s, preload=%s)",
            host,
            port,
            hssd_data_path or os.environ.get("HSSD_DATA_PATH") or "data/hssd-models",
            hssd_preprocessed_path
            or os.environ.get("HSSD_PREPROCESSED_PATH")
            or "data/preprocessed",
            preload,
        )
        server = HssdRetrievalServer(
            host=host,
            port=port,
            preload_retriever=preload,
            hssd_data_path=hssd_data_path,
            hssd_preprocessed_path=hssd_preprocessed_path,
        )
        server.start()
        server.wait_until_ready(timeout_s=wait_timeout_s)
        _SERVER_REGISTRY[(host, port)] = server
        atexit.register(_stop_server, host, port)
    return client


def _stop_server(host: str, port: int) -> None:
    server = _SERVER_REGISTRY.pop((host, port), None)
    if server is None:
        return
    try:
        server.stop()
        console_logger.info(
            "AssetSubstitution: stopped HSSD server on %s:%d at process exit",
            host,
            port,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort during atexit
        console_logger.warning(
            "AssetSubstitution: failed to stop HSSD server on %s:%d: %s",
            host,
            port,
            exc,
        )


# Singletons for the GLB→GLTF (Blender) + CoACD servers needed when the
# HSSD swap path runs in regenerate_sdf=True mode. Same lifecycle pattern
# as _ensure_hssd_server: health-check first; lazy-start if absent; stop
# at process exit. Stored under sentinel keys because clients/servers
# aren't tied to a host:port the same way HSSD is — only one at a time.
_BLENDER_SINGLETON: dict[str, object] = {}
_COLLISION_SINGLETON: dict[str, object] = {}


def _ensure_blender_server(port_range: tuple[int, int] = (8200, 8250)) -> object:
    """Return a running BlenderServer; start one in-process if needed."""
    if "server" in _BLENDER_SINGLETON:
        return _BLENDER_SINGLETON["server"]
    with _SERVER_LOCK:
        if "server" in _BLENDER_SINGLETON:
            return _BLENDER_SINGLETON["server"]
        from scenesmith.agent_utils.blender.server_manager import BlenderServer

        console_logger.info(
            "AssetSubstitution: starting Blender server (port_range=%s) for SDF regen",
            port_range,
        )
        server = BlenderServer(port_range=port_range, server_startup_delay=0.1)
        server.start()
        server.wait_until_ready()
        _BLENDER_SINGLETON["server"] = server
        atexit.register(_stop_blender_singleton)
        return server


def _stop_blender_singleton() -> None:
    server = _BLENDER_SINGLETON.pop("server", None)
    if server is None:
        return
    try:
        server.stop()
    except Exception as exc:  # noqa: BLE001
        console_logger.warning(
            "AssetSubstitution: failed to stop Blender server at exit: %s", exc
        )


def _ensure_collision_client(
    host: str = "127.0.0.1", port: int = 7100, wait_timeout_s: float = 60.0
) -> object:
    """Return a CoACD client, starting the server in-process once if the singleton is not populated."""
    from scenesmith.agent_utils.convex_decomposition_server import (
        ConvexDecompositionClient,
    )

    client = ConvexDecompositionClient(host=host, port=port)
    # health_check isn't defined on the client — probe by attempting a quick
    # round-trip via the singleton check, and start if absent.
    if "client" in _COLLISION_SINGLETON:
        return _COLLISION_SINGLETON["client"]
    with _SERVER_LOCK:
        if "client" in _COLLISION_SINGLETON:
            return _COLLISION_SINGLETON["client"]
        try:
            from scenesmith.agent_utils.convex_decomposition_server import (
                ConvexDecompositionServer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Cannot lazy-start ConvexDecompositionServer "
                "(ConvexDecompositionServer not exported). Either start one "
                "externally on %s:%d, or thread a pre-built client in via "
                "params['collision_client']." % (host, port)
            ) from exc

        console_logger.info(
            "AssetSubstitution: starting ConvexDecomposition server on %s:%d",
            host,
            port,
        )
        server = ConvexDecompositionServer(host=host, port=port)
        server.start()
        # Give it time to bind before first request.
        import time as _time

        _time.sleep(min(wait_timeout_s, 5.0))
        _COLLISION_SINGLETON["server"] = server
        _COLLISION_SINGLETON["client"] = client
        atexit.register(_stop_collision_singleton)
        return client


def _stop_collision_singleton() -> None:
    server = _COLLISION_SINGLETON.pop("server", None)
    _COLLISION_SINGLETON.pop("client", None)
    if server is None:
        return
    try:
        server.stop()
    except Exception as exc:  # noqa: BLE001
        console_logger.warning(
            "AssetSubstitution: failed to stop CoACD server at exit: %s", exc
        )


class AssetSubstitution(BaseRandomizationOp):
    """Swap object geometry/SDF with an alternative from a catalog or HSSD.

    Params:
        source (str): ``"catalog"`` (default) or ``"hssd"``.
        target_object_types (list[str]): Object type strings eligible for swap.
            Default: ``["manipuland", "furniture"]``.
        seed (int | None): Random seed.

        Catalog mode:
            catalog_dir (str): Path to a generated_assets-style directory.
                ``asset_registry.json`` files under it are indexed by ``name``.

        HSSD mode (requires ``source="hssd"``):
            hssd_host (str): Server host. Default ``127.0.0.1``.
            hssd_port (int): Server port. Default ``7001``.
            hssd_data_path (str | None): Override for ``HSSD_DATA_PATH`` env.
            hssd_preprocessed_path (str | None): Override for ``HSSD_PREPROCESSED_PATH``.
            output_dir (str): Where the retrieval server writes exported meshes.
                Defaults to ``<scene.scene_dir>/hssd_swaps``.
            num_candidates (int): How many candidates to request per object.
                The first non-matching candidate is selected so we get a real
                swap, not the same asset back. Default 3.
            wait_timeout_s (float): How long to wait for a freshly-started
                server to become ready. Default 180 s (CLIP preload is slow).
    """

    def apply(self, scene: RoomScene, params: dict, locked_ids: set[str]) -> RoomScene:
        source = str(params.get("source", "catalog")).lower()
        if source == "hssd":
            return self._apply_hssd(scene, params, locked_ids)
        return self._apply_catalog(scene, params, locked_ids)

    # ------------------------------------------------------------------
    # Backward-compatible catalog mode.
    # ------------------------------------------------------------------

    def _apply_catalog(
        self, scene: RoomScene, params: dict, locked_ids: set[str]
    ) -> RoomScene:
        catalog_dir = Path(params["catalog_dir"])
        target_types = _resolve_types(
            params.get("target_object_types", ["manipuland", "furniture"])
        )
        seed = params.get("seed")
        rng = random.Random(seed)

        target_object_ids = params.get("target_object_ids")
        if target_object_ids is not None:
            target_object_ids = {str(oid) for oid in target_object_ids}

        catalog = _build_catalog(catalog_dir)
        if not catalog:
            console_logger.warning(
                f"AssetSubstitution: empty catalog from {catalog_dir}"
            )
            return scene

        for obj in scene.objects.values():
            if obj.object_id in locked_ids:
                continue
            if target_types and obj.object_type not in target_types:
                continue
            if (
                target_object_ids is not None
                and str(obj.object_id) not in target_object_ids
            ):
                continue

            alternatives = [
                entry
                for entry in catalog.get(obj.name, [])
                if entry["geometry_path"] != obj.geometry_path
            ]
            if not alternatives:
                continue

            chosen = rng.choice(alternatives)
            # asset_registry.json stores paths as strings; SceneObject
            # expects Path so JSON serialization round-trips correctly.
            obj.geometry_path = Path(chosen["geometry_path"])
            obj.sdf_path = Path(chosen["sdf_path"])
            console_logger.debug(
                f"AssetSubstitution: {obj.object_id} ({obj.name}) → "
                f"{Path(chosen['sdf_path']).parent.name}"
            )

        return scene

    # ------------------------------------------------------------------
    # HSSD mode — query the retrieval server for alternatives.
    # ------------------------------------------------------------------

    def _apply_hssd(
        self, scene: RoomScene, params: dict, locked_ids: set[str]
    ) -> RoomScene:
        target_types = _resolve_types(
            params.get("target_object_types", ["manipuland", "furniture"])
        )
        # Optional per-id filter — when set, only these ids are eligible
        # even if they pass the type filter. Used by DRSubflowAgent's
        # per-object swap_asset tool so an LLM can target a single object.
        target_object_ids = params.get("target_object_ids")
        if target_object_ids is not None:
            target_object_ids = {str(oid) for oid in target_object_ids}
        seed = params.get("seed")
        rng = random.Random(seed)
        host = str(params.get("hssd_host", "127.0.0.1"))
        port = int(params.get("hssd_port", 7001))
        num_candidates = int(params.get("num_candidates", 3))
        output_dir = Path(params.get("output_dir") or (scene.scene_dir / "hssd_swaps"))
        output_dir.mkdir(parents=True, exist_ok=True)
        client = _ensure_hssd_server(
            host=host,
            port=port,
            hssd_data_path=params.get("hssd_data_path"),
            hssd_preprocessed_path=params.get("hssd_preprocessed_path"),
            wait_timeout_s=float(params.get("wait_timeout_s", 180.0)),
        )

        # Build the request list mirroring the main pipeline's shape
        # (asset_manager._retrieve_hssd_assets:533).
        eligible: list[tuple[str, Any]] = []
        retrieval_requests: list[HssdRetrievalServerRequest] = []
        for obj in scene.objects.values():
            if obj.object_id in locked_ids:
                continue
            if target_types and obj.object_type not in target_types:
                continue
            if (
                target_object_ids is not None
                and str(obj.object_id) not in target_object_ids
            ):
                continue
            description = (obj.description or obj.name or "").strip()
            if not description:
                console_logger.debug(
                    "AssetSubstitution: skipping %s (no description / name)",
                    obj.object_id,
                )
                continue
            dims = (
                float(obj.bbox_max[0] - obj.bbox_min[0]),
                float(obj.bbox_max[1] - obj.bbox_min[1]),
                float(obj.bbox_max[2] - obj.bbox_min[2]),
            )
            out = output_dir / str(obj.object_id)
            out.mkdir(parents=True, exist_ok=True)
            retrieval_requests.append(
                HssdRetrievalServerRequest(
                    object_description=description,
                    object_type=obj.object_type.value.upper(),
                    output_dir=str(out),
                    desired_dimensions=dims,
                    scene_id=str(scene.room_id),
                    num_candidates=num_candidates,
                )
            )
            eligible.append((obj.object_id, obj))

        if not retrieval_requests:
            console_logger.info(
                "AssetSubstitution(hssd): no eligible objects (all locked / no description)"
            )
            return scene

        console_logger.info(
            "AssetSubstitution(hssd): retrieving %d alternatives from %s:%d",
            len(retrieval_requests),
            host,
            port,
        )
        # Drain the stream; map each index back to its (id, obj) and pick
        # the best alternative whose mesh path differs from the current one.
        results_by_index: dict[int, list] = {}
        for index, response in client.retrieve_objects(
            retrieval_requests, timeout_s=int(params.get("request_timeout_s", 600))
        ):
            results_by_index[index] = response.results

        # Full-pipeline knobs — `regenerate_sdf=True` rebuilds the GLTF +
        # CoACD collision so downstream physics matches the new visual.
        # When the caller doesn't thread blender_server / collision_client
        # through, we lazy-start them in-process (same atexit-cleanup
        # pattern as the HSSD server). The lite fallback (regenerate_sdf=False)
        # path stays available via _rewrite_sdf_visual below.
        regenerate_sdf = bool(params.get("regenerate_sdf", False))
        blender_server = params.get("blender_server")
        collision_client = params.get("collision_client")
        if regenerate_sdf:
            if blender_server is None:
                try:
                    blender_server = _ensure_blender_server()
                except Exception as exc:  # noqa: BLE001
                    console_logger.warning(
                        "AssetSubstitution(hssd): lazy-start of Blender server "
                        "failed (%s); falling back to lite visual-only swap.",
                        exc,
                    )
                    regenerate_sdf = False
            if regenerate_sdf and collision_client is None:
                try:
                    collision_client = _ensure_collision_client()
                except Exception as exc:  # noqa: BLE001
                    console_logger.warning(
                        "AssetSubstitution(hssd): lazy-start of CoACD server "
                        "failed (%s); falling back to lite visual-only swap.",
                        exc,
                    )
                    regenerate_sdf = False

        swapped = 0
        for index, (oid, obj) in enumerate(eligible):
            results = results_by_index.get(index, [])
            if not results:
                console_logger.warning(
                    "AssetSubstitution(hssd): no results for %s", oid
                )
                continue
            current_geom = str(obj.geometry_path or "")
            alternatives = [r for r in results if str(r.mesh_path) != current_geom]
            if not alternatives:
                console_logger.debug(
                    "AssetSubstitution(hssd): %s — no alternative differs from "
                    "current geometry; skipping",
                    oid,
                )
                continue
            chosen = rng.choice(alternatives)
            old_geom = obj.geometry_path
            old_sdf = obj.sdf_path

            # SceneObject.to_dict calls safe_relative_path(obj.geometry_path,
            # scene_dir) which requires a Path. Assigning str() here breaks
            # subsequent scene_state.json serialization with AttributeError.
            # Both branches below must hand back Path objects.
            if regenerate_sdf:
                try:
                    new_geom, new_sdf = _regenerate_full_asset(
                        obj=obj,
                        chosen_mesh_path=Path(chosen.mesh_path),
                        out_dir=output_dir / str(oid),
                        blender_server=blender_server,
                        collision_client=collision_client,
                        material=str(params.get("default_material", "plastic")),
                        mass_kg=float(params.get("default_mass_kg", 0.5)),
                        mass_range_kg=tuple(
                            params.get("default_mass_range_kg", (0.05, 5.0))
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    console_logger.error(
                        "AssetSubstitution(hssd): SDF regen FAILED for %s: %s",
                        oid,
                        exc,
                        exc_info=True,
                    )
                    continue
                obj.geometry_path = Path(new_geom)
                obj.sdf_path = Path(new_sdf)
            else:
                # Lite mode: write a per-object SDF copy with the visual
                # mesh URI rewritten to the new HSSD asset (so renders
                # actually pick up the swap). Collision URIs are left
                # pointing at the original meshes — physics stays
                # approximate but visuals are correct.
                try:
                    new_sdf = _rewrite_sdf_visual(
                        orig_sdf_path=Path(obj.sdf_path) if obj.sdf_path else None,
                        new_mesh_path=Path(chosen.mesh_path),
                        out_dir=output_dir / str(oid),
                        object_id=str(oid),
                    )
                except Exception as exc:  # noqa: BLE001
                    console_logger.error(
                        "AssetSubstitution(hssd, lite): SDF rewrite FAILED for %s: %s",
                        oid,
                        exc,
                        exc_info=True,
                    )
                    continue
                obj.geometry_path = Path(chosen.mesh_path)
                if new_sdf is not None:
                    obj.sdf_path = Path(new_sdf)

            console_logger.info(
                "AssetSubstitution(hssd): %s (%s) %s → %s [hssd_id=%s, similarity=%.3f, "
                "sdf=%s]",
                oid,
                obj.name,
                Path(old_geom or "").name,
                Path(obj.geometry_path).name,
                chosen.hssd_id,
                chosen.similarity_score,
                (
                    "regenerated"
                    if regenerate_sdf
                    else f"kept ({Path(old_sdf or '').name})"
                ),
            )
            swapped += 1

        console_logger.info(
            "AssetSubstitution(hssd): swapped %d / %d eligible objects "
            "(regenerate_sdf=%s)",
            swapped,
            len(eligible),
            regenerate_sdf,
        )
        return scene

    # ------------------------------------------------------------------

    def validate(self, scene: RoomScene) -> list[str]:
        violations = []
        for obj in scene.objects.values():
            if obj.geometry_path and not Path(obj.geometry_path).exists():
                violations.append(
                    f"AssetSubstitution: geometry path missing for "
                    f"{obj.object_id}: {obj.geometry_path}"
                )
            if obj.sdf_path and not Path(obj.sdf_path).exists():
                violations.append(
                    f"AssetSubstitution: SDF path missing for "
                    f"{obj.object_id}: {obj.sdf_path}"
                )
        return violations


def _build_catalog(catalog_dir: Path) -> dict[str, list[dict]]:
    """Read asset_registry.json files and index by object name."""
    catalog: dict[str, list[dict]] = {}
    for registry_file in catalog_dir.rglob("asset_registry.json"):
        try:
            entries = json.loads(registry_file.read_text())
            if not isinstance(entries, list):
                entries = [entries]
            for entry in entries:
                name = entry.get("name", "")
                if name:
                    catalog.setdefault(name, []).append(entry)
        except Exception as e:
            console_logger.debug(f"AssetSubstitution: skipping {registry_file}: {e}")
    return catalog


def _resolve_types(type_strings: list[str]) -> set[ObjectType]:
    result = set()
    for s in type_strings:
        try:
            result.add(ObjectType(s))
        except ValueError:
            pass
    return result


# Match a <visual ...> ... </visual> block (non-greedy). Captures the
# whole element so we can rewrite only the URI + scale inside it without
# touching <collision> blocks or inertial / surface metadata.
_VISUAL_BLOCK_RE = re.compile(r"(<visual\b[^>]*>)(.*?)(</visual>)", re.DOTALL)
_URI_RE = re.compile(r"<uri>\s*([^<]+?)\s*</uri>")
_SCALE_RE = re.compile(r"<scale>\s*([^<]+?)\s*</scale>")


def _rewrite_sdf_visual(
    *,
    orig_sdf_path: Path | None,
    new_mesh_path: Path,
    out_dir: Path,
    object_id: str,
) -> Path | None:
    """Write a per-object SDF copy whose visual mesh references new_mesh_path.

    The original SDF is copied to ``out_dir / f"{object_id}.sdf"``, then
    inside its single ``<visual>`` block the first ``<uri>`` is
    rewritten to ``file://{abs new_mesh_path}`` and the first
    ``<scale>`` is reset to ``1 1 1`` (HSSD assets are pre-scaled to
    ``desired_dimensions``, so the original SDF's bbox-derived scale
    no longer applies).

    Collision blocks are intentionally left untouched: the renderer
    only reads visual geometry, and rewriting collision would require
    a fresh CoACD pass (which is what ``regenerate_sdf=True`` does).

    Returns the new SDF path. Returns ``None`` (no-op) if the original
    SDF doesn't exist or contains no ``<visual>`` block.
    """
    if orig_sdf_path is None or not orig_sdf_path.exists():
        console_logger.warning(
            "_rewrite_sdf_visual: %s has no readable sdf_path (%s); leaving "
            "obj.sdf_path unchanged — render will use the old SDF.",
            object_id,
            orig_sdf_path,
        )
        return None

    text = orig_sdf_path.read_text()
    new_uri = f"file://{new_mesh_path.resolve()}"

    def _rewrite_visual(match: re.Match) -> str:
        open_tag, body, close_tag = match.group(1), match.group(2), match.group(3)
        # Replace only the FIRST <uri> + <scale> inside the visual body.
        body_new, n_uri = _URI_RE.subn(f"<uri>{new_uri}</uri>", body, count=1)
        body_new, n_scale = _SCALE_RE.subn(
            "<scale>1.00000000 1.00000000 1.00000000</scale>", body_new, count=1
        )
        if n_uri == 0:
            console_logger.warning(
                "_rewrite_sdf_visual: %s — no <uri> inside <visual>, leaving as-is",
                object_id,
            )
        return f"{open_tag}{body_new}{close_tag}"

    new_text, n_blocks = _VISUAL_BLOCK_RE.subn(_rewrite_visual, text, count=1)
    if n_blocks == 0:
        console_logger.warning(
            "_rewrite_sdf_visual: %s — no <visual> block found in %s",
            object_id,
            orig_sdf_path,
        )
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    new_sdf_path = out_dir / f"{object_id}.sdf"
    new_sdf_path.write_text(new_text)

    # Mirror the collision meshes into the new SDF directory so their
    # original relative <uri>X_collision_N.obj</uri> references still
    # resolve (the SDF moved; the obj files would otherwise be lost).
    orig_dir = orig_sdf_path.parent
    for collision_match in _URI_RE.finditer(new_text):
        rel = collision_match.group(1)
        if rel.startswith(("file://", "http://", "https://", "/")):
            continue  # already absolute
        src = orig_dir / rel
        dst = out_dir / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return new_sdf_path


def _regenerate_full_asset(
    *,
    obj: Any,
    chosen_mesh_path: Path,
    out_dir: Path,
    blender_server: Any,
    collision_client: Any,
    material: str,
    mass_kg: float,
    mass_range_kg: tuple[float, float],
) -> tuple[Path, Path]:
    """Run the post-HSSD pipeline chunk: GLB->GLTF + convex decomp + SDF write.

    Mirrors the relevant subset of
    ``AssetManager._retrieve_hssd_assets`` (asset_manager.py:660+). The
    VLM physics analysis step is skipped — DR substitutions reuse a
    sensible default for material / mass so we don't pay an LLM call per
    swap. The mesh canonicalization step is also skipped because HSSD
    assets are already in the canonical Y-up orientation TaskSmith
    expects.

    Args:
        obj: SceneObject being swapped (used only for naming the output files).
        chosen_mesh_path: GLB or GLTF path returned by the HSSD server.
        out_dir: Per-object directory under the swap output_dir; final
            GLTF and SDF land here.
        blender_server: Running blender server (for GLB->GLTF conversion).
        collision_client: Running convex-decomposition client (for SDF
            collision geometry).
        material: Material label fed to ``generate_drake_sdf`` (selects
            friction coefficient).
        mass_kg: Object mass used in the SDF inertial block.
        mass_range_kg: Mass uncertainty range (kept in metadata; not used
            for the rigid-body inertia computation).

    Returns:
        ``(final_gltf_path, sdf_path)`` — both written to disk.
    """
    # Local imports keep the op module light when full-pipeline mode is unused.
    import trimesh

    from scenesmith.agent_utils.mesh_physics_analyzer import MeshPhysicsAnalysis
    from scenesmith.agent_utils.sdf_generator import generate_drake_sdf

    out_dir.mkdir(parents=True, exist_ok=True)
    asset_name = str(obj.object_id)

    # 1. GLB -> GLTF (Drake requires separate-textures GLTF for visuals).
    if chosen_mesh_path.suffix.lower() == ".glb":
        final_gltf_path = out_dir / f"{asset_name}.gltf"
        blender_server.convert_glb_to_gltf(
            input_path=chosen_mesh_path,
            output_path=final_gltf_path,
            export_yup=True,
        )
    else:
        # Already GLTF — copy into the per-object out_dir for tidy artifact layout.
        import shutil

        final_gltf_path = out_dir / f"{asset_name}.gltf"
        shutil.copy(chosen_mesh_path, final_gltf_path)

    # 2. Convex decomposition (CoACD with default-ish params — DR doesn't
    #    care about pixel-perfect collisions, we want a usable shape fast).
    collision_pieces = collision_client.generate_collision_geometry(
        mesh_path=final_gltf_path,
        method="coacd",
        threshold=0.05,
        max_convex_hull=-1,
        preprocess_mode="auto",
        preprocess_resolution=50,
        resolution=2000,
        mcts_nodes=20,
        mcts_iterations=150,
        mcts_max_depth=3,
        pca=False,
        merge=True,
        decimate=False,
        max_ch_vertex=256,
        extrude=False,
        extrude_margin=0.01,
        apx_mode="ch",
        seed=0,
    )

    # 3. Drake SDF with sane defaults — reuses the original mesh's bbox via
    #    trimesh re-load so the inertia block is consistent with the new visual.
    physics = MeshPhysicsAnalysis(
        up_axis="+Y",
        front_axis="+Z",
        material=material,
        mass_kg=mass_kg,
        mass_range_kg=mass_range_kg,
    )
    sdf_path = out_dir / f"{asset_name}.sdf"
    generate_drake_sdf(
        visual_mesh_path=final_gltf_path,
        collision_pieces=collision_pieces,
        physics_analysis=physics,
        output_path=sdf_path,
        asset_name=asset_name,
    )
    # Validate the mesh loaded correctly — exposes broken meshes before they reach the simulator.
    trimesh.load_mesh(final_gltf_path, force="mesh")
    return final_gltf_path, sdf_path
