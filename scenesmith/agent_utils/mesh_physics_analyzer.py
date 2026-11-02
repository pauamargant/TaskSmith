"""Mesh physics analysis using VLM for orientation, material, and mass prediction.

This module provides VLM-based analysis of 3D meshes to determine:
- Canonical orientation (up axis, front view)
- Material type (wood, metal, plastic, etc.)
- Mass estimation with confidence range
"""

import logging
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field

from scenesmith.agent_utils.vision_resolution import get_validation_render_dimensions
from scenesmith.prompts import MeshPhysicsPrompts, prompt_manager
from scenesmith.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenesmith.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


@dataclass
class MeshPhysicsAnalysis:
    """Results from VLM-based mesh physics analysis.

    All axes are in Blender coordinates (X-right, Y-forward, Z-up) since VLM
    analyzes Blender-rendered images.
    """

    up_axis: str
    """Up axis in Blender coordinates (e.g., "+Z", "-Y")."""

    front_axis: str
    """Front axis in Blender coordinates (e.g., "+Y", "+X")."""

    material: str
    """Material type (e.g., "wood", "metal", "plastic")."""

    mass_kg: float
    """Predicted mass in kilograms."""

    mass_range_kg: tuple[float, float]
    """Confidence range for mass (min, max) in kilograms."""


class _CanonicalOrientationSchema(BaseModel):
    """Sub-schema for the orientation block of the VLM response."""

    model_config = ConfigDict(extra="ignore")

    up_axis: str
    front_view_image_index: int
    # Visible reasoning slot: the VLM writes a 1-sentence rationale for why
    # the chosen image shows the front face. Surfaced in logs for auditing.
    front_view_reasoning: str = ""


class MeshPhysicsAnalysisSchema(BaseModel):
    """Agents-SDK structured output schema for mesh physics analysis.

    Mirrors the JSON shape the VLM prompt asks for. ``extra="ignore"`` keeps
    the schema permissive — the model can emit extra fields without rejecting
    the response.
    """

    model_config = ConfigDict(extra="ignore")

    material: str
    mass_kg: float
    mass_range_kg: list[float] = Field(min_length=2, max_length=2)
    # mass_source: one of "image_text" / "estimated" / "metadata". Optional
    # so a model that omits it doesn't fail validation.
    mass_source: str = "estimated"
    canonical_orientation: _CanonicalOrientationSchema


def get_view_direction_from_image_number(
    image_number: int,
    num_side_views: int = 4,
    include_diagonal_views: bool = False,
    include_vertical_views: bool = True,
) -> str:
    """Map an image number to its corresponding view direction and axis.

    Args:
        image_number: The image number (0-based index).
        num_side_views: Number of side views rendered.
        include_diagonal_views: Whether diagonal views are included.
        include_vertical_views: Whether top/bottom views are included. If False,
            image numbering starts at 0 for side views.

    Returns:
        String representing the view axis: "x", "-y", "z", "top", "bottom",
        "top_diagonal", "bottom_diagonal"
    """
    vertical_offset = 2 if include_vertical_views else 0

    if include_vertical_views and image_number == 0:
        return "top"
    elif include_vertical_views and image_number == 1:
        return "bottom"
    elif vertical_offset <= image_number < vertical_offset + num_side_views:
        # Side views are rendered in a circle around the XY plane.
        side_index = image_number - vertical_offset
        angle = 2 * np.pi * side_index / num_side_views

        # Map to closest primary axis.
        if angle <= np.pi / 4 or angle > 7 * np.pi / 4:
            return "x"
        elif np.pi / 4 < angle <= 3 * np.pi / 4:
            return "y"
        elif 3 * np.pi / 4 < angle <= 5 * np.pi / 4:
            return "-x"
        else:  # 5*np.pi/4 < angle <= 7*np.pi/4
            return "-y"
    elif include_diagonal_views and image_number == vertical_offset + num_side_views:
        return "top_diagonal"
    elif (
        include_diagonal_views and image_number == vertical_offset + num_side_views + 1
    ):
        return "bottom_diagonal"
    else:
        max_expected = (
            vertical_offset + num_side_views + (2 if include_diagonal_views else 0) - 1
        )
        raise ValueError(
            f"Invalid image number {image_number}. Expected 0 to {max_expected}"
        )


def get_front_axis_from_image_number(
    image_number: int,
    num_side_views: int = 4,
    include_diagonal_views: bool = False,
    include_vertical_views: bool = True,
) -> str:
    """Get the coordinate axis that corresponds to the front direction.

    This maps the image number that the VLM identified as showing the front view
    to the corresponding front-facing axis in Blender coordinates.

    Args:
        image_number: The image number that the VLM identified as the front view.
        num_side_views: Number of side views rendered.
        include_diagonal_views: Whether diagonal views are included.
        include_vertical_views: Whether top/bottom views are included.

    Returns:
        String representing the front axis: "x", "-x", "y", "-y", "z", or "-z"
    """
    view_axis = get_view_direction_from_image_number(
        image_number=image_number,
        num_side_views=num_side_views,
        include_diagonal_views=include_diagonal_views,
        include_vertical_views=include_vertical_views,
    )

    # Map view direction to front axis.
    if view_axis == "top":
        return "z"
    elif view_axis == "bottom":
        return "-z"
    elif view_axis == "top_diagonal":
        return "z"  # Diagonal from top, still primarily z-direction.
    elif view_axis == "bottom_diagonal":
        return "-z"  # Diagonal from bottom, still primarily -z-direction.
    return view_axis


def render_and_encode_multiview_images_for_mesh_physics(
    *,
    mesh_path: Path,
    cfg: DictConfig,
    elevation_degrees: float,
    blender_server: "BlenderServer",
    num_side_views: int = 4,
    debug_output_dir: Path | None = None,
    include_vertical_views: bool = True,
) -> list[str]:
    """Render multi-view images for mesh physics analysis and return base64 strings.

    This helper must be called sequentially because BlenderServer is single-process.
    """
    # Determine output directory for rendered images.
    if debug_output_dir is not None:
        output_dir = debug_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="mesh_analysis_")
        output_dir = Path(temp_dir)

    start_time = time.time()
    console_logger.info(f"Rendering mesh physics views for {mesh_path}")

    try:
        dims = get_validation_render_dimensions(
            cfg.asset_manager,
            quality_preset=getattr(cfg, "image_quality_preset", None),
        )
        image_paths = blender_server.render_multiview_for_analysis(
            mesh_path=mesh_path,
            output_dir=output_dir,
            elevation_degrees=elevation_degrees,
            num_side_views=num_side_views,
            include_vertical_views=include_vertical_views,
            width=dims.width,
            height=dims.height,
            taa_samples=cfg.asset_manager.validation_taa_samples,
        )
        console_logger.info(
            f"Rendered {len(image_paths)} multi-view images to {output_dir}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to render mesh views: {e}") from e

    console_logger.debug(f"Encoding {len(image_paths)} images for VLM")
    encoded_images = [encode_image_to_base64(img) for img in image_paths]
    console_logger.info(
        f"Render+encode complete in {time.time() - start_time:.2f}s for {mesh_path.stem}"
    )
    return encoded_images


def analyze_mesh_physics_from_encoded_images(
    *,
    encoded_images: list[str],
    mesh_stem: str,
    cfg: DictConfig,
    num_side_views: int = 4,
    prompt_type: Literal["generated", "hssd"] = "generated",
    include_vertical_views: bool = True,
) -> MeshPhysicsAnalysis:
    """Run VLM physics analysis given already-encoded multiview images."""
    start_time = time.time()

    prompt_enum = (
        MeshPhysicsPrompts.GENERATED
        if prompt_type == "generated"
        else MeshPhysicsPrompts.HSSD
    )
    system_prompt = prompt_manager.get_prompt(prompt_name=prompt_enum)
    user_text = (
        "Please analyze these multi-view renders of the object "
        f"(asset name: {mesh_stem}) and provide the physical properties "
        "analysis in the specified JSON format."
    )

    console_logger.debug("Calling Agents SDK for physics analysis")
    try:
        openai_config = cfg.openai
        model = openai_config.model
        reasoning_effort = openai_config.reasoning_effort.mesh_analysis
        verbosity = openai_config.verbosity.mesh_analysis

        # Migrated to the Agents-SDK structured-output path (used by
        # asset_router, scene_analyzer, articulated_physics_analyzer) so
        # mesh physics inherits the same timeout, retry, and reasoning
        # handling as the rest of the pipeline.
        from scenesmith.agent_utils.sdk_vlm_runner import run_structured_vlm_call

        schema = run_structured_vlm_call(
            model=model,
            instructions=system_prompt,
            user_text=user_text,
            images_b64=encoded_images,
            output_type=MeshPhysicsAnalysisSchema,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            subsystem="physics_analysis",
            agent_name="mesh_physics_oneshot",
        )
        response_json = schema.model_dump()
        console_logger.info(f"Mesh analysis response ({mesh_stem}): {response_json}")

    except Exception as e:
        raise RuntimeError(f"VLM analysis failed: {e}") from e

    try:
        material = response_json["material"]
        mass_kg = float(response_json["mass_kg"])
        mass_range_kg = tuple(response_json["mass_range_kg"])

        orientation = response_json["canonical_orientation"]
        up_axis_raw = orientation["up_axis"]
        front_view_idx = int(orientation["front_view_image_index"])

        if up_axis_raw.startswith("-"):
            up_axis = up_axis_raw.upper()
        else:
            up_axis = f"+{up_axis_raw.upper()}"

        if prompt_type == "hssd" and up_axis != "+Z":
            raise ValueError(
                f"HSSD object must have up_axis=z (got {up_axis_raw}). "
                "HSSD objects are already canonically upright."
            )

        front_axis_raw = get_front_axis_from_image_number(
            image_number=front_view_idx,
            num_side_views=num_side_views,
            include_diagonal_views=False,
            include_vertical_views=include_vertical_views,
        )

        if front_axis_raw.startswith("-"):
            front_axis = front_axis_raw.upper()
        else:
            front_axis = f"+{front_axis_raw.upper()}"

        console_logger.info(
            f"VLM analysis complete: material={material}, mass={mass_kg}kg, "
            f"up={up_axis}, front={front_axis}"
        )
        console_logger.info(
            f"VLM analysis complete in {time.time() - start_time:.2f} seconds"
        )

        return MeshPhysicsAnalysis(
            up_axis=up_axis,
            front_axis=front_axis,
            material=material,
            mass_kg=mass_kg,
            mass_range_kg=mass_range_kg,
        )

    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse VLM response: {e}") from e


def analyze_mesh_orientation_and_material(
    mesh_path: Path,
    cfg: DictConfig,
    elevation_degrees: float,
    blender_server: "BlenderServer",
    num_side_views: int = 4,
    debug_output_dir: Path | None = None,
    prompt_type: Literal["generated", "hssd"] = "generated",
    include_vertical_views: bool = True,
) -> MeshPhysicsAnalysis:
    """Analyze mesh orientation and material properties using VLM.

    This function:
    1. Renders multi-view images of the mesh using BlenderServer
    2. Sends images to VLM for analysis
    3. Parses structured response for orientation, material, and mass
    4. Saves rendered images to debug directory or temporary directory

    Args:
        mesh_path: Path to the mesh file to analyze.
        cfg: Configuration with OpenAI model settings.
        elevation_degrees: Elevation angle in degrees for side view cameras.
        blender_server: BlenderServer instance for rendering. REQUIRED - forked
            workers cannot safely use embedded bpy due to GPU/OpenGL state
            corruption from fork.
        num_side_views: Number of equidistant side views to render (default: 4).
        debug_output_dir: Optional directory to save rendered images for debugging.
            If None, a temporary directory will be used.
        prompt_type: Type of mesh physics prompt to use. "generated" for full 3D
            orientation analysis, "hssd" for Z-axis rotation only (assumes upright).
        include_vertical_views: If True, render top/bottom views. If False, only
            render side views (constrains rotation to Z-axis). Typically False for
            HSSD assets.

    Returns:
        MeshPhysicsAnalysis with orientation, material, and mass predictions.

    Raises:
        RuntimeError: If rendering or VLM analysis fails.
        ValueError: If VLM response cannot be parsed or if HSSD constraint is
            violated (up_axis != "z").
    """
    start_time = time.time()

    console_logger.info(f"Analyzing mesh physics for {mesh_path}")

    # Determine output directory for rendered images.
    if debug_output_dir is not None:
        output_dir = debug_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Create temporary directory for images.
        temp_dir = tempfile.mkdtemp(prefix="mesh_analysis_")
        output_dir = Path(temp_dir)

    # Render multi-view images using BlenderServer (required for fork safety).
    num_vertical_views = 2 if include_vertical_views else 0
    total_views = num_side_views + num_vertical_views
    view_description = (
        "(top, bottom, + side views)" if include_vertical_views else "(side views only)"
    )
    console_logger.debug(
        f"Rendering {total_views} views for VLM analysis {view_description}"
    )
    try:
        dims = get_validation_render_dimensions(
            cfg.asset_manager,
            quality_preset=getattr(cfg, "image_quality_preset", None),
        )
        # Render via BlenderServer HTTP endpoint.
        image_paths = blender_server.render_multiview_for_analysis(
            mesh_path=mesh_path,
            output_dir=output_dir,
            elevation_degrees=elevation_degrees,
            num_side_views=num_side_views,
            include_vertical_views=include_vertical_views,
            width=dims.width,
            height=dims.height,
            taa_samples=cfg.asset_manager.validation_taa_samples,
        )
        console_logger.info(
            f"Rendered {len(image_paths)} multi-view images to {output_dir}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to render mesh views: {e}") from e

    # Load appropriate prompt based on asset type.
    prompt_enum = (
        MeshPhysicsPrompts.GENERATED
        if prompt_type == "generated"
        else MeshPhysicsPrompts.HSSD
    )
    system_prompt = prompt_manager.get_prompt(prompt_name=prompt_enum)

    # Encode images to base64 for VLM.
    console_logger.debug(f"Encoding {len(image_paths)} images for VLM")
    encoded_images = [encode_image_to_base64(img) for img in image_paths]

    user_text = (
        f"Please analyze these multi-view renders of the object "
        f"(asset name: {mesh_path.stem}) and provide the physical properties "
        f"analysis in the specified JSON format."
    )

    console_logger.debug("Calling Agents SDK for physics analysis")
    try:
        openai_config = cfg.openai
        model = openai_config.model
        reasoning_effort = openai_config.reasoning_effort.mesh_analysis
        verbosity = openai_config.verbosity.mesh_analysis

        # Migrated to the Agents-SDK structured-output path; see sibling
        # function ``analyze_mesh_physics_from_encoded_images``.
        from scenesmith.agent_utils.sdk_vlm_runner import run_structured_vlm_call

        schema = run_structured_vlm_call(
            model=model,
            instructions=system_prompt,
            user_text=user_text,
            images_b64=encoded_images,
            output_type=MeshPhysicsAnalysisSchema,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            subsystem="physics_analysis",
            agent_name="mesh_physics_oneshot",
        )
        response_json = schema.model_dump()
        console_logger.info(
            f"Mesh analysis response ({mesh_path.stem}): {response_json}"
        )

    except Exception as e:
        raise RuntimeError(f"VLM analysis failed: {e}") from e

    # Parse VLM response.
    try:
        material = response_json["material"]
        mass_kg = float(response_json["mass_kg"])
        mass_range_kg = tuple(response_json["mass_range_kg"])

        orientation = response_json["canonical_orientation"]
        up_axis_raw = orientation["up_axis"]
        front_view_idx = int(orientation["front_view_image_index"])

        # Convert axis strings from VLM format to our format.
        if up_axis_raw.startswith("-"):
            up_axis = up_axis_raw.upper()
        else:
            up_axis = f"+{up_axis_raw.upper()}"

        # Validate HSSD constraint: up_axis must be "Z" for HSSD assets.
        if prompt_type == "hssd" and up_axis != "+Z":
            raise ValueError(
                f"HSSD object must have up_axis='z' (got '{up_axis_raw}'). "
                f"HSSD objects are already canonically upright."
            )

        # Map front view image index to axis.
        front_axis_raw = get_front_axis_from_image_number(
            image_number=front_view_idx,
            num_side_views=num_side_views,
            include_diagonal_views=False,
            include_vertical_views=include_vertical_views,
        )

        # Convert to uppercase format.
        if front_axis_raw.startswith("-"):
            front_axis = front_axis_raw.upper()
        else:
            front_axis = f"+{front_axis_raw.upper()}"

        console_logger.info(
            f"VLM analysis complete: material={material}, mass={mass_kg}kg, "
            f"up={up_axis}, front={front_axis}"
        )

        console_logger.info(
            f"VLM analysis complete in {time.time() - start_time:.2f} seconds"
        )

        return MeshPhysicsAnalysis(
            up_axis=up_axis,
            front_axis=front_axis,
            material=material,
            mass_kg=mass_kg,
            mass_range_kg=mass_range_kg,
        )

    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse VLM response: {e}") from e
