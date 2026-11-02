"""Resolution preset helpers for vision renders.

This module centralizes the mapping between human-friendly preset names and
pixel dimensions for:
- room observation renders (top + side views)
- per-object validation/physics renders (multiview)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

ResolutionPreset = Literal["low", "medium", "high"]

_PRESET_TO_PX: dict[ResolutionPreset, int] = {"low": 256, "medium": 512, "high": 1024}


def preset_to_px(preset: ResolutionPreset) -> int:
    return _PRESET_TO_PX[preset]


def _normalize_preset(value: Any, *, default: ResolutionPreset) -> ResolutionPreset:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _PRESET_TO_PX:
            return cast(ResolutionPreset, lowered)
    raise ValueError(
        f"Invalid resolution_preset={value!r}. Expected one of: "
        f"{', '.join(_PRESET_TO_PX.keys())}"
    )


@dataclass(frozen=True, slots=True)
class RoomRenderDimensions:
    top_view_width: int
    top_view_height: int
    side_view_width: int
    side_view_height: int


def get_room_render_dimensions(
    rendering_cfg: Any, *, quality_preset: str | None = None
) -> RoomRenderDimensions:
    """Return effective room render dimensions.

    Priority: quality_preset kwarg > rendering_cfg.resolution_preset > explicit dimensions.
    quality_preset is set by image_quality_preset at the agent config top level.
    """
    preset = quality_preset
    if preset in (None, "", "null"):
        preset = getattr(rendering_cfg, "resolution_preset", None)
    if preset in (None, "", "null"):
        return RoomRenderDimensions(
            top_view_width=int(rendering_cfg.top_view_width),
            top_view_height=int(rendering_cfg.top_view_height),
            side_view_width=int(rendering_cfg.side_view_width),
            side_view_height=int(rendering_cfg.side_view_height),
        )

    normalized = _normalize_preset(preset, default="high")
    top = preset_to_px(normalized)
    side = max(1, top // 2)
    return RoomRenderDimensions(
        top_view_width=top,
        top_view_height=top,
        side_view_width=side,
        side_view_height=side,
    )


@dataclass(frozen=True, slots=True)
class ValidationRenderDimensions:
    width: int
    height: int


def get_validation_render_dimensions(
    asset_manager_cfg: Any, *, quality_preset: str | None = None
) -> ValidationRenderDimensions:
    """Return effective per-object (validation/physics) render dimensions.

    Priority: quality_preset kwarg > asset_manager_cfg.validation_rendering.resolution_preset.
    quality_preset is set by image_quality_preset at the agent config top level.
    """
    preset = quality_preset
    if preset in (None, "", "null"):
        validation_cfg = getattr(asset_manager_cfg, "validation_rendering", None)
        preset = (
            getattr(validation_cfg, "resolution_preset", None)
            if validation_cfg
            else None
        )
    normalized = _normalize_preset(preset, default="medium")
    px = preset_to_px(normalized)
    return ValidationRenderDimensions(width=px, height=px)
