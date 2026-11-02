from omegaconf import OmegaConf

from scenesmith.agent_utils.vision_resolution import (
    get_room_render_dimensions,
    get_validation_render_dimensions,
    preset_to_px,
)


def test_preset_to_px_mapping() -> None:
    assert preset_to_px("low") == 256
    assert preset_to_px("medium") == 512
    assert preset_to_px("high") == 1024


def test_room_dimensions_no_preset_uses_explicit_sizes() -> None:
    cfg = OmegaConf.create(
        {
            "top_view_width": 111,
            "top_view_height": 222,
            "side_view_width": 333,
            "side_view_height": 444,
            "resolution_preset": None,
        }
    )
    dims = get_room_render_dimensions(cfg)
    assert dims.top_view_width == 111
    assert dims.top_view_height == 222
    assert dims.side_view_width == 333
    assert dims.side_view_height == 444


def test_room_dimensions_low_preset_overrides_sizes() -> None:
    cfg = OmegaConf.create(
        {
            "top_view_width": 999,
            "top_view_height": 999,
            "side_view_width": 999,
            "side_view_height": 999,
            "resolution_preset": "low",
        }
    )
    dims = get_room_render_dimensions(cfg)
    assert dims.top_view_width == 256
    assert dims.top_view_height == 256
    assert dims.side_view_width == 128
    assert dims.side_view_height == 128


def test_validation_dimensions_default_medium_when_unset() -> None:
    asset_manager_cfg = OmegaConf.create({})
    dims = get_validation_render_dimensions(asset_manager_cfg)
    assert dims.width == 512
    assert dims.height == 512


def test_validation_dimensions_high_preset() -> None:
    asset_manager_cfg = OmegaConf.create(
        {"validation_rendering": {"resolution_preset": "high"}}
    )
    dims = get_validation_render_dimensions(asset_manager_cfg)
    assert dims.width == 1024
    assert dims.height == 1024
