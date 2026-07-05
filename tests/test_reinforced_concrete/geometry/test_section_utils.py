"""Branch tests for section geometry factory helpers."""

from __future__ import annotations

import pytest

from section_design_checks.reinforced_concrete.geometry import (
    create_box_section,
    create_channel_section,
    create_circular_section,
    create_i_beam_section,
    create_inverted_t_beam_section,
    create_rectangular_section,
    create_t_beam_section,
    create_trapezoidal_section,
    create_voided_deck_section,
)


class TestFactoryInputValidation:
    """Public section factories must reject non-positive dimensions instead of
    silently building a degenerate (zero/negative-area) section."""

    @pytest.mark.parametrize("w,h", [(0.0, 500.0), (300.0, 0.0), (-10.0, 500.0), (300.0, -5.0)])
    def test_rectangular_rejects_nonpositive(self, w, h):
        with pytest.raises(ValueError, match="positive width and height"):
            create_rectangular_section(width=w, height=h)

    def test_circular_rejects_nonpositive_diameter(self):
        with pytest.raises(ValueError, match="positive diameter"):
            create_circular_section(diameter=0.0)
        with pytest.raises(ValueError, match="positive diameter"):
            create_circular_section(diameter=-100.0)

    def test_circular_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="n_points"):
            create_circular_section(diameter=400.0, n_points=2)


def test_create_t_beam_section_geometry() -> None:
    """Test create t beam section geometry."""
    section = create_t_beam_section(b_f=400.0, h_f=80.0, b_w=200.0, h_w=320.0)
    assert len(section.outline_coords) == 8
    x_min, y_min, x_max, y_max = section.get_bounding_box()
    assert x_max - x_min == pytest.approx(400.0)
    assert y_max - y_min == pytest.approx(400.0)
    assert section.outline_coords[0].x == pytest.approx(100.0)


def test_create_t_beam_section_invalid_web_width() -> None:
    """Test create t beam section invalid web width."""
    with pytest.raises(ValueError, match="must be <= flange width"):
        create_t_beam_section(b_f=300.0, h_f=80.0, b_w=350.0, h_w=250.0)


def test_create_inverted_t_beam_section_geometry() -> None:
    """Test create inverted t beam section geometry."""
    section = create_inverted_t_beam_section(b_f=420.0, h_f=90.0, b_w=180.0, h_w=310.0)
    assert len(section.outline_coords) == 8
    x_min, y_min, x_max, y_max = section.get_bounding_box()
    assert x_max - x_min == pytest.approx(420.0)
    assert y_max - y_min == pytest.approx(400.0)


def test_create_inverted_t_beam_section_invalid_web_width() -> None:
    """Test create inverted t beam section invalid web width."""
    with pytest.raises(ValueError, match="must be <= flange width"):
        create_inverted_t_beam_section(b_f=300.0, h_f=80.0, b_w=350.0, h_w=250.0)


def test_create_i_beam_section_geometry() -> None:
    """Test create i beam section geometry."""
    section = create_i_beam_section(
        b_f_top=500.0,
        h_f_top=80.0,
        b_f_bot=300.0,
        h_f_bot=100.0,
        b_w=200.0,
        h_w=250.0,
    )
    assert len(section.outline_coords) == 12
    x_min, y_min, x_max, y_max = section.get_bounding_box()
    assert x_max - x_min == pytest.approx(500.0)
    assert y_max - y_min == pytest.approx(430.0)


def test_create_i_beam_section_invalid_web_width_top() -> None:
    """Test create i beam section invalid web width top."""
    with pytest.raises(ValueError, match="top flange"):
        create_i_beam_section(
            b_f_top=300.0,
            h_f_top=70.0,
            b_f_bot=350.0,
            h_f_bot=90.0,
            b_w=320.0,
            h_w=200.0,
        )


def test_create_i_beam_section_invalid_web_width_bottom() -> None:
    """Test create i beam section invalid web width bottom."""
    with pytest.raises(ValueError, match="bottom flange"):
        create_i_beam_section(
            b_f_top=500.0,
            h_f_top=70.0,
            b_f_bot=300.0,
            h_f_bot=90.0,
            b_w=320.0,
            h_w=200.0,
        )


def test_create_box_section_geometry_and_default_bottom_flange() -> None:
    """Test create box section geometry and default bottom flange."""
    section = create_box_section(width=600.0, height=400.0, t_web=60.0, t_flange_top=50.0)
    assert len(section.outline_coords) == 4
    assert len(section.voids_coords) == 1
    void = section.voids_coords[0]
    xs = [p.x for p in void]
    ys = [p.y for p in void]
    assert max(xs) - min(xs) == pytest.approx(480.0)
    assert max(ys) - min(ys) == pytest.approx(300.0)


def test_create_box_section_invalid_web_thickness() -> None:
    """Test create box section invalid web thickness."""
    with pytest.raises(ValueError, match="2 \\* t_web"):
        create_box_section(width=200.0, height=300.0, t_web=100.0, t_flange_top=40.0, t_flange_bot=40.0)


def test_create_box_section_invalid_flange_thickness_sum() -> None:
    """Test create box section invalid flange thickness sum."""
    with pytest.raises(ValueError, match="must be < height"):
        create_box_section(width=300.0, height=200.0, t_web=30.0, t_flange_top=120.0, t_flange_bot=90.0)


def test_create_voided_deck_section_auto_spacing() -> None:
    """Test create voided deck section auto spacing."""
    section = create_voided_deck_section(
        width=1200.0,
        height=300.0,
        void_diameter=140.0,
        n_voids=3,
        n_points=16,
    )
    assert len(section.outline_coords) == 4
    assert len(section.voids_coords) == 3
    assert all(len(v) == 16 for v in section.voids_coords)


def test_create_voided_deck_section_custom_spacing() -> None:
    """Test create voided deck section custom spacing."""
    section = create_voided_deck_section(
        width=1000.0,
        height=280.0,
        void_diameter=120.0,
        n_voids=2,
        void_spacing=300.0,
        n_points=12,
    )
    assert len(section.voids_coords) == 2
    assert all(len(v) == 12 for v in section.voids_coords)


def test_create_voided_deck_section_invalid_inputs() -> None:
    """Test create voided deck section invalid inputs."""
    with pytest.raises(ValueError, match="n_voids must be >= 1"):
        create_voided_deck_section(width=1000.0, height=300.0, void_diameter=120.0, n_voids=0)

    with pytest.raises(ValueError, match="must be < height"):
        create_voided_deck_section(width=1000.0, height=300.0, void_diameter=300.0, n_voids=2)

    with pytest.raises(ValueError, match="exceeds slab width"):
        create_voided_deck_section(
            width=500.0,
            height=250.0,
            void_diameter=200.0,
            n_voids=2,
            void_spacing=400.0,
        )


def test_create_channel_section_top_and_bottom() -> None:
    """Test create channel section top and bottom."""
    top_open = create_channel_section(width=400.0, height=300.0, t_web=40.0, t_flange=60.0, open_side="top")
    bot_open = create_channel_section(width=400.0, height=300.0, t_web=40.0, t_flange=60.0, open_side="bottom")
    assert len(top_open.outline_coords) == 8
    assert len(bot_open.outline_coords) == 8


def test_create_channel_section_invalid_inputs() -> None:
    """Test create channel section invalid inputs."""
    with pytest.raises(ValueError, match="2 \\* t_web"):
        create_channel_section(width=200.0, height=300.0, t_web=100.0, t_flange=40.0)

    with pytest.raises(ValueError, match="must be < height"):
        create_channel_section(width=300.0, height=200.0, t_web=30.0, t_flange=200.0)

    with pytest.raises(ValueError, match="open_side must be"):
        create_channel_section(width=300.0, height=200.0, t_web=30.0, t_flange=50.0, open_side="left")


def test_create_trapezoidal_section_geometry() -> None:
    """Test create trapezoidal section geometry."""
    section = create_trapezoidal_section(b_top=300.0, b_bot=500.0, height=400.0, hook_ref=0)
    assert len(section.outline_coords) == 4
    x_min, y_min, x_max, y_max = section.get_bounding_box()
    assert x_max - x_min == pytest.approx(500.0)
    assert y_max - y_min == pytest.approx(400.0)

