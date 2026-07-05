"""
Additional branch-focused tests for reinforced_concrete.geometry.section.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

import materials.reinforced_concrete.geometry.section as section_mod
from materials.core.geometry import Point2D
from materials.reinforced_concrete.geometry import (
    RCSection,
    RebarGroup,
    create_circular_section,
    create_rectangular_section,
)
from materials.reinforced_concrete.geometry.reinforcement_reconcile import (
    ReinforcementInvalidPolicy,
)


def _section_with_top_bottom_layers(rebar) -> RCSection:
    sec = create_rectangular_section(300.0, 500.0, section_name="S")
    sec.add_rebar_group(
        RebarGroup(
            rebar=rebar,
            positions=(Point2D(x=50.0, y=50.0), Point2D(x=250.0, y=50.0)),
            layer_name="bottom",
        )
    )
    sec.add_rebar_group(
        RebarGroup(
            rebar=rebar,
            positions=(Point2D(x=50.0, y=450.0), Point2D(x=250.0, y=450.0)),
            layer_name="top",
        )
    )
    return sec


class TestSectionCoercionAndIntegralHelpers:
    """Tests for TestSectionCoercionAndIntegralHelpers."""
    def test_coercion_helpers_accept_sequence_inputs(self):
        """Test coercion helpers accept sequence inputs."""
        p = section_mod._coerce_point2d((1.0, 2.0))
        assert isinstance(p, Point2D)
        assert p == Point2D(x=1.0, y=2.0)
        assert section_mod._coerce_point2d("x") == "x"

        pts = section_mod._coerce_point2d_sequence([(0.0, 0.0), Point2D(x=1.0, y=1.0)])
        assert isinstance(pts, tuple)
        assert all(isinstance(v, Point2D) for v in pts)
        assert section_mod._coerce_point2d_sequence("x") == "x"

        voids = section_mod._coerce_voids([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])
        assert isinstance(voids, tuple)
        assert isinstance(voids[0], tuple)
        assert all(isinstance(v, Point2D) for v in voids[0])
        assert section_mod._coerce_voids("x") == "x"

    def test_ring_integrals_short_open_and_degenerate_rings(self):
        """Test ring integrals short open and degenerate rings."""
        short = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
        assert section_mod._ring_integrals_about_origin(short) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Open triangle ring exercises auto-closure branch.
        open_triangle = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=float)
        area, *_ = section_mod._ring_integrals_about_origin(open_triangle)
        assert area == pytest.approx(2.0, rel=1e-12)

        # Collinear points -> zero-area branch.
        degenerate = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
        assert section_mod._ring_integrals_about_origin(degenerate) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_polygon_integrals_zero_net_area(self):
        """Test polygon integrals zero net area."""
        outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        # Hole identical to exterior => zero net area (invalid polygon, but helper still handles it).
        poly = Polygon(outer, [outer])
        assert section_mod._polygon_integrals_about_origin(poly) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class TestSectionValidationAndRollbackBranches:
    """Tests for TestSectionValidationAndRollbackBranches."""
    @staticmethod
    def _base_coords() -> tuple[Point2D, ...]:
        return (
            Point2D(x=0.0, y=0.0),
            Point2D(x=100.0, y=0.0),
            Point2D(x=0.0, y=100.0),
        )

    def test_validate_outline_rejects_empty_polygon(self, monkeypatch):
        """Test validate outline rejects empty polygon."""
        monkeypatch.setattr(RCSection, "_build_outline_polygon", lambda self: Polygon())
        with pytest.raises(ValueError, match="outline is empty"):
            RCSection(outline_coords=self._base_coords())

    def test_validate_outline_rejects_invalid_polygon(self, monkeypatch):
        """Test validate outline rejects invalid polygon."""
        bad = Polygon([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)])
        monkeypatch.setattr(RCSection, "_build_outline_polygon", lambda self: bad)
        with pytest.raises(ValueError, match="not a valid polygon"):
            RCSection(outline_coords=self._base_coords())

    def test_validate_outline_rejects_zero_area_polygon(self, monkeypatch):
        """Test validate outline rejects zero area polygon."""
        class _ZeroAreaPoly:
            is_empty = False
            is_valid = True
            area = 0.0

        monkeypatch.setattr(RCSection, "_build_outline_polygon", lambda self: _ZeroAreaPoly())
        with pytest.raises(ValueError, match="zero or negative area"):
            RCSection(outline_coords=self._base_coords())

    def test_setattr_on_coords_invalidates_outline_cache(self, monkeypatch):
        """Test setattr on coords invalidates outline cache."""
        sec = create_rectangular_section(300.0, 500.0)
        calls = {"n": 0}
        original = RCSection._invalidate_outline_cache

        def _spy(self):
            calls["n"] += 1
            return original(self)

        monkeypatch.setattr(RCSection, "_invalidate_outline_cache", _spy)
        sec.outline_coords = (
            Point2D(x=0.0, y=0.0),
            Point2D(x=200.0, y=0.0),
            Point2D(x=200.0, y=500.0),
            Point2D(x=0.0, y=500.0),
        )
        assert calls["n"] >= 1

    def test_update_outline_restores_policy_on_failure_with_override(self):
        """Test update outline restores policy on failure with override."""
        sec = create_rectangular_section(300.0, 500.0)
        original_policy = sec.reinforcement_policy

        with pytest.raises(ValueError):
            sec.update_outline(
                outline_coords=(Point2D(x=0.0, y=0.0), Point2D(x=1.0, y=0.0)),
                reinforcement_policy=ReinforcementInvalidPolicy.ALLOW_INVALID,
            )

        assert sec.reinforcement_policy == original_policy

    def test_build_outline_polygon_error_branches_with_model_construct(self):
        """Test build outline polygon error branches with model construct."""
        sec_bad_outline = RCSection.model_construct(
            outline_coords=(Point2D(x=0.0, y=0.0), Point2D(x=1.0, y=0.0)),
            voids_coords=(),
            reinforcement_policy=ReinforcementInvalidPolicy.ERROR,
            rebar_groups=[],
            concrete_cover_override=None,
            section_name=None,
        )
        with pytest.raises(ValueError, match="at least 3 points"):
            sec_bad_outline._build_outline_polygon()

        sec_bad_void = RCSection.model_construct(
            outline_coords=(
                Point2D(x=0.0, y=0.0),
                Point2D(x=10.0, y=0.0),
                Point2D(x=10.0, y=10.0),
                Point2D(x=0.0, y=10.0),
            ),
            voids_coords=((Point2D(x=2.0, y=2.0), Point2D(x=3.0, y=3.0)),),
            reinforcement_policy=ReinforcementInvalidPolicy.ERROR,
            rebar_groups=[],
            concrete_cover_override=None,
            section_name=None,
        )
        with pytest.raises(ValueError, match="Each void ring must have at least 3 points"):
            sec_bad_void._build_outline_polygon()


class TestSectionTransformedAndCoverDepthBranches:
    """Tests for TestSectionTransformedAndCoverDepthBranches."""
    def test_transformed_centroid_and_inertia_guard_branches(self, rebar_20, monkeypatch):
        """Test transformed centroid and inertia guard branches."""
        sec = _section_with_top_bottom_layers(rebar_20)

        with pytest.raises(ValueError, match="must be positive"):
            sec.get_transformed_centroid(0.0)

        # factor == 0 branch: E_s / E_c - 1 = 0
        _, cx, cy = sec.get_transformed_centroid(rebar_20.E_s)
        gx, gy = sec.get_centroid()
        assert cx == pytest.approx(gx, rel=1e-12)
        assert cy == pytest.approx(gy, rel=1e-12)

        with pytest.raises(ValueError, match="zero/degenerate"):
            monkeypatch.setattr(
                section_mod,
                "_polygon_integrals_about_origin",
                lambda poly: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ) or sec.get_second_moment_area()

    def test_transformed_second_moment_area_branches(self, rebar_20):
        """Test transformed second moment area branches."""
        sec_no_rebar = create_rectangular_section(300.0, 500.0)

        with pytest.raises(ValueError, match="must be positive"):
            sec_no_rebar.get_transformed_second_moment_area(0.0)

        i_g = sec_no_rebar.get_second_moment_area()
        i_t = sec_no_rebar.get_transformed_second_moment_area(33_000.0)
        assert i_t == pytest.approx(i_g, rel=1e-12)

        sec = _section_with_top_bottom_layers(rebar_20)
        i_xx, i_yy, i_xy = sec.get_transformed_second_moment_area(33_000.0)
        assert i_xx > 0.0
        assert i_yy > 0.0
        assert np.isfinite(i_xy)

    def test_get_concrete_cover_policies(self, rebar_20, monkeypatch):
        """Test get concrete cover policies."""
        sec = _section_with_top_bottom_layers(rebar_20)

        sec.concrete_cover_override = 30.0
        assert sec.get_concrete_cover(reference="bottom") == pytest.approx(30.0, rel=1e-12)
        sec.concrete_cover_override = None

        with pytest.raises(ValueError, match="Unknown reference"):
            sec.get_concrete_cover(reference="side")

        assert sec.get_concrete_cover(reference="bottom", orthogonal_only=True) == pytest.approx(40.0, rel=1e-12)
        assert sec.get_concrete_cover(reference="top", orthogonal_only=True) == pytest.approx(40.0, rel=1e-12)
        assert sec.get_concrete_cover(reference="bottom", orthogonal_only=False) == pytest.approx(40.0, rel=1e-12)
        assert sec.get_concrete_cover(reference="top", orthogonal_only=False) == pytest.approx(40.0, rel=1e-12)

        # Force empty segmented-boundary selection to exercise fallback branch.
        monkeypatch.setattr(RCSection, "get_centroid", lambda self: (150.0, 1.0e9))
        assert sec.get_concrete_cover(reference="top", orthogonal_only=False) > 0.0

        empty = create_rectangular_section(300.0, 500.0)
        with pytest.raises(ValueError, match="no rebars"):
            empty.get_concrete_cover()

    def test_get_concrete_cover_handles_degenerate_boundary_segment(self, rebar_20):
        # Repeated vertex creates a zero-length segment in the top boundary list.
        """Test get concrete cover handles degenerate boundary segment."""
        sec = RCSection(
            outline_coords=(
                Point2D(x=0.0, y=0.0),
                Point2D(x=300.0, y=0.0),
                Point2D(x=300.0, y=500.0),
                Point2D(x=300.0, y=500.0),
                Point2D(x=0.0, y=500.0),
            ),
            section_name="DegenerateEdge",
        )
        sec.add_rebar_group(
            RebarGroup(
                rebar=rebar_20,
                positions=(Point2D(x=50.0, y=450.0),),
            )
        )
        assert sec.get_concrete_cover(reference="top", orthogonal_only=False) > 0.0

    def test_get_effective_depth_error_and_top_zone_branches(self, rebar_20, monkeypatch):
        """Test get effective depth error and top zone branches."""
        empty = create_rectangular_section(300.0, 500.0)
        with pytest.raises(ValueError, match="no rebars"):
            empty.get_effective_depth()

        sec = _section_with_top_bottom_layers(rebar_20)
        with pytest.raises(ValueError, match="compression_face must be"):
            sec.get_effective_depth("left")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="zone_fraction must be"):
            sec.get_effective_depth("top", zone_fraction=0.0)
        with pytest.raises(ValueError, match="tension_zone must be"):
            sec.get_effective_depth("top", tension_zone="left")  # type: ignore[arg-type]

        with monkeypatch.context() as m:
            m.setattr(RCSection, "get_bounding_box", lambda self: (0.0, 0.0, 300.0, 0.0))
            with pytest.raises(ValueError, match="Invalid section height"):
                sec.get_effective_depth("top")

        # Restore real method by creating a new section.
        sec2 = create_rectangular_section(300.0, 500.0)
        sec2.add_rebar_group(
            RebarGroup(
                rebar=rebar_20,
                positions=(Point2D(x=50.0, y=50.0), Point2D(x=250.0, y=50.0)),
                layer_name="bottom_only",
            )
        )
        with pytest.raises(ValueError, match="No reinforcement found in the selected tension zone"):
            sec2.get_effective_depth("top", tension_zone="top", zone_fraction=0.2)

        sec3 = _section_with_top_bottom_layers(rebar_20)
        d_top_zone = sec3.get_effective_depth("bottom", tension_zone="top", zone_fraction=0.2)
        assert d_top_zone == pytest.approx(450.0, rel=1e-12)

    def test_get_compression_rebar_depth_branches(self, rebar_20, monkeypatch):
        """Test get compression rebar depth branches."""
        empty = create_rectangular_section(300.0, 500.0)
        assert empty.get_compression_rebar_depth("top") is None

        sec = _section_with_top_bottom_layers(rebar_20)
        with pytest.raises(ValueError, match="compression_face must be"):
            sec.get_compression_rebar_depth("left")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="zone_fraction must be"):
            sec.get_compression_rebar_depth("top", zone_fraction=0.0)

        with monkeypatch.context() as m:
            m.setattr(RCSection, "get_bounding_box", lambda self: (0.0, 0.0, 300.0, 0.0))
            assert sec.get_compression_rebar_depth("top") is None

        sec_bottom_only = create_rectangular_section(300.0, 500.0)
        sec_bottom_only.add_rebar_group(
            RebarGroup(
                rebar=rebar_20,
                positions=(Point2D(x=50.0, y=50.0), Point2D(x=250.0, y=50.0)),
            )
        )
        assert sec_bottom_only.get_compression_rebar_depth("top", zone_fraction=0.1) is None

        sec2 = _section_with_top_bottom_layers(rebar_20)
        assert sec2.get_compression_rebar_depth("top") == pytest.approx(50.0, rel=1e-12)
        assert sec2.get_compression_rebar_depth("bottom") == pytest.approx(50.0, rel=1e-12)


class TestSectionFactoriesAndWrappers:
    """Tests for TestSectionFactoriesAndWrappers."""
    @pytest.mark.parametrize(
        ("hook_ref", "expected"),
        [
            (2, (-300.0, 0.0, 0.0, 500.0)),
            (3, (-300.0, -500.0, 0.0, 0.0)),
            (4, (0.0, -500.0, 300.0, 0.0)),
        ],
    )
    def test_create_rectangular_section_remaining_hook_refs(self, hook_ref, expected):
        """Test create rectangular section remaining hook refs."""
        sec = create_rectangular_section(300.0, 500.0, origin=(0.0, 0.0), hook_ref=hook_ref)
        assert sec.get_bounding_box() == pytest.approx(expected, rel=1e-12)

    def test_create_rectangular_section_invalid_hook_ref(self):
        """Test create rectangular section invalid hook ref."""
        with pytest.raises(ValueError, match="hook_ref must be 0, 1, 2, 3, or 4"):
            create_rectangular_section(300.0, 500.0, hook_ref=9)

    @pytest.mark.parametrize(
        ("hook_ref", "expected"),
        [
            (2, (-400.0, 0.0, 0.0, 400.0)),
            (3, (-400.0, -400.0, 0.0, 0.0)),
            (4, (0.0, -400.0, 400.0, 0.0)),
        ],
    )
    def test_create_circular_section_remaining_hook_refs(self, hook_ref, expected):
        """Test create circular section remaining hook refs."""
        sec = create_circular_section(400.0, origin=(0.0, 0.0), hook_ref=hook_ref)
        assert sec.get_bounding_box() == pytest.approx(expected, abs=1e-9)

    def test_create_circular_section_invalid_hook_ref(self):
        """Test create circular section invalid hook ref."""
        with pytest.raises(ValueError, match="hook_ref must be 0, 1, 2, 3, or 4"):
            create_circular_section(400.0, hook_ref=9)

    def test_str_and_plot_wrapper(self, monkeypatch):
        """Test str and plot wrapper."""
        sec = create_rectangular_section(300.0, 500.0, section_name="PlotSection")
        assert str(sec) == repr(sec)

        import materials.reinforced_concrete.geometry.section_viewer as sv_mod

        captured = {}

        class _DummyViewer:
            def __init__(self, section):
                captured["section"] = section

            def plot(self, **kwargs):
                captured["kwargs"] = kwargs
                return "dummy-figure"

        monkeypatch.setattr(sv_mod, "SectionViewer", _DummyViewer)
        out = sec.plot(show=False, title="T", width=321, height=654)
        assert out == "dummy-figure"
        assert captured["section"] is sec
        assert captured["kwargs"]["show"] is False
        assert captured["kwargs"]["title"] == "T"
        assert captured["kwargs"]["width"] == 321
        assert captured["kwargs"]["height"] == 654


class TestSectionClashAndRemoveBarsBranches:
    """Tests for TestSectionClashAndRemoveBarsBranches."""
    def test_validate_outline_rebars_raises_on_cross_group_clash(self, rebar_20):
        """Test validate outline rebars raises on cross group clash."""
        outline = (
            Point2D(x=0.0, y=0.0),
            Point2D(x=300.0, y=0.0),
            Point2D(x=300.0, y=500.0),
            Point2D(x=0.0, y=500.0),
        )
        g0 = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=100.0, y=100.0),),
            layer_name="g0",
        )
        g1 = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=105.0, y=100.0),),
            layer_name="g1",
        )

        with pytest.raises(ValueError, match="clash across groups"):
            RCSection(
                outline_coords=outline,
                rebar_groups=[g0, g1],
                reinforcement_policy=ReinforcementInvalidPolicy.ALLOW_INVALID,
            )

    def test_add_rebar_group_rejects_cross_group_clash(self, rebar_20):
        """Test add rebar group rejects cross group clash."""
        sec = create_rectangular_section(300.0, 500.0)
        sec.add_rebar_group(
            RebarGroup(
                rebar=rebar_20,
                positions=(Point2D(x=100.0, y=100.0),),
                layer_name="base",
            )
        )

        with pytest.raises(ValueError, match="clashes with"):
            sec.add_rebar_group(
                RebarGroup(
                    rebar=rebar_20,
                    positions=(Point2D(x=105.0, y=100.0),),
                    layer_name="clash",
                )
            )

    def test_remove_bars_filter_and_rebuild_branches(self, rebar_20):
        """Test remove bars filter and rebuild branches."""
        sec = create_rectangular_section(300.0, 500.0)
        sec.add_rebar_group(
            RebarGroup(
                rebar=rebar_20,
                positions=(Point2D(x=50.0, y=50.0), Point2D(x=100.0, y=50.0)),
                layer_name="g0",
            )
        )
        sec.add_rebar_group(
            RebarGroup(
                rebar=rebar_20,
                positions=(Point2D(x=200.0, y=50.0), Point2D(x=250.0, y=50.0)),
                layer_name="g1",
            )
        )

        # positions includes both Point2D and tuple types; bar_indices contributes too.
        removed_all_from_g0 = sec.remove_bars(
            group_index=0,
            positions=[Point2D(x=50.0, y=50.0), (100.0, 50.0)],
            bar_indices=[1],
        )
        assert removed_all_from_g0 == 2
        assert len(sec.rebar_groups) == 1
        assert sec.rebar_groups[0].layer_name == "g1"

        # No per-bar filter => remove whole matching group.
        removed_whole_group = sec.remove_bars(layer_name="g1")
        assert removed_whole_group == 2
        assert len(sec.rebar_groups) == 0

        # Partial keep path should rebuild a RebarGroup with remaining bars.
        sec2 = create_rectangular_section(300.0, 500.0)
        sec2.add_rebar_group(
            RebarGroup(
                rebar=rebar_20,
                positions=(
                    Point2D(x=60.0, y=60.0),
                    Point2D(x=120.0, y=60.0),
                    Point2D(x=180.0, y=60.0),
                ),
                layer_name="g2",
            )
        )
        removed_partial = sec2.remove_bars(group_index=0, bar_indices=[1])
        assert removed_partial == 1
        assert len(sec2.rebar_groups) == 1
        assert len(sec2.rebar_groups[0].positions) == 2
