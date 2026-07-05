"""Tests for cross-group rebar clash detection and remove_bars method."""

import pytest

from materials.core.geometry import Point2D
from materials.reinforced_concrete.geometry.reinforcement_reconcile import (
    ReinforcementInvalidPolicy,
    find_clashing_rebars,
)
from materials.reinforced_concrete.geometry.section import RCSection, RebarGroup
from materials.reinforced_concrete.materials.rebar import Rebar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _square_section(size: float = 500.0) -> RCSection:
    """300x300 square section with no bars."""
    return RCSection(
        outline_coords=(
            Point2D(x=0, y=0),
            Point2D(x=size, y=0),
            Point2D(x=size, y=size),
            Point2D(x=0, y=size),
        ),
        reinforcement_policy=ReinforcementInvalidPolicy.ALLOW_INVALID,
    )


def _rebar(dia: float = 20.0) -> Rebar:
    return Rebar(diameter=dia)


def _group(dia: float, positions: list[tuple[float, float]], layer: str | None = None) -> RebarGroup:
    return RebarGroup(
        rebar=_rebar(dia),
        positions=tuple(Point2D(x=x, y=y) for x, y in positions),
        layer_name=layer,
    )


# ===================================================================
# Cross-group clash detection
# ===================================================================

class TestCrossGroupClash:
    """Behavioral tests for clash detection across rebar groups."""

    def test_no_clash_separate_bars(self):
        """Non-overlapping bars in different groups pass."""
        sec = _square_section()
        g1 = _group(20, [(50, 50)])
        g2 = _group(20, [(100, 50)])  # 50mm apart, radius sum = 20 → no clash
        sec.add_rebar_group(g1)
        sec.add_rebar_group(g2)
        assert len(sec.rebar_groups) == 2

    def test_touching_bars_allowed(self):
        """Bars exactly touching (distance == sum of radii) are allowed."""
        sec = _square_section()
        g1 = _group(20, [(50, 50)])
        g2 = _group(20, [(70, 50)])  # 20mm apart = r1+r2 = 10+10 → touching
        sec.add_rebar_group(g1)
        sec.add_rebar_group(g2)
        assert len(sec.rebar_groups) == 2

    def test_overlapping_bars_raises_via_add(self):
        """Overlapping bars across groups raise ValueError via add_rebar_group."""
        sec = _square_section()
        g1 = _group(20, [(50, 50)])
        g2 = _group(20, [(60, 50)])  # 10mm apart < r1+r2 = 20 → clash
        sec.add_rebar_group(g1)
        with pytest.raises(ValueError, match="clashes"):
            sec.add_rebar_group(g2)

    def test_overlapping_bars_raises_via_assignment(self):
        """Overlapping bars across groups raise ValueError via direct assignment."""
        sec = _square_section()
        g1 = _group(20, [(50, 50)])
        g2 = _group(20, [(60, 50)])
        with pytest.raises(ValueError, match="clash"):
            sec.rebar_groups = [g1, g2]

    def test_different_diameters_clash(self):
        """Clash detection works with different bar diameters."""
        sec = _square_section()
        g1 = _group(32, [(50, 50)])   # r=16
        g2 = _group(20, [(70, 50)])   # r=10, distance=20 < 16+10=26 → clash
        sec.add_rebar_group(g1)
        with pytest.raises(ValueError, match="clashes"):
            sec.add_rebar_group(g2)

    def test_different_diameters_no_clash(self):
        """No clash when distance exceeds sum of radii for different diameters."""
        sec = _square_section()
        g1 = _group(32, [(50, 50)])   # r=16
        g2 = _group(20, [(80, 50)])   # r=10, distance=30 > 16+10=26 → ok
        sec.add_rebar_group(g1)
        sec.add_rebar_group(g2)
        assert len(sec.rebar_groups) == 2

    def test_find_clashing_rebars_function(self):
        """The standalone find_clashing_rebars function works."""
        sec = _square_section()
        g1 = _group(20, [(50, 50)])
        g2 = _group(20, [(55, 50)])  # overlap
        # Bypass validation to set up state for standalone function
        sec.rebar_groups.clear()
        object.__setattr__(sec, "rebar_groups", [g1, g2])
        details, clashes = find_clashing_rebars(sec)
        assert len(clashes) == 1
        assert clashes[0] == (0, 0, 1, 0)

    def test_within_group_still_checked(self):
        """Within-group overlap is still caught by RebarGroup validator."""
        with pytest.raises(ValueError, match="overlap"):
            _group(20, [(50, 50), (55, 50)])


# ===================================================================
# remove_bars
# ===================================================================

class TestRemoveBars:
    """Behavioral tests for `RCSection.remove_bars` filtering and counts."""

    def _section_with_bars(self) -> RCSection:
        sec = _square_section()
        sec.add_rebar_group(_group(20, [(50, 50), (100, 50)], layer="bottom"))
        sec.add_rebar_group(_group(16, [(50, 450), (100, 450), (150, 450)], layer="top"))
        return sec

    def test_remove_by_group_index(self):
        """Removing by group index should delete the full targeted group."""
        sec = self._section_with_bars()
        removed = sec.remove_bars(group_index=0)
        assert removed == 2
        assert len(sec.rebar_groups) == 1
        assert sec.rebar_groups[0].layer_name == "top"

    def test_remove_by_layer_name(self):
        """Removing by layer name should delete only bars from that named layer."""
        sec = self._section_with_bars()
        removed = sec.remove_bars(layer_name="top")
        assert removed == 3
        assert len(sec.rebar_groups) == 1
        assert sec.rebar_groups[0].layer_name == "bottom"

    def test_remove_by_bar_indices(self):
        """Removing selected indices should keep the remaining bars in the group."""
        sec = self._section_with_bars()
        removed = sec.remove_bars(group_index=1, bar_indices=[0, 2])
        assert removed == 2
        assert len(sec.rebar_groups) == 2
        assert len(sec.rebar_groups[1].positions) == 1
        assert sec.rebar_groups[1].positions[0].y == 450

    def test_remove_by_position(self):
        """Removing by tuple position should only match bars at that coordinate."""
        sec = self._section_with_bars()
        removed = sec.remove_bars(positions=[(50, 50)])
        # Should match bar in group 0 only (group 1 has y=450)
        assert removed == 1
        assert len(sec.rebar_groups) == 2
        assert len(sec.rebar_groups[0].positions) == 1

    def test_remove_by_position_tuple(self):
        """Removing by Point2D position should match equivalent bar coordinates."""
        sec = self._section_with_bars()
        removed = sec.remove_bars(positions=[Point2D(x=50, y=450)])
        assert removed == 1

    def test_remove_all_bars_in_group_drops_group(self):
        """If all bars in a group are removed, the empty group should be dropped."""
        sec = self._section_with_bars()
        removed = sec.remove_bars(group_index=0, bar_indices=[0, 1])
        assert removed == 2
        assert len(sec.rebar_groups) == 1

    def test_remove_no_match_returns_zero(self):
        """A non-matching filter should remove nothing and return zero."""
        sec = self._section_with_bars()
        removed = sec.remove_bars(layer_name="nonexistent")
        assert removed == 0
        assert len(sec.rebar_groups) == 2

    def test_remove_returns_correct_count(self):
        """Unfiltered removal should return the total number of removed bars."""
        sec = self._section_with_bars()
        total = sum(len(g.positions) for g in sec.rebar_groups)
        removed = sec.remove_bars()  # no filter = remove all
        assert removed == total
        assert len(sec.rebar_groups) == 0
