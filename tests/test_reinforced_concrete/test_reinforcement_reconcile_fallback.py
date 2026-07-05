"""
Extra coverage tests for reinforcement reconciliation fallback paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from materials.core.geometry import Point2D
from materials.reinforced_concrete.geometry.reinforcement_reconcile import (
    ReinforcementInvalidPolicy,
    find_clashing_rebars,
    find_invalid_rebars,
    prune_reinforcement_for_outline,
    update_outline,
)
from materials.reinforced_concrete.geometry.section import RCSection
from materials.reinforced_concrete.geometry.section import RebarGroup
from materials.reinforced_concrete.materials import Rebar


class _DummySectionWithMethod:
    def __init__(self):
        self.calls = []

    def update_outline(self, *, outline_coords, voids_coords, reinforcement_policy):
        self.calls.append((outline_coords, voids_coords, reinforcement_policy))
        return "sentinel-report"


class _DummySectionFallback:
    def __init__(self, *, with_suspend_flag: bool = True, with_invalidate: bool = True):
        self.outline_coords = None
        self.voids_coords = None
        self._outline_access_count = 0
        self._invalidate_calls = 0
        if with_suspend_flag:
            self._suspend_outline_reconcile = False
        if with_invalidate:
            self._invalidate_outline_cache = self._invalidate

    def _invalidate(self):
        self._invalidate_calls += 1

    @property
    def outline(self):
        self._outline_access_count += 1
        return object()


class TestUpdateOutlineHelper:
    def test_delegates_to_section_method_when_available(self):
        section = _DummySectionWithMethod()

        out = update_outline(
            section,
            outline_coords="new-outline",
            voids_coords=("void-a",),
            policy=ReinforcementInvalidPolicy.DROP_INVALID_BARS,
        )

        assert out == "sentinel-report"
        assert section.calls == [
            ("new-outline", ("void-a",), ReinforcementInvalidPolicy.DROP_INVALID_BARS),
        ]

    def test_fallback_path_updates_coords_rebuilds_outline_and_reconciles(self, monkeypatch):
        section = _DummySectionFallback(with_suspend_flag=True, with_invalidate=True)
        captured = {}

        def _fake_reconcile(sec, *, policy):
            captured["section"] = sec
            captured["policy"] = policy
            return "fallback-report"

        monkeypatch.setattr(
            "materials.reinforced_concrete.geometry.reinforcement_reconcile.reconcile_after_outline_change",
            _fake_reconcile,
        )

        out = update_outline(
            section,
            outline_coords="outline-1",
            voids_coords=("void-1",),
            policy=ReinforcementInvalidPolicy.ALLOW_INVALID,
        )

        assert out == "fallback-report"
        assert section.outline_coords == "outline-1"
        assert section.voids_coords == ("void-1",)
        assert section._invalidate_calls == 1
        assert section._outline_access_count == 1
        assert section._suspend_outline_reconcile is False
        assert captured["section"] is section
        assert captured["policy"] == ReinforcementInvalidPolicy.ALLOW_INVALID

    def test_fallback_path_without_optional_flags(self, monkeypatch):
        section = _DummySectionFallback(with_suspend_flag=False, with_invalidate=False)
        monkeypatch.setattr(
            "materials.reinforced_concrete.geometry.reinforcement_reconcile.reconcile_after_outline_change",
            lambda sec, *, policy: "ok",
        )

        out = update_outline(
            section,
            outline_coords="outline-2",
            policy=ReinforcementInvalidPolicy.DROP_INVALID_GROUPS,
        )

        assert out == "ok"
        assert section.outline_coords == "outline-2"
        assert section.voids_coords is None
        assert section._outline_access_count == 1
        assert not hasattr(section, "_suspend_outline_reconcile")

    def test_fallback_restores_suspend_flag_on_error(self, monkeypatch):
        section = _DummySectionFallback(with_suspend_flag=True, with_invalidate=True)

        def _boom(sec, *, policy):
            raise RuntimeError("reconcile failed")

        monkeypatch.setattr(
            "materials.reinforced_concrete.geometry.reinforcement_reconcile.reconcile_after_outline_change",
            _boom,
        )

        with pytest.raises(RuntimeError, match="reconcile failed"):
            update_outline(
                section,
                outline_coords="outline-3",
                policy=ReinforcementInvalidPolicy.ERROR,
            )

        assert section._suspend_outline_reconcile is False


class TestPruneDefensiveBranches:
    def test_drop_invalid_bars_keeps_unaffected_groups(self, monkeypatch):
        rebar = Rebar(diameter=16.0, grade="B500B")
        g0 = RebarGroup(
            rebar=rebar,
            positions=(Point2D(x=0.0, y=0.0), Point2D(x=20.0, y=0.0)),
            layer_name="g0",
        )
        g1 = RebarGroup(
            rebar=rebar,
            positions=(Point2D(x=100.0, y=0.0),),
            layer_name="g1",
        )
        section = SimpleNamespace(rebar_groups=[g0, g1])

        monkeypatch.setattr(
            "materials.reinforced_concrete.geometry.reinforcement_reconcile.find_invalid_rebars",
            lambda sec: (["g0 bar1 invalid"], [(0, 1)]),
        )

        report = prune_reinforcement_for_outline(
            section,
            policy=ReinforcementInvalidPolicy.DROP_INVALID_BARS,
        )

        assert report.invalid_groups == 1
        assert report.invalid_bars == 1
        assert report.removed_groups == 0
        assert report.removed_bars == 1
        assert len(section.rebar_groups) == 2
        assert len(section.rebar_groups[0].positions) == 1
        assert section.rebar_groups[1].layer_name == "g1"

    def test_unknown_policy_raises_value_error(self, monkeypatch):
        section = SimpleNamespace(rebar_groups=[])
        monkeypatch.setattr(
            "materials.reinforced_concrete.geometry.reinforcement_reconcile.find_invalid_rebars",
            lambda sec: (["invalid"], [(0, 0)]),
        )

        with pytest.raises(ValueError, match="Unknown policy"):
            prune_reinforcement_for_outline(section, policy="not-a-policy")  # type: ignore[arg-type]

    def test_find_clashing_rebars_returns_detail_strings_for_overlap(self):
        rebar = Rebar(diameter=20.0, grade="B500B")
        g0 = RebarGroup(
            rebar=rebar,
            positions=(Point2D(x=0.0, y=0.0),),
            layer_name="g0",
        )
        g1 = RebarGroup(
            rebar=rebar,
            positions=(Point2D(x=5.0, y=0.0),),
            layer_name="g1",
        )
        section = SimpleNamespace(rebar_groups=[g0, g1])

        details, clashes = find_clashing_rebars(section, _geom_tol=0.0)

        assert clashes == [(0, 0, 1, 0)]
        assert len(details) == 1
        assert "group[0] bar[0]" in details[0]
        assert "clashes with group[1] bar[0]" in details[0]

    def test_find_invalid_rebars_returns_entries_for_outside_bar(self):
        section = RCSection(
            outline_coords=(
                Point2D(x=0.0, y=0.0),
                Point2D(x=200.0, y=0.0),
                Point2D(x=200.0, y=200.0),
                Point2D(x=0.0, y=200.0),
            ),
            reinforcement_policy=ReinforcementInvalidPolicy.ALLOW_INVALID,
        )
        rebar = Rebar(diameter=20.0, grade="B500B")
        g0 = RebarGroup(
            rebar=rebar,
            positions=(Point2D(x=195.0, y=100.0),),
            layer_name="edge",
        )
        object.__setattr__(section, "rebar_groups", [g0])

        details, invalid = find_invalid_rebars(section)
        assert invalid == [(0, 0)]
        assert len(details) == 1
        assert "outside outline" in details[0]

    def test_prune_allow_invalid_returns_report_without_mutating(self, monkeypatch):
        section = SimpleNamespace(rebar_groups=[SimpleNamespace(), SimpleNamespace()])
        monkeypatch.setattr(
            "materials.reinforced_concrete.geometry.reinforcement_reconcile.find_invalid_rebars",
            lambda sec: (["invalid-a", "invalid-b"], [(0, 0), (1, 0)]),
        )

        report = prune_reinforcement_for_outline(
            section,
            policy=ReinforcementInvalidPolicy.ALLOW_INVALID,
        )
        assert report.invalid_groups == 2
        assert report.invalid_bars == 2
        assert report.removed_groups == 0
        assert report.removed_bars == 0
        assert len(section.rebar_groups) == 2

    def test_prune_error_policy_raises(self, monkeypatch):
        section = SimpleNamespace(rebar_groups=[SimpleNamespace()])
        monkeypatch.setattr(
            "materials.reinforced_concrete.geometry.reinforcement_reconcile.find_invalid_rebars",
            lambda sec: (["invalid"], [(0, 0)]),
        )

        with pytest.raises(ValueError, match="Outline update made some reinforcement invalid"):
            prune_reinforcement_for_outline(section, policy=ReinforcementInvalidPolicy.ERROR)

    def test_prune_drop_invalid_groups_removes_whole_groups(self, monkeypatch):
        rebar = Rebar(diameter=16.0, grade="B500B")
        g0 = RebarGroup(
            rebar=rebar,
            positions=(Point2D(x=0.0, y=0.0),),
            layer_name="g0",
        )
        g1 = RebarGroup(
            rebar=rebar,
            positions=(Point2D(x=10.0, y=0.0),),
            layer_name="g1",
        )
        section = SimpleNamespace(rebar_groups=[g0, g1])
        monkeypatch.setattr(
            "materials.reinforced_concrete.geometry.reinforcement_reconcile.find_invalid_rebars",
            lambda sec: (["group 0 invalid"], [(0, 0)]),
        )

        report = prune_reinforcement_for_outline(
            section,
            policy=ReinforcementInvalidPolicy.DROP_INVALID_GROUPS,
        )
        assert report.removed_groups == 1
        assert len(section.rebar_groups) == 1
        assert section.rebar_groups[0].layer_name == "g1"

    def test_prune_drop_invalid_bars_removes_group_when_no_positions_left(self, monkeypatch):
        rebar = Rebar(diameter=16.0, grade="B500B")
        g0 = RebarGroup(
            rebar=rebar,
            positions=(Point2D(x=0.0, y=0.0),),
            layer_name="g0",
        )
        section = SimpleNamespace(rebar_groups=[g0])
        monkeypatch.setattr(
            "materials.reinforced_concrete.geometry.reinforcement_reconcile.find_invalid_rebars",
            lambda sec: (["group 0 bar 0 invalid"], [(0, 0)]),
        )

        report = prune_reinforcement_for_outline(
            section,
            policy=ReinforcementInvalidPolicy.DROP_INVALID_BARS,
        )
        assert report.removed_groups == 1
        assert report.removed_bars == 1
        assert section.rebar_groups == []
