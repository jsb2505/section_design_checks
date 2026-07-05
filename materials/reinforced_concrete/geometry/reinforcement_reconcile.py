from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from shapely.geometry import Point as ShapelyPoint

if TYPE_CHECKING:
    from .section import RCSection, RebarGroup


class ReinforcementInvalidPolicy(StrEnum):
    '''
    Policy for handling rebar groups when section boundaries are mutated.

    Attributes:
        ERROR: Aborts the operation and raises a ValueError.
        DROP_INVALID_BARS: Removes individual bars that fall outside.
        DROP_INVALID_GROUPS: Removes an entire group if any bar is invalid.
        ALLOW_INVALID: No action taken; bars remain in their original coords.
    '''
    ERROR = "error" 
    '''Raise if any bar becomes invalid'''

    DROP_INVALID_BARS = "drop_bars"
    '''Remove only offending bars, keep groups if any bars remain'''

    DROP_INVALID_GROUPS = "drop_groups"
    '''drop any group with ≥1 invalid bar'''

    ALLOW_INVALID = "allow_invalid"
    '''Allows invalid bars to remain outside of section boundary'''


@dataclass(frozen=True)
class ReinforcementUpdateReport:
    invalid_groups: int
    invalid_bars: int
    removed_groups: int
    removed_bars: int
    details: list[str]  # human-readable messages


def find_invalid_rebars(section: "RCSection") -> tuple[list[str], list[tuple[int, int]]]:
    """
    Find rebars (as discs) not fully covered by the section outline.

    Returns:
        details: list of readable strings
        invalid: list of (group_index, bar_index)
    """
    details: list[str] = []
    invalid: list[tuple[int, int]] = []

    poly = section.outline

    for gi, group in enumerate(section.rebar_groups):
        r = float(group.rebar.diameter) / 2.0
        for bi, pos in enumerate(group.positions):
            disc = ShapelyPoint(pos.x, pos.y).buffer(r)
            if not poly.covers(disc):
                invalid.append((gi, bi))
                details.append(
                    f"group[{gi}] '{group.layer_name}' bar[{bi}] "
                    f"(ϕ{group.rebar.diameter:g}) at ({pos.x:.1f}, {pos.y:.1f}) is outside outline"
                )

    return details, invalid


def prune_reinforcement_for_outline(
    section: "RCSection",
    policy: ReinforcementInvalidPolicy,
) -> ReinforcementUpdateReport:
    details, invalid = find_invalid_rebars(section)

    if not invalid:
        return ReinforcementUpdateReport(0, 0, 0, 0, [])
    
    invalid_groups_set = {gi for gi, _ in invalid}
    invalid_bars_count = len(invalid)
    invalid_groups_count = len(invalid_groups_set)

    if policy == ReinforcementInvalidPolicy.ALLOW_INVALID:
        # no mutation, just report
        invalid_groups_set = {gi for gi, _ in invalid}
        return ReinforcementUpdateReport(
            invalid_groups=invalid_groups_count,
            invalid_bars=invalid_bars_count,
            removed_groups=0,
            removed_bars=0,
            details=details,
        )

    if policy == ReinforcementInvalidPolicy.ERROR:
        msg = "Outline update made some reinforcement invalid:\n- " + "\n- ".join(details)
        raise ValueError(msg)

    removed_groups = 0
    removed_bars = 0

    if policy == ReinforcementInvalidPolicy.DROP_INVALID_GROUPS:
        new_groups: list[RebarGroup] = []
        for gi, group in enumerate(section.rebar_groups):
            if gi in invalid_groups_set:
                removed_groups += 1
            else:
                new_groups.append(group)
        section.rebar_groups = new_groups

    elif policy == ReinforcementInvalidPolicy.DROP_INVALID_BARS:
        invalid_by_group: dict[int, set[int]] = {}
        for gi, bi in invalid:
            invalid_by_group.setdefault(gi, set()).add(bi)

        # local import avoids circular import pain if section.py
        # later imports this module at top-level
        from .section import RebarGroup

        new_groups: list[RebarGroup] = []
        for gi, group in enumerate(section.rebar_groups):
            bad = invalid_by_group.get(gi, set())
            if not bad:
                new_groups.append(group)
                continue

            kept_positions = tuple(
                p for idx, p in enumerate(group.positions) if idx not in bad
            )
            removed_bars += len(group.positions) - len(kept_positions)

            if kept_positions:
                new_groups.append(
                    RebarGroup(
                        rebar=group.rebar,
                        positions=kept_positions,
                        layer_name=group.layer_name,
                    )
                )
            else:
                removed_groups += 1

        section.rebar_groups = new_groups

    else:
        raise ValueError(f"Unknown policy: {policy}")

    return ReinforcementUpdateReport(
        invalid_groups=invalid_groups_count,
        invalid_bars=invalid_bars_count,
        removed_groups=removed_groups,
        removed_bars=removed_bars,
        details=details,
    )


def reconcile_after_outline_change(
    section: "RCSection",
    *,
    policy: ReinforcementInvalidPolicy,
) -> ReinforcementUpdateReport:
    """
    Enforce reinforcement policy after outline/void coords have changed
    and the outline polygon has been (re)built.
    """
    return prune_reinforcement_for_outline(section, policy)


def update_outline(
    section: "RCSection",
    *,
    outline_coords: Any,
    voids_coords: Any | None = None,
    policy: ReinforcementInvalidPolicy = ReinforcementInvalidPolicy.ERROR,
) -> ReinforcementUpdateReport:
    """
    Update section geometry, then reconcile reinforcement according to policy.
    
    NOTE:
        If RCSection auto-reconciles inside __setattr__ on outline/void assignment,
        prefer calling section.update_outline(...) to keep this atomic.
    """
    method = getattr(section, "update_outline", None)
    if callable(method):
        # RCSection.update_outline should do: set coords (atomically) + rebuild outline + reconcile once
        return cast(
            "ReinforcementUpdateReport",
                method(
                outline_coords=outline_coords,
                voids_coords=voids_coords,
                reinforcement_policy=policy,
            ),
        )

    # Fallback: try to avoid double-reconcile if RCSection has the suspend flag
    had_flag = hasattr(section, "_suspend_outline_reconcile")
    if had_flag:
        setattr(section, "_suspend_outline_reconcile", True)

    try:
        section.outline_coords = outline_coords
        if voids_coords is not None:
            section.voids_coords = voids_coords

        if hasattr(section, "_invalidate_outline_cache"):
            section._invalidate_outline_cache()
        _ = section.outline  # force build/validate polygon

        return reconcile_after_outline_change(section, policy=policy)
    finally:
        if had_flag:
            setattr(section, "_suspend_outline_reconcile", False)
