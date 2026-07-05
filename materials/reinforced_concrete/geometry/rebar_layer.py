"""
Helper functions for creating rebar layers in RC sections.

Provides utilities for positioning rebars in common configurations:
- Linear layers (bottom/top/side)
- Perimeter reinforcement
- Custom patterns
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from materials.core.geometry import Point2D
from materials.reinforced_concrete.geometry.section import RebarGroup
from materials.reinforced_concrete.materials.rebar import Rebar


def create_linear_rebar_layer(
    rebar: Rebar,
    n_bars: int,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    layer_name: Optional[str] = None,
    omit_start: bool = False,
    omit_end: bool = False,
) -> RebarGroup:
    """
    Create a linear layer of evenly-spaced rebars between two points.

    Spacing is always calculated as if all *n_bars* are present (including
    the endpoints).  The ``omit_start`` / ``omit_end`` flags then drop the
    first / last bar from the returned group **without changing the spacing
    of the remaining bars**.

    This is useful when two layers share a corner point: pass the same
    coordinates and *n_bars* to both calls, using ``omit_end=True`` on
    one and ``omit_start=True`` on the other to avoid a duplicated bar.

    Notes:
        - If n_bars == 1, a single bar is placed at the midpoint.
        - If n_bars >= 2, bars are evenly spaced including the endpoints.

    Args:
        rebar: Rebar specification
        n_bars: Number of bars (minimum 1)
        start_point: (x, y) coordinates of first bar (mm)
        end_point: (x, y) coordinates of last bar (mm)
        layer_name: Optional layer identifier
        omit_start: If True, exclude the bar at *start_point*
        omit_end: If True, exclude the bar at *end_point*

    Returns:
        RebarGroup with linear arrangement

    Example:
        >>> bar = Rebar(diameter=16, grade="B500B")
        >>> layer = create_linear_rebar_layer(bar, 4, (50, 50), (250, 50))
        >>> len(layer.positions)
        4
        >>> # Around a corner – omit the shared endpoint
        >>> leg1 = create_linear_rebar_layer(bar, 4, (50, 50), (250, 50), omit_end=True)
        >>> leg2 = create_linear_rebar_layer(bar, 4, (250, 50), (250, 250), omit_start=True)
        >>> len(leg1.positions), len(leg2.positions)
        (3, 3)
    """
    if n_bars < 1:
        raise ValueError("Number of bars must be at least 1")

    x0, y0 = start_point
    x1, y1 = end_point

    if n_bars == 1:
        # Single bar at midpoint
        positions = [Point2D(x=(x0 + x1) / 2.0, y=(y0 + y1) / 2.0)]
    else:
        # Multiple bars evenly spaced (including endpoints)
        x_coords = np.linspace(x0, x1, n_bars, dtype=float)
        y_coords = np.linspace(y0, y1, n_bars, dtype=float)
        positions = [Point2D(x=float(x), y=float(y)) for x, y in zip(x_coords, y_coords)]

    # Trim after computing full spacing
    if omit_start and len(positions) > 0:
        positions = positions[1:]
    if omit_end and len(positions) > 0:
        positions = positions[:-1]

    if len(positions) == 0:
        raise ValueError(
            "All bars were omitted — n_bars is too small for the requested omit_start/omit_end combination"
        )

    return RebarGroup(rebar=rebar, positions=tuple(positions), layer_name=layer_name)


def create_rectangular_perimeter_rebars(
    rebar: Rebar,
    width: float,
    height: float,
    cover: float,
    n_bars_width: int,
    n_bars_height: int,
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
) -> List[RebarGroup]:
    """
    Create perimeter reinforcement for a rectangular section.

    Corner bars are included in top/bottom layers only, not duplicated in side layers.

    Hook Reference Convention (must match section creation):
        hook_ref=0: Center (origin at center of rectangle)
        hook_ref=1: Bottom-left corner (section in +X, +Y quadrant) - DEFAULT
        hook_ref=2: Bottom-right corner (section in -X, +Y quadrant)
        hook_ref=3: Top-right corner (section in -X, -Y quadrant)
        hook_ref=4: Top-left corner (section in +X, -Y quadrant)

    Args:
        rebar: Rebar specification
        width: Section width (mm)
        height: Section height (mm)
        cover: Cover to rebar outer surface (mm)
        n_bars_width: Number of bars along width (top and bottom layers)
        n_bars_height: Number of bars along height (left and right sides, excluding corners)
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point (0=center, 1=bottom-left, etc.). Default: 1

    Returns:
        List of RebarGroups (bottom, top, left, right)

    Example:
        >>> bar = Rebar(diameter=12)
        >>> # Section with bottom-left at (0, 0)
        >>> groups = create_rectangular_perimeter_rebars(
        ...     bar, 300, 500, 30, n_bars_width=3, n_bars_height=2
        ... )
        >>> len(groups)
        4

    Note:
        Cover is measured to the outer surface of the rebar, not the centreline.
        Bar centrelines are positioned at cover + diameter/2 from section edges.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    if cover < 0:
        raise ValueError("cover must be >= 0")
    if n_bars_width < 0 or n_bars_height < 0:
        raise ValueError("n_bars_width and n_bars_height must be >= 0")

    ox, oy = origin

    # Calculate center point based on hook_ref (same logic as section creation)
    if hook_ref == 0:
        # Center
        cx, cy = ox, oy
    elif hook_ref == 1:
        # Bottom-left corner (section in +X, +Y)
        cx = ox + width / 2.0
        cy = oy + height / 2.0
    elif hook_ref == 2:
        # Bottom-right corner (section in -X, +Y)
        cx = ox - width / 2.0
        cy = oy + height / 2.0
    elif hook_ref == 3:
        # Top-right corner (section in -X, -Y)
        cx = ox - width / 2.0
        cy = oy - height / 2.0
    elif hook_ref == 4:
        # Top-left corner (section in +X, -Y)
        cx = ox + width / 2.0
        cy = oy - height / 2.0
    else:
        raise ValueError(f"hook_ref must be 0, 1, 2, 3, or 4, got {hook_ref}")

    half_width = width / 2.0
    half_height = height / 2.0

    # Cover is to outer surface, so centreline is at cover + radius
    centreline_offset = cover + rebar.diameter / 2.0

    # Geometry feasibility checks
    if 2.0 * centreline_offset > width or 2.0 * centreline_offset > height:
        raise ValueError(
            "cover + bar radius is too large for the section dimensions "
            "(bars would lie outside the section)."
        )

    groups: List[RebarGroup] = []

    # Bottom layer
    if n_bars_width >= 1:
        groups.append(
            create_linear_rebar_layer(
                rebar=rebar,
                n_bars=n_bars_width,
                start_point=(cx - half_width + centreline_offset, cy - half_height + centreline_offset),
                end_point=(cx + half_width - centreline_offset, cy - half_height + centreline_offset),
                layer_name="bottom",
            )
        )

    # Top layer
    if n_bars_width >= 1:
        groups.append(
            create_linear_rebar_layer(
                rebar=rebar,
                n_bars=n_bars_width,
                start_point=(cx - half_width + centreline_offset, cy + half_height - centreline_offset),
                end_point=(cx + half_width - centreline_offset, cy + half_height - centreline_offset),
                layer_name="top",
            )
        )

    # Side bars (excluding corners - they're in top/bottom)
    if n_bars_height >= 1:
        available_height = height - 2.0 * centreline_offset

        # Place side bars evenly between the top and bottom corner levels.
        # For n=1 this yields y_start == y_end (and the single bar lands at mid-height).
        step = available_height / (n_bars_height + 1)
        y_start = (cy - half_height + centreline_offset) + step
        y_end = (cy + half_height - centreline_offset) - step

        groups.append(
            create_linear_rebar_layer(
                rebar=rebar,
                n_bars=n_bars_height,
                start_point=(cx - half_width + centreline_offset, y_start),
                end_point=(cx - half_width + centreline_offset, y_end),
                layer_name="left",
            )
        )

        groups.append(
            create_linear_rebar_layer(
                rebar=rebar,
                n_bars=n_bars_height,
                start_point=(cx + half_width - centreline_offset, y_start),
                end_point=(cx + half_width - centreline_offset, y_end),
                layer_name="right",
            )
        )

    return groups


def create_circular_perimeter_rebars(
    rebar: Rebar,
    diameter: float,
    cover: float,
    n_bars: int,
    origin: Tuple[float, float] = (0.0, 0.0),
    start_angle: float = 0.0,
) -> RebarGroup:
    """
    Create perimeter reinforcement for a circular section.

    Args:
        rebar: Rebar specification
        diameter: Section diameter (mm)
        cover: Cover to rebar outer surface (mm)
        n_bars: Number of bars around perimeter (minimum 3)
        origin: Centre of section (default: (0, 0))
        start_angle: Starting angle in degrees (0 = right, counterclockwise)

    Returns:
        RebarGroup with circular arrangement

    Example:
        >>> bar = Rebar(diameter=16)
        >>> group = create_circular_perimeter_rebars(bar, 400, 40, 8)
        >>> len(group.positions)
        8

    Note:
        Cover is measured to the outer surface of the rebar, not the centreline.
        Bar centrelines are positioned at radius = (section_diameter/2) - cover - (bar_diameter/2).
    """
    if diameter <= 0:
        raise ValueError("diameter must be > 0")
    if cover < 0:
        raise ValueError("cover must be >= 0")
    if n_bars < 3:
        raise ValueError("Circular perimeter requires at least 3 bars")

    cx, cy = origin

    # Radius to bar centreline = section radius - cover - bar radius
    radius = diameter / 2.0 - cover - rebar.diameter / 2.0
    if radius <= 0:
        raise ValueError(
            "cover + bar radius is too large for the section diameter "
            "(bars would lie on/inside the centre)."
        )

    start = np.radians(start_angle)
    angles = np.linspace(start, start + 2.0 * np.pi, n_bars, endpoint=False, dtype=float)

    positions = [
        Point2D(x=float(cx + radius * np.cos(a)), y=float(cy + radius * np.sin(a)))
        for a in angles
    ]

    return RebarGroup(rebar=rebar, positions=tuple(positions), layer_name="perimeter")


def create_custom_rebar_layer(
    rebar: Rebar,
    positions: Sequence[Tuple[float, float]],
    layer_name: Optional[str] = None,
) -> RebarGroup:
    """
    Create a rebar layer at custom positions.

    Args:
        rebar: Rebar specification
        positions: Sequence of (x, y) coordinates (mm)
        layer_name: Optional layer identifier

    Returns:
        RebarGroup with custom positions

    Example:
        >>> bar = Rebar(diameter=20, grade="B500B")
        >>> layer = create_custom_rebar_layer(
        ...     bar,
        ...     [(50, 50), (150, 50), (250, 50), (125, 100)]
        ... )
    """
    point_positions = [Point2D(x=float(x), y=float(y)) for x, y in positions]
    return RebarGroup(rebar=rebar, positions=tuple(point_positions), layer_name=layer_name)
