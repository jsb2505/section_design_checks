"""
Helper functions for creating rebar layers in RC sections.

Provides utilities for positioning rebars in common configurations:
- Linear layers (bottom/top/side)
- Perimeter reinforcement
- Custom patterns
"""

from typing import List, Tuple, Optional, Literal
import numpy as np
from materials.core.geometry import Point2D
from materials.reinforced_concrete.materials.rebar import Rebar
from materials.reinforced_concrete.geometry.section import RebarGroup


def create_linear_rebar_layer(
    rebar: Rebar,
    n_bars: int,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    layer_name: Optional[str] = None,
) -> RebarGroup:
    """
    Create a linear layer of evenly-spaced rebars between two points.

    Args:
        rebar: Rebar specification
        n_bars: Number of bars (minimum 2)
        start_point: (x, y) coordinates of first bar (mm)
        end_point: (x, y) coordinates of last bar (mm)
        layer_name: Optional layer identifier

    Returns:
        RebarGroup with linear arrangement

    Example:
        >>> bar = Rebar(diameter=16, grade="B500B")
        >>> layer = create_linear_rebar_layer(bar, 4, (50, 50), (250, 50))
        >>> len(layer.positions)
        4
    """
    if n_bars < 1:
        raise ValueError("Number of bars must be at least 1")

    if n_bars == 1:
        # Single bar at midpoint
        x = (start_point[0] + end_point[0]) / 2.0
        y = (start_point[1] + end_point[1]) / 2.0
        positions = [Point2D(x=x, y=y)]
    else:
        # Multiple bars evenly spaced
        x_coords = np.linspace(start_point[0], end_point[0], n_bars)
        y_coords = np.linspace(start_point[1], end_point[1], n_bars)
        positions = [Point2D(x=float(x), y=float(y)) for x, y in zip(x_coords, y_coords)]

    return RebarGroup(
        rebar=rebar,
        positions=positions,
        layer_name=layer_name,
    )


def create_rectangular_perimeter_rebars(
    rebar: Rebar,
    width: float,
    height: float,
    cover: float,
    n_bars_width: int,
    n_bars_height: int,
    origin: Tuple[float, float] = (0.0, 0.0),
    include_corners_twice: bool = False,
) -> List[RebarGroup]:
    """
    Create perimeter reinforcement for a rectangular section.

    Args:
        rebar: Rebar specification
        width: Section width (mm)
        height: Section height (mm)
        cover: Cover to rebar centerline (mm)
        n_bars_width: Number of bars along width (top and bottom)
        n_bars_height: Number of bars along height (left and right sides, excluding corners)
        origin: Bottom-left corner of section
        include_corners_twice: If True, corner bars appear in both horizontal and vertical layers

    Returns:
        List of RebarGroups (bottom, top, left, right)

    Example:
        >>> bar = Rebar(diameter=12, grade="B500B")
        >>> groups = create_rectangular_perimeter_rebars(
        ...     bar, 300, 500, 30, n_bars_width=3, n_bars_height=2
        ... )
        >>> len(groups)
        4
    """
    x0, y0 = origin

    groups = []

    # Bottom layer
    if n_bars_width >= 1:
        bottom = create_linear_rebar_layer(
            rebar=rebar,
            n_bars=n_bars_width,
            start_point=(x0 + cover, y0 + cover),
            end_point=(x0 + width - cover, y0 + cover),
            layer_name="bottom",
        )
        groups.append(bottom)

    # Top layer
    if n_bars_width >= 1:
        top = create_linear_rebar_layer(
            rebar=rebar,
            n_bars=n_bars_width,
            start_point=(x0 + cover, y0 + height - cover),
            end_point=(x0 + width - cover, y0 + height - cover),
            layer_name="top",
        )
        groups.append(top)

    # Side bars (excluding corners unless include_corners_twice is True)
    if n_bars_height >= 1:
        # Left side
        if include_corners_twice:
            # Include corners
            n_left = n_bars_height + 2
            y_start = y0 + cover
            y_end = y0 + height - cover
        else:
            # Exclude corners (they're in top/bottom)
            n_left = n_bars_height
            # Position between top and bottom bars
            if n_left >= 1:
                y_start = y0 + cover + (height - 2*cover) / (n_bars_height + 1)
                y_end = y0 + height - cover - (height - 2*cover) / (n_bars_height + 1)
            else:
                y_start = y0 + cover
                y_end = y0 + height - cover

        if n_left > 0:
            left = create_linear_rebar_layer(
                rebar=rebar,
                n_bars=n_left,
                start_point=(x0 + cover, y_start),
                end_point=(x0 + cover, y_end),
                layer_name="left",
            )
            groups.append(left)

        # Right side
        n_right = n_left
        if n_right > 0:
            right = create_linear_rebar_layer(
                rebar=rebar,
                n_bars=n_right,
                start_point=(x0 + width - cover, y_start),
                end_point=(x0 + width - cover, y_end),
                layer_name="right",
            )
            groups.append(right)

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
        cover: Cover to rebar centerline (mm)
        n_bars: Number of bars around perimeter
        origin: Center of section
        start_angle: Starting angle in degrees (0 = right, counterclockwise)

    Returns:
        RebarGroup with circular arrangement

    Example:
        >>> bar = Rebar(diameter=16, grade="B500B")
        >>> group = create_circular_perimeter_rebars(bar, 400, 40, 8)
        >>> len(group.positions)
        8
    """
    if n_bars < 3:
        raise ValueError("Circular perimeter requires at least 3 bars")

    cx, cy = origin
    radius = diameter / 2.0 - cover

    angles = np.linspace(
        np.radians(start_angle),
        np.radians(start_angle) + 2 * np.pi,
        n_bars,
        endpoint=False
    )

    positions = [
        Point2D(x=cx + radius * np.cos(angle), y=cy + radius * np.sin(angle))
        for angle in angles
    ]

    return RebarGroup(
        rebar=rebar,
        positions=positions,
        layer_name="perimeter",
    )


def create_custom_rebar_layer(
    rebar: Rebar,
    positions: List[Tuple[float, float]],
    layer_name: Optional[str] = None,
) -> RebarGroup:
    """
    Create a rebar layer at custom positions.

    Args:
        rebar: Rebar specification
        positions: List of (x, y) coordinates (mm)
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
    point_positions = [Point2D(x=x, y=y) for x, y in positions]

    return RebarGroup(
        rebar=rebar,
        positions=point_positions,
        layer_name=layer_name,
    )


def create_single_rebar(
    rebar: Rebar,
    position: Tuple[float, float],
    layer_name: Optional[str] = None,
) -> RebarGroup:
    """
    Create a single rebar at a specific position.

    Args:
        rebar: Rebar specification
        position: (x, y) coordinates (mm)
        layer_name: Optional layer identifier

    Returns:
        RebarGroup with single bar
    """
    return RebarGroup(
        rebar=rebar,
        positions=[Point2D(x=position[0], y=position[1])],
        layer_name=layer_name,
    )
