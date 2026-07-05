"""
Utility functions for creating common RC section geometries.

All functions return an RCSection instance with the appropriate outline
(and voids where applicable). No rebar placement is done here.

Hook Reference Convention (shared by all functions):
    hook_ref=0: Centre (origin at centroid of bounding box)
    hook_ref=1: Bottom-left corner (section in +X, +Y quadrant) — DEFAULT
    hook_ref=2: Bottom-right corner (section in -X, +Y quadrant)
    hook_ref=3: Top-right corner (section in -X, -Y quadrant)
    hook_ref=4: Top-left corner (section in +X, -Y quadrant)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from materials.core.geometry import Point2D
from materials.reinforced_concrete.geometry.section import RCSection


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _resolve_hook_ref(
    width: float,
    height: float,
    origin: Tuple[float, float],
    hook_ref: int,
) -> Tuple[float, float]:
    """Convert (origin, hook_ref) into the bottom-left corner (x0, y0).

    ``width`` and ``height`` are the overall bounding-box dimensions.
    """
    ox, oy = origin

    if hook_ref == 0:
        # Origin is at centre → bottom-left = origin − half dims
        return ox - width / 2.0, oy - height / 2.0
    elif hook_ref == 1:
        # Origin is already at bottom-left
        return ox, oy
    elif hook_ref == 2:
        # Origin is at bottom-right
        return ox - width, oy
    elif hook_ref == 3:
        # Origin is at top-right
        return ox - width, oy - height
    elif hook_ref == 4:
        # Origin is at top-left
        return ox, oy - height
    else:
        raise ValueError(f"hook_ref must be 0, 1, 2, 3, or 4, got {hook_ref}")


def _points_to_outline(coords: Sequence[Tuple[float, float]]) -> Tuple[Point2D, ...]:
    """Convert a sequence of (x, y) tuples to a tuple of Point2D."""
    return tuple(Point2D(x=x, y=y) for x, y in coords)


# ---------------------------------------------------------------------------
# Rectangular section (moved from section.py)
# ---------------------------------------------------------------------------

def create_rectangular_section(
    width: float,
    height: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create a rectangular RC section.

    Hook Reference Convention:
        hook_ref=0: Centre (origin at centre of rectangle)
        hook_ref=1: Bottom-left corner (section in +X, +Y quadrant) - DEFAULT
        hook_ref=2: Bottom-right corner (section in -X, +Y quadrant)
        hook_ref=3: Top-right corner (section in -X, -Y quadrant)
        hook_ref=4: Top-left corner (section in +X, -Y quadrant)

    Args:
        width: Section width (mm)
        height: Section height (mm)
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point (0=centre, 1=bottom-left, 2=bottom-right,
                  3=top-right, 4=top-left). Default: 1 (bottom-left)
        section_name: Optional section name

    Returns:
        RCSection with rectangular outline

    Examples:
        >>> # Section with bottom-left at (0, 0), extends to (300, 500)
        >>> section = create_rectangular_section(300, 500)

        >>> # Section centred at (0, 0)
        >>> section = create_rectangular_section(300, 500, hook_ref=0)

        >>> # Section with bottom-left at (100, 50)
        >>> section = create_rectangular_section(300, 500, origin=(100, 50))
    """
    x0, y0 = _resolve_hook_ref(width, height, origin, hook_ref)

    coords = [
        (x0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height),
        (x0, y0 + height),
    ]

    return RCSection(
        outline_coords=_points_to_outline(coords),
        section_name=section_name or f"Rect {width}×{height}",
    )


# ---------------------------------------------------------------------------
# Circular section (moved from section.py)
# ---------------------------------------------------------------------------

def create_circular_section(
    diameter: float,
    n_points: int = 60,
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create a circular RC section.

    Hook Reference Convention:
        hook_ref=0: Centre (origin at centre of circle)
        hook_ref=1: Bottom-left of bounding box (section in +X, +Y quadrant) - DEFAULT
        hook_ref=2: Bottom-right of bounding box (section in -X, +Y quadrant)
        hook_ref=3: Top-right of bounding box (section in -X, -Y quadrant)
        hook_ref=4: Top-left of bounding box (section in +X, -Y quadrant)

    Args:
        diameter: Section diameter (mm)
        n_points: Number of points to approximate circle (default: 60)
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point (0=centre, 1=bottom-left, etc.). Default: 1
        section_name: Optional section name (default: None)

    Returns:
        RCSection with circular outline

    Examples:
        >>> # Circle with bounding box bottom-left at (0, 0)
        >>> section = create_circular_section(400)

        >>> # Circle centred at (0, 0)
        >>> section = create_circular_section(400, hook_ref=0)
    """
    radius = diameter / 2.0
    # _resolve_hook_ref gives bottom-left of bounding box; centre = BL + radius
    x0, y0 = _resolve_hook_ref(diameter, diameter, origin, hook_ref)
    cx = x0 + radius
    cy = y0 + radius

    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=float)
    coords = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]

    return RCSection(
        outline_coords=_points_to_outline(coords),
        section_name=section_name or f"Circular Ø{diameter}",
    )


# ---------------------------------------------------------------------------
# T-beam section
# ---------------------------------------------------------------------------

def create_t_beam_section(
    b_f: float,
    h_f: float,
    b_w: float,
    h_w: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create a T-beam RC section (flange at top).

    ::

          b_f
      |<--------->|
      +-----------+  ─┬─ h_f
      |   flange  |   │
      +--+-----+--+  ─┘
         | web |      h_w
         |     |
         +-----+
          b_w

    Args:
        b_f: Flange width (mm)
        h_f: Flange thickness (mm)
        b_w: Web width (mm)
        h_w: Web height (mm)
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point. Default: 1 (bottom-left of bounding box)
        section_name: Optional section name

    Returns:
        RCSection with T-beam outline
    """
    if b_w > b_f:
        raise ValueError(f"Web width b_w={b_w} must be <= flange width b_f={b_f}")

    total_height = h_f + h_w
    x0, y0 = _resolve_hook_ref(b_f, total_height, origin, hook_ref)

    # Web offset from left edge of bounding box (centred under flange)
    web_left = x0 + (b_f - b_w) / 2.0
    web_right = web_left + b_w

    # 8-point polygon, CCW from bottom-left of web
    coords = [
        (web_left, y0),              # bottom-left of web
        (web_right, y0),             # bottom-right of web
        (web_right, y0 + h_w),      # top-right of web / bottom-right of flange step
        (x0 + b_f, y0 + h_w),       # bottom-right of flange
        (x0 + b_f, y0 + total_height),  # top-right of flange
        (x0, y0 + total_height),    # top-left of flange
        (x0, y0 + h_w),             # bottom-left of flange
        (web_left, y0 + h_w),       # top-left of web / bottom-left of flange step
    ]

    return RCSection(
        outline_coords=_points_to_outline(coords),
        section_name=section_name or f"T-beam {b_f}×{h_f}/{b_w}×{h_w}",
    )


# ---------------------------------------------------------------------------
# Inverted T-beam section
# ---------------------------------------------------------------------------

def create_inverted_t_beam_section(
    b_f: float,
    h_f: float,
    b_w: float,
    h_w: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create an inverted T-beam RC section (flange at bottom).

    ::

          b_w
         +-----+
         | web |      h_w
         |     |
      +--+-----+--+  ─┬─ h_f
      |   flange  |   │
      +-----------+  ─┘
          b_f

    Args:
        b_f: Flange width (mm)
        h_f: Flange thickness (mm)
        b_w: Web width (mm)
        h_w: Web height (mm)
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point. Default: 1 (bottom-left of bounding box)
        section_name: Optional section name

    Returns:
        RCSection with inverted T-beam outline
    """
    if b_w > b_f:
        raise ValueError(f"Web width b_w={b_w} must be <= flange width b_f={b_f}")

    total_height = h_f + h_w
    x0, y0 = _resolve_hook_ref(b_f, total_height, origin, hook_ref)

    web_left = x0 + (b_f - b_w) / 2.0
    web_right = web_left + b_w

    # 8-point polygon, CCW from bottom-left of flange
    coords = [
        (x0, y0),                   # bottom-left of flange
        (x0 + b_f, y0),             # bottom-right of flange
        (x0 + b_f, y0 + h_f),      # top-right of flange
        (web_right, y0 + h_f),      # bottom-right of web step
        (web_right, y0 + total_height),  # top-right of web
        (web_left, y0 + total_height),   # top-left of web
        (web_left, y0 + h_f),       # bottom-left of web step
        (x0, y0 + h_f),             # top-left of flange
    ]

    return RCSection(
        outline_coords=_points_to_outline(coords),
        section_name=section_name or f"Inv T-beam {b_f}×{h_f}/{b_w}×{h_w}",
    )


# ---------------------------------------------------------------------------
# I-beam section
# ---------------------------------------------------------------------------

def create_i_beam_section(
    b_f_top: float,
    h_f_top: float,
    b_f_bot: float,
    h_f_bot: float,
    b_w: float,
    h_w: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create an I-beam RC section with potentially asymmetric flanges.

    ::

      +-------------+  ─┬─ h_f_top
      | top flange  |   │
      +--+-------+--+  ─┘
         |  web  |      h_w
      +--+-------+--+  ─┬─ h_f_bot
      | bot flange  |   │
      +-------------+  ─┘

    Args:
        b_f_top: Top flange width (mm)
        h_f_top: Top flange thickness (mm)
        b_f_bot: Bottom flange width (mm)
        h_f_bot: Bottom flange thickness (mm)
        b_w: Web width (mm)
        h_w: Web height (mm)
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point. Default: 1 (bottom-left of bounding box)
        section_name: Optional section name

    Returns:
        RCSection with I-beam outline
    """
    if b_w > b_f_top:
        raise ValueError(f"Web width b_w={b_w} must be <= top flange width b_f_top={b_f_top}")
    if b_w > b_f_bot:
        raise ValueError(f"Web width b_w={b_w} must be <= bottom flange width b_f_bot={b_f_bot}")

    total_width = max(b_f_top, b_f_bot)
    total_height = h_f_bot + h_w + h_f_top
    x0, y0 = _resolve_hook_ref(total_width, total_height, origin, hook_ref)

    # Centre each element horizontally within the bounding box
    centre_x = x0 + total_width / 2.0

    bot_left = centre_x - b_f_bot / 2.0
    bot_right = centre_x + b_f_bot / 2.0
    web_left = centre_x - b_w / 2.0
    web_right = centre_x + b_w / 2.0
    top_left = centre_x - b_f_top / 2.0
    top_right = centre_x + b_f_top / 2.0

    y_bot_flange_top = y0 + h_f_bot
    y_web_top = y0 + h_f_bot + h_w
    y_top = y0 + total_height

    # 12-point polygon, CCW from bottom-left of bottom flange
    coords = [
        (bot_left, y0),                   # 1  bottom-left of bot flange
        (bot_right, y0),                  # 2  bottom-right of bot flange
        (bot_right, y_bot_flange_top),    # 3  top-right of bot flange
        (web_right, y_bot_flange_top),    # 4  bottom-right of web
        (web_right, y_web_top),           # 5  top-right of web
        (top_right, y_web_top),           # 6  bottom-right of top flange
        (top_right, y_top),               # 7  top-right of top flange
        (top_left, y_top),                # 8  top-left of top flange
        (top_left, y_web_top),            # 9  bottom-left of top flange
        (web_left, y_web_top),            # 10 top-left of web
        (web_left, y_bot_flange_top),     # 11 bottom-left of web
        (bot_left, y_bot_flange_top),     # 12 top-left of bot flange
    ]

    return RCSection(
        outline_coords=_points_to_outline(coords),
        section_name=section_name or f"I-beam {b_f_top}/{b_w}/{b_f_bot}",
    )


# ---------------------------------------------------------------------------
# Box section
# ---------------------------------------------------------------------------

def create_box_section(
    width: float,
    height: float,
    t_web: float,
    t_flange_top: float,
    t_flange_bot: Optional[float] = None,
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create a box (hollow rectangular) RC section.

    Outer rectangle with a single rectangular void inside.

    Args:
        width: Overall section width (mm)
        height: Overall section height (mm)
        t_web: Web (side wall) thickness (mm)
        t_flange_top: Top flange thickness (mm)
        t_flange_bot: Bottom flange thickness (mm). Defaults to t_flange_top.
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point. Default: 1 (bottom-left of bounding box)
        section_name: Optional section name

    Returns:
        RCSection with box outline and rectangular void
    """
    if t_flange_bot is None:
        t_flange_bot = t_flange_top

    if 2 * t_web >= width:
        raise ValueError(f"2 * t_web ({2*t_web}) must be < width ({width})")
    if t_flange_top + t_flange_bot >= height:
        raise ValueError(
            f"t_flange_top + t_flange_bot ({t_flange_top + t_flange_bot}) must be < height ({height})"
        )

    x0, y0 = _resolve_hook_ref(width, height, origin, hook_ref)

    # Outer rectangle
    outer = [
        (x0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height),
        (x0, y0 + height),
    ]

    # Inner void
    void = [
        (x0 + t_web, y0 + t_flange_bot),
        (x0 + width - t_web, y0 + t_flange_bot),
        (x0 + width - t_web, y0 + height - t_flange_top),
        (x0 + t_web, y0 + height - t_flange_top),
    ]

    return RCSection(
        outline_coords=_points_to_outline(outer),
        voids_coords=[_points_to_outline(void)],
        section_name=section_name or f"Box {width}×{height}",
    )


# ---------------------------------------------------------------------------
# Voided deck section
# ---------------------------------------------------------------------------

def create_voided_deck_section(
    width: float,
    height: float,
    void_diameter: float,
    n_voids: int,
    void_spacing: Optional[float] = None,
    n_points: int = 32,
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create a rectangular deck slab with circular voids at mid-height.

    If ``void_spacing`` is None the voids are distributed evenly across the width.

    Args:
        width: Overall slab width (mm)
        height: Overall slab height (mm)
        void_diameter: Diameter of each circular void (mm)
        n_voids: Number of circular voids
        void_spacing: Centre-to-centre spacing of voids (mm). Auto if None.
        n_points: Points per circular void polygon (default: 32)
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point. Default: 1 (bottom-left of bounding box)
        section_name: Optional section name

    Returns:
        RCSection with rectangular outline and circular voids
    """
    if n_voids < 1:
        raise ValueError(f"n_voids must be >= 1, got {n_voids}")
    if void_diameter >= height:
        raise ValueError(f"void_diameter ({void_diameter}) must be < height ({height})")

    radius = void_diameter / 2.0
    x0, y0 = _resolve_hook_ref(width, height, origin, hook_ref)

    # Calculate void centre positions
    if void_spacing is None:
        spacing = width / (n_voids + 1)
    else:
        spacing = void_spacing

    cy_void = y0 + height / 2.0  # mid-height

    void_centres_x: List[float] = []
    if void_spacing is None:
        for i in range(n_voids):
            void_centres_x.append(x0 + spacing * (i + 1))
    else:
        # Centre the group
        total_span = spacing * (n_voids - 1) if n_voids > 1 else 0.0
        start_x = x0 + (width - total_span) / 2.0
        for i in range(n_voids):
            void_centres_x.append(start_x + spacing * i)

    # Validate voids fit
    for cx_v in void_centres_x:
        if cx_v - radius < x0 or cx_v + radius > x0 + width:
            raise ValueError(
                f"Void at x={cx_v - x0:.1f} with diameter {void_diameter} exceeds slab width"
            )

    # Outer rectangle
    outer = [
        (x0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height),
        (x0, y0 + height),
    ]

    # Circular voids
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=float)
    voids: List[Tuple[Point2D, ...]] = []
    for cx_v in void_centres_x:
        circle = [
            (cx_v + radius * np.cos(a), cy_void + radius * np.sin(a))
            for a in angles
        ]
        voids.append(_points_to_outline(circle))

    return RCSection(
        outline_coords=_points_to_outline(outer),
        voids_coords=voids,
        section_name=section_name or f"Voided deck {width}×{height} ({n_voids}×Ø{void_diameter})",
    )


# ---------------------------------------------------------------------------
# Channel section (U-shape)
# ---------------------------------------------------------------------------

def create_channel_section(
    width: float,
    height: float,
    t_web: float,
    t_flange: float,
    open_side: str = "top",
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create a channel (U-shape) RC section.

    ``open_side="top"`` gives a standard U (open at top).
    ``open_side="bottom"`` gives an inverted U (open at bottom).

    Args:
        width: Overall section width (mm)
        height: Overall section height (mm)
        t_web: Web (side wall) thickness (mm)
        t_flange: Flange (base) thickness (mm)
        open_side: ``"top"`` or ``"bottom"``
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point. Default: 1 (bottom-left of bounding box)
        section_name: Optional section name

    Returns:
        RCSection with channel outline
    """
    if 2 * t_web >= width:
        raise ValueError(f"2 * t_web ({2*t_web}) must be < width ({width})")
    if t_flange >= height:
        raise ValueError(f"t_flange ({t_flange}) must be < height ({height})")
    if open_side not in ("top", "bottom"):
        raise ValueError(f"open_side must be 'top' or 'bottom', got '{open_side}'")

    x0, y0 = _resolve_hook_ref(width, height, origin, hook_ref)

    if open_side == "top":
        # U-shape: base at bottom, open at top
        # 8-point polygon CCW
        coords = [
            (x0, y0),                                 # bottom-left outer
            (x0 + width, y0),                          # bottom-right outer
            (x0 + width, y0 + height),                 # top-right outer
            (x0 + width - t_web, y0 + height),         # top-right inner
            (x0 + width - t_web, y0 + t_flange),       # inner bottom-right
            (x0 + t_web, y0 + t_flange),               # inner bottom-left
            (x0 + t_web, y0 + height),                 # top-left inner
            (x0, y0 + height),                         # top-left outer
        ]
    else:
        # Inverted U: base at top, open at bottom
        coords = [
            (x0, y0),                                  # bottom-left outer
            (x0 + t_web, y0),                          # bottom-left inner
            (x0 + t_web, y0 + height - t_flange),      # inner top-left
            (x0 + width - t_web, y0 + height - t_flange),  # inner top-right
            (x0 + width - t_web, y0),                  # bottom-right inner
            (x0 + width, y0),                          # bottom-right outer
            (x0 + width, y0 + height),                 # top-right outer
            (x0, y0 + height),                         # top-left outer
        ]

    return RCSection(
        outline_coords=_points_to_outline(coords),
        section_name=section_name or f"Channel {width}×{height}",
    )


# ---------------------------------------------------------------------------
# Trapezoidal section
# ---------------------------------------------------------------------------

def create_trapezoidal_section(
    b_top: float,
    b_bot: float,
    height: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create a trapezoidal RC section.

    Both widths are centred on the same vertical axis.

    ::

           b_top
        +--------+
       /          \\     height
      /            \\
     +--------------+
          b_bot

    Args:
        b_top: Width at top (mm)
        b_bot: Width at bottom (mm)
        height: Section height (mm)
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point. Default: 1 (bottom-left of bounding box)
        section_name: Optional section name

    Returns:
        RCSection with trapezoidal outline
    """
    total_width = max(b_top, b_bot)
    x0, y0 = _resolve_hook_ref(total_width, height, origin, hook_ref)

    centre_x = x0 + total_width / 2.0

    coords = [
        (centre_x - b_bot / 2.0, y0),               # bottom-left
        (centre_x + b_bot / 2.0, y0),               # bottom-right
        (centre_x + b_top / 2.0, y0 + height),      # top-right
        (centre_x - b_top / 2.0, y0 + height),      # top-left
    ]

    return RCSection(
        outline_coords=_points_to_outline(coords),
        section_name=section_name or f"Trapezoid {b_top}/{b_bot}×{height}",
    )
