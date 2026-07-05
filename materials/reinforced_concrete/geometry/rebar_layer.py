"""
Helper functions for creating rebar layers in RC sections.

Provides utilities for positioning rebars in common configurations:
- Linear layers (bottom/top/side)
- Perimeter reinforcement
- Custom patterns
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from shapely.geometry import Point as ShapelyPoint

from materials.core.geometry import Point2D
from materials.reinforced_concrete.geometry.section import RebarGroup
from materials.reinforced_concrete.materials.rebar import Rebar

if TYPE_CHECKING:
    from materials.reinforced_concrete.geometry.section import RCSection


_FACE_NAMES: tuple[str, ...] = ("top", "bottom", "left", "right")
_GEOM_TOL = 1e-9


def _normalise_face(face: str) -> Literal["top", "bottom", "left", "right"]:
    """Validate and normalise face selector."""
    face_norm = face.strip().lower()
    if face_norm not in _FACE_NAMES:
        raise ValueError("face must be one of: 'top', 'bottom', 'left', 'right'")
    return cast(Literal["top", "bottom", "left", "right"], face_norm)


def _resolve_face_segment(
    section: RCSection, face: Literal["top", "bottom", "left", "right"]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Return the boundary segment that best represents a requested section face.

    Selection is based on the segment midpoint coordinate in the face direction,
    with segment length used as a tie-breaker.
    """
    coords = list(section.outline.exterior.coords)
    best_segment: tuple[tuple[float, float], tuple[float, float]] | None = None
    best_key: tuple[float, float] | None = None

    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        length = float(np.hypot(dx, dy))
        if length <= _GEOM_TOL:
            continue

        mid_x = 0.5 * float(x0 + x1)
        mid_y = 0.5 * float(y0 + y1)

        if face == "top":
            face_score = mid_y
        elif face == "bottom":
            face_score = -mid_y
        elif face == "right":
            face_score = mid_x
        else:  # face == "left"
            face_score = -mid_x

        key = (face_score, length)
        if best_key is None or key > best_key:
            best_key = key
            best_segment = ((float(x0), float(y0)), (float(x1), float(y1)))

    if best_segment is None:
        raise ValueError("Section outline has no valid boundary segments")

    return best_segment


def _orient_face_segment(
    face: Literal["top", "bottom", "left", "right"],
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Orient selected face segments consistently for predictable bar ordering."""
    x0, y0 = segment_start
    x1, y1 = segment_end

    if face in ("top", "bottom"):
        if x0 > x1:
            return segment_end, segment_start
    else:
        if y0 > y1:
            return segment_end, segment_start

    return segment_start, segment_end


def _signed_exterior_area(section: RCSection) -> float:
    """Signed area of exterior ring (positive for CCW winding)."""
    coords = list(section.outline.exterior.coords)
    area2 = 0.0
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        area2 += float(x0 * y1 - x1 * y0)
    return 0.5 * area2


def _get_inward_unit_normal(
    section: RCSection,
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> tuple[float, float]:
    """
    Get inward unit normal for a boundary segment.

    Uses a geometric probe test first and falls back to ring winding if both
    probe directions are ambiguous.
    """
    x0, y0 = segment_start
    x1, y1 = segment_end

    dx = x1 - x0
    dy = y1 - y0
    length = float(np.hypot(dx, dy))
    if length <= _GEOM_TOL:
        raise ValueError("Selected face segment has near-zero length")

    left_n = (-dy / length, dx / length)
    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)
    probe = max(1e-6, length * 1e-6)

    left_probe = ShapelyPoint(mx + left_n[0] * probe, my + left_n[1] * probe)
    right_probe = ShapelyPoint(mx - left_n[0] * probe, my - left_n[1] * probe)

    left_inside = bool(section.outline.covers(left_probe))
    right_inside = bool(section.outline.covers(right_probe))

    if left_inside and not right_inside:
        return left_n
    if right_inside and not left_inside:
        return (-left_n[0], -left_n[1])

    # Fallback to winding direction if probe test is ambiguous.
    if _signed_exterior_area(section) >= 0.0:
        return left_n
    return (-left_n[0], -left_n[1])


def _offset_and_trim_segment(
    *,
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
    inward_normal: tuple[float, float],
    offset_to_bar_centre: float,
    end_trim_to_bar_centre: float,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """
    Offset a face segment inward and trim equally at both ends.

    Returns:
        start, end, usable_length
    """
    if offset_to_bar_centre < 0.0:
        raise ValueError("cover must be >= 0")
    if end_trim_to_bar_centre < 0.0:
        raise ValueError("side_cover must be >= 0")

    x0, y0 = segment_start
    x1, y1 = segment_end
    dx = x1 - x0
    dy = y1 - y0
    seg_length = float(np.hypot(dx, dy))
    if seg_length <= _GEOM_TOL:
        raise ValueError("Selected face segment has near-zero length")

    if 2.0 * end_trim_to_bar_centre > seg_length + _GEOM_TOL:
        raise ValueError(
            "side_cover + bar radius is too large for the selected face length"
        )

    tx = dx / seg_length
    ty = dy / seg_length
    nx, ny = inward_normal

    offset_start = (
        x0 + nx * offset_to_bar_centre,
        y0 + ny * offset_to_bar_centre,
    )
    offset_end = (
        x1 + nx * offset_to_bar_centre,
        y1 + ny * offset_to_bar_centre,
    )

    usable_length = max(seg_length - 2.0 * end_trim_to_bar_centre, 0.0)
    half_usable = 0.5 * usable_length

    mx = 0.5 * (offset_start[0] + offset_end[0])
    my = 0.5 * (offset_start[1] + offset_end[1])

    start = (mx - tx * half_usable, my - ty * half_usable)
    end = (mx + tx * half_usable, my + ty * half_usable)
    return start, end, usable_length


def _resolve_n_bars(
    *,
    n_bars: int | None,
    bar_spacing: float | None,
    usable_length: float,
) -> int:
    """Resolve number of bars from mutually exclusive count/spacing inputs."""
    if (n_bars is None) == (bar_spacing is None):
        raise ValueError("Provide exactly one of n_bars or bar_spacing")

    if n_bars is not None:
        if n_bars < 1:
            raise ValueError("n_bars must be at least 1")
        return int(n_bars)

    spacing = float(bar_spacing)  # type: ignore[arg-type]
    if spacing <= 0.0:
        raise ValueError("bar_spacing must be > 0")

    return max(int(np.floor(usable_length / spacing)) + 1, 1)


def _create_linear_group_from_line(
    *,
    rebar: Rebar,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    n_bars: int,
    bar_spacing: float | None,
    layer_name: str | None,
    omit_start: bool,
    omit_end: bool,
) -> RebarGroup:
    """
    Create a linear group from a line using count or explicit spacing behaviour.

    - ``bar_spacing is None``: endpoint-based spacing (delegates to
      ``create_linear_rebar_layer``).
    - ``bar_spacing`` set: bars are centred on line midpoint with exact spacing.
    """
    if bar_spacing is None:
        return create_linear_rebar_layer(
            rebar=rebar,
            n_bars=n_bars,
            start_point=start_point,
            end_point=end_point,
            layer_name=layer_name,
            omit_start=omit_start,
            omit_end=omit_end,
        )

    x0, y0 = start_point
    x1, y1 = end_point
    dx = x1 - x0
    dy = y1 - y0
    line_length = float(np.hypot(dx, dy))

    if n_bars == 1 or line_length <= _GEOM_TOL:
        positions = [Point2D(x=(x0 + x1) / 2.0, y=(y0 + y1) / 2.0)]
    else:
        spacing = float(bar_spacing)
        # Guard: with explicit spacing the bars are centred on the line midpoint,
        # so a requested span larger than the (already side-cover-trimmed) line
        # would silently place bars beyond the face ends. Reject it here so every
        # call site is protected (mirrors the multi-layer span check).
        required_span = (n_bars - 1) * spacing
        if required_span > line_length + _GEOM_TOL:
            raise ValueError(
                f"Cannot fit {n_bars} bars at {spacing:.1f} mm spacing on a line of "
                f"length {line_length:.1f} mm: required span {required_span:.1f} mm exceeds "
                f"the available length (bars would fall outside the face). "
                f"Reduce n_bars or spacing."
            )
        tx = dx / line_length
        ty = dy / line_length
        mx = 0.5 * (x0 + x1)
        my = 0.5 * (y0 + y1)
        half_span = 0.5 * (n_bars - 1) * spacing

        positions = []
        for i in range(n_bars):
            dist = -half_span + i * spacing
            positions.append(Point2D(x=mx + tx * dist, y=my + ty * dist))

    if omit_start and len(positions) > 0:
        positions = positions[1:]
    if omit_end and len(positions) > 0:
        positions = positions[:-1]

    if len(positions) == 0:
        raise ValueError(
            "All bars were omitted — n_bars is too small for the requested omit_start/omit_end combination"
        )

    return RebarGroup(
        rebar=rebar,
        positions=tuple(positions),
        layer_name=layer_name,
    )


def create_linear_rebar_layer(
    rebar: Rebar,
    n_bars: int,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    layer_name: str | None = None,
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


def create_multi_layer_linear_rebars(
    rebars: Rebar | Sequence[Rebar],
    n_bars: int,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    gap: float | Sequence[float] = 25.0,
    gap_between_faces: bool = True,
    layer_names: Sequence[str] | None = None,
    omit_start: bool = False,
    omit_end: bool = False,
) -> list[RebarGroup]:
    """
    Create multiple parallel linear rebar layers offset perpendicular to the
    start→end direction.

    Layer 0 sits at the supplied *start_point* / *end_point*.  Subsequent
    layers are offset to the **left** of the start→end vector (90° CCW).
    For a typical horizontal layer running left→right this means layers
    stack **upward**.  A negative *gap* reverses the direction.

    Args:
        rebars: A single Rebar (same bar in every layer) or a sequence of
            Rebar objects — one per layer.  The length of the sequence
            determines the number of layers.
        n_bars: Number of bars per layer (same for every layer).
        start_point: (x, y) start of the base layer (mm).
        end_point: (x, y) end of the base layer (mm).
        gap: Spacing between layers.  A single float is reused for every
            gap; a sequence must have length ``n_layers - 1``.
            Interpretation depends on *gap_between_faces*.
        gap_between_faces: If True (default) *gap* is clear spacing between
            bar surfaces.  If False, *gap* is centre-to-centre distance.
        layer_names: Optional layer identifiers (length must equal n_layers).
            Defaults to "layer_0", "layer_1", …
        omit_start: Passed through to each ``create_linear_rebar_layer`` call.
        omit_end: Passed through to each ``create_linear_rebar_layer`` call.

    Returns:
        List of RebarGroups, one per layer (layer 0 first).

    Example:
        >>> bar = Rebar(diameter=16, grade="B500B")
        >>> # Two layers of H16, 25 mm clear gap, stacking upward
        >>> groups = create_multi_layer_linear_rebars(
        ...     [bar, bar], n_bars=4,
        ...     start_point=(50, 50), end_point=(250, 50), gap=25,
        ... )
        >>> # Layer 0 at y=50, layer 1 at y=50+8+25+8=91
        >>> groups[1].positions[0].y
        91.0
    """
    # --- normalise rebars to a list ---
    if isinstance(rebars, Rebar):
        raise TypeError(
            "rebars must be a sequence of Rebar objects (one per layer), "
            "not a single Rebar.  Wrap in a list: [rebar, rebar, ...]"
        )
    rebar_list: list[Rebar] = list(rebars)
    n_layers = len(rebar_list)
    if n_layers < 1:
        raise ValueError("At least one rebar layer is required")

    # --- normalise gap to a list of n_layers-1 values ---
    if isinstance(gap, (int, float)):
        gaps: list[float] = [float(gap)] * max(n_layers - 1, 0)
    else:
        gaps = [float(g) for g in gap]
        if len(gaps) != n_layers - 1:
            raise ValueError(
                f"gap sequence length ({len(gaps)}) must equal n_layers - 1 ({n_layers - 1})"
            )

    # --- layer names ---
    if layer_names is not None:
        if len(layer_names) != n_layers:
            raise ValueError(
                f"layer_names length ({len(layer_names)}) must equal number of layers ({n_layers})"
            )
        names: list[str | None] = list(layer_names)
    else:
        names = [f"layer_{i}" for i in range(n_layers)]

    # --- perpendicular unit normal (left of start→end, i.e. 90° CCW) ---
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    length = (dx**2 + dy**2) ** 0.5
    if length < 1e-12:
        raise ValueError("start_point and end_point are coincident — cannot determine offset direction")
    nx, ny = -dy / length, dx / length  # 90° CCW rotation

    # --- build layers ---
    groups: list[RebarGroup] = []
    cumulative_offset = 0.0

    for i in range(n_layers):
        if i > 0:
            if gap_between_faces:
                cumulative_offset += (
                    rebar_list[i - 1].diameter / 2.0
                    + gaps[i - 1]
                    + rebar_list[i].diameter / 2.0
                )
            else:
                cumulative_offset += gaps[i - 1]

        offset_x = nx * cumulative_offset
        offset_y = ny * cumulative_offset

        layer_start = (start_point[0] + offset_x, start_point[1] + offset_y)
        layer_end = (end_point[0] + offset_x, end_point[1] + offset_y)

        groups.append(
            create_linear_rebar_layer(
                rebar=rebar_list[i],
                n_bars=n_bars,
                start_point=layer_start,
                end_point=layer_end,
                layer_name=names[i],
                omit_start=omit_start,
                omit_end=omit_end,
            )
        )

    return groups


def create_linear_rebar_layer_on_face(
    section: RCSection,
    rebar: Rebar,
    face: Literal["top", "bottom", "left", "right"],
    cover: float,
    *,
    n_bars: int | None = None,
    bar_spacing: float | None = None,
    side_cover: float | None = None,
    layer_name: str | None = None,
    omit_start: bool = False,
    omit_end: bool = False,
) -> RebarGroup:
    """
    Create a linear rebar layer by referencing a section face.

    The selected face segment is offset inward by ``cover + bar_radius`` and
    trimmed at both ends by ``side_cover + bar_radius`` (or ``cover + bar_radius``
    when ``side_cover`` is not provided). Bars are then placed symmetrically
    about the face midpoint.

    Exactly one of ``n_bars`` or ``bar_spacing`` must be supplied.
    """
    face_norm = _normalise_face(face)

    if cover < 0.0:
        raise ValueError("cover must be >= 0")

    resolved_side_cover = cover if side_cover is None else float(side_cover)
    if resolved_side_cover < 0.0:
        raise ValueError("side_cover must be >= 0")

    segment_start, segment_end = _resolve_face_segment(section, face_norm)
    segment_start, segment_end = _orient_face_segment(
        face_norm, segment_start, segment_end
    )
    inward_normal = _get_inward_unit_normal(section, segment_start, segment_end)

    start_point, end_point, usable_length = _offset_and_trim_segment(
        segment_start=segment_start,
        segment_end=segment_end,
        inward_normal=inward_normal,
        offset_to_bar_centre=cover + rebar.diameter / 2.0,
        end_trim_to_bar_centre=resolved_side_cover + rebar.diameter / 2.0,
    )

    resolved_n_bars = _resolve_n_bars(
        n_bars=n_bars,
        bar_spacing=bar_spacing,
        usable_length=usable_length,
    )

    if resolved_n_bars > 1 and usable_length <= _GEOM_TOL:
        raise ValueError(
            "Not enough usable face length for multiple bars after applying side_cover"
        )

    return _create_linear_group_from_line(
        rebar=rebar,
        n_bars=resolved_n_bars,
        bar_spacing=bar_spacing,
        start_point=start_point,
        end_point=end_point,
        layer_name=layer_name if layer_name is not None else face_norm,
        omit_start=omit_start,
        omit_end=omit_end,
    )


def create_multi_layer_linear_rebars_on_face(
    section: RCSection,
    rebars: Rebar | Sequence[Rebar],
    face: Literal["top", "bottom", "left", "right"],
    cover: float,
    *,
    n_bars: int | None = None,
    bar_spacing: float | None = None,
    side_cover: float | None = None,
    gap: float | Sequence[float] = 25.0,
    gap_between_faces: bool = True,
    layer_names: Sequence[str] | None = None,
    omit_start: bool = False,
    omit_end: bool = False,
) -> list[RebarGroup]:
    """
    Create multiple parallel face-based linear rebar layers.

    Layers are stacked inward from the requested face. Spacing between layers
    follows ``gap`` and ``gap_between_faces`` semantics from
    ``create_multi_layer_linear_rebars``.

    Exactly one of ``n_bars`` or ``bar_spacing`` must be supplied. When
    ``bar_spacing`` is used, the bar count is derived from the first layer and
    reused for all layers.
    """
    face_norm = _normalise_face(face)

    if cover < 0.0:
        raise ValueError("cover must be >= 0")

    resolved_side_cover = cover if side_cover is None else float(side_cover)
    if resolved_side_cover < 0.0:
        raise ValueError("side_cover must be >= 0")

    if isinstance(rebars, Rebar):
        raise TypeError(
            "rebars must be a sequence of Rebar objects (one per layer), "
            "not a single Rebar.  Wrap in a list: [rebar, rebar, ...]"
        )
    rebar_list: list[Rebar] = list(rebars)
    n_layers = len(rebar_list)
    if n_layers < 1:
        raise ValueError("At least one rebar layer is required")

    if isinstance(gap, (int, float)):
        gaps: list[float] = [float(gap)] * max(n_layers - 1, 0)
    else:
        gaps = [float(g) for g in gap]
        if len(gaps) != n_layers - 1:
            raise ValueError(
                f"gap sequence length ({len(gaps)}) must equal n_layers - 1 ({n_layers - 1})"
            )

    if layer_names is not None:
        if len(layer_names) != n_layers:
            raise ValueError(
                f"layer_names length ({len(layer_names)}) must equal number of layers ({n_layers})"
            )
        names: list[str | None] = list(layer_names)
    else:
        names = [f"layer_{i}" for i in range(n_layers)]

    segment_start, segment_end = _resolve_face_segment(section, face_norm)
    segment_start, segment_end = _orient_face_segment(
        face_norm, segment_start, segment_end
    )
    inward_normal = _get_inward_unit_normal(section, segment_start, segment_end)

    _, _, first_layer_usable_length = _offset_and_trim_segment(
        segment_start=segment_start,
        segment_end=segment_end,
        inward_normal=inward_normal,
        offset_to_bar_centre=cover + rebar_list[0].diameter / 2.0,
        end_trim_to_bar_centre=resolved_side_cover + rebar_list[0].diameter / 2.0,
    )

    resolved_n_bars = _resolve_n_bars(
        n_bars=n_bars,
        bar_spacing=bar_spacing,
        usable_length=first_layer_usable_length,
    )

    groups: list[RebarGroup] = []
    cumulative_offset = cover + rebar_list[0].diameter / 2.0

    for i in range(n_layers):
        if i > 0:
            if gap_between_faces:
                cumulative_offset += (
                    rebar_list[i - 1].diameter / 2.0
                    + gaps[i - 1]
                    + rebar_list[i].diameter / 2.0
                )
            else:
                cumulative_offset += gaps[i - 1]

        layer_start, layer_end, layer_usable_length = _offset_and_trim_segment(
            segment_start=segment_start,
            segment_end=segment_end,
            inward_normal=inward_normal,
            offset_to_bar_centre=cumulative_offset,
            end_trim_to_bar_centre=resolved_side_cover + rebar_list[i].diameter / 2.0,
        )

        if resolved_n_bars > 1 and layer_usable_length <= _GEOM_TOL:
            raise ValueError(
                f"Not enough usable face length for multiple bars in layer {i} after applying side_cover"
            )

        if bar_spacing is not None and resolved_n_bars > 1:
            required_span = (resolved_n_bars - 1) * float(bar_spacing)
            if required_span > layer_usable_length + _GEOM_TOL:
                raise ValueError(
                    f"Requested bar_spacing does not fit usable face length in layer {i}"
                )

        groups.append(
            _create_linear_group_from_line(
                rebar=rebar_list[i],
                n_bars=resolved_n_bars,
                bar_spacing=bar_spacing,
                start_point=layer_start,
                end_point=layer_end,
                layer_name=names[i],
                omit_start=omit_start,
                omit_end=omit_end,
            )
        )

    return groups


def create_rectangular_perimeter_rebars(
    rebar: Rebar,
    width: float,
    height: float,
    cover: float,
    n_bars_width: int,
    n_bars_height: int,
    origin: tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
) -> list[RebarGroup]:
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

    groups: list[RebarGroup] = []

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
    origin: tuple[float, float] = (0.0, 0.0),
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
        start_angle: Starting angle in degrees (0 = right, counter-clockwise)

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
    positions: Sequence[tuple[float, float]],
    layer_name: str | None = None,
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
