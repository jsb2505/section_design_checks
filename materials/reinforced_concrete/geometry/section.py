"""
Reinforced concrete section geometry using Shapely for 2D polygonal shapes.

Provides flexible geometry definition with arbitrary polygonal outlines
and rebar positioning.

Supports:
- Solid polygon sections
- Polygon sections with voids/holes (interior rings)
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Literal

import numpy as np
from shapely.geometry import Polygon, Point as ShapelyPoint
from pydantic import BaseModel, Field, ConfigDict, model_validator

from materials.core.geometry import BaseGeometry, Point2D
from materials.reinforced_concrete.materials.rebar import Rebar


# Small tolerance used for geometric checks (mm)
_GEOM_TOL_MM = 1e-6


def _ring_integrals_about_origin(coords: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Compute signed area, centroid (about origin), and second moments (about origin)
    for a single closed polygon ring using standard shoelace-based formulas.

    Args:
        coords: (N,2) array of ring coordinates (x,y). May be closed or open.

    Returns:
        (A, Cx, Cy, Ixx0, Iyy0, Ixy0) where:
            A   = signed area (mm²)
            Cx  = centroid x (mm) (only meaningful if A != 0)
            Cy  = centroid y (mm)
            Ixx0, Iyy0, Ixy0 = second moments/products about origin (mm⁴), signed with A
    """
    if coords.shape[0] < 3:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Ensure ring is closed for consistent indexing
    if not (coords[0, 0] == coords[-1, 0] and coords[0, 1] == coords[-1, 1]):
        coords = np.vstack([coords, coords[0]])

    x = coords[:, 0]
    y = coords[:, 1]

    x0 = x[:-1]
    y0 = y[:-1]
    x1 = x[1:]
    y1 = y[1:]

    cross = x0 * y1 - x1 * y0  # signed
    A = 0.5 * float(np.sum(cross))

    if abs(A) < 1e-18:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Centroid (about origin)
    Cx = float(np.sum((x0 + x1) * cross)) / (6.0 * A)
    Cy = float(np.sum((y0 + y1) * cross)) / (6.0 * A)

    # Second moments about origin
    Ixx0 = float(np.sum((y0 * y0 + y0 * y1 + y1 * y1) * cross)) / 12.0
    Iyy0 = float(np.sum((x0 * x0 + x0 * x1 + x1 * x1) * cross)) / 12.0
    Ixy0 = float(
        np.sum(
            (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * cross
        )
    ) / 24.0

    return (A, Cx, Cy, Ixx0, Iyy0, Ixy0)


def _polygon_integrals_about_origin(poly: Polygon) -> Tuple[float, float, float, float, float, float]:
    """
    Compute signed area, centroid, and second moments about origin for a Polygon,
    including voids/holes (interior rings).

    Strategy:
        - Compute ring integrals for exterior ring.
        - Add ring integrals for each interior ring (holes). Orientation is handled
          via signed area in the formulas (holes should contribute negative area).
        - Combine to obtain polygon centroid and origin-referenced inertias.
    """
    # Exterior
    ext = np.asarray(poly.exterior.coords, dtype=float)
    A_e, Cx_e, Cy_e, Ixx_e, Iyy_e, Ixy_e = _ring_integrals_about_origin(ext)

    A_total = A_e
    Cx_num = Cx_e * A_e
    Cy_num = Cy_e * A_e
    Ixx0 = Ixx_e
    Iyy0 = Iyy_e
    Ixy0 = Ixy_e

    # Interiors (holes)
    for ring in poly.interiors:
        coords = np.asarray(ring.coords, dtype=float)
        A_i, Cx_i, Cy_i, Ixx_i, Iyy_i, Ixy_i = _ring_integrals_about_origin(coords)

        A_total += A_i
        Cx_num += Cx_i * A_i
        Cy_num += Cy_i * A_i
        Ixx0 += Ixx_i
        Iyy0 += Iyy_i
        Ixy0 += Ixy_i

    if abs(A_total) < 1e-18:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    Cx = Cx_num / A_total
    Cy = Cy_num / A_total
    return (A_total, Cx, Cy, Ixx0, Iyy0, Ixy0)


class RebarGroup(BaseModel):
    """
    Group of rebars with common properties.

    Represents one or more bars at specific locations.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    rebar: Rebar = Field(
        ...,
        description="Rebar specification (diameter and material)"
    )

    positions: List[Point2D] = Field(
        ...,
        description="List of (x, y) coordinates for bar centres (mm)",
        min_length=1,
    )

    layer_name: Optional[str] = Field(
        None,
        description="Optional layer identifier (e.g., 'bottom', 'top', 'side')"
    )

    @model_validator(mode="after")
    def validate_positions_non_overlapping(self) -> "RebarGroup":
        """
        Ensure bar geometries do not overlap.

        Bars are allowed to touch (distance == diameter), but not overlap
        (distance < diameter). All bars in the group share the same diameter.
        """
        if len(self.positions) < 2:
            return self

        d = float(self.rebar.diameter)
        min_dist = d - _GEOM_TOL_MM  # allow touching (+ tiny numerical tolerance)
        min_dist_sq = min_dist * min_dist

        for i, p1 in enumerate(self.positions):
            for p2 in self.positions[i + 1:]:
                dx = p1.x - p2.x
                dy = p1.y - p2.y
                if (dx * dx + dy * dy) < min_dist_sq:
                    raise ValueError(
                        "Rebars overlap (or are closer than diameter). "
                        f"Diameter={d:g} mm, "
                        f"positions=({p1.x:.3f},{p1.y:.3f}) and ({p2.x:.3f},{p2.y:.3f}). "
                        "Bars may touch, but must not overlap."
                    )
        return self

    @property
    def n_bars(self) -> int:
        """Number of bars in this group."""
        return len(self.positions)

    @property
    def total_area(self) -> float:
        """
        Total steel area for this group.

        Returns:
            Total area in mm²
        """
        return self.n_bars * self.rebar.area

    def get_centroid(self) -> Point2D:
        """
        Calculate centroid of bar group.

        Returns:
            Centroid coordinates
        """
        x_avg = sum(p.x for p in self.positions) / self.n_bars
        y_avg = sum(p.y for p in self.positions) / self.n_bars
        return Point2D(x=x_avg, y=y_avg)

    def __repr__(self) -> str:
        return f"RebarGroup({self.n_bars}×{self.rebar}, layer={self.layer_name})"


class RCSection(BaseGeometry):
    """
    Reinforced concrete section with arbitrary polygonal outline.

    Uses Shapely for robust geometric operations:
    - Area, centroid, moments of inertia (supports voids)
    - Containment checking (rebar discs fully within section, including voids)

    Coordinate system convention:
    - Origin is at the *centre* of the section by default for helper constructors.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    outline: Polygon = Field(
        ...,
        description="Section outline as Shapely Polygon (coordinates in mm)"
    )

    rebar_groups: List[RebarGroup] = Field(
        default_factory=list,
        description="List of rebar groups in the section"
    )

    concrete_cover_override: Optional[float] = Field(
        default=None,
        description="Optional override for concrete cover (mm). If None, calculate from geometry.",
        gt=0,
    )

    section_name: Optional[str] = Field(
        None,
        description="Section identifier"
    )

    @model_validator(mode="after")
    def validate_outline_and_rebars(self) -> "RCSection":
        """
        Validate geometry consistency after model creation.

        - outline must be a valid, non-empty polygon with positive area
        - all rebars (as discs) must lie fully within outline (respects voids)
        """
        if self.outline.is_empty:
            raise ValueError("Section outline is empty")
        if not self.outline.is_valid:
            raise ValueError("Section outline is not a valid polygon")
        if self.outline.area <= 0:
            raise ValueError("Section outline has zero or negative area")

        for group in self.rebar_groups:
            r = float(group.rebar.diameter) / 2.0
            for pos in group.positions:
                disc = ShapelyPoint(pos.x, pos.y).buffer(r)
                if not self.outline.covers(disc):
                    raise ValueError(
                        f"Rebar (ϕ{group.rebar.diameter:g}) at ({pos.x:.1f}, {pos.y:.1f}) "
                        "is not fully within the section outline (may cross boundary or enter a void)."
                    )
        return self

    def get_area(self) -> float:
        """
        Gross concrete area (excluding rebar).

        Returns:
            Area in mm²

        Note:
            Shapely's area correctly accounts for holes/voids.
        """
        return float(self.outline.area)

    def get_centroid(self) -> Tuple[float, float]:
        """
        Centroid of gross concrete section.

        Returns:
            (x, y) coordinates in mm

        Note:
            Shapely's centroid correctly accounts for holes/voids.
        """
        c = self.outline.centroid
        return (float(c.x), float(c.y))

    def get_second_moment_area(self) -> Tuple[float, float, float]:
        """
        Second moments of area about centroidal axes (gross concrete section only).

        Supports polygons with holes/voids by integrating:
            - exterior ring contribution
            - plus interior ring signed contributions

        Returns:
            (I_xx, I_yy, I_xy) in mm⁴
        """
        A, Cx, Cy, Ixx0, Iyy0, Ixy0 = _polygon_integrals_about_origin(self.outline)
        if abs(A) < 1e-18:
            raise ValueError("Cannot compute second moments: section area is zero/degenerate")

        # Shift to centroidal axes using parallel axis theorem:
        Ixx_c = Ixx0 - A * (Cy ** 2)
        Iyy_c = Iyy0 - A * (Cx ** 2)
        Ixy_c = Ixy0 - A * (Cx * Cy)

        # Return positive magnitudes (engineering convention)
        return (abs(Ixx_c), abs(Iyy_c), abs(Ixy_c))

    def get_transformed_second_moment_area(
        self,
        E_c: float,
        centroid: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Second moments of area for transformed section including reinforcement.

        Uses transformed section method with modular ratio α_e = E_s / E_c.
        Each steel bar contributes (α_e - 1) · A_s via parallel axis terms.
        """
        if E_c <= 0:
            raise ValueError(f"Concrete modulus E_c must be positive, got {E_c}")
        if not self.rebar_groups:
            raise ValueError("Cannot calculate transformed properties: no rebars in section")

        cx, cy = self.get_centroid() if centroid is None else centroid

        I_xx_conc, I_yy_conc, I_xy_conc = self.get_second_moment_area()

        I_xx_steel = 0.0
        I_yy_steel = 0.0
        I_xy_steel = 0.0

        for group in self.rebar_groups:
            alpha_e = group.rebar.E_s / E_c
            factor = alpha_e - 1.0

            for pos in group.positions:
                dx = pos.x - cx
                dy = pos.y - cy
                A_transformed = factor * group.rebar.area

                I_xx_steel += A_transformed * dy ** 2
                I_yy_steel += A_transformed * dx ** 2
                I_xy_steel += A_transformed * dx * dy

        return (I_xx_conc + I_xx_steel, I_yy_conc + I_yy_steel, I_xy_conc + I_xy_steel)

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Bounding box of section.

        Returns:
            (min_x, min_y, max_x, max_y) in mm
        """
        min_x, min_y, max_x, max_y = self.outline.bounds
        return (float(min_x), float(min_y), float(max_x), float(max_y))

    @property
    def total_steel_area(self) -> float:
        """Total area of all reinforcement in mm²."""
        return sum(group.total_area for group in self.rebar_groups)

    @property
    def reinforcement_ratio(self) -> float:
        """Reinforcement ratio (ρ = A_s / A_c)."""
        a_c = self.get_area()
        return 0.0 if a_c == 0.0 else (self.total_steel_area / a_c)

    def get_rebar_positions(self) -> List[Tuple[float, float, float]]:
        """
        Get all rebar positions with areas.

        Returns:
            List of (x, y, area) tuples for each bar
        """
        out: List[Tuple[float, float, float]] = []
        for group in self.rebar_groups:
            for pos in group.positions:
                out.append((pos.x, pos.y, group.rebar.area))
        return out

    def get_steel_centroid(self) -> Tuple[float, float]:
        """
        Calculate centroid of all reinforcement.

        Returns:
            (x, y) coordinates in mm, or (0, 0) if no reinforcement
        """
        if self.total_steel_area == 0.0:
            return (0.0, 0.0)

        total_area = 0.0
        mx = 0.0
        my = 0.0

        for group in self.rebar_groups:
            for pos in group.positions:
                a = group.rebar.area
                total_area += a
                mx += a * pos.x
                my += a * pos.y

        return (mx / total_area, my / total_area)

    def add_rebar_group(self, group: RebarGroup) -> None:
        """
        Add a rebar group to the section.

        Raises:
            ValueError: If any rebar disc is not fully within the outline
        """
        r = float(group.rebar.diameter) / 2.0
        for pos in group.positions:
            disc = ShapelyPoint(pos.x, pos.y).buffer(r)
            if not self.outline.covers(disc):
                raise ValueError(
                    f"Rebar (ϕ{group.rebar.diameter:g}) at ({pos.x:.1f}, {pos.y:.1f}) "
                    "is not fully within the section outline (may cross boundary or enter a void)."
                )
        self.rebar_groups.append(group)

    def get_concrete_cover(
        self,
        reference: Literal["top", "bottom"] = "bottom",
        orthogonal_only: bool = True,
    ) -> float:
        """
        Calculate concrete cover to a chosen face (top or bottom) for cracking checks.

        Cover is to the *outer surface* of rebar:
            cover = distance(boundary, bar_centre) - bar_radius

        Behaviour:
            - If concrete_cover_override is set, returns it.
            - If orthogonal_only=True: cover is computed orthogonally to the chosen
              face using the section's bounding box (typical beam design).
            - If orthogonal_only=False: cover is computed as the true minimum distance
              from bar centres to the polygon boundary, but filtered to the chosen face
              by selecting boundary segments on the top/bottom half relative to centroid.

        Args:
            reference: "top" or "bottom" (tension face to evaluate)
            orthogonal_only: If True (default), use bounding-box orthogonal cover.
                            If False, use true polygon boundary distance with face filtering.

        Returns:
            Cover in mm
        """
        if self.concrete_cover_override is not None:
            return self.concrete_cover_override

        if not self.rebar_groups:
            raise ValueError("Cannot calculate cover: no rebars in section")

        if reference not in ("top", "bottom"):
            raise ValueError(f"Unknown reference: {reference}")

        # Orthogonal-only cover to chosen face (top OR bottom, not min of both)
        if orthogonal_only:
            _, min_y, _, max_y = self.get_bounding_box()
            min_cover = float("inf")

            for group in self.rebar_groups:
                r = float(group.rebar.diameter) / 2.0
                for pos in group.positions:
                    c = (pos.y - min_y) - r if reference == "bottom" else (max_y - pos.y) - r
                    min_cover = min(min_cover, c)

            return min_cover

        # True polygon cover, but restricted to "top" or "bottom" boundary portions.
        _, cy = self.get_centroid()
        min_cover = float("inf")

        # Include exterior and interior boundaries (void boundaries matter too)
        rings = [self.outline.exterior, *self.outline.interiors]

        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for ring in rings:
            coords = list(ring.coords)
            for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
                ym = 0.5 * (y0 + y1)
                if reference == "top" and ym >= cy:
                    segments.append(((x0, y0), (x1, y1)))
                elif reference == "bottom" and ym <= cy:
                    segments.append(((x0, y0), (x1, y1)))

        # Fallback: if segmentation yields nothing, use full boundary distance
        if not segments:
            boundary = self.outline.boundary
            for group in self.rebar_groups:
                r = float(group.rebar.diameter) / 2.0
                for pos in group.positions:
                    d = boundary.distance(ShapelyPoint(pos.x, pos.y))
                    min_cover = min(min_cover, d - r)
            return min_cover

        def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
            """Distance from point P to segment AB in 2D."""
            vx = bx - ax
            vy = by - ay
            wx = px - ax
            wy = py - ay
            vv = vx * vx + vy * vy
            if vv <= 1e-18:
                dx = px - ax
                dy = py - ay
                return (dx * dx + dy * dy) ** 0.5
            t = (wx * vx + wy * vy) / vv
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            cx_ = ax + t * vx
            cy_ = ay + t * vy
            dx = px - cx_
            dy = py - cy_
            return (dx * dx + dy * dy) ** 0.5

        for group in self.rebar_groups:
            r = float(group.rebar.diameter) / 2.0
            for pos in group.positions:
                px, py = float(pos.x), float(pos.y)
                d_min = float("inf")
                for (a, b) in segments:
                    d = point_to_segment_distance(px, py, a[0], a[1], b[0], b[1])
                    d_min = min(d_min, d)
                min_cover = min(min_cover, d_min - r)

        return min_cover

    def get_effective_depth(
        self,
        compression_face: Literal["top", "bottom"] = "top",
        *,
        tension_zone: Optional[Literal["top", "bottom"]] = None,
        zone_fraction: float = 0.5,
    ) -> float:
        """
        Effective depth d = distance from compression face to centroid of *tension* reinforcement.

        This is a geometric helper used for beam-like checks.

        How tension bars are selected:
        - By default (tension_zone=None): bars on the opposite side of the section centroid
          from the compression face are treated as "tension side".
            * compression_face="top"    -> tension_zone="bottom"
            * compression_face="bottom" -> tension_zone="top"
        - Additionally, a bar must lie within the chosen "zone" thickness:
            zone = bottom fraction or top fraction of the section depth.
          This avoids including mid-depth bars (e.g., side bars in walls/columns) unless desired.

        Args:
            compression_face: "top" or "bottom" compression edge reference.
            tension_zone: Explicitly choose which side is tension ("top" or "bottom").
                         If None, inferred as the opposite side of compression_face.
            zone_fraction: Fraction of depth considered as the tension zone (0 < f <= 1).
                          0.5 means "lower half" (or upper half) only.
                          Use 1.0 to include *all* bars on the tension side.

        Returns:
            d (mm)

        Raises:
            ValueError if no rebars exist or no bars found in the chosen tension zone.
        """
        if not self.rebar_groups:
            raise ValueError("Cannot compute effective depth: no rebars in section")

        if compression_face not in ("top", "bottom"):
            raise ValueError(f"compression_face must be 'top' or 'bottom', got {compression_face}")

        if not (0.0 < zone_fraction <= 1.0):
            raise ValueError(f"zone_fraction must be in (0, 1], got {zone_fraction}")

        _, min_y, _, max_y = self.get_bounding_box()
        h = max_y - min_y
        if h <= 0.0:
            raise ValueError("Invalid section height (bounding box height <= 0)")

        # Infer tension zone if not provided
        if tension_zone is None:
            tension_zone = "bottom" if compression_face == "top" else "top"
        if tension_zone not in ("top", "bottom"):
            raise ValueError(f"tension_zone must be 'top' or 'bottom', got {tension_zone}")

        # Define the band (zone) to consider
        if tension_zone == "bottom":
            y_limit = min_y + zone_fraction * h  # include bars with y <= y_limit
            def in_zone(y: float) -> bool:
                return y <= y_limit + _GEOM_TOL_MM
        else:
            y_limit = max_y - zone_fraction * h  # include bars with y >= y_limit
            def in_zone(y: float) -> bool:
                return y >= y_limit - _GEOM_TOL_MM

        # Collect tension-zone bars (area-weighted)
        A = 0.0
        mx = 0.0
        my = 0.0

        for group in self.rebar_groups:
            a_bar = float(group.rebar.area)
            for pos in group.positions:
                y = float(pos.y)
                if in_zone(y):
                    A += a_bar
                    mx += a_bar * float(pos.x)
                    my += a_bar * y

        if A <= 0.0:
            raise ValueError(
                "No reinforcement found in the selected tension zone. "
                f"(compression_face={compression_face}, tension_zone={tension_zone}, zone_fraction={zone_fraction})"
            )

        c = Point2D(x=mx / A, y=my / A)

        # Effective depth from compression face to that centroid (orthogonal to face)
        d = (max_y - c.y) if compression_face == "top" else (c.y - min_y)
        d = float(d)

        return d

    def __repr__(self) -> str:
        name_str = f"'{self.section_name}'" if self.section_name else "unnamed"
        return (
            f"RCSection({name_str}, "
            f"A_c={self.get_area():.0f} mm², "
            f"A_s={self.total_steel_area:.0f} mm², "
            f"{len(self.rebar_groups)} groups)"
        )

    def __str__(self) -> str:
        return self.__repr__()


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
        hook_ref=0: Center (origin at center of rectangle)
        hook_ref=1: Bottom-left corner (section in +X, +Y quadrant) - DEFAULT
        hook_ref=2: Bottom-right corner (section in -X, +Y quadrant)
        hook_ref=3: Top-right corner (section in -X, -Y quadrant)
        hook_ref=4: Top-left corner (section in +X, -Y quadrant)

    Args:
        width: Section width (mm)
        height: Section height (mm)
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point (0=center, 1=bottom-left, 2=bottom-right,
                  3=top-right, 4=top-left). Default: 1 (bottom-left)
        section_name: Optional section name

    Returns:
        RCSection with rectangular outline

    Examples:
        >>> # Section with bottom-left at (0, 0), extends to (300, 500)
        >>> section = create_rectangular_section(300, 500)

        >>> # Section centered at (0, 0)
        >>> section = create_rectangular_section(300, 500, hook_ref=0)

        >>> # Section with bottom-left at (100, 50)
        >>> section = create_rectangular_section(300, 500, origin=(100, 50))
    """
    ox, oy = origin

    # Calculate center point based on hook_ref
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

    hw = width / 2.0
    hh = height / 2.0

    coords = [
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
        (cx - hw, cy - hh),
    ]

    return RCSection(
        outline=Polygon(coords),
        section_name=section_name or f"Rect {width}×{height}",
    )


def create_circular_section(
    diameter: float,
    n_points: int = 32,
    origin: Tuple[float, float] = (0.0, 0.0),
    hook_ref: int = 1,
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create a circular RC section.

    Hook Reference Convention:
        hook_ref=0: Center (origin at center of circle)
        hook_ref=1: Bottom-left of bounding box (section in +X, +Y quadrant) - DEFAULT
        hook_ref=2: Bottom-right of bounding box (section in -X, +Y quadrant)
        hook_ref=3: Top-right of bounding box (section in -X, -Y quadrant)
        hook_ref=4: Top-left of bounding box (section in +X, -Y quadrant)

    Args:
        diameter: Section diameter (mm)
        n_points: Number of points to approximate circle (default: 32)
        origin: Hook point coordinates (default: (0, 0))
        hook_ref: Hook reference point (0=center, 1=bottom-left, etc.). Default: 1
        section_name: Optional section name

    Returns:
        RCSection with circular outline

    Examples:
        >>> # Circle with bounding box bottom-left at (0, 0)
        >>> section = create_circular_section(400)

        >>> # Circle centered at (0, 0)
        >>> section = create_circular_section(400, hook_ref=0)
    """
    ox, oy = origin
    radius = diameter / 2.0

    # Calculate center point based on hook_ref
    if hook_ref == 0:
        # Center
        cx, cy = ox, oy
    elif hook_ref == 1:
        # Bottom-left corner of bounding box
        cx = ox + radius
        cy = oy + radius
    elif hook_ref == 2:
        # Bottom-right corner of bounding box
        cx = ox - radius
        cy = oy + radius
    elif hook_ref == 3:
        # Top-right corner of bounding box
        cx = ox - radius
        cy = oy - radius
    elif hook_ref == 4:
        # Top-left corner of bounding box
        cx = ox + radius
        cy = oy - radius
    else:
        raise ValueError(f"hook_ref must be 0, 1, 2, 3, or 4, got {hook_ref}")

    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=float)
    coords = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
    coords.append(coords[0])  # close

    return RCSection(
        outline=Polygon(coords),
        section_name=section_name or f"Circular Ø{diameter}",
    )
