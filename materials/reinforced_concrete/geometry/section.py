"""
Reinforced concrete section geometry using Shapely for 2D polygonal shapes.

Provides flexible geometry definition with arbitrary polygonal outlines
and rebar positioning.

Supports:
- Solid polygon sections
- Polygon sections with voids/holes (interior rings)
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

if TYPE_CHECKING:
    from materials.reinforced_concrete.materials.concrete import ConcreteMaterial

    from .reinforcement_reconcile import ReinforcementUpdateReport

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PrivateAttr, model_validator
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from materials.core.geometry import BaseGeometry, Point2D
from materials.reinforced_concrete.materials.rebar import Rebar

from .reinforcement_reconcile import ReinforcementInvalidPolicy

# Small tolerance used for geometric checks (mm)
_GEOM_TOL_MM = 1e-6


def _ring_integrals_about_origin(coords: np.ndarray) -> tuple[float, float, float, float, float, float]:
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


def _polygon_integrals_about_origin(poly: Polygon) -> tuple[float, float, float, float, float, float]:
    """
    Compute signed area, centroid, and second moments about origin for a Polygon,
    including voids/holes (interior rings).

    Strategy:
        - Compute ring integrals for exterior ring.
        - Subtract ring integrals for each interior ring (holes).

    Note:
        Ring integrals scale with the ring's SIGNED area. Shapely does NOT
        normalise interior-ring winding — it preserves whatever orientation the
        caller supplied — so a hole given in the conventional clockwise order has
        negative signed area and negated integrals. Each hole is therefore scaled
        back to a positive-area basis (via the sign of its signed area) before
        subtraction, so the result is correct for either winding.
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

    # Interiors (holes) - subtract their contributions.
    # All ring integrals scale with the SIGNED area, and Shapely preserves the
    # caller's interior-ring winding, so normalise each hole to a positive-area
    # basis via s = sign(A_i). This makes EVERY term subtract with a consistent
    # sign - including the product of inertia Ixy, which can legitimately be
    # negative and was previously subtracted with the raw (winding-dependent)
    # sign, corrupting I_xy for clockwise-wound voids.
    for ring in poly.interiors:
        coords = np.asarray(ring.coords, dtype=float)
        A_i, Cx_i, Cy_i, Ixx_i, Iyy_i, Ixy_i = _ring_integrals_about_origin(coords)

        s = 1.0 if A_i >= 0.0 else -1.0
        A_hole = s * A_i  # == |A_i|
        A_total -= A_hole
        Cx_num -= Cx_i * A_hole
        Cy_num -= Cy_i * A_hole
        Ixx0 -= s * Ixx_i
        Iyy0 -= s * Iyy_i
        Ixy0 -= s * Ixy_i

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
        frozen=True,  # Immutable and hashable
        extra="forbid",
    )

    rebar: Rebar = Field(
        ...,
        description="Rebar specification (diameter and material)"
    )

    positions: tuple[Point2D, ...] = Field(
        ...,
        description="Tuple of (x, y) coordinates for bar centres (mm)",
        min_length=1,
    )

    layer_name: str | None = Field(
        default=None,
        description="Optional layer identifier (e.g., 'bottom', 'top', 'side')"
    )

    @model_validator(mode="after")
    def validate_positions_non_overlapping(self) -> RebarGroup:
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


    @cached_property
    def n_bars(self) -> int:
        """Number of bars in this group."""
        return len(self.positions)


    @cached_property
    def total_area(self) -> float:
        """
        Total steel area for this group.

        Returns:
            Total area in mm²
        """
        return self.n_bars * self.rebar.area

    @cached_property
    def centroid(self) -> Point2D:
        """
        Centroid of bar group.

        Returns:
            Centroid coordinates
        """
        return self._get_centroid()

    def _get_centroid(self) -> Point2D:
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


def _coerce_point2d(v: Any) -> Any:
    """Accept Point2D, (x, y) tuple/list, or dict."""
    if isinstance(v, Point2D):
        return v
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return Point2D(x=v[0], y=v[1])
    return v  # let Pydantic validate/reject


def _coerce_point2d_sequence(v: Any) -> tuple[Point2D, ...]:
    """Coerce a sequence of Point2D-like items."""
    if isinstance(v, (tuple, list)):
        return tuple(_coerce_point2d(item) for item in v)
    return cast("tuple[Point2D, ...]", v)  # let Pydantic validate/reject


def _coerce_voids(v: Any) -> tuple[tuple[Point2D, ...], ...]:
    """Coerce nested sequence of Point2D-like items for voids."""
    if isinstance(v, (tuple, list)):
        return tuple(_coerce_point2d_sequence(ring) for ring in v)
    return cast("tuple[tuple[Point2D, ...], ...]", v)  # let Pydantic validate/reject


class RCSection(BaseGeometry):
    """
    Reinforced concrete section with arbitrary polygonal outline.

    Uses Shapely for robust geometric operations:
    - Area, centroid, moments of inertia (supports voids)
    - Containment checking (rebar discs fully within section, including voids)

    Coordinate system convention:
    - Origin is at the *centre* of the section by default for helper constructors.
    """
    _suspend_outline_reconcile: bool = PrivateAttr(default=False)

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    reinforcement_policy: ReinforcementInvalidPolicy = Field(
        default=ReinforcementInvalidPolicy.ERROR,
        description=(
            "Policy applied automatically when outline coords change. "
            "error: reject change if any rebar becomes invalid; "
            "drop_bars: remove invalid bars; "
            "drop_groups: remove any group with an invalid bar; "
            "allow_invalid: allow invalid reinforcement."
        ),
    )

    outline_coords: Annotated[tuple[Point2D, ...], BeforeValidator(_coerce_point2d_sequence)] = Field(
        ...,
        description="Exterior ring coordinates (mm). First/last may be same; will be closed.",
        min_length=3,
    )

    voids_coords: Annotated[tuple[tuple[Point2D, ...], ...], BeforeValidator(_coerce_voids)] = Field(
        default_factory=tuple,
        description="Interior rings (holes), each a tuple of coordinates (mm).",
    )

    @cached_property
    def outline(self) -> Polygon:
        """
        Public Shapely polygon for downstream geometry operations (cached).

        Recomputed only when the cache is invalidated (e.g. after reassigning
        outline_coords / voids_coords).
        """
        return self._build_outline_polygon()


    def _invalidate_outline_cache(self) -> None:
        """Clear cached Shapely outline so it rebuilds next time it's accessed."""
        self.__dict__.pop("outline", None)  # type: ignore[union-attr]  # cached_property stores value on the instance

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override to invalidate outline cache when coords change.

        Note: Reconciliation is handled by model_validator (which Pydantic calls
        after field assignment when validate_assignment=True). For atomic updates
        with rollback capability, use update_outline() instead.
        """
        super().__setattr__(name, value)

        # Invalidate cached outline when geometry coords change
        if name in {"outline_coords", "voids_coords"}:
            self._invalidate_outline_cache()


    rebar_groups: list[RebarGroup] = Field(
        default_factory=list,
        description="List of rebar groups in the section"
    )

    concrete_cover_override: float | None = Field(
        default=None,
        description="Optional override for concrete cover (mm). If None, calculate from geometry.",
        gt=0,
    )

    section_name: str | None = Field(
        default=None,
        description="Section identifier"
    )

    @model_validator(mode="after")
    def validate_outline_and_rebars(self) -> RCSection:
        """
        Validate polygon geometry and reconcile reinforcement.

        This runs:
        - During __init__
        - After any field assignment (when validate_assignment=True)

        For atomic updates with rollback, use update_outline() which sets
        _suspend_outline_reconcile=True to bypass reconciliation here.
        """
        # Skip if suspended (atomic updates via update_outline handle reconciliation themselves)
        if getattr(self, "_suspend_outline_reconcile", False):
            return self

        # Rebuild and validate the polygon
        self._invalidate_outline_cache()
        poly = self.outline

        if poly.is_empty:
            raise ValueError("Section outline is empty")
        if not poly.is_valid:
            raise ValueError("Section outline is not a valid polygon")
        if poly.area <= 0:
            raise ValueError("Section outline has zero or negative area")

        # Reconcile reinforcement according to policy
        if self.rebar_groups:
            self._auto_reconcile_reinforcement()

        # Check for cross-group bar clashes
        if len(self.rebar_groups) > 1:
            from .reinforcement_reconcile import find_clashing_rebars
            details, clashes = find_clashing_rebars(self)
            if clashes:
                msg = "Rebar bars clash across groups:\n- " + "\n- ".join(details)
                raise ValueError(msg)

        return self


    # --- RCSection: add a public atomic outline update method ---
    def update_outline(
        self,
        *,
        outline_coords: tuple[Point2D, ...],
        voids_coords: tuple[tuple[Point2D, ...], ...] | None = None,
        reinforcement_policy: ReinforcementInvalidPolicy | None = None,
    ) -> ReinforcementUpdateReport:
        """
        Atomically update outline/voids, rebuild outline, then reconcile reinforcement once.

        If policy is "error" and reinforcement becomes invalid, this rolls back to the
        previous coords and re-raises.
        """
        old_outline_coords = self.outline_coords
        old_voids_coords = self.voids_coords

        if voids_coords is None:
            voids_coords = old_voids_coords

        # allow temporary override, else use instance policy
        old_policy = self.reinforcement_policy
        override = reinforcement_policy is not None

        self._suspend_outline_reconcile = True
        try:
            if override:
                # Use super().__setattr__ to avoid triggering reconcile
                super().__setattr__("reinforcement_policy", reinforcement_policy)

            # Use super().__setattr__ to bypass __setattr__ reconcile logic
            super().__setattr__("outline_coords", outline_coords)
            super().__setattr__("voids_coords", voids_coords)

            self._invalidate_outline_cache()
            _ = self.outline  # force build / validate polygon

            return self._auto_reconcile_reinforcement()

        except Exception:
            # rollback everything using super().__setattr__ to avoid triggering reconcile
            super().__setattr__("outline_coords", old_outline_coords)
            super().__setattr__("voids_coords", old_voids_coords)
            if override:
                # Restore policy while still suspended
                super().__setattr__("reinforcement_policy", old_policy)
            self._invalidate_outline_cache()
            _ = self.outline
            raise

        finally:
            # Restore policy while still suspended (before turning off suspend)
            if override:
                super().__setattr__("reinforcement_policy", old_policy)
            self._suspend_outline_reconcile = False


    # --- RCSection: update _build_outline_polygon to work with tuples + ensure closure ---
    def _build_outline_polygon(self) -> Polygon:
        ext = [(float(p.x), float(p.y)) for p in self.outline_coords]
        if len(ext) < 3:
            raise ValueError("outline_coords must contain at least 3 points.")
        if ext[0] != ext[-1]:
            ext.append(ext[0])

        holes: list[list[tuple[float, float]]] = []
        for ring in self.voids_coords:
            if len(ring) < 3:
                raise ValueError("Each void ring must have at least 3 points.")
            coords = [(float(p.x), float(p.y)) for p in ring]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            holes.append(coords)

        return Polygon(ext, holes=holes)


    def _auto_reconcile_reinforcement(self) -> ReinforcementUpdateReport:
        """
        Enforce reinforcement_policy after outline coords changes.
        Local import avoids circular imports.
        """
        from .reinforcement_reconcile import reconcile_after_outline_change
        return reconcile_after_outline_change(self, policy=self.reinforcement_policy)


    def invalid_rebars(self) -> tuple[list[str], list[tuple[int, int]]]:
        '''
        Utility method to check if there are any invalid bars outside the bounds of the section

        Returns: (tuple)
            details: a list of reports invalid bars as str
            indices: a list of tuples containing 'group' index and 'bar' index of invalid bars
        '''
        from .reinforcement_reconcile import find_invalid_rebars
        details, indices = find_invalid_rebars(self)
        return details, indices


    #--------------------------
    # Geometry Utility Methods
    #--------------------------

    def get_area(self) -> float:
        """
        Gross concrete area (excluding rebar).

        Returns:
            Area in mm²

        Note:
            Shapely's area correctly accounts for holes/voids.
        """
        return float(self.outline.area)


    def get_transformed_area(self, E_cm: float) -> float:
        """
        Calculate transformed area of the section using transformed section method.

        Assumes self.get_area() returns GROSS concrete polygon area (i.e. it includes
        the area occupied by rebars, since rebars are not modelled as holes).

        Therefore steel contributes (α_e - 1) * A_s, not α_e * A_s.

        Args:
            E_cm: The elastic modulus of concrete in MPa

        Returns:
            Transformed area in mm²
        """
        if E_cm <= 0:
            raise ValueError(f"Concrete modulus E_cm must be positive, got {E_cm}")

        A_eff = self.get_area()  # mm², includes bar regions as concrete

        # Transformed area. Because get_area() is the GROSS area (the bar regions
        # are already counted as concrete), the steel contributes the EXTRA
        # (n-1)·A_steel, not n·A_steel: A_c,tr = A_gross + (alpha_e - 1)·A_steel.
        # Calculate weighted average E_s for all steel
        total_steel_stiffness = 0.0
        total_steel_area = 0.0

        for group in self.rebar_groups:
            group_area = len(group.positions) * group.rebar.area
            group_E_s = group.rebar.E_s
            total_steel_stiffness += group_area * group_E_s
            total_steel_area += group_area

        if total_steel_area > 0:
            E_s_avg = total_steel_stiffness / total_steel_area
            alpha_e = E_s_avg / E_cm
            A_eff += (alpha_e - 1.0) * total_steel_area

        return A_eff


    def get_centroid(self) -> tuple[float, float]:
        """
        Centroid of gross concrete section.

        Returns:
            (x, y) coordinates in mm

        Note:
            Shapely's centroid correctly accounts for holes/voids.
        """
        c = self.outline.centroid
        return (float(c.x), float(c.y))


    def get_transformed_centroid(self, E_cm: float) -> tuple[float, float, float]:
        """
        Centroid of transformed section (gross concrete + (n-1) steel areas).

        Assumes self.get_area() / self.get_centroid() are for the gross concrete
        polygon (i.e. rebar is not modelled as holes), so steel contributes only
        (n-1) * A_s (the extra over the concrete already counted).

        Args:
            E_cm: The elastic modulus of concrete in MPa

        Returns:
            (A_tr, cx_tr, cy_tr)
        """
        if E_cm <= 0:
            raise ValueError(f"Concrete modulus E_c must be positive, got {E_cm}")

        A_gross = self.get_area()
        cx_g, cy_g = self.get_centroid()

        # Start with gross concrete polygon contribution
        A_tr = A_gross
        Sx = A_gross * cx_g
        Sy = A_gross * cy_g

        # Add "extra" transformed steel areas at bar locations
        for group in self.rebar_groups:
            factor = group.rebar.E_s / E_cm - 1.0
            if factor == 0.0:
                continue

            # add first moments of the "extra" transformed area at bar positions
            A_extra_bar = factor * group.rebar.area
            for pos in group.positions:
                A_tr += A_extra_bar
                Sx += A_extra_bar * pos.x
                Sy += A_extra_bar * pos.y

        cx_tr = Sx / A_tr
        cy_tr = Sy / A_tr
        return A_tr, cx_tr, cy_tr


    def get_second_moment_area(self) -> tuple[float, float, float]:
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
        return (abs(Ixx_c), abs(Iyy_c), Ixy_c)


    def get_transformed_second_moment_area(
        self,
        E_cm: float,
    ) -> tuple[float, float, float]:
        """
        Second moments of area for transformed section including reinforcement.

        Uses transformed section method with modular ratio α_e = E_s / E_c.
        Each steel bar contributes (α_e - 1) · A_s via parallel axis terms.

        Args:
            E_cm: The elastic modulus of concrete in MPa

        Returns:
            (I_xx, I_yy, I_xy) in mm⁴
        """
        if E_cm <= 0:
            raise ValueError(f"Concrete modulus E_c must be positive, got {E_cm}")
        if not self.rebar_groups:
            return self.get_second_moment_area()

        # Gross concrete centroid and second moments (assumed about gross centroid axes)
        cx_g, cy_g = self.get_centroid()
        I_xx_g, I_yy_g, I_xy_g = self.get_second_moment_area()
        A_gross = self.get_area()

        _, cx_t, cy_t = self.get_transformed_centroid(E_cm)

        # Shift concrete I from gross centroid to transformed centroid
        dx_c = cx_g - cx_t
        dy_c = cy_g - cy_t
        I_xx = I_xx_g + A_gross * dy_c**2
        I_yy = I_yy_g + A_gross * dx_c**2
        I_xy = I_xy_g + A_gross * dx_c * dy_c

        # Add steel "extra stiffness" terms about transformed centroid
        for group in self.rebar_groups:
            factor = group.rebar.E_s / E_cm - 1.0

            A_extra = factor * group.rebar.area
            for pos in group.positions:
                dx = pos.x - cx_t
                dy = pos.y - cy_t
                I_xx += A_extra * dy**2
                I_yy += A_extra * dx**2
                I_xy += A_extra * dx * dy

        return I_xx, I_yy, I_xy


    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """
        Bounding box of section.

        Returns:
            (min_x, min_y, max_x, max_y) in mm
        """
        min_x, min_y, max_x, max_y = self.outline.bounds
        return (float(min_x), float(min_y), float(max_x), float(max_y))


    def is_symmetric_about_vertical_axis(self, tol: float = 0.01) -> bool:
        """
        Check whether the section (geometry + rebar) is approximately symmetric
        about the vertical axis through the gross concrete centroid.

        When this returns False, the 2D M-N solver (horizontal NA only) cannot
        satisfy minor-axis equilibrium. The biaxial solver should be used instead.

        Uses two independent checks:
        1. Geometry: the product of inertia I_xy ≈ 0 relative to I_xx and I_yy
        2. Rebar: for each bar at (x, y), a matching bar exists at (2·cx − x, y)

        Args:
            tol: Relative tolerance for geometry check and absolute tolerance (mm)
                 for rebar position matching.

        Returns:
            True if section is symmetric about the vertical centroidal axis.
        """
        import math

        # 1. Geometry check via product of inertia
        I_xx, I_yy, I_xy = self.get_second_moment_area()
        I_ref = math.sqrt(max(I_xx * I_yy, 1e-18))
        if abs(I_xy) > tol * I_ref:
            return False

        # 2. Rebar symmetry check
        if self.rebar_groups:
            cx, _ = self.get_centroid()

            # Collect all bar positions and areas
            bars: list[tuple[float, float, float]] = []
            for group in self.rebar_groups:
                for pos in group.positions:
                    bars.append((float(pos.x), float(pos.y), float(group.rebar.area)))

            if bars:
                matched = [False] * len(bars)
                for i, (xi, yi, ai) in enumerate(bars):
                    if matched[i]:
                        continue
                    # Find mirror bar at (2*cx - xi, yi)
                    mirror_x = 2.0 * cx - xi
                    found = False
                    for j, (xj, yj, aj) in enumerate(bars):
                        if j == i or matched[j]:
                            continue
                        if (abs(xj - mirror_x) < tol
                                and abs(yj - yi) < tol
                                and abs(aj - ai) < max(tol * ai, 1e-6)):
                            matched[i] = True
                            matched[j] = True
                            found = True
                            break
                    if not found:
                        # Bar on centroidal axis counts as its own mirror
                        if abs(xi - cx) < tol:
                            matched[i] = True
                        else:
                            return False

        return True

    @property
    def total_steel_area(self) -> float:
        """Total area of all reinforcement in mm²."""
        return sum(group.total_area for group in self.rebar_groups)


    @property
    def reinforcement_ratio(self) -> float:
        """Reinforcement ratio (ρ = A_s / A_c)."""
        a_c = self.get_area()
        return 0.0 if a_c == 0.0 else (self.total_steel_area / a_c)


    def get_rebar_positions(self) -> list[tuple[float, float, float]]:
        """
        Get all rebar positions with areas.

        Returns:
            List of (x, y, area) tuples for each bar
        """
        out: list[tuple[float, float, float]] = []
        for group in self.rebar_groups:
            for pos in group.positions:
                out.append((pos.x, pos.y, group.rebar.area))
        return out


    def get_steel_centroid(self) -> tuple[float, float]:
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
                        or clashes with existing bars in other groups.
        """
        r = float(group.rebar.diameter) / 2.0
        for pos in group.positions:
            disc = ShapelyPoint(pos.x, pos.y).buffer(r)
            if not self.outline.covers(disc):
                raise ValueError(
                    f"Rebar (ϕ{group.rebar.diameter:g}) at ({pos.x:.1f}, {pos.y:.1f}) "
                    "is not fully within the section outline (may cross boundary or enter a void)."
                )

        # Check for clashes with existing groups
        r_new = float(group.rebar.diameter) / 2.0
        for gi, existing in enumerate(self.rebar_groups):
            r_ex = float(existing.rebar.diameter) / 2.0
            min_dist = (r_new + r_ex) - _GEOM_TOL_MM
            min_dist_sq = min_dist * min_dist
            for p_new in group.positions:
                for bi, p_ex in enumerate(existing.positions):
                    dx = p_new.x - p_ex.x
                    dy = p_new.y - p_ex.y
                    if (dx * dx + dy * dy) < min_dist_sq:
                        raise ValueError(
                            f"New bar (ϕ{group.rebar.diameter:g}) at "
                            f"({p_new.x:.1f}, {p_new.y:.1f}) clashes with "
                            f"group[{gi}] bar[{bi}] "
                            f"(ϕ{existing.rebar.diameter:g}) at "
                            f"({p_ex.x:.1f}, {p_ex.y:.1f}). "
                            "Bars may touch but must not overlap."
                        )

        self.rebar_groups.append(group)


    def remove_bars(
        self,
        *,
        group_index: int | None = None,
        layer_name: str | None = None,
        positions: Sequence[Point2D | tuple[float, float]] | None = None,
        bar_indices: Sequence[int] | None = None,
    ) -> int:
        """
        Remove bars matching the given criteria.

        Filtering:
            ``group_index`` and/or ``layer_name`` select which group(s) to
            target.  Within those groups, ``positions`` (matched by coordinate
            within tolerance) or ``bar_indices`` select individual bars.  If
            neither is given the entire matching group(s) are removed.

        Returns:
            Number of bars removed.
        """
        # Normalise position targets to (x, y) floats
        pos_targets: list[tuple[float, float]] | None = None
        if positions is not None:
            pos_targets = []
            for p in positions:
                if isinstance(p, Point2D):
                    pos_targets.append((p.x, p.y))
                else:
                    pos_targets.append((float(p[0]), float(p[1])))

        removed = 0
        new_groups: list[RebarGroup] = []

        for gi, group in enumerate(self.rebar_groups):
            # Check if this group is targeted
            match_gi = group_index is None or gi == group_index
            match_ln = layer_name is None or group.layer_name == layer_name
            targeted = match_gi and match_ln

            if not targeted:
                new_groups.append(group)
                continue

            # If no per-bar filter, remove entire group
            if pos_targets is None and bar_indices is None:
                removed += len(group.positions)
                continue

            # Build set of bar indices to remove
            remove_set: set[int] = set()
            if bar_indices is not None:
                remove_set.update(bar_indices)
            if pos_targets is not None:
                tol_sq = _GEOM_TOL_MM * _GEOM_TOL_MM
                for bi, bp in enumerate(group.positions):
                    for tx, ty in pos_targets:
                        dx = bp.x - tx
                        dy = bp.y - ty
                        if (dx * dx + dy * dy) <= tol_sq:
                            remove_set.add(bi)
                            break

            kept = tuple(p for i, p in enumerate(group.positions) if i not in remove_set)
            removed += len(group.positions) - len(kept)

            if kept:
                new_groups.append(
                    RebarGroup(
                        rebar=group.rebar,
                        positions=kept,
                        layer_name=group.layer_name,
                    )
                )

        self.rebar_groups = new_groups
        return removed


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

        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
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
                return float((dx * dx + dy * dy) ** 0.5)
            t = (wx * vx + wy * vy) / vv
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            cx_ = ax + t * vx
            cy_ = ay + t * vy
            dx = px - cx_
            dy = py - cy_
            return float((dx * dx + dy * dy) ** 0.5)

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
        tension_zone: Literal["top", "bottom"] | None = None,
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

    def get_compression_rebar_depth(
        self,
        compression_face: Literal["top", "bottom"] = "top",
        *,
        zone_fraction: float = 0.5,
    ) -> float | None:
        """
        Distance d_2 from compression face to centroid of compression reinforcement.

        Used for German NA lever arm cap: z_cap = max(d - 2·d_2, d - d_2 - 30)

        Args:
            compression_face: "top" or "bottom" compression edge reference.
            zone_fraction: Fraction of depth considered as compression zone (0 < f <= 1).
                          0.5 means upper/lower half only.

        Returns:
            d_2 (mm) or None if no compression reinforcement found.
        """
        if not self.rebar_groups:
            return None

        if compression_face not in ("top", "bottom"):
            raise ValueError(f"compression_face must be 'top' or 'bottom', got {compression_face}")

        if not (0.0 < zone_fraction <= 1.0):
            raise ValueError(f"zone_fraction must be in (0, 1], got {zone_fraction}")

        _, min_y, _, max_y = self.get_bounding_box()
        h = max_y - min_y
        if h <= 0.0:
            return None

        # Define the compression zone band
        if compression_face == "top":
            y_limit = max_y - zone_fraction * h  # include bars with y >= y_limit
            def in_zone(y: float) -> bool:
                return y >= y_limit - _GEOM_TOL_MM
        else:
            y_limit = min_y + zone_fraction * h  # include bars with y <= y_limit
            def in_zone(y: float) -> bool:
                return y <= y_limit + _GEOM_TOL_MM

        # Collect compression-zone bars (area-weighted)
        A = 0.0
        my = 0.0

        for group in self.rebar_groups:
            a_bar = float(group.rebar.area)
            for pos in group.positions:
                y = float(pos.y)
                if in_zone(y):
                    A += a_bar
                    my += a_bar * y

        if A <= 0.0:
            # No compression reinforcement found
            return None

        centroid_y = my / A

        # Distance from compression face to compression rebar centroid
        d_2 = (max_y - centroid_y) if compression_face == "top" else (centroid_y - min_y)
        return float(d_2)

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


    def plot(
        self,
        *,
        concrete: ConcreteMaterial | None = None,
        show: bool = True,
        title: str | None = None,
        width: int = 700,
        height: int = 700,
    ) -> Any:
        """
        Create an interactive Plotly figure of the section cross-section.

        Thin wrapper that delegates to SectionViewer in a separate module
        to keep RCSection focused on geometry rather than plotting.
        """
        from materials.reinforced_concrete.geometry.section_viewer import SectionViewer

        viewer = SectionViewer(self)
        return viewer.plot(
            concrete=concrete,
            show=show,
            title=title,
            width=width,
            height=height,
        )


# ---------------------------------------------------------------------------
# Section creation helpers — moved to section_utils.py, re-exported here
# for backward compatibility.
# ---------------------------------------------------------------------------
