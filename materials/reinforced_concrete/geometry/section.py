"""
Reinforced concrete section geometry using Shapely for 2D polygonal shapes.

Provides flexible geometry definition with arbitrary polygonal outlines
and rebar positioning.
"""

from typing import List, Tuple, Optional, Literal
import numpy as np
from shapely.geometry import Polygon, Point as ShapelyPoint
from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict
from materials.core.geometry import BaseGeometry, Point2D
from materials.reinforced_concrete.materials.rebar import Rebar


class RebarGroup(BaseModel):
    """
    Group of rebars with common properties.

    Represents one or more bars at specific locations.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
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

    @field_validator("positions")
    @classmethod
    def validate_positions_unique(cls, v: List[Point2D]) -> List[Point2D]:
        """Ensure bar positions are reasonably unique (no overlapping bars)."""
        if len(v) < 2:
            return v

        # Check for duplicate positions (within 1mm tolerance)
        for i, p1 in enumerate(v):
            for p2 in v[i+1:]:
                distance = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5
                if distance < 1.0:
                    raise ValueError(
                        f"Bars too close together at ({p1.x:.1f}, {p1.y:.1f}) "
                        f"and ({p2.x:.1f}, {p2.y:.1f}). Minimum spacing 1mm."
                    )
        return v

    @computed_field
    @property
    def n_bars(self) -> int:
        """Number of bars in this group."""
        return len(self.positions)

    @computed_field
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
    - Area, centroid, moments of inertia
    - Containment checking (bar positions within section)
    - Geometric transformations

    Coordinate system:
    - Origin at user-defined location
    - X-axis: typically section width direction
    - Y-axis: typically section height direction
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
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

    @field_validator("outline")
    @classmethod
    def validate_outline(cls, v: Polygon) -> Polygon:
        """Validate outline polygon."""
        if not v.is_valid:
            raise ValueError("Section outline is not a valid polygon")
        if v.is_empty:
            raise ValueError("Section outline is empty")
        if v.area <= 0:
            raise ValueError("Section outline has zero or negative area")
        return v

    @field_validator("rebar_groups")
    @classmethod
    def validate_rebars_in_section(cls, v: List[RebarGroup], info) -> List[RebarGroup]:
        """Validate that all rebars are within the section outline."""
        if "outline" not in info.data:
            return v

        outline: Polygon = info.data["outline"]

        for group in v:
            for pos in group.positions:
                point = ShapelyPoint(pos.x, pos.y)
                if not outline.contains(point):
                    # Allow bars on the boundary
                    if not outline.boundary.distance(point) < 1e-6:
                        raise ValueError(
                            f"Rebar at ({pos.x:.1f}, {pos.y:.1f}) is outside section outline"
                        )
        return v

    def get_area(self) -> float:
        """
        Gross concrete area (excluding rebar).

        Returns:
            Area in mm²
        """
        return self.outline.area

    def get_centroid(self) -> Tuple[float, float]:
        """
        Centroid of gross concrete section.

        Returns:
            (x, y) coordinates in mm
        """
        centroid = self.outline.centroid
        return (centroid.x, centroid.y)

    def get_second_moment_area(self) -> Tuple[float, float, float]:
        """
        Second moments of area about centroidal axes (gross concrete section only).

        Uses parallel axis theorem with Shapely.

        Returns:
            (I_xx, I_yy, I_xy) in mm⁴

        Note:
            This returns the gross concrete section properties only.
            For transformed section properties including steel, use
            get_transformed_second_moment_area().
        """
        # Get centroid
        cx, cy = self.get_centroid()

        # Get coordinates of polygon boundary
        coords = np.array(self.outline.exterior.coords[:-1])  # Exclude last point (duplicate)
        x = coords[:, 0] - cx  # Translate to centroid
        y = coords[:, 1] - cy

        # Calculate using shoelace formula for polygon moments
        n = len(x)
        I_xx = 0.0
        I_yy = 0.0
        I_xy = 0.0

        for i in range(n):
            j = (i + 1) % n
            cross = x[i] * y[j] - x[j] * y[i]

            I_xx += (y[i]**2 + y[i]*y[j] + y[j]**2) * cross
            I_yy += (x[i]**2 + x[i]*x[j] + x[j]**2) * cross
            I_xy += (x[i]*y[j] + 2*x[i]*y[i] + 2*x[j]*y[j] + x[j]*y[i]) * cross

        I_xx = abs(I_xx) / 12.0
        I_yy = abs(I_yy) / 12.0
        I_xy = abs(I_xy) / 24.0

        return (I_xx, I_yy, I_xy)

    def get_transformed_second_moment_area(
        self,
        E_c: float,
        centroid: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Second moments of area for transformed section including reinforcement.

        Uses the transformed section method where steel is converted to equivalent
        concrete using the modular ratio α_e = E_s / E_c. Each steel bar contributes
        (α_e - 1) · A_s to the transformed area.

        Formulation:
            I_transformed = I_concrete + Σ[(α_e - 1) · A_s · d²]

        where d is the distance from the bar to the centroid axis.

        Args:
            E_c: Concrete elastic modulus in MPa (typically E_cm from ConcreteMaterial)
            centroid: Optional centroid to use. If None, uses gross section centroid.

        Returns:
            (I_xx, I_yy, I_xy) in mm⁴

        Raises:
            ValueError: If E_c <= 0 or no rebar groups present

        Note:
            - Uses (α_e - 1) factor to account for concrete already present at bar location
            - Assumes bars are small relative to section (point area approximation)
            - For uncracked section analysis, use gross section centroid
            - For cracked section, calculate neutral axis position first
        """
        if E_c <= 0:
            raise ValueError(f"Concrete modulus E_c must be positive, got {E_c}")

        if not self.rebar_groups:
            raise ValueError("Cannot calculate transformed properties: no rebars in section")

        # Use provided centroid or gross section centroid
        if centroid is None:
            cx, cy = self.get_centroid()
        else:
            cx, cy = centroid

        # Start with gross concrete section
        I_xx_concrete, I_yy_concrete, I_xy_concrete = self.get_second_moment_area()

        # Add contribution from each rebar group
        I_xx_steel = 0.0
        I_yy_steel = 0.0
        I_xy_steel = 0.0

        for group in self.rebar_groups:
            # Get modular ratio for this rebar material
            E_s = group.rebar.E_s
            alpha_e = E_s / E_c

            # Use (α_e - 1) to account for concrete already at bar location
            factor = alpha_e - 1.0

            for pos in group.positions:
                # Distance from bar to centroid
                dx = pos.x - cx
                dy = pos.y - cy

                # Contribution using parallel axis theorem
                # I = Σ[A · d²] where A is the transformed area
                A_transformed = factor * group.rebar.area

                I_xx_steel += A_transformed * dy**2
                I_yy_steel += A_transformed * dx**2
                I_xy_steel += A_transformed * dx * dy

        return (
            I_xx_concrete + I_xx_steel,
            I_yy_concrete + I_yy_steel,
            I_xy_concrete + I_xy_steel,
        )

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Bounding box of section.

        Returns:
            (min_x, min_y, max_x, max_y) in mm
        """
        bounds = self.outline.bounds
        return bounds

    @computed_field
    @property
    def total_steel_area(self) -> float:
        """
        Total area of all reinforcement.

        Returns:
            Total steel area in mm²
        """
        return sum(group.total_area for group in self.rebar_groups)

    @computed_field
    @property
    def reinforcement_ratio(self) -> float:
        """
        Reinforcement ratio (ρ = A_s / A_c).

        Returns:
            Ratio (dimensionless)
        """
        if self.get_area() == 0:
            return 0.0
        return self.total_steel_area / self.get_area()

    @computed_field
    @property
    def concrete_cover(self) -> float:
        """
        Minimum concrete cover (top/bottom faces only).

        Convenient property for accessing cover with default settings.
        Equivalent to get_concrete_cover(orthogonal_only=True).

        Returns:
            Minimum concrete cover in mm

        Raises:
            ValueError: If no rebars in section and no override set

        Note:
            For custom calculations (specific face, all faces), use get_concrete_cover() method.
            This property is not cached and recalculates on each access.
        """
        return self.get_concrete_cover(orthogonal_only=True)

    def get_rebar_positions(self) -> List[Tuple[float, float, float]]:
        """
        Get all rebar positions with areas.

        Returns:
            List of (x, y, area) tuples for each bar
        """
        positions = []
        for group in self.rebar_groups:
            for pos in group.positions:
                positions.append((pos.x, pos.y, group.rebar.area))
        return positions

    def get_steel_centroid(self) -> Tuple[float, float]:
        """
        Calculate centroid of all reinforcement.

        Returns:
            (x, y) coordinates in mm, or (0, 0) if no reinforcement
        """
        if self.total_steel_area == 0:
            return (0.0, 0.0)

        total_area = 0.0
        moment_x = 0.0
        moment_y = 0.0

        for group in self.rebar_groups:
            for pos in group.positions:
                area = group.rebar.area
                total_area += area
                moment_x += area * pos.x
                moment_y += area * pos.y

        return (moment_x / total_area, moment_y / total_area)

    def add_rebar_group(self, group: RebarGroup) -> None:
        """
        Add a rebar group to the section.

        Args:
            group: RebarGroup to add

        Raises:
            ValueError: If rebars are outside section
        """
        # Validate positions are within section
        for pos in group.positions:
            point = ShapelyPoint(pos.x, pos.y)
            if not self.outline.contains(point) and self.outline.boundary.distance(point) > 1e-6:
                raise ValueError(
                    f"Rebar at ({pos.x:.1f}, {pos.y:.1f}) is outside section outline"
                )

        self.rebar_groups.append(group)

    def get_concrete_cover(
        self,
        orthogonal_only: bool = True,
        face: Optional[Literal["top", "bottom", "left", "right"]] = None
    ) -> float:
        """
        Calculate minimum concrete cover from section boundary to rebar outer surface.

        Cover is defined as the shortest distance from section boundary to the
        outer diameter of any rebar (not centreline)

        Args:
            orthogonal_only: If True (default), only consider top/bottom faces.
                           This avoids edge bars corrupting the calculation.
            face: If specified, only calculate cover for specific face.
                 Overrides orthogonal_only.

        Returns:
            Minimum concrete cover in mm

        Raises:
            ValueError: If no rebars in section

        Note:
            If concrete_cover_override is set, returns that value instead.
        """
        # Use override if provided
        if self.concrete_cover_override is not None:
            return self.concrete_cover_override

        if not self.rebar_groups:
            raise ValueError("Cannot calculate cover: no rebars in section")

        min_x, min_y, max_x, max_y = self.get_bounding_box()
        min_cover = float('inf')

        for group in self.rebar_groups:
            bar_radius = group.rebar.diameter / 2.0

            for pos in group.positions:
                # Calculate distance from bar outer surface to each face
                covers = {}

                if face is None or face == "bottom":
                    covers["bottom"] = pos.y - min_y - bar_radius
                if face is None or face == "top":
                    covers["top"] = max_y - pos.y - bar_radius
                if not orthogonal_only or face == "left":
                    covers["left"] = pos.x - min_x - bar_radius
                if not orthogonal_only or face == "right":
                    covers["right"] = max_x - pos.x - bar_radius

                # Find minimum cover for this bar
                bar_min_cover = min(covers.values())
                min_cover = min(min_cover, bar_min_cover)

        if min_cover == float('inf'):
            raise ValueError("Could not calculate cover")

        return min_cover

    def get_effective_depth(self, reference: Literal["top", "bottom", "left", "right"] = "top") -> float:
        """
        Calculate effective depth from reference edge to steel centroid.

        Args:
            reference: Edge to measure from

        Returns:
            Effective depth in mm
        """
        steel_cx, steel_cy = self.get_steel_centroid()
        min_x, min_y, max_x, max_y = self.get_bounding_box()

        if reference == "top":
            return max_y - steel_cy
        elif reference == "bottom":
            return steel_cy - min_y
        elif reference == "left":
            return steel_cx - min_x
        elif reference == "right":
            return max_x - steel_cx
        else:
            raise ValueError(f"Unknown reference: {reference}")

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
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create a rectangular RC section.

    Args:
        width: Section width (mm)
        height: Section height (mm)
        origin: Bottom-left corner coordinates (default: origin)
        section_name: Optional section name

    Returns:
        RCSection with rectangular outline

    Example:
        >>> section = create_rectangular_section(300, 500)
        >>> section.get_area()
        150000.0
    """
    x0, y0 = origin
    coords = [
        (x0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height),
        (x0, y0 + height),
        (x0, y0),  # Close the polygon
    ]
    outline = Polygon(coords)

    return RCSection(
        outline=outline,
        section_name=section_name or f"Rect {width}×{height}",
    )


def create_circular_section(
    diameter: float,
    n_points: int = 32,
    origin: Tuple[float, float] = (0.0, 0.0),
    section_name: Optional[str] = None,
) -> RCSection:
    """
    Create a circular RC section.

    Args:
        diameter: Section diameter (mm)
        n_points: Number of points to approximate circle (default: 32)
        origin: Centre coordinates (default: origin)
        section_name: Optional section name

    Returns:
        RCSection with circular outline

    Example:
        >>> section = create_circular_section(400)
        >>> round(section.get_area())
        125664
    """
    cx, cy = origin
    radius = diameter / 2.0

    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    coords = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
    coords.append(coords[0])  # Close the polygon

    outline = Polygon(coords)

    return RCSection(
        outline=outline,
        section_name=section_name or f"Circular Ø{diameter}",
    )
