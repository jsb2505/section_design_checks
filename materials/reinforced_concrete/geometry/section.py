"""
Reinforced concrete section geometry using Shapely for 2D polygonal shapes.

Provides flexible geometry definition with arbitrary polygonal outlines
and rebar positioning.
"""

from typing import List, Tuple, Optional, Sequence
import numpy as np
from shapely.geometry import Polygon, Point as ShapelyPoint, MultiPoint
from shapely.affinity import translate, rotate, scale
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
        description="List of (x, y) coordinates for bar centers (mm)",
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

    concrete_cover: float = Field(
        default=30.0,
        description="Nominal concrete cover to rebar centerline (mm)",
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
        Second moments of area about centroidal axes (gross section).

        Uses parallel axis theorem with Shapely.

        Returns:
            (I_xx, I_yy, I_xy) in mm⁴
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
        origin: Center coordinates (default: origin)
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
