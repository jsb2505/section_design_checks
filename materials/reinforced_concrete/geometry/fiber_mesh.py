"""
Fiber mesh generation for section analysis (M-N interaction diagrams).

Divides RC sections into discrete fibers for strain compatibility analysis.
Each fiber has:
- Position (centroid)
- Area
- Material type (concrete or steel)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List, Tuple

import numpy as np
import numpy.typing as npt
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.geometry.base import BaseGeometry


from materials.reinforced_concrete.geometry.section import RCSection


@dataclass(frozen=True, slots=True)
class Fiber:
    """
    Single fiber element for section analysis.

    Attributes:
        x: X-coordinate of fiber centroid (mm)
        y: Y-coordinate of fiber centroid (mm)
        area: Fiber area (mm²)
        material_type: 'concrete' or 'steel'
        material_index: Index into material array (e.g. rebar group index)
    """

    x: float
    y: float
    area: float
    material_type: Literal["concrete", "steel"]
    material_index: int = 0


class FiberMesh:
    """
    Fiber mesh for RC section analysis.

    Divides the section into discrete fibers:
    - Concrete fibers: rectangular grid across section
    - Steel fibers: individual rebars

    Notes:
        If exclude_steel_area=True, concrete fiber areas are reduced by the
        *geometric overlap* between the concrete region in the cell and the union
        of rebar discs (Shapely circles). This is more accurate than allocating
        full bar area to a single cell.

        Derived properties are stored as attributes on the instance:
            - concrete_fibers
            - steel_fibers
    """

    def __init__(
        self,
        section: RCSection,
        n_fibers_width: int = 20,
        n_fibers_height: int = 20,
        exclude_steel_area: bool = True,
    ) -> None:
        self.section = section
        self.n_fibers_width = n_fibers_width
        self.n_fibers_height = n_fibers_height
        self.exclude_steel_area = exclude_steel_area

        self.concrete_fibers: List[Fiber] = []
        self.steel_fibers: List[Fiber] = []

        # Precompute geometry for speed
        self._outline = self.section.outline
        self._outline_prepared = prep(self._outline)

        # Precompute rebar circles and union (used for concrete subtraction)
        self._bar_circles = self._build_rebar_circles()
        self._bar_union = unary_union(self._bar_circles) if self._bar_circles else None
        self._bar_union_prepared = prep(self._bar_union) if self._bar_union else None

        self._generate_mesh()

    def _build_rebar_circles(self) -> list[BaseGeometry]:
        circles: list[BaseGeometry] = []
        for group in self.section.rebar_groups:
            r = float(group.rebar.diameter) / 2.0
            if r <= 0:
                continue

            for pos in group.positions:
                # buffer() returns a Polygon (which is a BaseGeometry)
                circles.append(Point(pos.x, pos.y).buffer(r))

        return circles

    def _generate_mesh(self) -> None:
        """Generate concrete and steel fibers."""
        self._generate_concrete_fibers()
        self._generate_steel_fibers()

    def _generate_concrete_fibers(self) -> None:
        """
        Generate concrete fibers using a rectangular grid.

        Algorithm:
            1) Divide section bounding box into a regular grid of cells.
            2) For each cell, compute concrete region within the cell:
                   cell_concrete = outline ∩ cell_box
            3) If exclude_steel_area=True, subtract the *overlap* between
               cell_concrete and the union of rebar discs.
            4) Use the centroid of the remaining concrete region as fiber position.

        Notes:
            - Edge cells produce partial fibers (clipped by outline).
            - If the remaining area is negligible, the fiber is skipped.
        """
        min_x, min_y, max_x, max_y = self.section.get_bounding_box()

        # Guard against degenerate bbox / invalid mesh sizes
        if self.n_fibers_width <= 0 or self.n_fibers_height <= 0:
            raise ValueError("n_fibers_width and n_fibers_height must be > 0")

        dx = (max_x - min_x) / self.n_fibers_width
        dy = (max_y - min_y) / self.n_fibers_height

        if dx <= 0 or dy <= 0:
            raise ValueError("Section bounding box is degenerate (zero width/height).")

        # Small area threshold (mm²)
        eps_area = 1e-6

        for i in range(self.n_fibers_width):
            x0 = min_x + i * dx
            x1 = x0 + dx

            for j in range(self.n_fibers_height):
                y0 = min_y + j * dy
                y1 = y0 + dy

                cell = box(x0, y0, x1, y1)

                # Quick reject: if outline doesn't intersect cell bbox, skip
                if not self._outline_prepared.intersects(cell):
                    continue

                cell_concrete = self._outline.intersection(cell)
                if cell_concrete.is_empty:
                    continue

                area_concrete = cell_concrete.area
                if area_concrete < eps_area:
                    continue

                # Subtract steel overlap geometrically (accurate)
                if self.exclude_steel_area and self._bar_union is not None:
                    # Quick reject: if bar union doesn't intersect cell, avoid intersection call
                    if self._bar_union_prepared is None or self._bar_union_prepared.intersects(cell):
                        steel_overlap = cell_concrete.intersection(self._bar_union)
                        if not steel_overlap.is_empty:
                            area_concrete -= steel_overlap.area

                    if area_concrete < eps_area:
                        continue

                # Use centroid of remaining concrete region
                c = cell_concrete.centroid
                fiber = Fiber(
                    x=float(c.x),
                    y=float(c.y),
                    area=float(area_concrete),
                    material_type="concrete",
                    material_index=0,
                )
                self.concrete_fibers.append(fiber)

    def _generate_steel_fibers(self) -> None:
        """
        Generate steel fibers (one per rebar).

        Each rebar is treated as a discrete fiber at its centroid.
        material_index tracks which rebar group the fiber belongs to.

        Note:
            Bar areas are taken from the group's rebar definition (group.rebar.area).
            Groups can have different diameters/areas from each other.
        """
        for group_idx, group in enumerate(self.section.rebar_groups):
            for pos in group.positions:
                self.steel_fibers.append(
                    Fiber(
                        x=float(pos.x),
                        y=float(pos.y),
                        area=float(group.rebar.area),
                        material_type="steel",
                        material_index=group_idx,
                    )
                )

    @property
    def all_fibers(self) -> List[Fiber]:
        """Get all fibers (concrete + steel)."""
        return self.concrete_fibers + self.steel_fibers

    @property
    def n_concrete_fibers(self) -> int:
        return len(self.concrete_fibers)

    @property
    def n_steel_fibers(self) -> int:
        return len(self.steel_fibers)

    @property
    def total_fibers(self) -> int:
        return self.n_concrete_fibers + self.n_steel_fibers

    def get_fiber_arrays(
        self,
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.str_],
        npt.NDArray[np.int32],
    ]:
        """
        Get fiber data as numpy arrays for vectorized calculations.

        Returns:
            Tuple of (x, y, area, material_type, material_index) arrays
        """
        all_f = self.all_fibers

        x = np.array([f.x for f in all_f], dtype=np.float64)
        y = np.array([f.y for f in all_f], dtype=np.float64)
        area = np.array([f.area for f in all_f], dtype=np.float64)
        material_type = np.array([f.material_type for f in all_f], dtype=np.str_)
        material_index = np.array([f.material_index for f in all_f], dtype=np.int32)

        return x, y, area, material_type, material_index

    def __repr__(self) -> str:
        return (
            f"FiberMesh("
            f"concrete={self.n_concrete_fibers}, "
            f"steel={self.n_steel_fibers}, "
            f"total={self.total_fibers})"
        )
