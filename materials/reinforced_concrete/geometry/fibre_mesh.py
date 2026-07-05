"""
Fibre mesh generation for section analysis (M-N interaction diagrams).

Divides RC sections into discrete fibres for strain compatibility analysis.
Each fibre has:
- Position (centroid)
- Area
- Material type (concrete or steel)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from materials.reinforced_concrete.geometry.section import RCSection


@dataclass(frozen=True, slots=True)
class Fibre:
    """
    Single fibre element for section analysis.

    Attributes:
        x: X-coordinate of fibre centroid (mm)
        y: Y-coordinate of fibre centroid (mm)
        area: Fibre area (mm²)
        material_type: 'concrete' or 'steel'
        material_index: Index into material array (e.g. rebar group index)
        i: width index (grid column). -1 means "not on grid" (e.g. steel fibre).
        j: height index (grid row). -1 means "not on grid" (e.g. steel fibre).
    """

    x: float
    y: float
    area: float
    material_type: Literal["concrete", "steel"]
    material_index: int = 0
    i: int = -1
    j: int = -1

    def __repr__(self) -> str:
        return (
            "Fibre("
            f"x={self.x:.1f}, "
            f"y={self.y:.1f}, "
            f"area={self.area:.1f}, "
            f"material_type={self.material_type}, "
            f"material_index={self.material_index}, "
            f"i={self.i}, "
            f"j={self.j}"
            ")"
        )


class FibreMesh:
    """
    Fibre mesh for RC section analysis.

    Divides the section into discrete fibres:
    - Concrete fibres: rectangular grid across section
    - Steel fibres: individual rebars

    Notes:
        If exclude_steel_area=True, concrete fibre areas are reduced by the
        *geometric overlap* between the concrete region in the cell and the union
        of rebar discs (Shapely circles). This is more accurate than allocating
        full bar area to a single cell.

        Derived properties are stored as attributes on the instance:
            - concrete_fibres
            - steel_fibres
    """

    def __init__(
        self,
        section: RCSection,
        n_fibres_width: int = 20,
        n_fibres_height: int = 20,
        exclude_steel_area: bool = True,
    ) -> None:
        # Validate cheap things early (fail fast)
        if n_fibres_width <= 0 or n_fibres_height <= 0:
            raise ValueError("n_fibres_width and n_fibres_height must be > 0")

        self.section = section
        self.n_fibres_width = int(n_fibres_width)
        self.n_fibres_height = int(n_fibres_height)
        self.exclude_steel_area = bool(exclude_steel_area)

        self.concrete_fibres: list[Fibre] = []
        self.steel_fibres: list[Fibre] = []

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
            if r <= 0.0:
                continue

            for pos in group.positions:
                # buffer() returns a Polygon (which is a BaseGeometry)
                circles.append(Point(float(pos.x), float(pos.y)).buffer(r))

        return circles

    def _generate_mesh(self) -> None:
        """Generate concrete and steel fibres."""
        self._generate_concrete_fibres()
        self._generate_steel_fibres()

        # Optional sanity checks (lightweight)
        # - Helps catch accidental empty meshes early.
        if not self.concrete_fibres:
            raise ValueError(
                "FibreMesh generation produced no concrete fibres. "
                "Check section outline geometry and fibre counts."
            )

    def _generate_concrete_fibres(self) -> None:
        """
        Generate concrete fibres using a rectangular grid.

        Algorithm:
            1) Divide section bounding box into a regular grid of cells.
            2) For each cell, compute concrete region within the cell:
                   cell_concrete = outline ∩ cell_box
            3) If exclude_steel_area=True, subtract the *overlap* between
               cell_concrete and the union of rebar discs.
            4) Use the centroid of the remaining concrete region as fibre position.

        Notes:
            - Edge cells produce partial fibres (clipped by outline).
            - If the remaining area is negligible, the fibre is skipped.
            - When excluding steel area, centroid is computed from the *remaining*
              concrete geometry (after subtraction) to be consistent.
        """
        min_x, min_y, max_x, max_y = self.section.get_bounding_box()

        dx = (max_x - min_x) / self.n_fibres_width
        dy = (max_y - min_y) / self.n_fibres_height

        if dx <= 0.0 or dy <= 0.0:
            raise ValueError("Section bounding box is degenerate (zero width/height).")

        # Small area threshold (mm²)
        eps_area = 1e-6

        for i in range(self.n_fibres_width):
            x0 = min_x + i * dx
            x1 = x0 + dx

            for j in range(self.n_fibres_height):
                y0 = min_y + j * dy
                y1 = y0 + dy

                cell = box(x0, y0, x1, y1)

                # Quick reject: if outline doesn't intersect cell bbox, skip
                if not self._outline_prepared.intersects(cell):
                    continue

                cell_concrete = self._outline.intersection(cell)
                if cell_concrete.is_empty:
                    continue

                # Subtract steel overlap geometrically (accurate)
                # Use difference() so area and centroid are consistent.
                remaining = cell_concrete
                if self.exclude_steel_area and self._bar_union is not None:
                    # Quick reject: if bar union doesn't intersect cell, avoid difference call
                    if self._bar_union_prepared is None or self._bar_union_prepared.intersects(cell):
                        remaining = cell_concrete.difference(self._bar_union)

                if remaining.is_empty:
                    continue

                area_concrete = float(remaining.area)
                if area_concrete < eps_area:
                    continue

                # Use centroid of remaining concrete region
                c = remaining.centroid
                self.concrete_fibres.append(
                    Fibre(
                        x=float(c.x),
                        y=float(c.y),
                        area=area_concrete,
                        material_type="concrete",
                        material_index=0,
                        i=i,
                        j=j,
                    )
                )

    def _generate_steel_fibres(self) -> None:
        """
        Generate steel fibres (one per rebar).

        Each rebar is treated as a discrete fibre at its centroid.
        material_index tracks which rebar group the fibre belongs to.

        Note:
            Bar areas are taken from the group's rebar definition (group.rebar.area).
            Groups can have different diameters/areas from each other.
        """
        for group_idx, group in enumerate(self.section.rebar_groups):
            a_bar = float(group.rebar.area)
            if a_bar <= 0.0:
                continue

            for pos in group.positions:
                self.steel_fibres.append(
                    Fibre(
                        x=float(pos.x),
                        y=float(pos.y),
                        area=a_bar,
                        material_type="steel",
                        material_index=group_idx,
                    )
                )

    @property
    def all_fibres(self) -> list[Fibre]:
        """Get all fibres (concrete + steel)."""
        # Note: this creates a new list each call. Fine for occasional use.
        return self.concrete_fibres + self.steel_fibres

    @property
    def n_concrete_fibres(self) -> int:
        return len(self.concrete_fibres)

    @property
    def n_steel_fibres(self) -> int:
        return len(self.steel_fibres)

    @property
    def total_fibres(self) -> int:
        return self.n_concrete_fibres + self.n_steel_fibres

    def __repr__(self) -> str:
        return (
            f"FibreMesh({self.total_fibres} fibres: "
            f"{self.n_concrete_fibres} concrete, {self.n_steel_fibres} steel)"
        )

    def get_fibre_arrays(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],  # x
        npt.NDArray[np.float64],  # y
        npt.NDArray[np.float64],  # area
        npt.NDArray[np.str_],     # material_type
        npt.NDArray[np.int32],    # material_index
        npt.NDArray[np.int32],    # i
        npt.NDArray[np.int32],    # j
    ]:
        """
        Get fibre data as numpy arrays for vectorized calculations.

        Returns:
            (x, y, area, material_type, material_index, i, j)

        Notes:
            - i/j indices apply to concrete fibres (grid indices).
            - Steel fibres have i=j=-1.
            - This function always returns i/j to keep callers simple and avoid
            union return types that confuse type checkers.
        """
        all_f = self.all_fibres

        x = np.array([f.x for f in all_f], dtype=np.float64)
        y = np.array([f.y for f in all_f], dtype=np.float64)
        area = np.array([f.area for f in all_f], dtype=np.float64)
        material_type = np.array([f.material_type for f in all_f], dtype=np.str_)
        material_index = np.array([f.material_index for f in all_f], dtype=np.int32)

        # With your new Fibre defaults (i/j = -1), these are always ints already
        ii = np.array([f.i for f in all_f], dtype=np.int32)
        jj = np.array([f.j for f in all_f], dtype=np.int32)

        return x, y, area, material_type, material_index, ii, jj
