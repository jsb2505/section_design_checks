"""
Fiber mesh generation for section analysis (M-N interaction diagrams).

Divides RC sections into discrete fibers for strain compatibility analysis.
Each fiber has:
- Position (centroid)
- Area
- Material type (concrete or steel)
"""

from typing import List, Tuple, Literal
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from shapely.geometry import box
from materials.reinforced_concrete.geometry.section import RCSection


@dataclass
class Fiber:
    """
    Single fiber element for section analysis.

    Attributes:
        x: X-coordinate of fiber centroid (mm)
        y: Y-coordinate of fiber centroid (mm)
        area: Fiber area (mm²)
        material_type: 'concrete' or 'steel'
        material_index: Index into material array (for tracking different steel types)
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

    Used for fiber-based strain compatibility analysis in M-N diagrams.
    """

    def __init__(
        self,
        section: RCSection,
        n_fibers_width: int = 20,
        n_fibers_height: int = 20,
        exclude_steel_area: bool = True,
    ):
        """
        Create fiber mesh for a section.

        Args:
            section: RC section to mesh
            n_fibers_width: Number of concrete fibers across width
            n_fibers_height: Number of concrete fibers across height
            exclude_steel_area: If True, subtract steel area from concrete fibers
        """
        self.section = section
        self.n_fibers_width = n_fibers_width
        self.n_fibers_height = n_fibers_height
        self.exclude_steel_area = exclude_steel_area

        self.concrete_fibers: List[Fiber] = []
        self.steel_fibers: List[Fiber] = []

        self._generate_mesh()

    def _generate_mesh(self) -> None:
        """Generate concrete and steel fibers."""
        self._generate_concrete_fibers()
        self._generate_steel_fibers()

    def _generate_concrete_fibers(self) -> None:
        """
        Generate concrete fibers using rectangular grid.

        Divides section bounding box into grid and keeps only fibers
        that overlap with the section outline.
        """
        # Get bounding box
        min_x, min_y, max_x, max_y = self.section.get_bounding_box()

        # Calculate fiber dimensions
        fiber_width = (max_x - min_x) / self.n_fibers_width
        fiber_height = (max_y - min_y) / self.n_fibers_height
        fiber_area = fiber_width * fiber_height

        # Generate grid
        for i in range(self.n_fibers_width):
            for j in range(self.n_fibers_height):
                # Fiber boundaries
                x_min = min_x + i * fiber_width
                x_max = x_min + fiber_width
                y_min = min_y + j * fiber_height
                y_max = y_min + fiber_height

                # Fiber centroid
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0

                # Create fiber box
                fiber_box = box(x_min, y_min, x_max, y_max)

                # Check intersection with section outline
                intersection = self.section.outline.intersection(fiber_box)

                if intersection.is_empty:
                    continue

                # Calculate actual fiber area (may be partial for edge fibers)
                actual_area = intersection.area

                if actual_area < 1e-6:  # Skip negligible fibers
                    continue

                # Subtract steel area if requested
                if self.exclude_steel_area:
                    # Check for steel in this fiber
                    for group in self.section.rebar_groups:
                        for pos in group.positions:
                            # Check if bar center is in this fiber
                            if (x_min <= pos.x <= x_max) and (y_min <= pos.y <= y_max):
                                # Subtract bar area (or fiber area, whichever is smaller)
                                actual_area -= min(group.rebar.area, actual_area)

                # Skip if area is now negligible
                if actual_area < 1e-6:
                    continue

                # Adjust centroid if fiber was clipped
                if intersection.area < fiber_area * 0.99:  # Clipped fiber
                    if hasattr(intersection, 'centroid'):
                        x_center = intersection.centroid.x
                        y_center = intersection.centroid.y

                fiber = Fiber(
                    x=x_center,
                    y=y_center,
                    area=actual_area,
                    material_type="concrete",
                    material_index=0,
                )
                self.concrete_fibers.append(fiber)

    def _generate_steel_fibers(self) -> None:
        """
        Generate steel fibers (one per rebar).

        Each rebar is treated as a discrete fiber at its centroid.
        Material index tracks which rebar group the fiber belongs to.
        """
        for group_idx, group in enumerate(self.section.rebar_groups):
            for pos in group.positions:
                fiber = Fiber(
                    x=pos.x,
                    y=pos.y,
                    area=group.rebar.area,
                    material_type="steel",
                    material_index=group_idx,  # Track which group this bar belongs to
                )
                self.steel_fibers.append(fiber)

    @property
    def all_fibers(self) -> List[Fiber]:
        """Get all fibers (concrete + steel)."""
        return self.concrete_fibers + self.steel_fibers

    @property
    def n_concrete_fibers(self) -> int:
        """Number of concrete fibers."""
        return len(self.concrete_fibers)

    @property
    def n_steel_fibers(self) -> int:
        """Number of steel fibers."""
        return len(self.steel_fibers)

    @property
    def total_fibers(self) -> int:
        """Total number of fibers."""
        return self.n_concrete_fibers + self.n_steel_fibers

    def get_fiber_arrays(self) -> Tuple[
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

        x = np.array([f.x for f in all_f])
        y = np.array([f.y for f in all_f])
        area = np.array([f.area for f in all_f])
        material_type = np.array([f.material_type for f in all_f])
        material_index = np.array([f.material_index for f in all_f], dtype=np.int32)

        return x, y, area, material_type, material_index

    def calculate_section_forces(
        self,
        stresses: npt.NDArray[np.float64],
    ) -> Tuple[float, float]:
        """
        Calculate axial force and moment from fiber stresses.

        Args:
            stresses: Stress at each fiber (must match fiber order)

        Returns:
            Tuple of (N, M) where:
                N: Axial force in kN (positive = compression)
                M: Moment about section centroid in kN·m (positive = hogging)
        """
        x, y, area, _, _ = self.get_fiber_arrays()

        # Axial force: N = Σ(σ · A)
        # Convert from N to kN
        N = np.sum(stresses * area) / 1000.0

        # Moment about section centroid
        section_cx, section_cy = self.section.get_centroid()

        # M = Σ(σ · A · y_offset)
        # Using y-offset from section centroid
        y_offset = y - section_cy

        # Convert from N·mm to kN·m
        M = np.sum(stresses * area * y_offset) / 1_000_000.0

        return N, M

    def __repr__(self) -> str:
        return (
            f"FiberMesh("
            f"concrete={self.n_concrete_fibers}, "
            f"steel={self.n_steel_fibers}, "
            f"total={self.total_fibers})"
        )
