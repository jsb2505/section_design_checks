"""
Biaxial M-M-N interaction surface generator using fiber-based strain compatibility.

Implements EC2 ultimate limit state analysis for combined axial force and biaxial bending.
"""

from typing import List, Tuple, Optional, Literal, Dict, Any
import json
import csv
from pathlib import Path
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict

from materials.reinforced_concrete.geometry import RCSection, FiberMesh
from materials.reinforced_concrete.constitutive import (
    create_concrete_stress_strain,
    create_steel_stress_strain,
)
from materials.reinforced_concrete.materials import ConcreteMaterial


class BiaxialInteractionPoint(BaseModel):
    """Single point on biaxial M-M-N interaction surface."""

    model_config = ConfigDict(frozen=True)

    N: float = Field(..., description="Axial force in kN (positive = compression)")
    Mx: float = Field(..., description="Moment about x-axis in kN·m")
    My: float = Field(..., description="Moment about y-axis in kN·m")
    neutral_axis_depth: float = Field(..., description="Neutral axis depth from centroid (mm)")
    neutral_axis_angle: float = Field(..., description="Neutral axis angle from x-axis (degrees)")
    max_concrete_strain: float = Field(..., description="Maximum concrete strain")
    max_steel_strain: float = Field(..., description="Maximum steel strain")

    def __repr__(self) -> str:
        return f"BiaxialPoint(N={self.N:.1f} kN, Mx={self.Mx:.1f} kN·m, My={self.My:.1f} kN·m)"

    def to_dict(self) -> Dict[str, Any]:
        """Export biaxial interaction point to dictionary."""
        return {
            "N_kN": self.N,
            "Mx_kNm": self.Mx,
            "My_kNm": self.My,
            "neutral_axis_depth_mm": self.neutral_axis_depth,
            "neutral_axis_angle_deg": self.neutral_axis_angle,
            "max_concrete_strain": self.max_concrete_strain,
            "max_steel_strain": self.max_steel_strain,
        }


class BiaxialMNInteractionSurface:
    """
    Biaxial M-M-N interaction surface generator using fiber-based strain compatibility.

    The surface represents all combinations of axial force (N) and biaxial moments (Mx, My)
    that bring the section to its ultimate limit state per EC2.

    Method:
    1. Assume a neutral axis depth and angle
    2. Calculate strain distribution (plane sections remain plane)
    3. Strains perpendicular to the neutral axis
    4. Get stresses from constitutive models
    5. Integrate forces over fibers
    6. Result is one (N, Mx, My) point on the surface
    7. Repeat for different neutral axis depths and angles
    """

    def __init__(
        self,
        section: RCSection,
        concrete: ConcreteMaterial,
        concrete_model_type: Literal["parabola-rectangle", "bilinear"] = "parabola-rectangle",
        steel_branch_type: Literal["inclined", "horizontal"] = "inclined",
        n_fibers_width: int = 20,
        n_fibers_height: int = 30,
    ):
        """
        Initialize biaxial M-M-N surface generator.

        Args:
            section: RC section with reinforcement
            concrete: Concrete material properties
            concrete_model_type: Stress-strain model for concrete
            steel_branch_type: Stress-strain model for steel
            n_fibers_width: Fiber mesh resolution (width)
            n_fibers_height: Fiber mesh resolution (height)
        """
        self.section = section
        self.concrete = concrete

        # Create constitutive models
        self.concrete_model = create_concrete_stress_strain(
            concrete=concrete,
            model_type=concrete_model_type,
            use_characteristic=False,
        )

        if len(section.rebar_groups) == 0:
            raise ValueError("Section must have at least one rebar group")

        first_rebar = section.rebar_groups[0].rebar
        self.steel_model = create_steel_stress_strain(
            steel=first_rebar,
            branch_type=steel_branch_type,
            use_characteristic=False,
        )

        # Generate fiber mesh
        self.mesh = FiberMesh(
            section=section,
            n_fibers_width=n_fibers_width,
            n_fibers_height=n_fibers_height,
            exclude_steel_area=True,
        )

        # Get section properties
        self.section_centroid_x, self.section_centroid_y = section.get_centroid()
        min_x, min_y, max_x, max_y = section.get_bounding_box()
        self.section_width = max_x - min_x
        self.section_height = max_y - min_y

    def calculate_point(
        self,
        neutral_axis_depth: float,
        neutral_axis_angle: float = 0.0,
        max_concrete_strain: Optional[float] = None,
    ) -> BiaxialInteractionPoint:
        """
        Calculate single point on biaxial interaction surface.

        Uses strain compatibility with rotated neutral axis:
        - Plane sections remain plane
        - Strain varies linearly perpendicular to neutral axis
        - Maximum concrete strain at extreme fiber

        Args:
            neutral_axis_depth: Distance from centroid to neutral axis (mm)
                                Positive = compression side
            neutral_axis_angle: Angle of neutral axis from x-axis (degrees)
                                0° = NA horizontal (bending about y-axis, Mx)
                                90° = NA vertical (bending about x-axis, My)
            max_concrete_strain: Maximum concrete compressive strain (default: ε_cu2)

        Returns:
            BiaxialInteractionPoint with N, Mx, My, and strain information
        """
        if max_concrete_strain is None:
            max_concrete_strain = self.concrete_model.get_ultimate_strain()

        # Get fiber coordinates relative to section centroid
        x, y, area, material_type, material_index = self.mesh.get_fiber_arrays()

        # Fiber positions relative to centroid
        x_rel = x - self.section_centroid_x
        y_rel = y - self.section_centroid_y

        # Neutral axis angle in radians
        theta = np.radians(neutral_axis_angle)

        # Perpendicular distance from each fiber to the neutral axis
        # NA passes through a point at distance 'neutral_axis_depth' from centroid
        # in the direction perpendicular to the NA (angle theta + 90°)
        # Distance = x*sin(theta) - y*cos(theta) + neutral_axis_depth
        # (using signed distance, positive = compression side)

        # Normal vector to neutral axis (pointing toward compression zone)
        nx = np.sin(theta)
        ny = -np.cos(theta)

        # Signed distance from neutral axis (positive = compression)
        distance_from_na = nx * x_rel + ny * y_rel - neutral_axis_depth

        # Calculate strain at each fiber
        # ε = ε_max * distance_from_na / (max_distance_from_na)
        # For simplicity, use fixed compression zone depth
        # Strain varies linearly from NA

        if neutral_axis_depth > 0:
            # Compression zone exists
            # Maximum compression at the extreme fiber on compression side
            # ε = ε_cu * distance_from_na / neutral_axis_depth
            strains = max_concrete_strain * distance_from_na / neutral_axis_depth
        else:
            # Pure tension (NA beyond section)
            # All fibers in tension
            strains = -max_concrete_strain * distance_from_na / abs(neutral_axis_depth)

        # Calculate stresses from constitutive models
        stresses = np.zeros_like(strains)

        # Concrete fibers
        concrete_mask = material_type == 'concrete'
        stresses[concrete_mask] = self.concrete_model.get_stress_array(strains[concrete_mask])

        # Steel fibers
        steel_mask = material_type == 'steel'
        stresses[steel_mask] = self.steel_model.get_stress_array(strains[steel_mask])

        # Calculate resultant forces
        # N = Σ(σ · A)
        N = np.sum(stresses * area) / 1000.0  # kN

        # Moments about centroid
        # Mx = Σ(σ · A · y_rel)  (moment about x-axis, caused by y-direction forces)
        # My = Σ(σ · A · x_rel)  (moment about y-axis, caused by x-direction forces)
        Mx = np.sum(stresses * area * y_rel) / 1e6  # kN·m
        My = np.sum(stresses * area * x_rel) / 1e6  # kN·m

        # Get maximum strains for reporting
        max_conc_strain = np.max(strains[concrete_mask]) if np.any(concrete_mask) else 0.0
        max_steel_strain = np.max(np.abs(strains[steel_mask])) if np.any(steel_mask) else 0.0

        return BiaxialInteractionPoint(
            N=N,
            Mx=Mx,
            My=My,
            neutral_axis_depth=neutral_axis_depth,
            neutral_axis_angle=neutral_axis_angle,
            max_concrete_strain=max_conc_strain,
            max_steel_strain=max_steel_strain,
        )

    def generate_surface(
        self,
        n_angles: int = 16,
        n_depths: int = 30,
        include_tension: bool = True,
    ) -> List[BiaxialInteractionPoint]:
        """
        Generate complete biaxial M-M-N interaction surface.

        Creates points covering:
        - Full range of neutral axis angles (0° to 360°)
        - Full range of neutral axis depths (compression to tension)

        Args:
            n_angles: Number of neutral axis angles to evaluate
            n_depths: Number of neutral axis depths per angle
            include_tension: Include pure tension branch

        Returns:
            List of BiaxialInteractionPoint covering the 3D surface
        """
        points: List[BiaxialInteractionPoint] = []

        # Range of neutral axis angles (0° to 360°, but due to symmetry, 0° to 180° is sufficient)
        # Actually, for full generality, use 0° to 360° to capture any asymmetry
        angles = np.linspace(0, 360, n_angles, endpoint=False)

        # For each angle, generate points at different NA depths
        for angle in angles:
            # Define neutral axis depth range for this angle
            max_depth = max(self.section_width, self.section_height)

            # 1. Pure compression
            points.append(self.calculate_point(max_depth * 10, angle))

            # 2. Compression-controlled points
            depths_compression = np.linspace(
                max_depth * 2,
                max_depth * 0.1,
                n_depths // 2
            )
            for depth in depths_compression:
                points.append(self.calculate_point(depth, angle))

            # 3. Transition zone
            depths_transition = np.linspace(
                max_depth * 0.1,
                -max_depth * 0.1,
                n_depths // 4
            )
            for depth in depths_transition:
                points.append(self.calculate_point(depth, angle))

            if include_tension:
                # 4. Tension-controlled points
                depths_tension = np.linspace(
                    -max_depth * 0.1,
                    -max_depth * 2,
                    n_depths // 4
                )
                for depth in depths_tension:
                    points.append(self.calculate_point(depth, angle))

        return points

    def export_to_json(
        self,
        file_path: str | Path,
        n_angles: int = 16,
        n_depths: int = 30,
        include_metadata: bool = True,
        indent: int = 2,
    ) -> None:
        """Export biaxial M-M-N surface to JSON file."""
        points = self.generate_surface(n_angles=n_angles, n_depths=n_depths)

        data: Dict[str, Any] = {
            "surface_points": [p.to_dict() for p in points],
        }

        if include_metadata:
            data["metadata"] = {
                "section_name": self.section.section_name,
                "concrete_grade": self.concrete.grade,
                "n_angles": n_angles,
                "n_depths": n_depths,
                "total_points": len(points),
            }

        file_path = Path(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)

    def export_to_csv(
        self,
        file_path: str | Path,
        n_angles: int = 16,
        n_depths: int = 30,
    ) -> None:
        """Export biaxial M-M-N surface to CSV file."""
        points = self.generate_surface(n_angles=n_angles, n_depths=n_depths)

        file_path = Path(file_path)
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'N_kN',
                'Mx_kNm',
                'My_kNm',
                'neutral_axis_depth_mm',
                'neutral_axis_angle_deg',
                'max_concrete_strain',
                'max_steel_strain',
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for point in points:
                writer.writerow(point.to_dict())

    def __repr__(self) -> str:
        return (
            f"BiaxialMNInteractionSurface("
            f"section={self.section.section_name}, "
            f"concrete={self.concrete.grade})"
        )


def create_biaxial_interaction_surface(
    section: RCSection,
    concrete: ConcreteMaterial,
    **kwargs,
) -> BiaxialMNInteractionSurface:
    """
    Factory function to create biaxial M-M-N interaction surface.

    Args:
        section: RC section with reinforcement
        concrete: Concrete material
        **kwargs: Additional arguments passed to BiaxialMNInteractionSurface

    Returns:
        BiaxialMNInteractionSurface instance

    Example:
        >>> from materials.reinforced_concrete.geometry import create_rectangular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial
        >>>
        >>> section = create_rectangular_section(300, 500)
        >>> # ... add reinforcement ...
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>>
        >>> surface = create_biaxial_interaction_surface(section, concrete)
        >>> points = surface.generate_surface(n_angles=16, n_depths=30)
    """
    return BiaxialMNInteractionSurface(section=section, concrete=concrete, **kwargs)
