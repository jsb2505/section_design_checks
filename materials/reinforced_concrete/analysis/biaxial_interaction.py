"""
Biaxial M-M-N interaction surface generator using fiber-based strain compatibility.

Implements EC2 ultimate limit state analysis for combined axial force and biaxial bending.
"""

from typing import List, Optional, Literal, Dict, Any
import json
import csv
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from materials.reinforced_concrete.geometry import RCSection, FiberMesh
from materials.reinforced_concrete.constitutive import (
    create_concrete_stress_strain,
    create_steel_stress_strain,
)
from materials.reinforced_concrete.materials import ConcreteMaterial


class BiaxialInteractionPoint(BaseModel):
    """
    Single point on biaxial M-M-N interaction surface.

    Uses 3D FEA axis convention:
    - x: longitudinal axis (along member)
    - y: horizontal axis in cross-section (minor axis, width)
    - z: vertical axis in cross-section (major axis, height)
    - My: moment about y-axis (major axis bending, from z-forces)
    - Mz: moment about z-axis (minor axis bending, from y-forces)
    """

    model_config = ConfigDict(frozen=True)

    N: float = Field(..., description="Axial force in kN (positive = compression)")
    My: float = Field(..., description="Moment about y-axis (major axis) in kN·m")
    Mz: float = Field(..., description="Moment about z-axis (minor axis) in kN·m")
    neutral_axis_depth: float = Field(..., description="Neutral axis depth from centroid (mm)")
    neutral_axis_angle: float = Field(..., description="Neutral axis angle from y-axis (degrees)")
    max_concrete_strain: float = Field(..., description="Maximum concrete strain")
    max_steel_strain: float = Field(..., description="Maximum steel strain")

    def __repr__(self) -> str:
        return f"BiaxialPoint(N={self.N:.1f} kN, My={self.My:.1f} kN·m, Mz={self.Mz:.1f} kN·m)"

    def to_dict(self) -> Dict[str, Any]:
        """Export biaxial interaction point to dictionary."""
        return {
            "N_kN": self.N,
            "My_kNm": self.My,
            "Mz_kNm": self.Mz,
            "neutral_axis_depth_mm": self.neutral_axis_depth,
            "neutral_axis_angle_deg": self.neutral_axis_angle,
            "max_concrete_strain": self.max_concrete_strain,
            "max_steel_strain": self.max_steel_strain,
        }


class BiaxialMNInteractionSurface:
    """
    Biaxial M-M-N interaction surface generator using fiber-based strain compatibility.

    The surface represents all combinations of axial force (N) and biaxial moments (My, Mz)
    that bring the section to its ultimate limit state per EC2.

    Axis Convention (3D FEA standard):
    - x: longitudinal axis (along member)
    - y: horizontal axis in cross-section (minor axis, width)
    - z: vertical axis in cross-section (major axis, height)
    - My: moment about y-axis (major axis bending)
    - Mz: moment about z-axis (minor axis bending)

    Method:
    1. Assume a neutral axis depth and angle
    2. Calculate strain distribution (plane sections remain plane)
    3. Strains perpendicular to the neutral axis
    4. Get stresses from constitutive models
    5. Integrate forces over fibers
    6. Result is one (N, My, Mz) point on the surface
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

        # Create steel models for each rebar group (to support different steel grades)
        self.steel_models = []
        for group in section.rebar_groups:
            steel_model = create_steel_stress_strain(
                steel=group.rebar,
                branch_type=steel_branch_type,
                use_characteristic=False,  # Use f_yd
            )
            self.steel_models.append(steel_model)

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
            neutral_axis_angle: Angle of neutral axis (degrees)
                                0° = NA horizontal → vertical forces → My (major axis)
                                90° = NA vertical → horizontal forces → Mz (minor axis)
            max_concrete_strain: Maximum concrete compressive strain (default: ε_cu2)

        Returns:
            BiaxialInteractionPoint with N, My, Mz, and strain information
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

        # Steel fibers - apply correct steel model for each rebar group
        steel_mask = material_type == 'steel'
        steel_strains = strains[steel_mask]
        steel_indices = material_index[steel_mask]

        # Calculate stresses for each rebar group separately
        steel_stresses = np.zeros_like(steel_strains)
        for group_idx in range(len(self.steel_models)):
            # Find fibers belonging to this rebar group
            group_mask = steel_indices == group_idx
            if np.any(group_mask):
                # Apply the steel model for this specific group
                steel_stresses[group_mask] = self.steel_models[group_idx].get_stress_array(
                    steel_strains[group_mask]
                )

        stresses[steel_mask] = steel_stresses

        # Calculate resultant forces
        # N = Σ(σ · A)
        N = np.sum(stresses * area) / 1000.0  # kN

        # Moments about centroid using 3D FEA convention
        # Section coords: x_rel = horizontal (width), y_rel = vertical (height)
        # FEA coords: y = horizontal (minor), z = vertical (major)
        # My = Σ(σ · A · z_rel) = Σ(σ · A · y_rel_section)  (major axis moment)
        # Mz = Σ(σ · A · y_rel) = Σ(σ · A · x_rel_section)  (minor axis moment)
        My = np.sum(stresses * area * y_rel) / 1e6  # kN·m (major axis)
        Mz = np.sum(stresses * area * x_rel) / 1e6  # kN·m (minor axis)

        # Get maximum strains for reporting
        max_conc_strain = np.max(strains[concrete_mask]) if np.any(concrete_mask) else 0.0
        max_steel_strain = np.max(np.abs(strains[steel_mask])) if np.any(steel_mask) else 0.0

        return BiaxialInteractionPoint(
            N=N,
            My=My,
            Mz=Mz,
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
        - Pure compression point (all concrete and steel at compression capacity)
        - Pure tension point (all steel at tension capacity, concrete cracked)

        Args:
            n_angles: Number of neutral axis angles to evaluate
            n_depths: Number of neutral axis depths per angle
            include_tension: Include pure tension branch

        Returns:
            List of BiaxialInteractionPoint covering the 3D surface
        """
        points: List[BiaxialInteractionPoint] = []

        # Range of neutral axis angles (0° to 360°)
        # For full generality, use 0° to 360° to capture any asymmetry
        angles = np.linspace(0, 360, n_angles, endpoint=False)

        # For each angle, generate points at different NA depths
        for angle in angles:
            # Define neutral axis depth range for this angle
            max_depth = max(self.section_width, self.section_height)

            # Use sophisticated sampling similar to uniaxial M-N diagram
            # Denser sampling in high-curvature regions (balanced failure zone)
            na_depths = np.concatenate([
                # Very deep compression (approaching pure compression)
                np.linspace(max_depth * 10, max_depth * 2, max(3, n_depths // 8)),
                # Deep compression (approaching section)
                np.linspace(max_depth * 2, max_depth * 1.5, max(5, n_depths // 8)),
                # Compression controlled (high curvature region)
                np.linspace(max_depth * 1.5, max_depth * 0.5, max(10, n_depths // 4)),
                # Balanced region (very high curvature - needs dense sampling)
                np.linspace(max_depth * 0.5, max_depth * 0.05, max(20, n_depths // 2)),
                # Tension controlled - stop at NA just inside section
                np.linspace(max_depth * 0.05, max_depth * 0.001, max(5, n_depths // 8)),
            ])

            # Generate points for this angle
            for depth in na_depths:
                points.append(self.calculate_point(depth, angle))

            # Add tension branch if requested
            if include_tension:
                # Tension-controlled points (NA beyond section on compression side)
                depths_tension = np.linspace(
                    -max_depth * 0.001,
                    -max_depth * 2,
                    max(10, n_depths // 4)
                )
                for depth in depths_tension:
                    points.append(self.calculate_point(depth, angle))

        # Add pure compression point
        # Pure compression: entire section at ultimate concrete strain
        # N = concrete_area × f_c + Σ(A_s × f_yd)
        # My and Mz arise from eccentric steel placement

        section_cx, section_cy = self.section.get_centroid()

        # Concrete contribution (uniform compression at design strength)
        concrete_area = self.section.outline.area  # mm²
        f_c = self.concrete_model.get_yield_stress()  # f_cd for design models
        N_concrete = concrete_area * f_c / 1000.0  # kN

        # Steel contribution (all steel at yield in compression)
        N_steel = 0.0
        My_steel = 0.0
        Mz_steel = 0.0
        max_steel_strain = 0.0

        for group_idx, group in enumerate(self.section.rebar_groups):
            A_s = group.rebar.area  # Area per bar
            f_yd = self.steel_models[group_idx].get_yield_stress()  # Yield stress

            for pos in group.positions:
                # Compression force (positive)
                bar_force = A_s * f_yd / 1000.0  # kN
                N_steel += bar_force

                # Moments about section centroid (3D FEA convention)
                x_offset = pos.x - section_cx  # Horizontal offset
                y_offset = pos.y - section_cy  # Vertical offset

                # My = moment about y-axis (from z-direction forces)
                My_steel += bar_force * y_offset / 1000.0  # kN·m

                # Mz = moment about z-axis (from y-direction forces)
                Mz_steel += bar_force * x_offset / 1000.0  # kN·m

            # Track maximum steel yield strain
            max_steel_strain = max(max_steel_strain, self.steel_models[group_idx].epsilon_y)

        pure_compression_N = N_concrete + N_steel
        pure_compression_My = My_steel  # Concrete symmetric → no moment
        pure_compression_Mz = Mz_steel

        pure_compression_point = BiaxialInteractionPoint(
            N=pure_compression_N,
            My=pure_compression_My,
            Mz=pure_compression_Mz,
            neutral_axis_depth=max_depth * 1000,  # NA very deep
            neutral_axis_angle=0.0,  # Arbitrary for pure compression
            max_concrete_strain=self.concrete_model.get_ultimate_strain(),
            max_steel_strain=max_steel_strain,
        )
        points.append(pure_compression_point)

        # Add pure tension point if requested
        if include_tension:
            # Pure tension: all steel at yield in tension, concrete fully cracked
            # N = -Σ(A_s × f_yd)
            # My and Mz arise from eccentric steel placement

            pure_tension_N = 0.0
            pure_tension_My = 0.0
            pure_tension_Mz = 0.0
            max_steel_strain = 0.0

            for group_idx, group in enumerate(self.section.rebar_groups):
                A_s = group.rebar.area
                f_yd = self.steel_models[group_idx].get_yield_stress()

                for pos in group.positions:
                    # Tension force (negative)
                    bar_force = -A_s * f_yd / 1000.0  # kN (negative for tension)
                    pure_tension_N += bar_force

                    # Moments about section centroid
                    x_offset = pos.x - section_cx
                    y_offset = pos.y - section_cy

                    pure_tension_My += bar_force * y_offset / 1000.0  # kN·m
                    pure_tension_Mz += bar_force * x_offset / 1000.0  # kN·m

                max_steel_strain = max(max_steel_strain, self.steel_models[group_idx].epsilon_y)

            pure_tension_point = BiaxialInteractionPoint(
                N=pure_tension_N,
                My=pure_tension_My,
                Mz=pure_tension_Mz,
                neutral_axis_depth=-max_depth * 10,  # NA far beyond section
                neutral_axis_angle=0.0,  # Arbitrary for pure tension
                max_concrete_strain=0.0,
                max_steel_strain=max_steel_strain,
            )
            points.append(pure_tension_point)

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
                "concrete_fck": self.concrete.f_ck,
                "concrete_fcd": self.concrete.f_cd,
                "n_rebar_groups": len(self.section.rebar_groups),
                "n_fibers": self.mesh.total_fibers,
                "concrete_model": type(self.concrete_model).__name__,
                "steel_models": [type(sm).__name__ for sm in self.steel_models],
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
                'My_kNm',
                'Mz_kNm',
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
