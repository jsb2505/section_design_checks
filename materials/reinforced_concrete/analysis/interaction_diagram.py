"""
M-N interaction diagram generator using fiber-based strain compatibility.

Implements EC2 ultimate limit state analysis for combined axial force and bending.
"""

from typing import List, Tuple, Optional, Literal
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict

from materials.reinforced_concrete.geometry import RCSection, FiberMesh
from materials.reinforced_concrete.constitutive import (
    BaseConstitutiveModel,
    create_concrete_stress_strain,
    create_steel_stress_strain,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar


class InteractionPoint(BaseModel):
    """Single point on M-N interaction diagram."""

    model_config = ConfigDict(frozen=True)

    N: float = Field(..., description="Axial force in kN (positive = compression)")
    M: float = Field(..., description="Moment about centroid in kN·m")
    neutral_axis_depth: float = Field(..., description="Neutral axis depth from top (mm)")
    max_concrete_strain: float = Field(..., description="Maximum concrete strain")
    max_steel_strain: float = Field(..., description="Maximum steel strain")

    def __repr__(self) -> str:
        return f"InteractionPoint(N={self.N:.1f} kN, M={self.M:.1f} kN·m)"


class MNInteractionDiagram:
    """
    M-N interaction diagram generator using fiber-based strain compatibility.

    The diagram represents all combinations of axial force (N) and moment (M)
    that bring the section to its ultimate limit state per EC2.

    Method:
    1. Assume a neutral axis depth
    2. Calculate strain distribution (plane sections remain plane)
    3. Get stresses from constitutive models
    4. Integrate forces over fibers
    5. Result is one (N, M) point on the diagram
    6. Repeat for different neutral axis depths

    Special cases handled:
    - Pure compression (NA at infinity)
    - Pure tension (NA at -infinity)
    - Balanced failure
    - Small/large eccentricity
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
        Initialize M-N diagram generator.

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

        # Create constitutive models (using design strengths)
        self.concrete_model = create_concrete_stress_strain(
            concrete=concrete,
            model_type=concrete_model_type,
            use_characteristic=False,  # Use f_cd
        )

        # Assume all steel has same properties (use first rebar group)
        if len(section.rebar_groups) == 0:
            raise ValueError("Section must have at least one rebar group")

        # Get steel from first rebar group
        first_rebar = section.rebar_groups[0].rebar
        self.steel_model = create_steel_stress_strain(
            steel=first_rebar,
            branch_type=steel_branch_type,
            use_characteristic=False,  # Use f_yd
        )

        # Generate fiber mesh
        self.mesh = FiberMesh(
            section=section,
            n_fibers_width=n_fibers_width,
            n_fibers_height=n_fibers_height,
            exclude_steel_area=True,
        )

        # Get section properties
        _, self.section_centroid_y = section.get_centroid()
        _, min_y, _, max_y = section.get_bounding_box()
        self.section_height = max_y - min_y
        self.section_top = max_y
        self.section_bottom = min_y

    def calculate_point(
        self,
        neutral_axis_depth: float,
        max_concrete_strain: Optional[float] = None,
    ) -> InteractionPoint:
        """
        Calculate single point on interaction diagram.

        Uses strain compatibility:
        - Plane sections remain plane
        - Strain varies linearly from neutral axis
        - Concrete strain at top = max_concrete_strain (typically ε_cu2)

        Args:
            neutral_axis_depth: Depth from section top to neutral axis (mm)
                                Positive = NA inside section
                                Negative = NA above section (pure tension)
                                Very large = NA below section (pure compression)
            max_concrete_strain: Maximum concrete compressive strain (default: ε_cu2)

        Returns:
            InteractionPoint with N, M, and strain information
        """
        if max_concrete_strain is None:
            max_concrete_strain = self.concrete_model.get_ultimate_strain()

        # Get fiber coordinates
        x, y, area, material_type, material_index = self.mesh.get_fiber_arrays()

        # Calculate strain at each fiber using plane sections remain plane
        # ε = ε_top * (NA_depth - y_fiber) / NA_depth
        # where y is measured from section top

        # Distance from top of section to each fiber
        y_from_top = self.section_top - y

        # Strain distribution (compression positive)
        if neutral_axis_depth > 0:
            # NA inside or below section
            strains = max_concrete_strain * (neutral_axis_depth - y_from_top) / neutral_axis_depth
        else:
            # NA above section (tension throughout)
            # Use similar triangles with NA above section
            strains = -max_concrete_strain * y_from_top / abs(neutral_axis_depth)

        # Calculate stresses from constitutive models
        stresses = np.zeros_like(strains)

        # Concrete fibers
        concrete_mask = material_type == 'concrete'
        stresses[concrete_mask] = self.concrete_model.get_stress_array(strains[concrete_mask])

        # Steel fibers
        steel_mask = material_type == 'steel'
        stresses[steel_mask] = self.steel_model.get_stress_array(strains[steel_mask])

        # Calculate resultant forces
        N, M = self.mesh.calculate_section_forces(strains, stresses)

        # Get maximum strains for reporting
        max_conc_strain = np.max(strains[concrete_mask]) if np.any(concrete_mask) else 0.0
        max_steel_strain = np.max(np.abs(strains[steel_mask])) if np.any(steel_mask) else 0.0

        return InteractionPoint(
            N=N,
            M=M,
            neutral_axis_depth=neutral_axis_depth,
            max_concrete_strain=max_conc_strain,
            max_steel_strain=max_steel_strain,
        )

    def generate_diagram(
        self,
        n_points: int = 50,
        include_tension: bool = True,
    ) -> List[InteractionPoint]:
        """
        Generate complete M-N interaction diagram.

        Creates points covering:
        1. Pure compression (NA at infinity)
        2. Compression with small eccentricity
        3. Balanced failure
        4. Tension with compression block
        5. Pure tension (NA above section)

        Args:
            n_points: Number of points on diagram
            include_tension: Include pure tension branch

        Returns:
            List of InteractionPoint ordered from pure compression to pure tension
        """
        points: List[InteractionPoint] = []

        # Define neutral axis depth range
        # From deep in compression to tension zone

        # 1. Pure compression point (NA very deep)
        na_pure_compression = self.section_height * 10  # NA well below section
        points.append(self.calculate_point(na_pure_compression))

        # 2. Compression-controlled points (NA from deep to shallow)
        # Range from 2×height down to just below section bottom
        na_compression = np.linspace(
            self.section_height * 2,
            self.section_height * 0.1,
            n_points // 2
        )
        for na_depth in na_compression:
            points.append(self.calculate_point(na_depth))

        # 3. Transition zone (NA through section)
        # This captures balanced failure and transition region
        na_transition = np.linspace(
            self.section_height * 0.1,
            -self.section_height * 0.1,
            n_points // 4
        )
        for na_depth in na_transition:
            points.append(self.calculate_point(na_depth))

        if include_tension:
            # 4. Tension-controlled points (NA above section)
            na_tension = np.linspace(
                -self.section_height * 0.1,
                -self.section_height * 2,
                n_points // 4
            )
            for na_depth in na_tension:
                points.append(self.calculate_point(na_depth))

        return points

    def get_capacity(self, N_Ed: float) -> Tuple[float, float]:
        """
        Get moment capacity for given axial force.

        Finds the maximum moment capacity on the interaction diagram
        corresponding to the applied axial force.

        Args:
            N_Ed: Applied axial force in kN (positive = compression)

        Returns:
            Tuple of (M_Rd_pos, M_Rd_neg) - moment capacity in kN·m
            (positive and negative bending)
        """
        # Generate diagram with fine resolution
        diagram = self.generate_diagram(n_points=100)

        # Extract N and M values
        N_values = np.array([p.N for p in diagram])
        M_values = np.array([p.M for p in diagram])

        # Find points close to target N (within 5% tolerance or nearest neighbors)
        # M-N diagram is not monotonic, so we can't use simple interpolation
        N_tolerance = max(abs(N_Ed) * 0.05, 10.0)  # 5% or 10 kN minimum

        # Find indices where N is close to N_Ed
        close_mask = np.abs(N_values - N_Ed) <= N_tolerance

        if np.any(close_mask):
            # Use maximum M among close points
            M_Rd = np.max(np.abs(M_values[close_mask]))
        else:
            # If no close points, find nearest neighbor
            nearest_idx = np.argmin(np.abs(N_values - N_Ed))
            M_Rd = abs(M_values[nearest_idx])

        return (M_Rd, -M_Rd)  # Symmetric for rectangular sections

    def check_capacity(
        self,
        N_Ed: float,
        M_Ed: float,
    ) -> Tuple[bool, float]:
        """
        Check if applied loads are within capacity.

        Args:
            N_Ed: Applied axial force in kN
            M_Ed: Applied moment in kN·m

        Returns:
            Tuple of (is_safe, utilization)
            where utilization = demand/capacity
        """
        M_Rd, _ = self.get_capacity(N_Ed)

        utilization = abs(M_Ed) / M_Rd if M_Rd > 0 else float('inf')
        is_safe = utilization <= 1.0

        return (is_safe, utilization)

    def get_diagram_arrays(
        self,
        n_points: int = 50,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get M-N diagram as numpy arrays for plotting.

        Args:
            n_points: Number of points to generate

        Returns:
            Tuple of (N_array, M_array) in kN and kN·m
        """
        points = self.generate_diagram(n_points=n_points)

        N = np.array([p.N for p in points])
        M = np.array([p.M for p in points])

        return (N, M)

    def __repr__(self) -> str:
        return (
            f"MNInteractionDiagram("
            f"section={self.section.section_name}, "
            f"concrete={self.concrete.grade}, "
            f"fibers={self.mesh.total_fibers})"
        )


def create_interaction_diagram(
    section: RCSection,
    concrete: ConcreteMaterial,
    **kwargs,
) -> MNInteractionDiagram:
    """
    Factory function to create M-N interaction diagram.

    Args:
        section: RC section with reinforcement
        concrete: Concrete material
        **kwargs: Additional arguments passed to MNInteractionDiagram

    Returns:
        MNInteractionDiagram instance

    Example:
        >>> from materials.reinforced_concrete.geometry import create_rectangular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial
        >>>
        >>> section = create_rectangular_section(300, 500)
        >>> # ... add reinforcement ...
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>>
        >>> diagram = create_interaction_diagram(section, concrete)
        >>> N, M = diagram.get_diagram_arrays(n_points=100)
        >>>
        >>> # Check capacity
        >>> is_safe, util = diagram.check_capacity(N_Ed=500, M_Ed=150)
    """
    return MNInteractionDiagram(section=section, concrete=concrete, **kwargs)
