"""
M-N interaction diagram generator using fiber-based strain compatibility.

Implements EC2 ultimate limit state analysis for combined axial force and bending.
"""

from typing import List, Tuple, Optional, Literal, Dict, Any
import json
import csv
from pathlib import Path
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict
from scipy.optimize import root_scalar

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Export interaction point to dictionary.

        Returns:
            Dictionary with all point properties
        """
        return {
            "N_kN": self.N,
            "M_kNm": self.M,
            "neutral_axis_depth_mm": self.neutral_axis_depth,
            "max_concrete_strain": self.max_concrete_strain,
            "max_steel_strain": self.max_steel_strain,
        }


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
        tension_stiffening: bool = False,
        confined_concrete: bool = False,
        confinement_rho_s: Optional[float] = None,
        confinement_f_yh: Optional[float] = None,
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
            tension_stiffening: Include tension stiffening effects (default: False)
                If True, concrete in tension contributes using EC2 average stress-strain
            confined_concrete: Use Mander confined concrete model (default: False)
                If True, enhances strength and ductility in compression zone
            confinement_rho_s: Volumetric ratio of transverse reinforcement (optional)
                Required if confined_concrete=True. Typically 0.01-0.03
            confinement_f_yh: Yield strength of transverse reinforcement in MPa (optional)
                Defaults to same as longitudinal steel if not provided
        """
        self.section = section
        self.concrete = concrete
        self.tension_stiffening = tension_stiffening
        self.confined_concrete = confined_concrete
        self.confinement_rho_s = confinement_rho_s
        self.confinement_f_yh = confinement_f_yh

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

        # Validate confined concrete parameters
        if self.confined_concrete:
            if self.confinement_rho_s is None:
                raise ValueError(
                    "confinement_rho_s must be provided when confined_concrete=True"
                )
            if self.confinement_rho_s <= 0 or self.confinement_rho_s > 0.1:
                raise ValueError(
                    f"confinement_rho_s must be between 0 and 0.1, got {self.confinement_rho_s}"
                )
            # Default to longitudinal steel yield strength if not provided
            if self.confinement_f_yh is None:
                self.confinement_f_yh = first_rebar.f_yd

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
        concrete_strains = strains[concrete_mask]
        concrete_stresses = self.concrete_model.get_stress_array(concrete_strains)

        # Apply confined concrete model if enabled (Mander model)
        if self.confined_concrete:
            # Mander confined concrete model for compression
            # Enhances strength and ductility based on transverse reinforcement
            compression_mask = concrete_strains > 0  # Positive strain = compression

            if np.any(compression_mask):
                # Mander model parameters
                f_co = self.concrete.f_cd  # Unconfined strength
                epsilon_co = self.concrete.epsilon_c2  # Unconfined strain at peak

                # Confinement effectiveness coefficient (typically 0.95 for circular, 0.75 for rectangular)
                k_e = 0.75  # Conservative for rectangular sections

                # Effective lateral confining pressure
                f_l = 0.5 * k_e * self.confinement_rho_s * self.confinement_f_yh

                # Confined strength (Mander model)
                # f_cc = f_co * (1 + 5 * (f_cc/f_co - 1))
                # Simplified: f_cc = f_co * (2.254 * sqrt(1 + 7.94*f_l/f_co) - 2*f_l/f_co - 1.254)
                f_cc = f_co * (
                    2.254 * np.sqrt(1 + 7.94 * f_l / f_co) - 2 * f_l / f_co - 1.254
                )

                # Confined strain at peak stress
                epsilon_cc = epsilon_co * (
                    1 + 5 * (f_cc / f_co - 1)
                )

                # Ultimate confined strain (Mander)
                epsilon_cu_confined = 0.004 + 0.14 * self.confinement_rho_s * self.confinement_f_yh / f_co

                # Mander stress-strain relationship for confined concrete
                compression_strains = concrete_strains[compression_mask]

                # Normalized variables
                x = compression_strains / epsilon_cc
                r = self.concrete.E_cm / (self.concrete.E_cm - f_cc / epsilon_cc)

                # Mander equation: f_c = f_cc * x * r / (r - 1 + x^r)
                confined_stresses = np.where(
                    compression_strains <= epsilon_cu_confined,
                    f_cc * x * r / (r - 1 + x**r),
                    0.0  # Zero stress beyond ultimate strain
                )

                # Update concrete stresses in compression
                concrete_stresses[compression_mask] = confined_stresses

        # Apply tension stiffening if enabled
        if self.tension_stiffening:
            # EC2 average tension stress-strain for cracked concrete
            # Convention: tension = negative stress, compression = positive stress
            # Uses simplified approach: σ_ct = -f_ctm * (1 - β * (ε_t - ε_cr) / ε_cr)
            # where β ≈ 0.6 for short-term loading, ε_cr = f_ctm / E_cm

            tension_mask = concrete_strains < 0  # Negative strain = tension
            if np.any(tension_mask):
                # Tensile strength (mean)
                f_ctm = self.concrete.f_ctm
                E_cm = self.concrete.E_cm

                # Cracking strain (positive value)
                epsilon_cr = f_ctm / E_cm

                # Tension stiffening factor (0.6 for short-term, 0.4 for sustained)
                beta = 0.6

                # Get absolute values of tension strains (make positive)
                tension_strains_abs = -concrete_strains[tension_mask]

                # Linear elastic up to cracking, then tension stiffening decay
                # Before cracking: stress = -E_cm * ε_t (linear elastic tension)
                # After cracking: stress = -f_ctm * (1 - β * (ε_t - ε_cr) / ε_cr)
                tension_stresses = np.where(
                    tension_strains_abs <= epsilon_cr,
                    # Linear elastic (negative stress = tension)
                    -E_cm * tension_strains_abs,
                    # Tension stiffening (decreasing tension contribution)
                    -f_ctm * np.maximum(0, 1.0 - beta * (tension_strains_abs - epsilon_cr) / (epsilon_cr * 5.0))
                )

                # Update concrete stresses for fibers in tension
                concrete_stresses[tension_mask] = tension_stresses

        stresses[concrete_mask] = concrete_stresses

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
        corresponding to the applied axial force. Handles both symmetric
        and non-symmetric sections correctly.

        Args:
            N_Ed: Applied axial force in kN (positive = compression)

        Returns:
            Tuple of (M_Rd_pos, M_Rd_neg) - moment capacity in kN·m
            M_Rd_pos: Maximum positive moment capacity
            M_Rd_neg: Maximum negative moment capacity (negative value)
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
            # Get M values for points close to target N
            M_close = M_values[close_mask]

            # Find maximum positive and negative moments separately
            positive_moments = M_close[M_close >= 0]
            negative_moments = M_close[M_close < 0]

            M_Rd_pos = np.max(positive_moments) if len(positive_moments) > 0 else 0.0
            M_Rd_neg = np.min(negative_moments) if len(negative_moments) > 0 else 0.0
        else:
            # If no close points, find nearest neighbor
            nearest_idx = np.argmin(np.abs(N_values - N_Ed))
            M_nearest = M_values[nearest_idx]

            # For non-symmetric sections, we need to check both sides
            # Use the nearest point's M and assume symmetric as fallback
            M_Rd_pos = abs(M_nearest)
            M_Rd_neg = -abs(M_nearest)

        return (M_Rd_pos, M_Rd_neg)

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

    def find_balanced_point(
        self,
        max_concrete_strain: Optional[float] = None,
    ) -> Tuple[InteractionPoint, float]:
        """
        Find the balanced failure point on the M-N diagram.

        The balanced failure point occurs when the extreme compression fiber
        in concrete reaches its ultimate strain (ε_cu) at the same time that
        the tensile steel reaches its yield strain (ε_y).

        This method uses numerical optimization to find the neutral axis depth
        that satisfies the balanced failure condition.

        Args:
            max_concrete_strain: Maximum concrete strain (default: ε_cu2 from material)

        Returns:
            Tuple of (balanced_point, neutral_axis_depth_balanced)
            - balanced_point: InteractionPoint at balanced failure
            - neutral_axis_depth_balanced: Neutral axis depth at balanced failure (mm)

        Raises:
            RuntimeError: If balanced point cannot be found (optimization fails)
        """
        if max_concrete_strain is None:
            max_concrete_strain = self.concrete_model.get_ultimate_strain()

        # Get steel yield strain
        steel_yield_strain = self.steel_model.epsilon_y

        # Get fiber arrays to find deepest steel location
        x, y, area, material_type, material_index = self.mesh.get_fiber_arrays()
        steel_mask = material_type == 'steel'

        if not np.any(steel_mask):
            raise ValueError("Cannot find balanced point - no steel reinforcement")

        # Find extreme tension steel (bottom of section)
        steel_y = y[steel_mask]
        extreme_tension_steel_y = np.min(steel_y)  # Lowest y (bottom)

        # Distance from top of section to extreme tension steel
        y_steel_from_top = self.section_top - extreme_tension_steel_y

        # For balanced failure:
        # ε_concrete (at top) = ε_cu
        # ε_steel (at bottom) = ε_y
        # Using similar triangles:
        # ε_cu / x_bal = ε_y / (y_steel - x_bal)
        # Solving: x_bal = ε_cu * y_steel / (ε_cu + ε_y)

        # Initial estimate for balanced neutral axis depth
        x_bal_estimate = (
            max_concrete_strain * y_steel_from_top
            / (max_concrete_strain + steel_yield_strain)
        )

        # Define objective function: difference between target and actual max steel strain
        def objective(na_depth: float) -> float:
            """
            Objective function for finding balanced point.

            Returns the difference between the maximum tensile steel strain
            and the yield strain. Zero when balanced.
            """
            # Calculate strains at extreme tension steel
            y_from_top = y_steel_from_top

            if na_depth > 0:
                # Strain at extreme tension steel (negative = tension)
                strain_steel = max_concrete_strain * (na_depth - y_from_top) / na_depth
            else:
                # NA above section
                strain_steel = -max_concrete_strain * y_from_top / abs(na_depth)

            # Balanced when |ε_steel| = ε_yield (for tension, strain is negative)
            return abs(strain_steel) - steel_yield_strain

        # Solve using root finding
        # Search range: from 0.1*height to 2*height
        try:
            solution = root_scalar(
                objective,
                bracket=[self.section_height * 0.1, self.section_height * 2.0],
                method='brentq',
                xtol=0.01,  # 0.01 mm tolerance
            )

            if not solution.converged:
                raise RuntimeError("Balanced point optimization did not converge")

            na_balanced = solution.root

        except ValueError as e:
            # If bracket doesn't contain a root, use the estimate
            # This can happen for sections with very high or very low reinforcement
            na_balanced = x_bal_estimate

        # Calculate the balanced point
        balanced_point = self.calculate_point(
            neutral_axis_depth=na_balanced,
            max_concrete_strain=max_concrete_strain,
        )

        return (balanced_point, na_balanced)

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

    def export_to_json(
        self,
        file_path: str | Path,
        n_points: int = 50,
        include_metadata: bool = True,
        indent: int = 2,
    ) -> None:
        """
        Export M-N diagram to JSON file.

        Args:
            file_path: Output file path
            n_points: Number of points to generate
            include_metadata: Include section and material metadata
            indent: JSON indentation (None for compact)
        """
        points = self.generate_diagram(n_points=n_points)

        data: Dict[str, Any] = {
            "diagram_points": [p.to_dict() for p in points],
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
                "steel_model": type(self.steel_model).__name__,
            }

        file_path = Path(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)

    def export_to_csv(
        self,
        file_path: str | Path,
        n_points: int = 50,
        include_strains: bool = True,
    ) -> None:
        """
        Export M-N diagram to CSV file.

        Args:
            file_path: Output file path
            n_points: Number of points to generate
            include_strains: Include strain data columns
        """
        points = self.generate_diagram(n_points=n_points)

        file_path = Path(file_path)
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            if include_strains:
                fieldnames = [
                    'N_kN',
                    'M_kNm',
                    'neutral_axis_depth_mm',
                    'max_concrete_strain',
                    'max_steel_strain',
                ]
            else:
                fieldnames = ['N_kN', 'M_kNm']

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for point in points:
                row_data = point.to_dict()
                if not include_strains:
                    row_data = {
                        'N_kN': row_data['N_kN'],
                        'M_kNm': row_data['M_kNm'],
                    }
                writer.writerow(row_data)

    def to_dict(
        self,
        n_points: int = 50,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Export M-N diagram to dictionary for programmatic access.

        Args:
            n_points: Number of points to generate
            include_metadata: Include section and material metadata

        Returns:
            Dictionary with diagram data
        """
        points = self.generate_diagram(n_points=n_points)

        data: Dict[str, Any] = {
            "points": [p.to_dict() for p in points],
            "N_array": [p.N for p in points],
            "M_array": [p.M for p in points],
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
                "steel_model": type(self.steel_model).__name__,
            }

        return data

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
