"""
Bending (flexure) check using M-N interaction diagrams.

This is a FIRST PRINCIPLES check based on strain compatibility and force equilibrium.
Uses the fiber-based M-N interaction diagram infrastructure.
"""

from typing import Literal, Optional
from pydantic import Field

from materials.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial
from materials.reinforced_concrete.analysis import create_interaction_diagram


class BendingCheck(BaseCodeCheck):
    """
    EC2 bending check using M-N interaction diagram (§6.1).

    This check uses FIRST PRINCIPLES:
    1. Strain compatibility (plane sections remain plane)
    2. Force equilibrium (ΣF = N, ΣM = M)
    3. Constitutive models (stress-strain with codified factors γ_c, γ_s)

    The M-N diagram already handles:
    - Fiber-based integration
    - Design strengths (f_cd, f_yd)
    - Ultimate limit state strains
    - Stress-strain models per EC2 Figs 3.2-3.8

    Attributes:
        section: RC section geometry with reinforcement
        concrete: Concrete material (with γ_c factor)
        concrete_model_type: EC2 constitutive model to use
        steel_branch_type: Steel post-yield behaviour
        n_fibers_width: Mesh resolution (width)
        n_fibers_height: Mesh resolution (height)

    Example:
        >>> from materials.reinforced_concrete.geometry import create_rectangular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
        >>>
        >>> # Create section
        >>> section = create_rectangular_section(width=300, height=500)
        >>> # ... add reinforcement ...
        >>>
        >>> # Create check
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>> check = BendingCheck(section=section, concrete=concrete)
        >>>
        >>> # Perform check for applied loads (all parameters must be keyword arguments)
        >>> result = check.perform_check(M_Ed=150, N_Ed=500)  # kN·m, kN
        >>> print(result)
        >>> # Bending check (EC2 §6.1): PASS (utilization: 68.5%)
    """

    section: RCSection = Field(
        ...,
        description="RC section with reinforcement",
    )

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material (γ_c applied to get f_cd)",
    )

    concrete_model_type: Literal["parabola-rectangle", "bilinear", "schematic"] = Field(
        default="parabola-rectangle",
        description="EC2 concrete stress-strain model (Fig 3.3, 3.4, 3.2)",
    )

    steel_branch_type: Literal["horizontal", "inclined"] = Field(
        default="inclined",
        description="Steel post-yield behaviour (Fig 3.8)",
    )

    n_fibers_width: int = Field(
        default=20,
        description="Number of concrete fibers across width",
    )

    n_fibers_height: int = Field(
        default=30,
        description="Number of concrete fibers across height",
    )

    def perform_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        warning_threshold: float = 0.95,
        **kwargs,
    ) -> CheckResult:
        """
        Check section capacity against applied bending moment and axial force.

        This is a FIRST PRINCIPLES check:
        - Generates M-N interaction diagram using strain compatibility
        - Checks if (N_Ed, M_Ed) point lies within the interaction surface
        - Returns utilization ratio

        Args:
            M_Ed: Design bending moment in kN·m (factored loads)
            N_Ed: Design axial force in kN (positive = compression, negative = tension)
            warning_threshold: Utilization threshold for warning (default 0.95)

        Returns:
            CheckResult with pass/fail status and utilization
        """
        # Create M-N interaction diagram (strain compatibility analysis)
        diagram = create_interaction_diagram(
            section=self.section,
            concrete=self.concrete,
            concrete_model_type=self.concrete_model_type,
            steel_branch_type=self.steel_branch_type,
            n_fibers_width=self.n_fibers_width,
            n_fibers_height=self.n_fibers_height,
        )

        # Check capacity (returns is_safe and utilization)
        is_safe, utilization = diagram.get_utilization_vector(N_Ed=N_Ed, M_Ed=M_Ed)

        # Get moment capacity at this axial force for reporting
        N_cap, M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed=N_Ed)

        # Determine which capacity is relevant based on sign of M_Ed
        # Handle case where N_Ed is outside interaction diagram bounds
        if M_Rd_pos is None and M_Rd_neg is None:
            # N_Ed is outside the interaction diagram - section has no capacity
            M_Rd = 0.0
        elif M_Ed >= 0:
            M_Rd = M_Rd_pos if M_Rd_pos is not None else 0.0
        else:
            M_Rd = abs(M_Rd_neg) if M_Rd_neg is not None else 0.0

        # Create detailed message
        if is_safe:
            if utilization >= warning_threshold:
                message = f"High utilization - consider increasing section or reinforcement"
            else:
                message = "Section capacity adequate"
        else:
            deficit = (utilization - 1.0) * 100
            message = f"Section capacity exceeded - increase section or reinforcement by ~{deficit:.0f}%"

        # Additional details
        details = {
            "N_Ed": N_Ed,
            "M_Ed": M_Ed,
            "M_Rd": M_Rd,
            "concrete_model": self.concrete_model_type,
            "steel_model": self.steel_branch_type,
            "section_name": self.section.section_name or "unnamed",
            "concrete_grade": self.concrete.grade,
            "reinforcement_ratio": self.section.reinforcement_ratio,
        }

        return self._create_result(
            check_name="Bending check (EC2 §6.1)",
            demand=abs(M_Ed),
            capacity=M_Rd,
            units="kN·m",
            code_reference="EC2 §6.1",
            warning_threshold=warning_threshold,
            message=message,
            details=details,
        )

    def get_moment_capacity(self, N_Ed: float = 0.0) -> tuple[Optional[float], Optional[float]]:
        """
        Get moment capacity at specified axial force.

        Args:
            N_Ed: Design axial force in kN

        Returns:
            Tuple of (M_Rd_positive, M_Rd_negative) in kN·m
            Returns (None, None) if N_Ed is outside the interaction diagram bounds.
        """
        diagram = create_interaction_diagram(
            section=self.section,
            concrete=self.concrete,
            concrete_model_type=self.concrete_model_type,
            steel_branch_type=self.steel_branch_type,
            n_fibers_width=self.n_fibers_width,
            n_fibers_height=self.n_fibers_height,
        )

        N_cap, M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed=N_Ed)
        
        return (M_Rd_pos, M_Rd_neg)

    def generate_interaction_diagram(self, n_points: int = 100):
        """
        Generate complete M-N interaction diagram for visualization.

        Args:
            n_points: Number of points on the curve

        Returns:
            Tuple of (N_array, M_array) for plotting
        """
        diagram = create_interaction_diagram(
            section=self.section,
            concrete=self.concrete,
            concrete_model_type=self.concrete_model_type,
            steel_branch_type=self.steel_branch_type,
            n_fibers_width=self.n_fibers_width,
            n_fibers_height=self.n_fibers_height,
        )

        return diagram.get_diagram_arrays(n_points=n_points)
