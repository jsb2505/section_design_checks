"""
Bending (flexure) check using M-N interaction diagrams.

This is a FIRST PRINCIPLES check based on strain compatibility and force equilibrium.
Uses the fiber-based M-N interaction diagram infrastructure.
"""

from typing import Literal, Optional
from pydantic import Field
import numpy as np

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
        ge=10,
        le=500,
    )

    n_fibers_height: int = Field(
        default=30,
        description="Number of concrete fibers across height",
        ge=10,
        le=500,
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

        Uses M–N interaction diagram and ray intersection:
        - Finds boundary point (N_Rd, M_Rd) along the load vector (M_Ed, N_Ed)
        - Utilization = 1 / t_cap where (M_Rd, N_Rd) = t_cap * (M_Ed, N_Ed)
        """
        diagram = create_interaction_diagram(
            section=self.section,
            concrete=self.concrete,
            concrete_model_type=self.concrete_model_type,
            steel_branch_type=self.steel_branch_type,
            n_fibers_width=self.n_fibers_width,
            n_fibers_height=self.n_fibers_height,
        )

        N_Rd, M_Rd, _, utilization = diagram.get_capacity_vector(N_Ed=N_Ed, M_Ed=M_Ed)

        demand_components = {"N": float(N_Ed), "M": float(M_Ed)}
        units_components = {"N": "kN", "M": "kN·m"}

        # Outside diagram / no intersection
        if (
            N_Rd is None
            or M_Rd is None
            or utilization is None
            or utilization == float("inf")
            or utilization != utilization  # NaN check without numpy
        ):
            message = "Load point outside interaction diagram domain (no capacity found)"
            details = {
                "N_Ed": float(N_Ed),
                "M_Ed": float(M_Ed),
                "N_Rd": N_Rd,
                "M_Rd": M_Rd,
                "utilization": float(utilization) if utilization is not None else None,
                "concrete_model": self.concrete_model_type,
                "steel_model": self.steel_branch_type,
                "section_name": self.section.section_name or "unnamed",
                "concrete_grade": self.concrete.grade,
                "reinforcement_ratio": self.section.reinforcement_ratio,
            }

            return self._create_result(
                check_name="Bending check (EC2 §6.1)",
                code_reference="EC2 §6.1",
                warning_threshold=warning_threshold,
                utilization=float("inf"),
                demand_components=demand_components,
                capacity_components=None,
                units_components=units_components,
                message=message,
                details=details,
            )

        utilization_f = float(utilization)
        capacity_components = {"N": float(N_Rd), "M": float(M_Rd)}

        # Messaging (optional; base_check will set status based on utilization)
        if utilization_f <= 1.0:
            message = (
                "High utilization - consider increasing section or reinforcement"
                if utilization_f >= warning_threshold
                else "Section capacity adequate"
            )
        else:
            deficit = (utilization_f - 1.0) * 100.0
            message = f"Section capacity exceeded - increase section or reinforcement by ~{deficit:.0f}%"

        details = {
            "N_Ed": float(N_Ed),
            "M_Ed": float(M_Ed),
            "N_Rd": float(N_Rd),
            "M_Rd": float(M_Rd),
            "utilization": utilization_f,
            "concrete_model": self.concrete_model_type,
            "steel_model": self.steel_branch_type,
            "section_name": self.section.section_name or "unnamed",
            "concrete_grade": self.concrete.grade,
            "reinforcement_ratio": self.section.reinforcement_ratio,
        }

        return self._create_result(
            check_name="Bending check (EC2 §6.1)",
            code_reference="EC2 §6.1",
            warning_threshold=warning_threshold,
            utilization=utilization_f,
            demand_components=demand_components,
            capacity_components=capacity_components,
            units_components=units_components,
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

        if N_cap is not None:
            if N_cap >=0:
                if N_Ed > N_cap:
                    M_Rd_pos = None
                    M_Rd_neg = None
            else:
                if N_Ed < N_cap:
                    M_Rd_pos = None
                    M_Rd_neg = None
        return (M_Rd_pos, M_Rd_neg)

    def generate_interaction_diagram(self, n_points: int = 100) -> tuple["np.ndarray", "np.ndarray"]:
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
