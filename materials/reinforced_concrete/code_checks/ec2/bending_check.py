"""
Bending (flexure) check using M-N interaction diagrams.

This is a FIRST PRINCIPLES check based on strain compatibility and force equilibrium.
Uses the fiber-based M-N interaction diagram infrastructure.
"""

from typing import Literal, Optional
from pydantic import Field, PrivateAttr
import numpy as np

from materials.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
from materials.reinforced_concrete.analysis import create_interaction_diagram
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from materials.reinforced_concrete.code_checks.ec2.shear_utils import find_cot_theta_for_V_Ed


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

    # ===========================
    # Internal state (private)
    # ===========================

    _diagram: Optional[MNInteractionDiagram] = PrivateAttr(default=None)

    def model_post_init(self, __context):
        """
        Create M-N interaction diagram on initialization for reuse across multiple checks.

        This significantly improves performance when checking multiple load cases against
        the same section, as the diagram (mesh + material models) only needs to be created once.
        """
        super().model_post_init(__context)

        # Create and cache the diagram for reuse
        self._diagram = create_interaction_diagram(
            section=self.section,
            concrete=self.concrete,
            concrete_model_type=self.concrete_model_type,
            steel_branch_type=self.steel_branch_type,
            n_fibers_width=self.n_fibers_width,
            n_fibers_height=self.n_fibers_height,
        )

    def perform_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        V_Ed: Optional[float] = None,
        M_cap: Optional[float] = None,
        shear_reinforcement: Optional[ShearRebar] = None,
        warning_threshold: float = 0.95,
        **kwargs,
    ) -> CheckResult:
        """
        Check section capacity against applied bending moment and axial force.

        Uses M–N interaction diagram and ray intersection:
        - Finds boundary point (N_Rd, M_Rd) along the load vector (M_Ed, N_Ed)
        - Utilization = 1 / t_cap where (M_Rd, N_Rd) = t_cap * (M_Ed, N_Ed)

        Tension Shift Rule (EC2 §9.2.1.3):
        When M_cap is provided, automatically applies tension shift rule to account for
        additional tensile force in longitudinal reinforcement due to shear (truss analogy):

        With shear reinforcement:
        - Calculates optimal cot(θ) from V_Ed using V_Rd,max formula
        - a_l = 0.5 · z · cot(θ)  [shift distance, vertical links]
        - M_add = V_Ed · a_l

        Without shear reinforcement:
        - a_l = d  [EC2 §9.2.1.3(2)]
        - M_add = V_Ed · d

        Design moment: M_design = min(M_cap, M_Ed + M_add)

        Args:
            M_Ed: Design bending moment (kN·m)
            N_Ed: Design axial force (kN, positive = compression)
            V_Ed: Design shear force (kN) - required if M_cap is provided
            M_cap: Moment capacity cap (kN·m) from envelope analysis.
                   If provided, enables tension shift rule. Limits M_design = min(M_cap, M_Ed + M_add)
            shear_reinforcement: Optional ShearReinforcement object.
                                If provided, calculates cot(θ) from V_Ed.
                                If not provided, uses a_l = d (no shear reinforcement)
            warning_threshold: Utilization threshold for warnings (default 0.95)

        Returns:
            CheckResult with pass/fail status and utilization
        """
        # Validate tension shift inputs
        # If M_cap provided, tension shift is enabled
        apply_tension_shift = M_cap is not None

        if apply_tension_shift:
            if V_Ed is None:
                raise ValueError("V_Ed must be provided when M_cap is provided (tension shift enabled)")

        # Apply tension shift rule if requested
        M_Ed_original = M_Ed
        M_add = 0.0
        z_lever_arm = None
        cot_theta_used = None
        shift_distance_a_l = None

        if apply_tension_shift and V_Ed is not None:
            # Use diagram methods to calculate effective depth and lever arm
            assert self._diagram is not None, "Diagram should be initialized in model_post_init"

            # Solve strains once if it helps (moment is what determines compression face)
            if abs(M_Ed_original) > 1e-6:
                eps_top, eps_bottom = self._diagram.find_strains_for_MN(M_Ed_original, N_Ed)
            else:
                eps_top, eps_bottom = None, None

            # Get effective depth from diagram (accounts for compression face)
            d = self._diagram.get_effective_depth(
                M_Ed=M_Ed_original,
                N_Ed=N_Ed,
                eps_top=eps_top,
                eps_bottom=eps_bottom,
            )

            z_lever_arm, _ = self._diagram.get_lever_arm(
                M_Ed=M_Ed_original,
                N_Ed=N_Ed,
                d=d,
                eps_top=eps_top,
                eps_bottom=eps_bottom,
                prefer_rigorous=False,  # for tension shift
                cap_to_09d=True,
                warn_on_fallback=False,
            )

            if shear_reinforcement is not None:
                # With shear reinforcement - calculate cot_theta from V_Ed
                # Need to import and use ShearCheck's method for calculating cot_theta
                from materials.reinforced_concrete.code_checks.ec2.shear_check import ShearCheck

                # Create temporary shear check instance just for cot_theta calculation
                temp_shear_check = ShearCheck(
                    section=self.section,
                    concrete=self.concrete,
                    shear_reinforcement=shear_reinforcement,
                )

                # Estimate sigma_cp
                sigma_cp = 0.0  # Conservative assumption TODO update this
                if N_Ed > 0:  # Compression
                    # Rough estimate: σ_cp = N_Ed / A_c
                    A_c = self.section.get_area()  # mm² TODO update this
                    sigma_cp = (N_Ed * 1000) / A_c  # Convert kN to N, get MPa TODO update this

                # TODO calculate K for cot_theta
                K =

                # Calculate optimal cot_theta from V_Ed
                cot_theta_used = find_cot_theta_for_V_Ed(
                    V_Ed=V_Ed,
                    K=K,
                    link_angle_degrees=shear_reinforcement.angle
                )

                # EC2 §9.2.1.3(2): a_l = 0.5 · z · cot(θ) for vertical links
                shift_distance_a_l = 0.5 * z_lever_arm * cot_theta_used
            else:
                # Without shear reinforcement - use a_l = d
                # EC2 §9.2.1.3(2): For members without shear reinforcement, a_l = d
                shift_distance_a_l = d
                cot_theta_used = None  # Not applicable without shear reinforcement

            # Calculate additional moment from shear
            # M_add = V_Ed · a_l (shift distance)
            # Convert: V_Ed in kN, a_l in mm -> M_add in kN·m
            M_add = abs(V_Ed) * shift_distance_a_l / 1000.0  # Convert mm to m

            # Apply moment cap
            M_Ed = min(M_cap, M_Ed_original + M_add)

        # Use cached diagram for capacity check (created in model_post_init)
        assert self._diagram is not None, "Diagram should be initialized in model_post_init"

        # Perform capacity check with (potentially modified) M_Ed
        N_Rd, M_Rd, is_safe, utilization = self._diagram.get_capacity_vector(N_Ed=N_Ed, M_Ed=M_Ed, return_details=False)  # type: ignore[misc]

        demand_components = {"N": float(N_Ed), "M": float(M_Ed_original)}  # Show original demand
        units_components = {"N": "kN", "M": "kN·m"}

        # Outside diagram / no intersection
        if (
            N_Rd is None
            or M_Rd is None
            or is_safe == False
            or utilization is None
            or utilization == float("inf")
            or utilization != utilization  # NaN check without numpy
        ):
            message = "Load point outside interaction diagram domain (no capacity found)"
            details = {
                "N_Ed": float(N_Ed),
                "M_Ed_original": float(M_Ed_original),
                "M_Ed_design": float(M_Ed),
                "M_add": float(M_add) if apply_tension_shift else None,
                "V_Ed": float(V_Ed) if V_Ed is not None else None,
                "cot_theta": float(cot_theta_used) if cot_theta_used is not None else None,
                "shift_distance_a_l_mm": float(shift_distance_a_l) if shift_distance_a_l is not None else None,
                "z_lever_arm_mm": float(z_lever_arm) if z_lever_arm is not None else None,
                "M_cap": float(M_cap) if M_cap is not None else None,
                "tension_shift_applied": apply_tension_shift,
                "shear_reinforcement_provided": shear_reinforcement is not None,
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
            "M_Ed_original": float(M_Ed_original),
            "M_Ed_design": float(M_Ed),
            "M_add": float(M_add) if apply_tension_shift else None,
            "V_Ed": float(V_Ed) if V_Ed is not None else None,
            "cot_theta": float(cot_theta_used) if cot_theta_used is not None else None,
            "shift_distance_a_l_mm": float(shift_distance_a_l) if shift_distance_a_l is not None else None,
            "z_lever_arm_mm": float(z_lever_arm) if z_lever_arm is not None else None,
            "M_cap": float(M_cap) if M_cap is not None else None,
            "tension_shift_applied": apply_tension_shift,
            "shear_reinforcement_provided": shear_reinforcement is not None,
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
        # Use cached diagram
        assert self._diagram is not None, "Diagram should be initialized in model_post_init"
        N_cap, M_Rd_pos, M_Rd_neg = self._diagram.get_capacity_fixed_n(N_Ed=N_Ed)

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
        # Use cached diagram
        assert self._diagram is not None, "Diagram should be initialized in model_post_init"
        return self._diagram.get_diagram_arrays(n_points=n_points)
