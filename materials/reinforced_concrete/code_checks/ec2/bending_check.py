"""
Bending (flexure) check using M-N interaction diagrams.

This is a FIRST PRINCIPLES check based on strain compatibility and force equilibrium.
Uses the fibre-based M-N interaction diagram infrastructure.
"""

from typing import Optional
from dataclasses import dataclass
from math import copysign
from pydantic import Field, PrivateAttr
import numpy as np

from materials.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from materials.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
from materials.reinforced_concrete.analysis import create_interaction_diagram
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
import materials.reinforced_concrete.code_checks.ec2.shear_utils as shear_utils


class BendingCheck(BaseCodeCheck):
    """
    EC2 bending check using M-N interaction diagram (§6.1).

    This check uses FIRST PRINCIPLES:
    1. Strain compatibility (plane sections remain plane)
    2. Force equilibrium (ΣF = N, ΣM = M)
    3. Constitutive models (stress-strain with codified factors γ_c, γ_s)

    The M-N diagram already handles:
    - fibre-based integration
    - Design strengths (f_cd, f_yd)
    - Ultimate limit state strains
    - Stress-strain models per EC2 Figs 3.2-3.8

    Attributes:
        section: RC section geometry with reinforcement
        concrete: Concrete material (with γ_c factor)
        concrete_model_type: EC2 constitutive model to use
        steel_model_type: Steel post-yield behaviour
        n_fibres_width: Mesh resolution (width)
        n_fibres_height: Mesh resolution (height)

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

    concrete_model_type: ConcreteModelType = Field(
        default=ConcreteModelType.PARABOLA_RECTANGLE,
        description="EC2 concrete stress-strain model (Fig 3.3, 3.4, 3.2)",
    )

    steel_model_type: SteelModelType = Field(
        default=SteelModelType.INCLINED,
        description="Steel post-yield behaviour (Fig 3.8)",
    )

    n_fibres_width: int = Field(
        default=20,
        description="Number of concrete fibres across width",
        ge=10,
        le=500,
    )

    n_fibres_height: int = Field(
        default=30,
        description="Number of concrete fibres across height",
        ge=10,
        le=500,
    )


    # ===========================
    # Limit state factors
    # ===========================

    use_accidental: bool = Field(
        default=False,
        description="Use accidental limit state partial factors (gamma_c_accidental, gamma_s_accidental)",
    )


    # ===========================
    # Internal state (private)
    # ===========================

    _diagram: MNInteractionDiagram = PrivateAttr()
    _diagram_no_comp_steel: Optional[MNInteractionDiagram] = PrivateAttr(default=None)

    def model_post_init(self, __context):
        """
        Create M-N interaction diagram on initialization for reuse across multiple checks.

        This significantly improves performance when checking multiple load cases against
        the same section, as the diagram (mesh + material models) only needs to be created once.
        """
        super().model_post_init(__context)

        # Create and cache the diagram for reuse (with compression steel)
        self._diagram = create_interaction_diagram(
            section=self.section,
            concrete=self.concrete,
            concrete_model_type=self.concrete_model_type,
            steel_model_type=self.steel_model_type,
            n_fibres_width=self.n_fibres_width,
            n_fibres_height=self.n_fibres_height,
            use_accidental=self.use_accidental,
            ignore_compression_steel=False,
        )
        # Diagram without compression steel is created lazily on first use
        self._diagram_no_comp_steel = None

        # cached properties to save time later
        self._A_transformed = self.section.get_transformed_area(self.concrete.E_cm)  # mm²
        self._A_gross = self.section.get_area()  # mm²

    def _get_diagram(self, ignore_compression_steel: bool = False) -> MNInteractionDiagram:
        """Get the appropriate cached diagram based on ignore_compression_steel flag."""
        if not ignore_compression_steel:
            return self._diagram

        # Lazily create the diagram without compression steel
        if self._diagram_no_comp_steel is None:
            self._diagram_no_comp_steel = create_interaction_diagram(
                section=self.section,
                concrete=self.concrete,
                concrete_model_type=self.concrete_model_type,
                steel_model_type=self.steel_model_type,
                n_fibres_width=self.n_fibres_width,
                n_fibres_height=self.n_fibres_height,
                use_accidental=self.use_accidental,
                ignore_compression_steel=True,
            )
        return self._diagram_no_comp_steel


    # ===============================================
    # Properties (immutable - don't depend on loads)
    # ===============================================

    @property
    def f_cd_design(self) -> float:
        """Design concrete strength (accidental or persistent) in MPa."""
        return self.concrete.f_cd_accidental if self.use_accidental else self.concrete.f_cd


    def perform_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        V_Ed: Optional[float] = None,
        M_cap: Optional[float] = None,
        shear_reinforcement: Optional[ShearRebar] = None,
        warning_threshold: float = 0.95,
        ignore_compression_steel: bool = False,
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
            ignore_compression_steel: If True, steel in compression contributes zero force.
                                     This is a conservative option used by some commercial software.

        Returns:
            CheckResult with pass/fail status and utilization
        """
        # Validate tension shift inputs
        # If M_cap provided, tension shift is enabled
        apply_tension_shift = M_cap is not None
        if apply_tension_shift and V_Ed is None:
            raise ValueError("V_Ed must be provided when M_cap is provided (tension shift enabled)")

        return self._check_single_case(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            V_Ed=V_Ed,
            M_cap=M_cap,
            shear_reinforcement=shear_reinforcement,
            warning_threshold=warning_threshold,
            ignore_compression_steel=ignore_compression_steel,
        )


    def _check_single_case(
        self,
        *,
        M_Ed: float,
        N_Ed: float,
        V_Ed: Optional[float],
        M_cap: Optional[float],
        shear_reinforcement: Optional[ShearRebar],
        warning_threshold: float,
        ignore_compression_steel: bool = False,
    ) -> CheckResult:
        apply_tension_shift = M_cap is not None

        M_Ed_original = float(M_Ed)

        # --- Step 1: tension shift (pure-ish, easy to test) ---
        shift = self._apply_tension_shift_if_enabled(
            M_Ed_original=M_Ed_original,
            N_Ed=float(N_Ed),
            V_Ed=float(V_Ed) if V_Ed is not None else None,
            M_cap=float(M_cap) if M_cap is not None else None,
            shear_reinforcement=shear_reinforcement,
            enabled=apply_tension_shift,
        )

        # shift.M_design is the moment used for the capacity check
        M_design = shift.M_design

        # --- Step 2: capacity check against diagram ---
        diagram = self._get_diagram(ignore_compression_steel)
        cap = diagram.get_capacity_vector(N_Ed=N_Ed, M_Ed=M_design, return_details=False)
        N_Rd, M_Rd, is_safe, utilization = cap.N_Rd, cap.M_Rd, cap.is_safe, cap.utilization

        demand_components = {"N": float(N_Ed), "M": float(M_Ed_original)}
        units_components = {"N": "kN", "M": "kN·m"}

        # --- Step 3: handle “no intersection / invalid” outcome ---
        if (
            N_Rd is None
            or M_Rd is None
            or is_safe is False
            or utilization is None
            or utilization == float("inf")
            or utilization != utilization  # NaN
        ):
            message = "Load point outside interaction diagram domain (no capacity found)"
            details = {
                "N_Ed": float(N_Ed),
                "M_Ed_original": float(M_Ed_original),
                "M_Ed_design": float(M_design),
                **shift.details_dict(),  # <- includes M_add, z, cot_theta, etc (or None)
                "N_Rd": N_Rd,
                "M_Rd": M_Rd,
                "utilization": float(utilization) if utilization is not None else None,
                "concrete_model": self.concrete_model_type,
                "steel_model": self.steel_model_type,
                "section_name": self.section.section_name or "unnamed",
                "concrete_grade": self.concrete.grade,
                "reinforcement_ratio": self.section.reinforcement_ratio,
                "ignore_compression_steel": ignore_compression_steel,
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
            "M_Ed_design": float(M_design),
            **shift.details_dict(),
            "N_Rd": float(N_Rd),
            "M_Rd": float(M_Rd),
            "utilization": utilization_f,
            "concrete_model": self.concrete_model_type,
            "steel_model": self.steel_model_type,
            "section_name": self.section.section_name or "unnamed",
            "concrete_grade": self.concrete.grade,
            "reinforcement_ratio": self.section.reinforcement_ratio,
            "ignore_compression_steel": ignore_compression_steel,
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
    

    @dataclass(frozen=True)
    class _TensionShiftResult:
        """Internal container for tension shift rule outputs (EC2 §9.2.1.3)."""

        M_design: float
        applied: bool

        # Optional details (None when not applied)
        M_add: Optional[float] = None
        V_Ed: Optional[float] = None
        M_cap: Optional[float] = None
        cot_theta: Optional[float] = None
        shift_distance_a_l_mm: Optional[float] = None
        z_lever_arm_mm: Optional[float] = None
        shear_reinforcement_provided: bool = False

        def details_dict(self) -> dict:
            """Serialize shift info into the `details` dict format used by the check result."""
            return {
                "tension_shift_applied": self.applied,
                "M_add": float(self.M_add) if self.M_add is not None else None,
                "V_Ed": float(self.V_Ed) if self.V_Ed is not None else None,
                "M_cap": float(self.M_cap) if self.M_cap is not None else None,
                "cot_theta": float(self.cot_theta) if self.cot_theta is not None else None,
                "shift_distance_a_l_mm": float(self.shift_distance_a_l_mm)
                if self.shift_distance_a_l_mm is not None
                else None,
                "z_lever_arm_mm": float(self.z_lever_arm_mm) if self.z_lever_arm_mm is not None else None,
                "shear_reinforcement_provided": self.shear_reinforcement_provided,
            }


    def _solve_strains_if_helpful(self, *, M_Ed: float, N_Ed: float) -> tuple[Optional[float], Optional[float]]:
        """
        Solve strains once (if moment is meaningful) to determine compression face etc.

        Returns:
            (eps_top, eps_bottom) or (None, None) if not solved.
        """
        if abs(M_Ed) > 1e-6:
            return self._diagram.find_strains_for_MN(M_Ed, N_Ed)
        return (None, None)


    def _apply_tension_shift_if_enabled(
        self,
        *,
        M_Ed_original: float,
        N_Ed: float,
        V_Ed: Optional[float],
        M_cap: Optional[float],
        shear_reinforcement: Optional[ShearRebar],
        enabled: bool,
    ) -> _TensionShiftResult:
        """
        Apply EC2 §9.2.1.3 tension shift rule if enabled (i.e., M_cap is provided).

        Behaviour matches your previous inline implementation:
        - Computes d and z (capped to 0.9d) from the diagram.
        - With shear reinforcement: computes cot(theta) from V_Ed via V_Rd,max-form K term.
        - Without shear reinforcement: uses a_l = d.
        - Adds M_add = |V_Ed| * a_l and caps magnitude by |M_cap|, then restores sign of M_Ed_original.
        """
        if not enabled:
            return self._TensionShiftResult(M_design=float(M_Ed_original), applied=False)

        # By contract: wrapper validated these when enabled
        if V_Ed is None or M_cap is None:
            raise ValueError("Internal error: V_Ed and M_cap must be provided when tension shift is enabled")

        # 1) Solve strains once (moment determines compression face; axial alone doesn't)
        eps_top, eps_bottom = self._solve_strains_if_helpful(M_Ed=float(M_Ed_original), N_Ed=float(N_Ed))

        # 2) Effective depth
        d = self._diagram.get_effective_depth(
            M_Ed=float(M_Ed_original),
            N_Ed=float(N_Ed),
            eps_top=eps_top,
            eps_bottom=eps_bottom,
        )

        # 3) Lever arm for tension shift: prefer approximate ok, cap to 0.9d
        z_ec2, _ = self._diagram.get_lever_arm(
            M_Ed=float(M_Ed_original),
            N_Ed=float(N_Ed),
            d=d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            prefer_rigorous=False,
            cap_to_09d=True,
            warn_on_fallback=False,
        )

        cot_theta: Optional[float]
        shift_distance_a_l_mm: float

        if shear_reinforcement is not None:
            # With shear reinforcement - calculate cot(theta) from V_Ed

            f_cd = self.f_cd_design  # N/mm² (MPa)

            # Allow negative N_Ed into sigma_cp (compression positive)
            sigma_cp_uncapped = shear_utils.sigma_cp_from_N_and_area(N_Ed=float(N_Ed), A_mm2=float(self._A_transformed))
            sigma_cp = shear_utils.cap_sigma_cp_upper(sigma_cp=sigma_cp_uncapped, f_cd=f_cd)

            alpha_cw = shear_utils.find_alpha_cw(f_cd=f_cd, sigma_cp=sigma_cp)
            nu = shear_utils.find_nu_factor(f_ck=self.concrete.f_ck)
            b_w = shear_utils.calculate_section_breadth(section=self.section)  # mm

            # K = alpha_cw * b_w * z * nu * f_cd
            K = alpha_cw * b_w * z_ec2 * nu * f_cd

            cot_theta = shear_utils.find_cot_theta_for_V_Ed(
                V_Ed=float(V_Ed),
                K=K,
                link_angle_degrees=shear_reinforcement.angle,
            )

            # EC2 §9.2.1.3(2): a_l = 0.5 * z * cot(theta) for vertical links
            shift_distance_a_l_mm = 0.5 * z_ec2 * cot_theta

        else:
            # Without shear reinforcement: EC2 §9.2.1.3(2): a_l = d
            cot_theta = None
            shift_distance_a_l_mm = d

        # 4) Additional moment magnitude (V in kN, a_l in mm -> kN·m)
        M_add = abs(float(V_Ed)) * (shift_distance_a_l_mm / 1000.0)

        # 5) Apply cap to magnitude and restore sign of original
        abs_M_Ed_orig = abs(float(M_Ed_original))
        abs_M_cap = abs(float(M_cap))

        abs_M_Ed_shifted = min(abs_M_cap, abs_M_Ed_orig + M_add)
        M_design = copysign(abs_M_Ed_shifted, float(M_Ed_original))

        return self._TensionShiftResult(
            M_design=float(M_design),
            applied=True,
            M_add=float(M_add),
            V_Ed=float(V_Ed),
            M_cap=float(M_cap),
            cot_theta=float(cot_theta) if cot_theta is not None else None,
            shift_distance_a_l_mm=float(shift_distance_a_l_mm),
            z_lever_arm_mm=float(z_ec2),
            shear_reinforcement_provided=(shear_reinforcement is not None),
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
        N_cap, M_Rd_pos, M_Rd_neg = self._diagram.get_capacity_fixed_n(N_Ed=N_Ed)

        if N_cap is not None:
            if N_cap >= 0:  # N_cap is positive
                if N_Ed > N_cap:  # N_Ed outside upper bound
                    M_Rd_pos = None
                    M_Rd_neg = None
            else:  # N_cap is negative
                if N_Ed < N_cap:  # N_Ed outside lower bound
                    M_Rd_pos = None
                    M_Rd_neg = None
        return (M_Rd_pos, M_Rd_neg)


    def generate_interaction_diagram_arrays(self, n_points: int = 120) -> tuple["np.ndarray", "np.ndarray"]:
        """
        Generate complete M-N interaction diagram for visualization.

        Args:
            n_points: Number of points on the curve

        Returns:
            Tuple of (N_array, M_array) for plotting
        """
        return self._diagram.get_diagram_arrays(n_points=n_points)
