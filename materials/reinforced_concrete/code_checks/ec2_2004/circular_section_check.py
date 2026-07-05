"""
Circular section design checks following Orr (2012) approach.

Wraps BendingCheck, ShearCheck, and CrackingCheck with circular-specific
modifications for piles, columns, and other circular RC members.

Key modifications from standard EC2:
- Shear reinforcement efficiency factors λ1 (closed links) and λ2 (spirals)
- Equivalent web width for V_Rd_max (concrete strut crushing)
- Uncracked V_Rd_c using principal stress approach (Eq.17)
- Lever arm from first-principles (not capped to 0.9d)
- Optional k_f factor for cast-in-place piles (EC2 §2.4.2.5(2))

Reference:
    Orr, J.J. (2012). "Shear design of circular concrete sections."
    University of Bath.
"""

import warnings
from math import atan, degrees, pi, sqrt
from typing import Any, Dict, Optional, cast

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, computed_field, model_validator

from materials.reinforced_concrete.code_checks.base_check import (
    CheckResult,
    CheckStatus,
)
from materials.reinforced_concrete.code_checks.ec2_2004.bending_check import BendingCheck
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck, ShearLoadCase
from materials.reinforced_concrete.code_checks.ec2_2004.cracking_check import (
    CrackingCheck,
    LoadDuration,
    SLSCombination,
)
from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import (
    find_cot_theta_for_V_Ed,
    find_alpha_cw,
    find_nu_1_factor,
    find_nu_1_factor_note_2,
    find_nu_factor,
    sigma_cp_from_N_and_area,
    cap_sigma_cp_upper,
    clamp_cot_theta,
)
from materials.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
from materials.reinforced_concrete.ndp import get_ndp
from materials.core.units import ForceUnit, to_kn


class CircularSectionCheck(BaseModel):
    """
    EC2-compliant design checks for circular sections (piles/columns).

    Wraps BendingCheck, ShearCheck, and CrackingCheck with circular-specific
    modifications following Orr (2012).

    - **Bending**: Forwarded to BendingCheck with iterate_z=True by default.
      Tension shift uses circular equivalent web width for cot(θ) computation.
    - **Shear**: Custom implementation with λ1/λ2 efficiency factors, circular
      web width, and uncracked V_Rd_c per Eq.17.
    - **Cracking**: Forwarded to CrackingCheck (no circular modifications).

    The sub-checks are accessible via the ``bending`` and ``cracking`` properties
    for advanced operations (plotting, capacity queries, detailed results).

    Attributes:
        section: Circular RC section geometry with reinforcement
        concrete: Concrete material properties
        diameter: Section diameter (mm)
        cover: Cover to outer face of shear links (mm)
        shear_reinforcement: Shear links/spirals (optional)
        is_spiral: If True, ShearRebar.spacing is treated as spiral pitch for λ2
        apply_k_f: If True, multiply γ_c by k_f for cast-in-place piles (EC2 §2.4.2.5)

    Example:
        >>> from materials.reinforced_concrete.geometry import create_circular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
        >>>
        >>> section = create_circular_section(diameter=600)
        >>> # ... add perimeter reinforcement ...
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>> links = ShearRebar(diameter=12, spacing=200, n_legs=2, grade="B500B")
        >>>
        >>> check = CircularSectionCheck(
        ...     section=section, concrete=concrete, diameter=600,
        ...     cover=50, shear_reinforcement=links,
        ... )
        >>>
        >>> bending_result = check.perform_bending_check(M_Ed=150, N_Ed=500)
        >>> shear_result = check.perform_shear_check(
        ...     load_case=ShearLoadCase(V_Ed=200, M_Ed=150, N_Ed=500)
        ... )
        >>> cracking_result = check.perform_cracking_check(M_Ed=80, N_Ed=300)
    """

    # ===========================
    # Core inputs
    # ===========================

    section: RCSection = Field(
        ...,
        description="Circular RC section geometry with reinforcement",
    )

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material properties",
    )

    diameter: float = Field(
        ...,
        description="Section diameter (mm)",
        gt=0,
    )

    cover: float = Field(
        ...,
        description="Cover to outer face of shear links (mm)",
        gt=0,
    )

    # ===========================
    # Shear reinforcement
    # ===========================

    shear_reinforcement: Optional[ShearRebar] = Field(
        default=None,
        description="Shear links/spirals (None if unreinforced)",
    )

    is_spiral: bool = Field(
        default=False,
        description=(
            "If True, treat ShearRebar.spacing as the spiral pitch for λ2 "
            "calculation. When False (default), λ2 = 1.0 (closed links)."
        ),
    )

    r_sv_override: Optional[float] = Field(
        default=None,
        description=(
            "Manual override for r_sv (mm) — radius from section centre to "
            "shear reinforcement centreline. If None, computed as "
            "D/2 - cover - link_dia/2."
        ),
        gt=0,
    )

    use_simplified_lambda_1: bool = Field(
        default=False,
        description=(
            "If True, use the simplified λ1 = 0.85. "
            "If False (default), compute λ1 by numerical integration (Eq.6)."
        ),
    )

    use_increased_nu_1: bool = Field(
        default=False,
        description=(
            "Use increased ν₁ factor per EC2 §6.2.3(3) Note 2 when shear "
            "reinforcement stress is below 80% of f_yk (σ_s < 0.8·f_yk). "
            "This allows higher V_Rd,max capacity but requires iterative "
            "calculation and reduces f_ywd to 0.8·f_ywk for V_Rd,s."
        ),
    )

    # ===========================
    # Pile / foundation
    # ===========================

    apply_k_f: bool = Field(
        default=False,
        description=(
            "If True, multiply γ_c by k_f (NDP) for cast-in-place piles "
            "without permanent casing (EC2 §2.4.2.5(2)). Only affects ULS "
            "checks (bending, shear), not SLS cracking."
        ),
    )

    # ===========================
    # Forwarded to sub-checks
    # ===========================

    concrete_model_type: ConcreteModelType = Field(
        default=ConcreteModelType.PARABOLA_RECTANGLE,
        description="EC2 concrete stress-strain model for ULS",
    )

    steel_model_type: SteelModelType = Field(
        default=SteelModelType.INCLINED,
        description="Steel post-yield behaviour",
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

    use_accidental: bool = Field(
        default=False,
        description="Use accidental limit state partial factors",
    )

    # Cracking-specific
    w_k_limit: float = Field(
        default=0.3,
        description="Allowable crack width in mm (EC2 Table 7.1N)",
        gt=0.0,
    )

    load_duration: LoadDuration = Field(
        default=LoadDuration.LONG_TERM,
        description="Load duration: SHORT_TERM (k_t=0.6) or LONG_TERM (k_t=0.4)",
    )

    creep_coefficient: float = Field(
        default=1.5,
        description="Linear creep coefficient φ for long-term SLS",
        ge=0.0,
    )

    is_high_bond_bar: bool = Field(
        default=True,
        description="True for ribbed bars (k_1=0.8), False for plain bars (k_1=1.6)",
    )

    sls_combination: SLSCombination = Field(
        default=SLSCombination.QUASI_PERMANENT,
        description="SLS load combination type for stress limitation checks",
    )

    # ===========================
    # Private sub-checks
    # ===========================

    _bending_check: Optional[BendingCheck] = PrivateAttr(default=None)
    _shear_check: Optional[ShearCheck] = PrivateAttr(default=None)
    _cracking_check: Optional[CrackingCheck] = PrivateAttr(default=None)
    _concrete_uls: Optional[ConcreteMaterial] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _init_sub_checks(self) -> "CircularSectionCheck":
        # Warn if shear reinforcement angle is not 90° (ineffective for circular)
        if (
            self.shear_reinforcement is not None
            and abs(self.shear_reinforcement.angle - 90.0) > 1e-9
        ):
            warnings.warn(
                f"ShearRebar.angle={self.shear_reinforcement.angle}° is ignored for "
                f"circular sections — links must be 90° (vertical)."
                f"Spirals use spacing for pitch, angle makes no difference. "
                f"The λ1/λ2 efficiency factors account for circular geometry.",
                UserWarning,
                stacklevel=2,
            )

        # Apply k_f to gamma_c for ULS if requested
        concrete_uls = self.concrete
        if self.apply_k_f:
            k_f = cast(float, get_ndp("k_f"))
            concrete_uls = self.concrete.model_copy(
                update={"gamma_c": self.concrete.gamma_c * k_f}
            )
        self._concrete_uls = concrete_uls

        # Create sub-checks
        self._bending_check = BendingCheck(
            section=self.section,
            concrete=concrete_uls,
            concrete_model_type=self.concrete_model_type,
            steel_model_type=self.steel_model_type,
            n_fibres_width=self.n_fibres_width,
            n_fibres_height=self.n_fibres_height,
            use_accidental=self.use_accidental,
        )

        self._shear_check = ShearCheck(
            section=self.section,
            concrete=concrete_uls,
            shear_reinforcement=self.shear_reinforcement,
            use_accidental=self.use_accidental,
            use_rigorous=True,
            cap_lever_arm=True,  # z ≤ 0.9d safety cap. Circular z_mech is typically ~0.77d so rarely activates.
            concrete_model_type=self.concrete_model_type,
            steel_model_type=self.steel_model_type,
        )

        self._cracking_check = CrackingCheck(
            section=self.section,
            concrete=self.concrete,  # SLS uses characteristic properties (no k_f)
            w_k_limit=self.w_k_limit,
            load_duration=self.load_duration,
            concrete_model_type=ConcreteModelType.LINEAR_ELASTIC,
            steel_model_type=self.steel_model_type,
            n_fibres_width=self.n_fibres_width,
            n_fibres_height=self.n_fibres_height,
            is_high_bond_bar=self.is_high_bond_bar,
            creep_coefficient=self.creep_coefficient,
            sls_combination=self.sls_combination,
        )

        return self

    # ===========================
    # Computed properties
    # ===========================

    @computed_field
    @property
    def r_sv(self) -> float:
        """Radius from section centre to shear reinforcement centreline (mm)."""
        if self.r_sv_override is not None:
            return self.r_sv_override
        if self.shear_reinforcement is not None:
            return self.diameter / 2 - self.cover - self.shear_reinforcement.diameter / 2
        return self.diameter / 2 - self.cover

    @property
    def bending(self) -> BendingCheck:
        """Direct access to the internal BendingCheck (for plots, capacity queries, etc.)."""
        assert self._bending_check is not None
        return self._bending_check

    @property
    def cracking(self) -> CrackingCheck:
        """Direct access to the internal CrackingCheck (for detailed results, etc.)."""
        assert self._cracking_check is not None
        return self._cracking_check

    @property
    def _f_cd_design(self) -> float:
        """Design concrete compressive strength for ULS (accounts for k_f)."""
        assert self._concrete_uls is not None
        if self.use_accidental:
            return self._concrete_uls.f_cd_accidental
        return self._concrete_uls.f_cd

    @property
    def _f_ctd_design(self) -> float:
        """Design concrete tensile strength for ULS (accounts for k_f)."""
        assert self._concrete_uls is not None
        if self.use_accidental:
            return self._concrete_uls.f_ctd_accidental
        return self._concrete_uls.f_ctd

    @property
    def _f_ywd_design(self) -> float:
        """Design yield strength of shear reinforcement (MPa)."""
        if self.shear_reinforcement is None:
            return 0.0
        if self.use_accidental:
            return self.shear_reinforcement.f_yd_accidental
        return self.shear_reinforcement.f_yd


    # ===========================
    # Circular-specific methods
    # ===========================

    def calculate_lambda_1(self, z_0: float, z: float) -> float:
        """
        Link efficiency factor λ1 for circular sections (Orr 2012, Eq.6).

        Computed by numerical integration over the lever arm depth. Represents
        the fraction of link force that effectively resists vertical shear.

        Args:
            z_0: Distance from section centre to tension centroid (mm),
                 typically d - D/2.
            z: Lever arm (mm).

        Returns:
            λ1 efficiency factor (0 to 1). Typically ≈ 0.85 for common geometries.
        """
        if self.use_simplified_lambda_1:
            return 0.85

        r_sv = self.r_sv
        if r_sv <= 0:
            return 0.85  # Fallback

        n_points = 1000
        X = np.linspace(0, 1, n_points)
        y = z_0 - z * X  # distance from section centre at each integration point

        # Clip to avoid sqrt of negative (point outside link circle)
        arg = 1.0 - (y / r_sv) ** 2
        arg = np.clip(arg, 0.0, None)

        integrand = np.sqrt(arg)
        lambda_1 = float(np.trapezoid(integrand, X))

        # Sanity: clamp to [0, 1]
        return max(0.0, min(1.0, lambda_1))


    def calculate_lambda_2(self) -> float:
        """
        Spiral link efficiency factor λ2 (Orr 2012, Eq.8).

        For closed links (is_spiral=False), returns 1.0.
        For spiral links, accounts for the helix angle reduction:
            λ2 = 1 / √((p / (2π·r_sv))² + 1)
        where p = spiral pitch (= ShearRebar.spacing).

        Returns:
            λ2 efficiency factor (0 to 1).
        """
        if not self.is_spiral or self.shear_reinforcement is None:
            return 1.0

        p = self.shear_reinforcement.spacing
        r_sv = self.r_sv
        if r_sv <= 0:
            return 1.0

        return 1 / sqrt((p / (2 * pi * r_sv)) ** 2 + 1)


    def calculate_equivalent_web_width(
        self, d: float, z: float
    ) -> tuple[float, float, float]:
        """
        Equivalent web width for circular sections (Orr 2012, Eq.10-13).

        b_w = min(b_wc, b_wt) where:
        - b_wc = chord width at compression centroid depth (Eq.10)
        - b_wt = chord width inside shear reinforcement at tension centroid (Eq.12)

        Args:
            d: Effective depth from compression face to tension centroid (mm)
            z: Lever arm (mm)

        Returns:
            (b_w, b_wc, b_wt) all in mm
        """
        r = self.diameter / 2  # radius to extreme fibre
        r_sv = self.r_sv

        # b_wc: width at compression centroid (Eq.10)
        c = d - z  # depth of compression centroid from compression face (Eq.11)
        c = max(c, 0.0)
        arg_c = c * (2 * r - c)
        b_wc = 2 * sqrt(max(arg_c, 0.0))

        # b_wt: width inside shear reinforcement at tension centroid (Eq.12-13)
        e = r + r_sv - d  # Eq.13: (D/2 + r_sv) - d
        e = max(e, 0.0)
        arg_t = e * (2 * r_sv - e)
        b_wt = 2 * sqrt(max(arg_t, 0.0))

        b_w = min(b_wc, b_wt) if b_wc > 0 and b_wt > 0 else max(b_wc, b_wt)
        return b_w, b_wc, b_wt


    def calculate_V_Rd_c_uncracked(self, sigma_cp: float) -> float:
        """
        Unreinforced shear capacity for uncracked circular sections (Orr 2012, Eq.17).
        This is conservative for low axial forces (i.e. a risk of being cracked) since
        the contribution from the longitudinal steel is disregarded.

        Based on principal tensile stress limited to f_ctd:
            V_Rd_c = (3·π·r²/4) · √(f_ctd² + σ_cp · f_ctd)

        Args:
            sigma_cp: Axial compressive stress (MPa), compression positive.

        Returns:
            V_Rd_c in kN
        """
        r = self.diameter / 2  # mm
        f_ctd = self._f_ctd_design

        # Eq.17: V_Rd,c = (3 · π · r² / 4) · √(f_ctd² + σ_cp · f_ctd)
        # σ_cp contribution: compression delays cracking. If σ_cp < 0 (tension),
        # the argument may become negative → clamp to 0.
        # Note: Eq.6.5 upper bound (V_Rd_c ≤ 0.5·b_w·d·ν·f_cd) is applied at the
        # call site in perform_shear_check where b_w and d are available.
        inner = f_ctd ** 2 + sigma_cp * f_ctd
        if inner < 0:
            return 0.0

        V_Rd_c_N = (3 * pi * r ** 2 / 4) * sqrt(inner)
        return to_kn(V_Rd_c_N, ForceUnit.N)


    # ===========================
    # Check methods
    # ===========================

    def perform_bending_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        V_Ed: Optional[float] = None,
        M_cap: Optional[float] = None,
        shear_reinforcement: Optional[ShearRebar] = None,
        cot_theta_override: Optional[float] = None,
        warning_threshold: float = 0.95,
        ignore_compression_steel: bool = False,
        iterate_z: bool = True,
    ) -> CheckResult:
        """
        Bending check for circular section.

        Forwards to internal BendingCheck with iterate_z=True by default.
        When tension shift is active (V_Ed + M_cap provided) and no
        cot_theta_override is given, automatically computes cot(θ) from the
        circular equivalent web width to avoid using the standard rectangular
        web width internally.

        Args:
            M_Ed: Design bending moment (kN·m)
            N_Ed: Design axial force (kN, compression positive)
            V_Ed: Design shear force (kN) — required if M_cap is provided
            M_cap: Moment capacity cap (kN·m) from envelope analysis
            shear_reinforcement: Override for shear reinforcement (defaults to self)
            cot_theta_override: User-supplied cot(θ) for tension shift
            warning_threshold: Utilization threshold for warnings
            ignore_compression_steel: If True, ignore compression reinforcement
            iterate_z: If True, iteratively recalculate z (default True for circular)

        Returns:
            CheckResult with bending utilization
        """
        shear_reinf = (
            shear_reinforcement
            if shear_reinforcement is not None
            else self.shear_reinforcement
        )

        # Auto-compute cot_theta from circular b_w when tension shift is active
        # and user hasn't supplied their own override
        effective_cot_theta = cot_theta_override
        if (
            M_cap is not None
            and V_Ed is not None
            and cot_theta_override is None
            and shear_reinf is not None
        ):
            effective_cot_theta = self._compute_cot_theta_for_tension_shift(
                M_Ed=M_Ed, N_Ed=N_Ed, V_Ed=V_Ed
            )

        assert self._bending_check is not None
        return self._bending_check.perform_check(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            V_Ed=V_Ed,
            M_cap=M_cap,
            shear_reinforcement=shear_reinf,
            cot_theta_override=effective_cot_theta,
            warning_threshold=warning_threshold,
            ignore_compression_steel=ignore_compression_steel,
            iterate_z=iterate_z,
        )


    def perform_cracking_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        warning_threshold: float = 0.95,
        ignore_compression_steel: bool = False,
        force_cracked: bool = False,
    ) -> CheckResult:
        """
        Cracking check for circular section (wrapper — no circular modifications).

        Forwards to internal CrackingCheck.

        Args:
            M_Ed: Design moment at SLS (kN·m)
            N_Ed: Design axial force at SLS (kN, compression positive)
            warning_threshold: Utilization threshold for warnings
            ignore_compression_steel: If True, ignore compression reinforcement
            force_cracked: If True, skip cracking moment check and proceed
                to cracked analysis regardless

        Returns:
            CheckResult with crack width utilization
        """
        assert self._cracking_check is not None
        return self._cracking_check.perform_check(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            warning_threshold=warning_threshold,
            ignore_compression_steel=ignore_compression_steel,
            force_cracked=force_cracked,
        )


    def perform_shear_check(
        self,
        *,
        load_case: ShearLoadCase,
        cot_theta_override: Optional[float] = None,
        warning_threshold: float = 0.95,
        force_cracked: bool = False,
    ) -> CheckResult:
        """
        Shear check for circular section (Orr 2012).

        Custom implementation using:
        - λ1, λ2 efficiency factors for V_Rd_s
        - Circular equivalent web width for V_Rd_max
        - Uncracked V_Rd_c from principal stress (Eq.17)
        - Solver-based d and z from the interaction diagram

        Args:
            load_case: ShearLoadCase with V_Ed, M_Ed, N_Ed
            cot_theta_override: User-supplied cot(θ). If None, computed from
                V_Rd_max equation with circular b_w.
            warning_threshold: Utilization threshold for warnings
            force_cracked: If True, skip the cracking moment check and go
                straight to the reinforced shear check

        Returns:
            CheckResult with shear utilization and detailed breakdown
        """
        assert self._shear_check is not None
        assert self._cracking_check is not None
        assert self._concrete_uls is not None

        V_Ed = abs(load_case.V_Ed)
        M_Ed = load_case.M_Ed
        N_Ed = load_case.N_Ed

        # 1. Get d and z from the solver
        d = self._shear_check.find_effective_depth(M_Ed, N_Ed)
        z_ec2, z_mech = self._shear_check.find_lever_arm(M_Ed, N_Ed, d)
        z = z_mech if z_mech is not None else z_ec2

        # 2. sigma_cp
        A_c = self.section.get_area()  # Gross concrete area (mm²)
        sigma_cp = sigma_cp_from_N_and_area(N_Ed=N_Ed, area=A_c)
        sigma_cp_capped = cap_sigma_cp_upper(sigma_cp=sigma_cp, f_cd=self._f_cd_design)

        # 3. Check if section is cracked (unless forced)
        is_cracked = force_cracked
        M_cr: Optional[float] = None
        if not force_cracked:
            M_cr = self._cracking_check.find_cracking_moment(N_Ed=N_Ed)
            is_cracked = abs(M_Ed) > abs(M_cr)

        # 4. Uncracked section — use Eq.17
        if not is_cracked:
            # TODO should V_Rd_c_max only be found if the section is unreinforced? that is how ShearCheck works
            V_Rd_c = self.calculate_V_Rd_c_uncracked(sigma_cp_capped)
            # Eq.6.5 upper bound on unreinforced shear resistance
            b_w_uc, _, _ = self.calculate_equivalent_web_width(d, z)
            nu = find_nu_factor(self._concrete_uls.f_ck)
            V_Rd_c_max = to_kn(0.5 * b_w_uc * d * nu * self._f_cd_design, ForceUnit.N)
            V_Rd_c = min(V_Rd_c, V_Rd_c_max)
            return self._build_check_result(
                check_name="Circular shear (uncracked, Eq.17)",
                code_reference="Orr (2012) Eq.17, based on EC2 §6.2",
                demand=V_Ed,
                capacity=V_Rd_c,
                units="kN",
                warning_threshold=warning_threshold,
                details=self._shear_details(
                    V_Ed=V_Ed, M_Ed=M_Ed, N_Ed=N_Ed, V_Rd=V_Rd_c,
                    d=d, z=z, sigma_cp=sigma_cp_capped,
                    is_cracked=False, M_cr=M_cr, V_Rd_c=V_Rd_c,
                    V_Rd_c_max=V_Rd_c_max,
                    governing_mode="uncracked concrete",
                    section_name=self.section.section_name or "",
                ),
            )

        # 5. Cracked but no reinforcement — Eq.17 as conservative lower bound
        if self.shear_reinforcement is None:
            # TODO the code in this whole conditional is potentially being needlessly be calculated twice
            # - need to link with the check above with an OR statement, only details differ.
            V_Rd_c = self.calculate_V_Rd_c_uncracked(sigma_cp_capped)
            # Eq.6.5 upper bound on unreinforced shear resistance
            b_w_cr, _, _ = self.calculate_equivalent_web_width(d, z)
            nu_cr = find_nu_factor(self._concrete_uls.f_ck)
            V_Rd_c_max = to_kn(0.5 * b_w_cr * d * nu_cr * self._f_cd_design, ForceUnit.N)
            V_Rd_c = min(V_Rd_c, V_Rd_c_max)
            return self._build_check_result(
                check_name="Circular shear (cracked, no reinforcement)",
                code_reference="Orr (2012) Eq.17 (conservative)",
                demand=V_Ed,
                capacity=V_Rd_c,
                units="kN",
                warning_threshold=warning_threshold,
                message=(
                    "Section is cracked but has no shear reinforcement. "
                    "Empirical V_Rd_c for cracked sections is not validated "
                    "for circular sections (Orr 2012). Using uncracked Eq.17 "
                    "as conservative lower bound."
                ),
                details=self._shear_details(
                    V_Ed=V_Ed, M_Ed=M_Ed, N_Ed=N_Ed, V_Rd=V_Rd_c,
                    d=d, z=z, sigma_cp=sigma_cp_capped,
                    is_cracked=True, M_cr=M_cr, V_Rd_c=V_Rd_c,
                    V_Rd_c_max=V_Rd_c_max,
                    governing_mode="concrete (no shear reinforcement)",
                    section_name=self.section.section_name or "",
                ),
            )

        # 6. Cracked with shear reinforcement
        z_0 = d - self.diameter / 2  # distance from section centre to tension centroid
        lambda_1 = self.calculate_lambda_1(z_0, z)
        lambda_2 = self.calculate_lambda_2()

        # Circular equivalent web width
        b_w, b_wc, b_wt = self.calculate_equivalent_web_width(d, z)

        # Strut parameters
        f_cd = self._f_cd_design
        f_ck = self._concrete_uls.f_ck
        alpha_cw = find_alpha_cw(f_cd, sigma_cp_capped)

        used_note_2 = False
        if cot_theta_override is not None:
            # User override — compute V_Rd_max/V_Rd_s directly (no iteration)
            cot_theta = clamp_cot_theta(cot_theta_override)
            nu_1 = find_nu_1_factor(f_ck, link_angle_degrees=90.0)
            K = alpha_cw * b_w * z * nu_1 * f_cd
            tan_theta = 1 / cot_theta
            V_Rd_max = to_kn(K / (cot_theta + tan_theta), ForceUnit.N)
            f_ywd = self._f_ywd_design
            A_sw_over_s = self.shear_reinforcement.area_per_unit_length
            V_Rd_s = to_kn(
                lambda_1 * lambda_2 * A_sw_over_s * z * f_ywd * cot_theta,
                ForceUnit.N,
            )
        elif self.use_increased_nu_1:
            # Note 2 iteration: may increase ν₁ if σ_s < 0.8·f_yk
            V_Rd_max, V_Rd_s, cot_theta, nu_1, used_note_2 = (
                self._find_V_Rd_max_with_note_2_iteration(
                    V_Ed, z, sigma_cp_capped, b_w, lambda_1, lambda_2,
                )
            )
            K = alpha_cw * b_w * z * nu_1 * f_cd
            f_ywd = 0.8 * self.shear_reinforcement.f_yk if used_note_2 else self._f_ywd_design
        else:
            # Standard Note 1
            nu_1 = find_nu_1_factor(f_ck, link_angle_degrees=90.0)
            K = alpha_cw * b_w * z * nu_1 * f_cd
            cot_theta = find_cot_theta_for_V_Ed(
                V_Ed=V_Ed,  # already in kN; function converts internally
                K=K,
                link_angle_degrees=90.0,
            )
            tan_theta = 1 / cot_theta
            V_Rd_max = to_kn(K / (cot_theta + tan_theta), ForceUnit.N)
            f_ywd = self._f_ywd_design
            A_sw_over_s = self.shear_reinforcement.area_per_unit_length
            V_Rd_s = to_kn(
                lambda_1 * lambda_2 * A_sw_over_s * z * f_ywd * cot_theta,
                ForceUnit.N,
            )

        # Governing capacity
        V_Rd = min(V_Rd_s, V_Rd_max)
        governing = "V_Rd_s" if V_Rd_s <= V_Rd_max else "V_Rd_max"

        return self._build_check_result(
            check_name=f"Circular shear ({governing})",
            code_reference="Orr (2012) Eq.7/8/14, based on EC2 §6.2.3",
            demand=V_Ed,
            capacity=V_Rd,
            units="kN",
            warning_threshold=warning_threshold,
            details=self._shear_details(
                V_Ed=V_Ed, M_Ed=M_Ed, N_Ed=N_Ed, V_Rd=V_Rd,
                d=d, z=z, sigma_cp=sigma_cp_capped,
                is_cracked=True, M_cr=M_cr,
                V_Rd_s=V_Rd_s, V_Rd_max=V_Rd_max,
                governing_mode=governing,
                section_name=self.section.section_name or "",
                cot_theta=cot_theta,
                b_w=b_w, b_wc=b_wc, b_wt=b_wt,
                alpha_cw=alpha_cw, nu_1=nu_1, K=K,
                f_ywd=f_ywd, used_note_2=used_note_2,
                lambda_1=lambda_1, lambda_2=lambda_2, z_0=z_0,
            ),
        )


    # ===========================
    # Internal helpers
    # ===========================

    def _find_V_Rd_max_with_note_2_iteration(
        self,
        V_Ed: float,
        z: float,
        sigma_cp: float,
        b_w: float,
        lambda_1: float,
        lambda_2: float,
    ) -> tuple[float, float, float, float, bool]:
        """
        Calculate V_Rd_max and V_Rd_s with ν₁ Note 2 iteration per EC2 §6.2.3(3).

        Iterates to check if σ_s < 0.8·f_yk, allowing increased ν₁ factor.
        Detects oscillation and reverts to Note 1 if needed.

        The circular V_Rd_s includes λ₁/λ₂ efficiency factors, so the stress
        check accounts for the actual reinforcement contribution.

        Args:
            V_Ed: Design shear force in kN
            z: Lever arm in mm
            sigma_cp: Capped compressive stress in MPa
            b_w: Circular equivalent web width in mm
            lambda_1: Link efficiency factor (Orr Eq.6)
            lambda_2: Spiral efficiency factor (Orr Eq.9)

        Returns:
            Tuple of (V_Rd_max kN, V_Rd_s kN, cot_theta, nu_1, used_note_2 bool)
        """
        assert self.shear_reinforcement is not None
        # TODO need an assert not none for _concrete_uls here to remove f_ck pylance error
        f_ck = self._concrete_uls.f_ck
        f_cd = self._f_cd_design
        f_yk = self.shear_reinforcement.f_yk
        f_ywd = self._f_ywd_design
        threshold = 0.8 * f_yk
        alpha_cw = find_alpha_cw(f_cd, sigma_cp)
        A_sw_over_s = self.shear_reinforcement.area_per_unit_length

        # --- Iteration 1: Note 1 (standard ν₁) ---
        nu_1_n1 = find_nu_1_factor(f_ck, link_angle_degrees=90.0)
        K_n1 = alpha_cw * b_w * z * nu_1_n1 * f_cd

        cot_theta_n1 = find_cot_theta_for_V_Ed(
            V_Ed=V_Ed, K=K_n1, link_angle_degrees=90.0,
        )
        tan_theta_n1 = 1 / cot_theta_n1
        V_Rd_max_n1 = to_kn(K_n1 / (cot_theta_n1 + tan_theta_n1), ForceUnit.N)
        V_Rd_s_n1 = to_kn(
            lambda_1 * lambda_2 * A_sw_over_s * z * f_ywd * cot_theta_n1,
            ForceUnit.N,
        )

        # Stress in reinforcement: σ_s = f_ywd · (V_Ed / V_Rd_s)
        sigma_s_1 = f_ywd * (V_Ed / V_Rd_s_n1) if V_Rd_s_n1 > 0 else f_yk

        if sigma_s_1 >= threshold:
            # Stress too high — Note 2 not applicable
            return V_Rd_max_n1, V_Rd_s_n1, cot_theta_n1, nu_1_n1, False

        # --- Iteration 2: Note 2 (increased ν₁, reduced f_ywd) ---
        nu_1_n2 = find_nu_1_factor_note_2(f_ck, link_angle_degrees=90.0)
        K_n2 = alpha_cw * b_w * z * nu_1_n2 * f_cd
        f_ywd_n2 = 0.8 * f_yk  # Reduced per Note 2

        cot_theta_n2 = find_cot_theta_for_V_Ed(
            V_Ed=V_Ed, K=K_n2, link_angle_degrees=90.0,
        )
        tan_theta_n2 = 1 / cot_theta_n2
        V_Rd_max_n2 = to_kn(K_n2 / (cot_theta_n2 + tan_theta_n2), ForceUnit.N)
        V_Rd_s_n2 = to_kn(
            lambda_1 * lambda_2 * A_sw_over_s * z * f_ywd_n2 * cot_theta_n2,
            ForceUnit.N,
        )

        sigma_s_2 = f_ywd_n2 * (V_Ed / V_Rd_s_n2) if V_Rd_s_n2 > 0 else f_yk

        if sigma_s_2 >= threshold:
            # Oscillation — revert to Note 1
            warnings.warn(
                f"EC2 §6.2.3(3) Note 2: Oscillation detected. "
                f"Note 1: σ_s={sigma_s_1:.1f} MPa < {threshold:.1f} MPa, "
                f"Note 2: σ_s={sigma_s_2:.1f} MPa >= {threshold:.1f} MPa. "
                f"Reverting to Note 1 (conservative).",
                stacklevel=3,
            )
            return V_Rd_max_n1, V_Rd_s_n1, cot_theta_n1, nu_1_n1, False

        # Converged with Note 2
        return V_Rd_max_n2, V_Rd_s_n2, cot_theta_n2, nu_1_n2, True

    def _compute_cot_theta_for_tension_shift(
        self, M_Ed: float, N_Ed: float, V_Ed: float
    ) -> float:
        """
        Compute cot(θ) from circular equivalent web width for use in
        the BendingCheck tension shift rule.
        """
        assert self._shear_check is not None
        assert self._concrete_uls is not None

        d = self._shear_check.find_effective_depth(M_Ed, N_Ed)
        _, z_mech = self._shear_check.find_lever_arm(M_Ed, N_Ed, d)
        z = z_mech if z_mech is not None else 0.9 * d

        b_w, _, _ = self.calculate_equivalent_web_width(d, z)

        f_cd = self._f_cd_design
        f_ck = self._concrete_uls.f_ck
        A_c = self.section.get_area()
        sigma_cp = sigma_cp_from_N_and_area(N_Ed=N_Ed, area=A_c)
        sigma_cp_capped = cap_sigma_cp_upper(sigma_cp=sigma_cp, f_cd=f_cd)

        alpha_cw = find_alpha_cw(f_cd, sigma_cp_capped)
        nu_1 = find_nu_1_factor(f_ck, 90.0)
        K = alpha_cw * b_w * z * nu_1 * f_cd

        return find_cot_theta_for_V_Ed(
            V_Ed=abs(V_Ed),  # already in kN; function converts internally
            K=K,
            link_angle_degrees=90.0,
        )


    @staticmethod
    def _build_check_result(
        *,
        check_name: str,
        code_reference: str,
        demand: float,
        capacity: float,
        units: str,
        warning_threshold: float = 0.95,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> CheckResult:
        """Build a CheckResult with automatic status from utilization."""
        utilization = demand / capacity if capacity > 0 else float("inf")

        if utilization <= 1.0:
            if utilization >= warning_threshold:
                status = CheckStatus.WARNING
                if not message:
                    message = f"High utilization ({utilization:.1%})"
            else:
                status = CheckStatus.PASS
                if not message:
                    message = "Check satisfied"
        else:
            status = CheckStatus.FAIL
            if not message:
                message = f"Capacity exceeded by {(utilization - 1.0) * 100:.1f}%"

        return CheckResult(
            check_name=check_name,
            status=status,
            utilization=float(utilization),
            demand=demand,
            capacity=capacity,
            units=units,
            message=message,
            details=details or {},
            code_reference=code_reference,
        )

    @staticmethod
    def _shear_details(
        *,
        V_Ed: float,
        M_Ed: float,
        N_Ed: float,
        V_Rd: float,
        d: float,
        z: float,
        sigma_cp: float,
        is_cracked: bool,
        section_name: str = "",
        governing_mode: str = "",
        M_cr: Optional[float] = None,
        V_Rd_c: Optional[float] = None,
        V_Rd_c_max: Optional[float] = None,
        V_Rd_s: Optional[float] = None,
        V_Rd_max: Optional[float] = None,
        cot_theta: Optional[float] = None,
        b_w: Optional[float] = None,
        b_wc: Optional[float] = None,
        b_wt: Optional[float] = None,
        alpha_cw: Optional[float] = None,
        nu_1: Optional[float] = None,
        K: Optional[float] = None,
        f_ywd: Optional[float] = None,
        used_note_2: Optional[bool] = None,
        lambda_1: Optional[float] = None,
        lambda_2: Optional[float] = None,
        z_0: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Assemble details dict for shear check results.

        Key names match ShearCheck.perform_check details for consistency.
        Circular-specific keys (lambda_1, lambda_2, b_wc, b_wt, z_0, is_cracked,
        M_cr) are appended after the common fields.
        """
        # Common fields — same names and order as ShearCheck
        details: Dict[str, Any] = {
            "V_Ed": V_Ed,
            "M_Ed": M_Ed,
            "N_Ed": N_Ed,
            "V_Rd": V_Rd,
            "V_Rd_c": V_Rd_c,
            "V_Rd_c_max_unreinforced": V_Rd_c_max,
            "V_Rd_s": V_Rd_s,
            "V_Rd_max": V_Rd_max,
            "governing_mode": governing_mode,
            "cot_theta": cot_theta,
            "theta_deg": None if cot_theta is None else degrees(atan(1 / cot_theta)),
            "section_name": section_name,
            "d": d,
            "z": z,
            "b_w": b_w,
            "sigma_cp": sigma_cp,
            "alpha_cw": alpha_cw,
            "nu_1": nu_1,
            "K": K,
            "f_ywd": f_ywd,
            "used_note_2": used_note_2,
        }
        # Circular-specific fields
        details["is_cracked"] = is_cracked
        if M_cr is not None:
            details["M_cr"] = M_cr
        if lambda_1 is not None:
            details["lambda_1"] = lambda_1
        if lambda_2 is not None:
            details["lambda_2"] = lambda_2
        if b_wc is not None:
            details["b_wc"] = b_wc
        if b_wt is not None:
            details["b_wt"] = b_wt
        if z_0 is not None:
            details["z_0"] = z_0
        return details
