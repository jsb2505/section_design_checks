"""
Circular section design checks following Orr (2012) approach.

Wraps BendingCheck, ShearCheck, CrackingCheck, and StressLimitsCheck with
circular-specific
modifications for piles, columns, and other circular RC members.

Key modifications from standard EC2:
- Shear reinforcement efficiency factors λ1 (closed links) and λ2 (spirals)
- Equivalent web width for V_Rd_max (concrete strut crushing)
- Uncracked V_Rd_c using principal stress approach (Eq.17)
- Lever arm from distance between resultant opposing force centroids
- (but z capped to 0.9d)
- Optional k_f factor for cast-in-place piles (EC2 §2.4.2.5(2))

Reference:
    Orr, J.J. (2012). "Shear design of circular concrete sections."
    University of Bath.
"""

import warnings
from math import atan, degrees, pi, sqrt
from typing import Any, Dict, Literal, Optional, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, model_validator

from materials.reinforced_concrete.code_checks.base_check import (
    CheckResult,
    CheckStatus,
)
from materials.reinforced_concrete.code_checks.ec2_2004.bending_check import BendingCheck
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck, ShearLoadCase
from materials.reinforced_concrete.code_checks.ec2_2004.cracking_check import (
    CrackingCheck,
    LoadDuration,
)
from materials.reinforced_concrete.code_checks.ec2_2004.stress_limits_check import (
    StressLimitsCheck,
)
from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import (
    find_max_allowable_link_spacing,
    find_max_allowable_leg_spacing,
    find_cot_theta_for_V_Ed_from_V_Rd_max,
    find_cot_theta_for_V_Ed_from_V_Rd_s,
    find_alpha_cw,
    find_nu_1_factor,
    find_nu_1_factor_note_2,
    find_V_Rd_c_cracked,
    find_rho_l_from_strains,
    sigma_cp_from_N_and_area,
    cap_sigma_cp_upper,
    clamp_cot_theta,
)
from materials.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
from materials.reinforced_concrete.ndp import get_ndp, get_ndp_context
from materials.core.units import ForceUnit, to_kn


class CircularSectionCheck(BaseModel):
    """
    EC2-compliant design checks for circular sections (piles/columns).

    Wraps BendingCheck, ShearCheck, CrackingCheck, and StressLimitsCheck with
    circular-specific
    modifications following Orr (2012).

    - **Bending**: Forwarded to BendingCheck with iterate_z=True by default.
      Tension shift uses circular equivalent web width for cot(θ) computation.
    - **Shear**: Custom implementation with λ1/λ2 efficiency factors, circular
      web width, and uncracked V_Rd_c per Eq.17.
    - **Cracking**: Forwarded to CrackingCheck (no circular modifications).
    - **Stress limits**: Forwarded to StressLimitsCheck (no circular modifications).

    The sub-checks are accessible via the ``bending``, ``cracking``, and
    ``stress_limits`` properties for advanced operations (plotting, capacity
    queries, detailed results).

    Attributes:
        section: Circular RC section geometry with reinforcement
        concrete: Concrete material properties
        diameter: Section diameter (mm)
        cover:
            Cover to outer face of shear links (mm).
            Links are assumed to be on the outer layer.
            If no shear reinforcement, cover is not used.
        shear_reinforcement: Shear links/spirals (optional)
        is_spiral: If True, ShearRebar.link_spacing is treated as spiral pitch for λ2
        apply_k_f: If True, multiply γ_c by k_f for cast-in-place piles (EC2 §2.4.2.5)

    Example:
        >>> from materials.reinforced_concrete.geometry import create_circular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
        >>>
        >>> section = create_circular_section(diameter=600)
        >>> # ... add perimeter reinforcement ...
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>> links = ShearRebar(diameter=12, link_spacing=200, n_legs=2, grade="B500B")
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
        >>> stress_result = check.perform_stress_limits_check(M_Ed=80, N_Ed=300)
    """

    model_config = ConfigDict(frozen=True)

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
            "If True, treat ShearRebar.link_spacing as the spiral pitch for λ2 "
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
            "Policy toggle for EC2 §6.2.3(3) Note 2 iteration. If True, the "
            "check attempts Note 2 (increased ν₁) when eligible by stress "
            "criterion (σ_s < 0.8·f_yk), but may revert to Note 1 if "
            "oscillation/non-convergence is detected. When Note 2 is used, "
            "f_ywd is reduced to 0.8·f_ywk for V_Rd,s. Ignored when "
            "cot_theta_override is provided."
        ),
    )

    use_sigma_cp_for_alpha_cw: bool = Field(
        default=False,
        description=(
            "If True, include σ_cp in α_cw for V_Rd,max calculations. "
            "If False (default), α_cw is calculated with σ_cp = 0."
        ),
    )

    apply_tension_cot_theta_limit: bool = Field(
        default=True,
        description=(
            "Apply reduced cot(θ) upper limit when shear co-exists with externally "
            "applied tension (UK NA §6.2.3(2): cot θ ≤ 1.25). Default True "
            "(conservative). Set to False when tension arises from restraint, "
            "not external loading. Only has effect when the NDP provides "
            "cot_theta_upper_lim_tension (e.g. EU_UK)."
        ),
    )

    d_fallback: Literal["ratio_of_h", "centroid"] = Field(
        default="ratio_of_h",
        description=(
            "Policy for effective depth when strain state is ambiguous "
            "(net compression, net tension, pure axial). "
            "'ratio_of_h': d = d_ratio * h (default 0.9h). "
            "'centroid': min(d_top, d_bot) from rebar centroids, "
            "falls back to ratio_of_h if rebar missing on one face."
        ),
    )

    d_ratio: float = Field(
        default=0.9,
        description=(
            "Ratio of section depth h used when d_fallback='ratio_of_h' "
            "or as ultimate fallback for 'centroid' policy."
        ),
        gt=0.0,
        le=1.0,
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

    check_k1_stress: bool = Field(
        default=False,
        description="EC2 §7.2(2) characteristic concrete stress limit.",
    )

    check_k2_stress: bool = Field(
        default=True,
        description="EC2 §7.2(3) quasi-permanent concrete stress limit.",
    )

    check_k3_stress: bool = Field(
        default=False,
        description="EC2 §7.2(5) reinforcement stress limit.",
    )

    check_yielding: bool = Field(
        default=True,
        description="EC2 §7.2(4)P inelastic strain check.",
    )

    check_k4_stress: bool = Field(
        default=False,
        description="EC2 §7.2(5) imposed deformation stress limit.",
    )

    net_tension_face: Optional[Literal["top", "bottom"]] = Field(
        default=None,
        description=(
            "Face-checking policy for net tension cracking. "
            "None (default): check both faces, report the worst. "
            "'top' or 'bottom': only check the specified face."
        ),
    )

    # ===========================
    # Private sub-checks
    # ===========================

    @model_validator(mode="after")
    def _validate_geometry(self) -> "CircularSectionCheck":    
        r = self.diameter / 2
        if self.cover >= r:
            raise ValueError("cover must be < D/2")
        
        if self.r_sv_override is not None:
            if self.r_sv_override <= 0 or self.r_sv_override >= r:
                raise ValueError("r_sv_override must be > 0 and < D/2")

        if self.shear_reinforcement is not None:
            if self.shear_reinforcement.diameter <= 0:
                raise ValueError("ShearRebar.diameter must be > 0")
            if self.shear_reinforcement.link_spacing <= 0:
                raise ValueError("ShearRebar.link_spacing must be > 0")

            r_sv = self.diameter / 2 - self.cover - self.shear_reinforcement.diameter / 2
            if self.r_sv_override is None and r_sv <= 0:
                raise ValueError(
                    "Computed r_sv <= 0. Check cover and shear link diameter "
                    "(expected cover to outer face of links)."
                )
        return self
    

    _bending_check: Optional[BendingCheck] = PrivateAttr(default=None)
    _shear_check: Optional[ShearCheck] = PrivateAttr(default=None)
    _cracking_check: Optional[CrackingCheck] = PrivateAttr(default=None)
    _stress_limits_check: Optional[StressLimitsCheck] = PrivateAttr(default=None)
    _concrete_uls: Optional[ConcreteMaterial] = PrivateAttr(default=None)
    _ndp_snapshot: tuple = PrivateAttr(default=())

    @model_validator(mode="after")
    def _post_init(self) -> "CircularSectionCheck":
        # Warn if shear reinforcement angle is not 90° (ineffective for circular)
        if (
            self.shear_reinforcement is not None
            and abs(self.shear_reinforcement.angle - 90.0) > 1e-9
        ):
            warnings.warn(
                f"ShearRebar.angle={self.shear_reinforcement.angle}° is ignored for "
                f"circular sections — links must be 90° (vertical). "
                f"Spirals use spacing for pitch, angle makes no difference. "
                f"The λ1/λ2 efficiency factors account for circular geometry.",
                UserWarning,
                stacklevel=2,
            )

        # Apply k_f to concrete partial factors for ULS if requested
        concrete_uls = self.concrete
        if self.apply_k_f:
            k_f = cast(float, get_ndp("k_f"))
            concrete_uls = self.concrete.model_copy(
                update={
                    "gamma_c": self.concrete.gamma_c * k_f,
                    "gamma_c_accidental": self.concrete.gamma_c_accidental * k_f,
                }
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
            apply_tension_cot_theta_limit=self.apply_tension_cot_theta_limit,
            d_fallback=self.d_fallback,
            d_ratio=self.d_ratio,
        )

        self._shear_check = ShearCheck(
            section=self.section,
            concrete=concrete_uls,
            shear_reinforcement=self.shear_reinforcement,
            use_accidental=self.use_accidental,
            use_mechanical_lever_arm=True,
            z_d_ratio=0.77,           # Circular z_mech ≈ 0.77d (Orr 2012)
            z_d_ratio_upper=0.95,
            z_d_ratio_lower=0.65,
            use_sigma_cp_for_alpha_cw=self.use_sigma_cp_for_alpha_cw,
            concrete_model_type=self.concrete_model_type,
            steel_model_type=self.steel_model_type,
            d_fallback=self.d_fallback,
            d_ratio=self.d_ratio,
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
            check_k1_stress=self.check_k1_stress,
            check_k2_stress=self.check_k2_stress,
            check_k3_stress=self.check_k3_stress,
            check_yielding=self.check_yielding,
            check_k4_stress=self.check_k4_stress,
            net_tension_face=self.net_tension_face,
        )

        self._stress_limits_check = StressLimitsCheck(
            section=self.section,
            concrete=self.concrete,  # SLS uses characteristic properties (no k_f)
            creep_coefficient=self.creep_coefficient,
            concrete_model_type=ConcreteModelType.LINEAR_ELASTIC,
            steel_model_type=self.steel_model_type,
            n_fibres_width=self.n_fibres_width,
            n_fibres_height=self.n_fibres_height,
            check_k1_stress=self.check_k1_stress,
            check_k2_stress=self.check_k2_stress,
            check_k3_stress=self.check_k3_stress,
            check_yielding=self.check_yielding,
            check_k4_stress=self.check_k4_stress,
        )

        self._ndp_snapshot = get_ndp_context()

        return self

    def _check_ndp_context(self) -> None:
        """Warn if the active NDP context differs from the one at construction."""
        current = get_ndp_context()
        if current != self._ndp_snapshot:
            warnings.warn(
                f"NDP context has changed since this CircularSectionCheck was "
                f"constructed (was {self._ndp_snapshot}, now {current}). "
                f"Sub-check parameters (gamma_c, f_cd, k_f, etc.) reflect the "
                f"original context. Reconstruct the check for the new context.",
                UserWarning,
                stacklevel=3,
            )

    def with_updates(self, **changes: Any) -> "CircularSectionCheck":
        """
        Return a new CircularSectionCheck with the given fields replaced.

        Unlike ``model_copy``, this calls the constructor so that ``_post_init``
        runs and all sub-check delegates are fully rebuilt with the new values.
        There is no risk of stale sub-check state after changing section,
        concrete, etc.

        Example::

            new_check = check.with_updates(concrete=c40, diameter=700)
        """
        current = {name: getattr(self, name) for name in type(self).model_fields}
        current.update(changes)
        return type(self)(**current)

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
    def stress_limits(self) -> StressLimitsCheck:
        """Direct access to the internal StressLimitsCheck."""
        assert self._stress_limits_check is not None
        return self._stress_limits_check

    @property
    def _f_cd_design(self) -> float:
        """Design concrete compressive strength for shear (accounts for k_f)."""
        assert self._concrete_uls is not None
        if self.use_accidental:
            return self._concrete_uls.f_cd_shear_accidental
        return self._concrete_uls.f_cd_shear

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

    def calculate_lambda_1(self, z_0: float, z: float, integration_points: int = 100) -> float:
        """
        Link efficiency factor λ1 for circular sections (Orr 2012, Eq.6).

        Computed by numerical integration over the lever arm depth. Represents
        the fraction of link force that effectively resists vertical shear.

        Args:
            z_0: Distance from section centre to tension centroid (mm),
                 typically d - D/2.
            z: Lever arm (mm).
            integration_points: The number of steps to integrate over

        Returns:
            λ1 efficiency factor (0 to 1). Typically ≈ 0.85 for common geometries.
        """
        if self.use_simplified_lambda_1:
            return 0.85

        r_sv = self.r_sv
        if r_sv <= 0:
            return 0.85  # Fallback

        X = np.linspace(0, 1, integration_points)
        y = z_0 - z * X  # distance from section centre at each integration point

        # Clip to avoid sqrt of negative (point outside link circle)
        arg = 1.0 - (y / r_sv) ** 2
        arg = np.clip(arg, 0.0, None)

        integrand = np.sqrt(arg)
        # Avoid NumPy reduction sentinels by integrating via Python floats.
        lambda_1 = float(
            sum(
                0.5
                * (float(integrand[i]) + float(integrand[i + 1]))
                * float(X[i + 1] - X[i])
                for i in range(len(X) - 1)
            )
        )

        # Sanity: clamp to [0, 1]
        return max(0.0, min(1.0, lambda_1))


    def calculate_lambda_2(self) -> float:
        """
        Spiral link efficiency factor λ2 (Orr 2012, Eq.8).

        For closed links (is_spiral=False), returns 1.0.
        For spiral links, accounts for the helix angle reduction:
            λ2 = 1 / √((p / (2π·r_sv))² + 1)
        where p = spiral pitch (= ShearRebar.link_spacing).

        Returns:
            λ2 efficiency factor (0 to 1).
        """
        if not self.is_spiral or self.shear_reinforcement is None:
            return 1.0

        p = self.shear_reinforcement.link_spacing
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

        For members without shear reinforcement, the tension chord is taken
        at the concrete perimeter (r_sv = D/2) so b_w does not depend on cover.

        Args:
            d: Effective depth from compression face to tension centroid (mm)
            z: Lever arm (mm)

        Returns:
            (b_w, b_wc, b_wt) all in mm
        """
        r = self.diameter / 2  # radius to extreme fibre
        if self.shear_reinforcement is None:
            # No links/spiral: use concrete perimeter for the tension chord radius.
            r_sv = r
        else:
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

        # Explicit degeneracy handling avoids relying on min/max side-effects.
        if b_wc <= 0.0 and b_wt <= 0.0:
            b_w = 0.0
        elif b_wc <= 0.0:
            b_w = b_wt
        elif b_wt <= 0.0:
            b_w = b_wc
        else:
            b_w = min(b_wc, b_wt)
        return b_w, b_wc, b_wt

    def _find_rho_l(
        self,
        *,
        M_Ed: float,
        N_Ed: float,
        b_w: float,
        d: float,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
        ignore_compression_steel: bool = False,
    ) -> float:
        """Longitudinal reinforcement ratio for EC2 §6.2.2.

        Uses bars in tension (strain < 0) from the section strain state.
        Capped at 0.02 per EC2 §6.2.2(1).

        Args:
            M_Ed: Design bending moment (kN·m)
            N_Ed: Design axial force (kN)
            b_w: Equivalent web width (mm)
            d: Effective depth (mm)
            eps_top: Pre-computed top strain (optional)
            eps_bottom: Pre-computed bottom strain (optional)
            ignore_compression_steel: If True, use diagram without compression steel.

        Returns:
            rho_l, capped at 0.02
        """
        if b_w <= 0 or d <= 0:
            return 0.0

        if eps_top is None or eps_bottom is None:
            assert self._shear_check is not None
            eps_top, eps_bottom = self._shear_check._get_diagram(
                ignore_compression_steel
            ).find_strains_for_MN(M_Ed, N_Ed)

        return find_rho_l_from_strains(
            section=self.section,
            b_w=b_w,
            d=d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            rho_l_max=0.02,
        )

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
        use_v_rd_s_for_cot_theta: bool = False,
        warning_threshold: float = 0.95,
        suppress_warnings: bool = False,
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
            use_v_rd_s_for_cot_theta: If True, determine cot(θ) from rearranged
                EC2 Eq. 6.13 (V_Rd,s = V_Ed). If False (default), determine cot(θ)
                from rearranged EC2 Eq. 6.14 / V_Rd,max.
            warning_threshold: Utilization threshold for warnings
            suppress_warnings: If True, suppress warnings emitted during this check.
            ignore_compression_steel: If True, ignore compression reinforcement
            iterate_z: If True, iteratively recalculate z (default True for circular)

        Returns:
            CheckResult with bending utilization
        """
        self._check_ndp_context()
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
                M_Ed=M_Ed,
                N_Ed=N_Ed,
                V_Ed=V_Ed,
                use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
                ignore_compression_steel=ignore_compression_steel,
            )

        assert self._bending_check is not None
        return self._bending_check.perform_check(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            V_Ed=V_Ed,
            M_cap=M_cap,
            shear_reinforcement=shear_reinf,
            cot_theta_override=effective_cot_theta,
            use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
            warning_threshold=warning_threshold,
            suppress_warnings=suppress_warnings,
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
        self._check_ndp_context()
        assert self._cracking_check is not None
        return self._cracking_check.perform_check(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            warning_threshold=warning_threshold,
            ignore_compression_steel=ignore_compression_steel,
            force_cracked=force_cracked,
        )


    def perform_stress_limits_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        warning_threshold: float = 0.95,
        ignore_compression_steel: bool = False,
        suppress_warnings: bool = False,
        **kwargs,
    ) -> CheckResult:
        """
        Stress limitation check for circular section (wrapper - no circular modifications).

        Forwards to internal StressLimitsCheck.

        Args:
            M_Ed: Design moment at SLS (kN.m)
            N_Ed: Design axial force at SLS (kN, compression positive)
            warning_threshold: Utilization threshold for warnings
            ignore_compression_steel: If True, ignore compression reinforcement
            suppress_warnings: If True, suppress warnings emitted during this check.

        Returns:
            CheckResult with governing stress-limit utilization
        """
        self._check_ndp_context()
        assert self._stress_limits_check is not None
        return self._stress_limits_check.perform_check(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            warning_threshold=warning_threshold,
            ignore_compression_steel=ignore_compression_steel,
            suppress_warnings=suppress_warnings,
            **kwargs,
        )


    def _get_cot_theta_limits(self, sigma_cp: float) -> tuple[float, float]:
        """Return (cot_min, cot_max) with tension override applied if applicable."""
        cot_min = float(cast(float, get_ndp("cot_theta_lower_lim")))
        cot_max = float(cast(float, get_ndp("cot_theta_upper_lim")))

        if self.apply_tension_cot_theta_limit and sigma_cp < 0:
            tension_lim = get_ndp("cot_theta_upper_lim_tension")
            if tension_lim is not None and not callable(tension_lim):
                cot_max = min(cot_max, float(tension_lim))

        cot_max = max(cot_min, cot_max)
        return cot_min, cot_max

    def perform_shear_check(
        self,
        *,
        load_case: ShearLoadCase,
        cot_theta_override: Optional[float] = None,
        use_v_rd_s_for_cot_theta: bool = False,
        use_uncracked_V_Rd_c: bool = False,
        warning_threshold: float = 0.95,
        suppress_warnings: bool = False,
        ignore_compression_steel: bool = False,
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
                V_Rd_max equation with circular b_w. When provided, this
                bypasses the Note 2 iteration logic and uses Note 1 values.
            use_v_rd_s_for_cot_theta: If True, solve cot(θ) from rearranged EC2
                Eq. 6.13 (V_Rd,s = V_Ed). If False (default), solve from
                rearranged EC2 Eq. 6.14 / V_Rd,max.
            use_uncracked_V_Rd_c:
                If True, use uncracked Eq.17 V_Rd,c when passing V_Rd,c into
                NDP note-based spacing rules and reporting selected V_Rd,c.
                If False (default), use cracked V_Rd,c for this purpose.
                In both cases, reinforced shear capacity still governs from
                min(V_Rd,s, V_Rd,max).
            warning_threshold: Utilization threshold for warnings
            suppress_warnings: If True, suppress warnings emitted during this check.
            ignore_compression_steel: If True, ignore compression reinforcement.

        Returns:
            CheckResult with shear utilization and detailed breakdown
        """
        self._check_ndp_context()
        assert self._shear_check is not None
        assert self._concrete_uls is not None

        V_Ed = abs(load_case.V_Ed)
        M_Ed = load_case.M_Ed
        N_Ed = load_case.N_Ed

        if self.shear_reinforcement is None:
            raise ValueError(
                "Circular shear check requires shear_reinforcement. "
                "Unreinforced circular shear design is not supported in this flow."
            )

        # 1. Solve strains once for this load case; reuse across d, z and rho_l.
        eps_top: Optional[float]
        eps_bottom: Optional[float]
        if abs(M_Ed) > 1e-6:
            eps_top, eps_bottom = self._shear_check._get_diagram(
                ignore_compression_steel
            ).find_strains_for_MN(M_Ed, N_Ed)
        else:
            eps_top, eps_bottom = None, None

        d = self._shear_check.find_effective_depth(
            M_Ed,
            N_Ed,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
        )
        z_ec2, z_mech = self._shear_check.find_lever_arm(
            M_Ed,
            N_Ed,
            d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
        )
        z = z_mech if z_mech is not None else z_ec2

        # 2. sigma_cp
        A_c = self.section.get_area()  # Gross concrete area (mm²)
        sigma_cp = sigma_cp_from_N_and_area(N_Ed=N_Ed, area=A_c)
        sigma_cp_capped = cap_sigma_cp_upper(sigma_cp=sigma_cp, f_cd=self._f_cd_design)
        # 3. Reinforced shear checks.
        # Concrete V_Rd,c values are still calculated for reporting and spacing
        # note rules, but design capacity is always from V_Rd,s / V_Rd,max.
        z_0 = d - self.diameter / 2  # distance from section centre to tension centroid
        lambda_1 = self.calculate_lambda_1(z_0, z)
        lambda_2 = self.calculate_lambda_2()

        # Circular equivalent web width
        b_w, b_wc, b_wt = self.calculate_equivalent_web_width(d, z)

        rho_l = self._find_rho_l(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            b_w=b_w,
            d=d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
        )
        V_Rd_c_cracked = find_V_Rd_c_cracked(
            b_w=b_w,
            d=d,
            rho_l=rho_l,
            sigma_cp=sigma_cp_capped,
            f_ck=self._concrete_uls.f_ck,
            gamma_c=self._concrete_uls.gamma_c,
        )
        V_Rd_c_uncracked = self.calculate_V_Rd_c_uncracked(sigma_cp_capped)
        V_Rd_c = V_Rd_c_uncracked if use_uncracked_V_Rd_c else V_Rd_c_cracked

        # Strut parameters
        f_cd = self._f_cd_design
        f_ck = self._concrete_uls.f_ck
        alpha_cw = find_alpha_cw(
            f_cd,
            sigma_cp_capped,
            use_sigma_cp_for_alpha_cw=self.use_sigma_cp_for_alpha_cw,
        )
        link_spacing_max_allowable: Optional[float] = None
        link_spacing_satisfied: Optional[bool] = None
        leg_spacing_max_allowable: Optional[float] = None
        leg_spacing_satisfied: Optional[bool] = None

        cot_min, cot_max = self._get_cot_theta_limits(sigma_cp_capped)

        used_note_2 = False
        if cot_theta_override is not None:
            # User override — compute V_Rd_max/V_Rd_s directly (no iteration)
            cot_theta = clamp_cot_theta(cot_theta_override, cot_min=cot_min, cot_max=cot_max)
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
                    use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
                    suppress_warnings=suppress_warnings,
                    cot_min=cot_min,
                    cot_max=cot_max,
                )
            )
            K = alpha_cw * b_w * z * nu_1 * f_cd
            f_ywd = 0.8 * self.shear_reinforcement.f_yk if used_note_2 else self._f_ywd_design
        else:
            # Standard Note 1
            nu_1 = find_nu_1_factor(f_ck, link_angle_degrees=90.0)
            K = alpha_cw * b_w * z * nu_1 * f_cd
            if use_v_rd_s_for_cot_theta:
                A_sw_over_s_eff = lambda_1 * lambda_2 * self.shear_reinforcement.area_per_unit_length
                cot_theta = find_cot_theta_for_V_Ed_from_V_Rd_s(
                    V_Ed=V_Ed,
                    A_sw_over_s=A_sw_over_s_eff,
                    z=z,
                    f_ywd=self._f_ywd_design,
                    link_angle_degrees=90.0,
                    cot_min=cot_min,
                    cot_max=cot_max,
                )
            else:
                cot_theta = find_cot_theta_for_V_Ed_from_V_Rd_max(
                    V_Ed=V_Ed,
                    K=K,
                    link_angle_degrees=90.0,
                    cot_min=cot_min,
                    cot_max=cot_max,
                )
            tan_theta = 1 / cot_theta
            V_Rd_max = to_kn(K / (cot_theta + tan_theta), ForceUnit.N)
            f_ywd = self._f_ywd_design
            A_sw_over_s = self.shear_reinforcement.area_per_unit_length
            V_Rd_s = to_kn(
                lambda_1 * lambda_2 * A_sw_over_s * z * f_ywd * cot_theta,
                ForceUnit.N,
            )

        link_spacing_max_allowable = find_max_allowable_link_spacing(
            effective_depth=d,
            section_depth=self.diameter,
            f_ck=f_ck,
            V_Ed=V_Ed,
            V_Rd_max=V_Rd_max,
            V_Rd_c=V_Rd_c,
            link_angle_degrees=self.shear_reinforcement.angle,
        )
        link_spacing_satisfied = self.shear_reinforcement.link_spacing <= link_spacing_max_allowable + 1e-9
        if not link_spacing_satisfied and not suppress_warnings:
            warnings.warn(
                "Provided shear link spacing exceeds the maximum allowable spacing: "
                f"s={self.shear_reinforcement.link_spacing:.1f} mm > s_max={link_spacing_max_allowable:.1f} mm.",
                stacklevel=2,
            )

        if self.shear_reinforcement.leg_spacing is not None:
            leg_spacing_max_allowable = find_max_allowable_leg_spacing(
                effective_depth=d,
                section_depth=self.diameter,
                f_ck=f_ck,
                V_Ed=V_Ed,
                V_Rd_max=V_Rd_max,
                V_Rd_c=V_Rd_c,
                link_angle_degrees=self.shear_reinforcement.angle,
            )
            leg_spacing_satisfied = self.shear_reinforcement.leg_spacing <= leg_spacing_max_allowable + 1e-9
            if not leg_spacing_satisfied and not suppress_warnings:
                warnings.warn(
                    "Provided shear leg spacing exceeds the maximum allowable spacing: "
                    f"s_t={self.shear_reinforcement.leg_spacing:.1f} mm > s_t,max={leg_spacing_max_allowable:.1f} mm.",
                    stacklevel=2,
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
                V_Rd_c=V_Rd_c,
                V_Rd_c_cracked=V_Rd_c_cracked, V_Rd_c_uncracked=V_Rd_c_uncracked,
                use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
                V_Rd_s=V_Rd_s, V_Rd_max=V_Rd_max,
                governing_mode=governing,
                section_name=self.section.section_name or "",
                cot_theta=cot_theta,
                b_w=b_w, b_wc=b_wc, b_wt=b_wt,
                alpha_cw=alpha_cw, nu_1=nu_1, K=K,
                f_ywd=f_ywd, used_note_2=used_note_2,
                cot_theta_from_v_rd_s=use_v_rd_s_for_cot_theta,
                lambda_1=lambda_1, lambda_2=lambda_2, z_0=z_0,
                link_spacing_satisfied=link_spacing_satisfied,
                link_spacing_provided=self.shear_reinforcement.link_spacing,
                link_spacing_max_allowable=link_spacing_max_allowable,
                leg_spacing_satisfied=leg_spacing_satisfied,
                leg_spacing_provided=self.shear_reinforcement.leg_spacing,
                leg_spacing_max_allowable=leg_spacing_max_allowable,
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
        use_v_rd_s_for_cot_theta: bool = False,
        suppress_warnings: bool = False,
        cot_min: float = 1.0,
        cot_max: float = 2.5,
    ) -> tuple[float, float, float, float, bool]:
        """
        Calculate V_Rd_max and V_Rd_s with ν₁ Note 2 iteration per EC2 §6.2.3(3).

        This is an "attempt Note 2" policy, not a forced Note 2 policy.
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
            use_v_rd_s_for_cot_theta: If True, determine cot(θ) from
                rearranged Eq. 6.13 (V_Rd,s = V_Ed). If False, use rearranged
                Eq. 6.14 / V_Rd,max.

        Returns:
            Tuple of (V_Rd_max kN, V_Rd_s kN, cot_theta, nu_1, used_note_2 bool)
        """
        assert self.shear_reinforcement is not None
        assert self._concrete_uls is not None
        f_ck = self._concrete_uls.f_ck
        f_cd = self._f_cd_design
        f_yk = self.shear_reinforcement.f_yk
        f_ywd = self._f_ywd_design
        threshold = 0.8 * f_yk
        alpha_cw = find_alpha_cw(
            f_cd,
            sigma_cp,
            use_sigma_cp_for_alpha_cw=self.use_sigma_cp_for_alpha_cw,
        )
        A_sw_over_s = self.shear_reinforcement.area_per_unit_length

        # --- Iteration 1: Note 1 (standard ν₁) ---
        nu_1_n1 = find_nu_1_factor(f_ck, link_angle_degrees=90.0)
        K_n1 = alpha_cw * b_w * z * nu_1_n1 * f_cd

        if use_v_rd_s_for_cot_theta:
            A_sw_over_s_eff = lambda_1 * lambda_2 * A_sw_over_s
            cot_theta_n1 = find_cot_theta_for_V_Ed_from_V_Rd_s(
                V_Ed=V_Ed,
                A_sw_over_s=A_sw_over_s_eff,
                z=z,
                f_ywd=f_ywd,
                link_angle_degrees=90.0,
                cot_min=cot_min,
                cot_max=cot_max,
            )
        else:
            cot_theta_n1 = find_cot_theta_for_V_Ed_from_V_Rd_max(
                V_Ed=V_Ed, K=K_n1, link_angle_degrees=90.0,
                cot_min=cot_min, cot_max=cot_max,
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
        f_ywd_n2 = 0.8 * f_yk  # Reduced per Note under expression (6.8)

        if use_v_rd_s_for_cot_theta:
            A_sw_over_s_eff = lambda_1 * lambda_2 * A_sw_over_s
            cot_theta_n2 = find_cot_theta_for_V_Ed_from_V_Rd_s(
                V_Ed=V_Ed,
                A_sw_over_s=A_sw_over_s_eff,
                z=z,
                f_ywd=f_ywd_n2,
                link_angle_degrees=90.0,
                cot_min=cot_min,
                cot_max=cot_max,
            )
        else:
            cot_theta_n2 = find_cot_theta_for_V_Ed_from_V_Rd_max(
                V_Ed=V_Ed, K=K_n2, link_angle_degrees=90.0,
                cot_min=cot_min, cot_max=cot_max,
            )
        tan_theta_n2 = 1 / cot_theta_n2
        V_Rd_max_n2 = to_kn(K_n2 / (cot_theta_n2 + tan_theta_n2), ForceUnit.N)
        V_Rd_s_n2 = to_kn(
            lambda_1 * lambda_2 * A_sw_over_s * z * f_ywd_n2 * cot_theta_n2,
            ForceUnit.N,
        )

        sigma_s_2 = f_ywd_n2 * (V_Ed / V_Rd_s_n2) if V_Rd_s_n2 > 0 else f_yk

        if sigma_s_2 >= threshold:
            if suppress_warnings:
                return V_Rd_max_n1, V_Rd_s_n1, cot_theta_n1, nu_1_n1, False
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
        self,
        M_Ed: float,
        N_Ed: float,
        V_Ed: float,
        use_v_rd_s_for_cot_theta: bool = False,
        ignore_compression_steel: bool = False,
    ) -> float:
        """
        Compute cot(θ) from circular equivalent web width for use in
        the BendingCheck tension shift rule.

        If use_v_rd_s_for_cot_theta=True, cot(θ) is determined from
        rearranged Eq. 6.13 (V_Rd,s = V_Ed). Otherwise Eq. 6.14 / V_Rd,max
        is used.
        """
        assert self._shear_check is not None
        assert self._concrete_uls is not None

        # Solve strains once; both find_effective_depth and find_lever_arm reuse them.
        _eps_top: Optional[float]
        _eps_bottom: Optional[float]
        if abs(M_Ed) > 1e-6:
            _eps_top, _eps_bottom = self._shear_check._get_diagram(
                ignore_compression_steel
            ).find_strains_for_MN(M_Ed, N_Ed)
        else:
            _eps_top, _eps_bottom = None, None

        d = self._shear_check.find_effective_depth(
            M_Ed,
            N_Ed,
            eps_top=_eps_top,
            eps_bottom=_eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
        )
        _, z_mech = self._shear_check.find_lever_arm(
            M_Ed,
            N_Ed,
            d,
            eps_top=_eps_top,
            eps_bottom=_eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
        )
        z = z_mech if z_mech is not None else 0.9 * d

        b_w, _, _ = self.calculate_equivalent_web_width(d, z)

        f_cd = self._f_cd_design
        f_ck = self._concrete_uls.f_ck
        A_c = self.section.get_area()
        sigma_cp = sigma_cp_from_N_and_area(N_Ed=N_Ed, area=A_c)
        sigma_cp_capped = cap_sigma_cp_upper(sigma_cp=sigma_cp, f_cd=f_cd)

        alpha_cw = find_alpha_cw(
            f_cd,
            sigma_cp_capped,
            use_sigma_cp_for_alpha_cw=self.use_sigma_cp_for_alpha_cw,
        )
        nu_1 = find_nu_1_factor(f_ck, 90.0)
        K = alpha_cw * b_w * z * nu_1 * f_cd

        cot_min, cot_max = self._get_cot_theta_limits(sigma_cp_capped)

        if use_v_rd_s_for_cot_theta and self.shear_reinforcement is not None:
            return find_cot_theta_for_V_Ed_from_V_Rd_s(
                V_Ed=abs(V_Ed),
                A_sw_over_s=self.shear_reinforcement.area_per_unit_length,
                z=z,
                f_ywd=self._f_ywd_design,
                link_angle_degrees=90.0,
                cot_min=cot_min,
                cot_max=cot_max,
            )

        return find_cot_theta_for_V_Ed_from_V_Rd_max(
            V_Ed=abs(V_Ed),
            K=K,
            link_angle_degrees=90.0,
            cot_min=cot_min,
            cot_max=cot_max,
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
        section_name: str = "",
        governing_mode: str = "",
        V_Rd_c: Optional[float] = None,
        V_Rd_c_cracked: Optional[float] = None,
        V_Rd_c_uncracked: Optional[float] = None,
        use_uncracked_V_Rd_c: Optional[bool] = None,
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
        cot_theta_from_v_rd_s: Optional[bool] = None,
        lambda_1: Optional[float] = None,
        lambda_2: Optional[float] = None,
        z_0: Optional[float] = None,
        link_spacing_satisfied: Optional[bool] = None,
        link_spacing_provided: Optional[float] = None,
        link_spacing_max_allowable: Optional[float] = None,
        leg_spacing_satisfied: Optional[bool] = None,
        leg_spacing_provided: Optional[float] = None,
        leg_spacing_max_allowable: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Assemble details dict for shear check results.

        Key names match ShearCheck.perform_check details for consistency.
        Circular-specific keys (lambda_1, lambda_2, b_wc, b_wt, z_0) are
        appended after the common fields.
        """
        # Common fields — same names and order as ShearCheck
        details: Dict[str, Any] = {
            "V_Ed": V_Ed,
            "M_Ed": M_Ed,
            "N_Ed": N_Ed,
            "V_Rd": V_Rd,
            "V_Rd_c": V_Rd_c,
            "V_Rd_c_cracked": V_Rd_c_cracked,
            "V_Rd_c_uncracked": V_Rd_c_uncracked,
            "use_uncracked_V_Rd_c": use_uncracked_V_Rd_c,
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
            "cot_theta_from_v_rd_s": cot_theta_from_v_rd_s,
            "link_spacing_satisfied": link_spacing_satisfied,
            "link_spacing_provided": link_spacing_provided,
            "link_spacing_max_allowable": link_spacing_max_allowable,
            "leg_spacing_satisfied": leg_spacing_satisfied,
            "leg_spacing_provided": leg_spacing_provided,
            "leg_spacing_max_allowable": leg_spacing_max_allowable,
        }
        # Circular-specific fields
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

