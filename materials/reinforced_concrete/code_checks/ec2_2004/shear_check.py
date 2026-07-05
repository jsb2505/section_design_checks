"""
Shear check using codified EC2 stress approach.

This is a CODIFIED check with business logic - uses EC2 formulas directly
rather than first principles. Implements §6.2 Variable Strut Inclination Method.

N_Ed, M_Ed, and V_Ed are now parameters to perform_check(),not fields.
This enables checking multiple load cases against the same section efficiently.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union, Dict, TYPE_CHECKING
from math import atan, degrees, radians, sin, sqrt
import warnings
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from materials.core.units import ForceUnit, to_kn, from_kn
from materials.reinforced_concrete.ndp import get_ndp

from materials.utils.helpers import cot
from materials.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from materials.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import (
    calculate_section_breadth,
    find_rho_l_from_strains,
    find_max_allowable_link_spacing,
    find_max_allowable_leg_spacing,
    find_cot_theta_for_V_Ed_from_V_Rd_max,
    find_cot_theta_for_V_Ed_from_V_Rd_s,
    find_alpha_cw,
    find_V_Rd_c_cracked,
    find_nu_1_factor,
    find_nu_1_factor_note_2,
    sigma_cp_from_N_and_area,
    cap_sigma_cp_upper,
    clamp_cot_theta,
    find_V_Rd_c_max_unreinforced,
    find_minimum_ratio_of_shear_reinforcement
)
from materials.reinforced_concrete.analysis.interaction_diagram import (
    MNInteractionDiagram,
)
from materials.reinforced_concrete.code_checks.ec2_2004.flexure_utils import (
    EffectiveDepthFallback,
    find_effective_depth_for_flexure,
)

if TYPE_CHECKING:
    from pathlib import Path

class ShearLoadCase(BaseModel):
    """
    Single shear load case for checking.

    Attributes:
        V_Ed: Design shear force in kN
        M_Ed: Design moment in kN·m (defaults to 0.0)
              - In rigorous mode: used for accurate NA and lever arm via M-N solver
              - In approximate mode: if non-zero, used to determine compression face (still uses z=0.9d)
        N_Ed: Design axial force in kN (compression positive, default 0.0)
    """
    V_Ed: float = Field(..., description="Design shear force in kN")
    M_Ed: float = Field(default=0.0, description="Design moment in kN·m")
    N_Ed: float = Field(default=0.0, description="Design axial force in kN (compression positive)")


class ShearCheck(BaseCodeCheck):
    """
    EC2 shear check using Variable Strut Inclination Method (§6.2).

    Supports two modes:
    - **Rigorous mode** (use_mechanical_lever_arm=True, default): Uses M-N interaction solver for
      accurate neutral axis, compression face detection, and lever arm computation from
      force resultant centroids. Most accurate. Initialization: ~100ms.
    - **Approximate mode** (use_mechanical_lever_arm=False): Uses M-N solver only for compression
      face detection (when M_Ed or N_Ed provided), but always uses z=z_d_ratio*d for
      lever arm (default 0.9d, configurable). Faster but less accurate for eccentric loading.

    N_Ed, M_Ed, V_Ed are  parameters to perform_check(), not fields.
    This allows efficiently checking many load cases against the same section.

    This is a CODIFIED approach with business logic:
    - Uses EC2 empirical formulas (§6.2.2, §6.2.3)
    - Concrete shear resistance V_Rd,c (Eq. 6.2)
    - Shear reinforcement resistance V_Rd,s (Eq. 6.8)
    - Compression strut resistance V_Rd,max (Eq. 6.9, 6.14)
    - Variable strut angle 21.8° ≤ θ ≤ 45° (cot θ = 1.0 to 2.5)

    Attributes:
        section:RC section geometry
        concrete: Concrete material
        shear_reinforcement: Shear links/stirrups (optional)
        use_accidental:
            Use accidental limit state partial factors (default: False)
        use_mechanical_lever_arm:
            Use solver-based approach for NA and lever arm (default: True)
        allow_negative_sigma_cp:
            Allow negative σ_cp from tensile axial forces (default: True)
            If True, negative σ_cp reduces shear capacity
            If False, σ_cp is limited to a minimum of 0.0 MPa
        use_transformed_area_for_sigma_cp: 
            Use transformed area (concrete + n·steel) for σ_cp calculation (default: True)
        z_d_ratio:
            Lever arm ratio z/d for approximate mode (default: 0.9, circular ~0.77)
        z_d_ratio_upper:
            Upper bound for z/d in rigorous mode (default: 0.95)
        z_d_ratio_lower:
            Lower bound for z/d in rigorous mode (default: 0.65)
        breadth_policy:
            Section breadth policy for automatic b_w calculation when
            ``breadth_override`` is not set:
            ``"minimum"`` (EC2-style minimum width) or ``"average"``.
        breadth_average_height_ratio:
            Relative depth window used for ``breadth_policy="average"``,
            centred at mid-depth along ``breadth_shear_direction``.
        breadth_shear_direction:
            Shear direction vector ``(vx, vy)`` used for directional breadth
            slicing (default vertical shear: ``(0, 1)``).
        concrete_model_type: Concrete stress-strain model (for rigorous mode)
        steel_model_type: Steel stress-strain branch (for rigorous mode)

    Example:
        >>> from materials.reinforced_concrete.geometry import create_rectangular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
        >>> from materials.reinforced_concrete.code_checks.ec2.shear_check import ShearCheck, ShearLoadCase
        >>>
        >>> section = create_rectangular_section(width=300, height=500)
        >>> # ... add tension reinforcement ...
        >>>
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>> shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")
        >>>
        >>> # Create check once (diagram created on init if use_mechanical_lever_arm=True)
        >>> check = ShearCheck(
        ...     section=section,
        ...     concrete=concrete,
        ...     shear_reinforcement=shear_rebar,
        ...     use_mechanical_lever_arm=True,  # Default - accurate NA and lever arm
        ... )
        >>>
        >>> # Simple check - just shear force (M_Ed and N_Ed default to 0)
        >>> result = check.perform_check(load_case=ShearLoadCase(V_Ed=150))
        >>>
        >>> # With moment and axial force
        >>> result = check.perform_check(
        ...     load_case=ShearLoadCase(V_Ed=150, M_Ed=50, N_Ed=100)
        ... )
        >>>
        >>> # Check multiple load cases (use list comprehension)
        >>> load_cases = [
        ...     ShearLoadCase(V_Ed=150, M_Ed=50, N_Ed=100),   # Sagging
        ...     ShearLoadCase(V_Ed=120, M_Ed=-30, N_Ed=80),   # Hogging
        ...     ShearLoadCase(V_Ed=100),                       # Pure shear
        ... ]
        >>> results = [check.perform_check(load_case=case) for case in load_cases]
    """

    # ===============================
    # Section definition (immutable)
    # ===============================

    section: RCSection = Field(
        ...,
        description="RC section geometry",
    )

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material",
    )

    shear_reinforcement: Optional[ShearRebar] = Field(
        default=None,
        description="Shear links/stirrups (None if unreinforced)",
    )


    # ===========================
    # Limit state and rigour mode
    # ===========================

    use_accidental: bool = Field(
        default=False,
        description="Use accidental limit state partial factors (gamma_c_accidental, gamma_s_accidental)",
    )

    use_mechanical_lever_arm: bool = Field(
        default=False,
        description=(
            "Use rigorous mode: compute lever arm from force centroids. "
            "If False (approximate mode): always use z=0.9d for lever arm. "
            "Both modes use M-N solver for compression face detection when M_Ed or N_Ed provided."
        ),
    )

    allow_negative_sigma_cp: bool = Field(
        default=True,
        description=(
            "Allow negative σ_cp from tensile axial forces (default: True). "
            "If True, negative σ_cp reduces shear capacity. "
            "If False, σ_cp is limited to a minimum of 0.0 MPa."
        ),
    )

    use_transformed_area_for_sigma_cp: bool = Field(
        default=True,
        description=(
            "Use transformed area (concrete + n·steel) for σ_cp calculation (default: True). "
        ),
    )

    use_sigma_cp_for_alpha_cw: bool = Field(
        default=False,
        description=(
            "If True, include σ_cp in α_cw for V_Rd,max calculations. "
            "If False (default), α_cw is calculated with σ_cp = 0."
        ),
    )

    z_d_ratio: float = Field(
        default=0.9,
        description=(
            "Lever arm ratio z/d for approximate mode (use_mechanical_lever_arm=False). "
            "Default 0.9 per EC2 §6.2.3(1). For circular sections use ~0.77."
        ),
        gt=0.0,
        le=1.0,
    )

    z_d_ratio_upper: float = Field(
        default=0.95,
        description=(
            "Upper bound for lever arm ratio z/d in rigorous mode. "
            "z_mech is clamped to z <= z_d_ratio_upper * d."
        ),
        gt=0.0,
        le=1.0,
    )

    z_d_ratio_lower: float = Field(
        default=0.60,
        description=(
            "Lower bound for lever arm ratio z/d in rigorous mode. "
            "z_mech is clamped to z >= z_d_ratio_lower * d. "
            "Prevents fallback to a single default value for extreme axial states."
        ),
        gt=0.0,
        le=1.0,
    )

    breadth_override: Optional[float] = Field(
        default=None,
        description=(
            "User-supplied web breadth b_w (mm). If provided, overrides the automatic "
            "minimum-width calculation from section geometry. Useful for non-standard "
            "sections or when the automatic slicing does not capture the intended width."
        ),
        gt=0,
    )

    breadth_policy: Literal["minimum", "average"] = Field(
        default="minimum",
        description=(
            "Automatic b_w policy when breadth_override is not provided. "
            "'minimum' matches EC2 minimum-width intent; 'average' returns the "
            "mean width over a central depth window."
        ),
    )

    breadth_average_height_ratio: float = Field(
        default=1.0,
        description=(
            "Depth ratio used for breadth_policy='average', measured along "
            "breadth_shear_direction and centered at section mid-depth. "
            "Example: 0.5 uses the middle 50% depth window."
        ),
        gt=0.0,
        le=1.0,
    )

    breadth_shear_direction: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description=(
            "Shear direction vector (vx, vy) used by automatic breadth slicing. "
            "(0, 1) gives horizontal slices for major-axis shear; (1, 0) gives "
            "vertical slices for minor-axis shear."
        ),
    )

    use_increased_nu_1: bool = Field(
        default=False,
        description=(
            "Use increased ν₁ factor per EC2 §6.2.3(3) Note 2 when shear reinforcement "
            "stress is below 80% of f_yk (σ_s < 0.8·f_yk). This allows higher V_Rd,max "
            "capacity but requires iterative calculation."
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

    d_fallback: EffectiveDepthFallback = Field(
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


    # ==================================
    # Material models for rigorous mode
    # ==================================

    concrete_model_type: ConcreteModelType = Field(
        default=ConcreteModelType.PARABOLA_RECTANGLE,
        description="Concrete stress-strain model type (used if use_mechanical_lever_arm=True)",
    )

    steel_model_type: SteelModelType = Field(
        default=SteelModelType.INCLINED,
        description="Steel stress-strain branch type (used if use_mechanical_lever_arm=True)",
    )

    concrete_model_override: Optional[Any] = Field(
        default=None, exclude=True,
        description="Pre-built custom concrete constitutive model (bypasses factory).",
    )
    steel_models_override: Optional[list] = Field(
        default=None, exclude=True,
        description="Pre-built custom steel constitutive models, one per rebar group (bypasses factory).",
    )


    # =========================
    # Internal state (private)
    # =========================

    _diagram: Optional[MNInteractionDiagram] = PrivateAttr(default=None)
    _diagram_no_comp_steel: Optional[MNInteractionDiagram] = PrivateAttr(default=None)
    _diagram_snapshot: Optional[dict] = PrivateAttr(default=None)
    _diagram_no_comp_snapshot: Optional[dict] = PrivateAttr(default=None)
    _breadth_cache: Optional[float] = PrivateAttr(default=None)
    _breadth_snapshot: Optional[dict[str, Any]] = PrivateAttr(default=None)

    def _take_section_snapshot(self) -> dict[str, Any]:
        """
        Capture current section state for cache invalidation.

        Uses ``model_dump`` when available so in-place section mutations
        can be detected without requiring field reassignment on ``ShearCheck``.
        """
        section = self.section
        if hasattr(section, "model_dump"):
            return section.model_dump()  # type: ignore[no-any-return]
        return {"section_obj_id": id(section)}

    def _take_breadth_snapshot(self) -> dict[str, Any]:
        """Capture current breadth inputs for cache invalidation."""
        try:
            policy = self.breadth_policy
            average_height_ratio = self.breadth_average_height_ratio
            shear_direction = self.breadth_shear_direction
        except AttributeError:
            # Compatibility for tests that instantiate via object.__new__
            policy = "minimum"
            average_height_ratio = 1.0
            shear_direction = (0.0, 1.0)

        return {
            "section": self._take_section_snapshot(),
            "policy": policy,
            "average_height_ratio": float(average_height_ratio),
            "shear_direction": (float(shear_direction[0]), float(shear_direction[1])),
        }

    @model_validator(mode="after")
    def _validate_z_d_ratios(self) -> "ShearCheck":
        if self.z_d_ratio_lower >= self.z_d_ratio_upper:
            raise ValueError(
                f"z_d_ratio_lower ({self.z_d_ratio_lower}) must be < "
                f"z_d_ratio_upper ({self.z_d_ratio_upper})"
            )
        return self

    @model_validator(mode="after")
    def _validate_concrete_model_type(self) -> "ShearCheck":
        if self.concrete_model_override is not None:
            return self
        if self.concrete_model_type == ConcreteModelType.LINEAR_ELASTIC:
            raise ValueError(
                "LINEAR_ELASTIC concrete model is only valid for SLS checks "
                "(e.g. CrackingCheck), not for ULS shear checks."
            )
        return self

    @model_validator(mode="after")
    def _validate_breadth_direction(self) -> "ShearCheck":
        vx, vy = self.breadth_shear_direction
        if abs(float(vx)) <= 1e-12 and abs(float(vy)) <= 1e-12:
            raise ValueError("breadth_shear_direction must be a non-zero vector")
        return self

    def _take_snapshot(self) -> dict:
        """Capture current state of inputs that affect the interaction diagram."""
        snapshot = {
            "section": self.section.model_dump(),
            "concrete": self.concrete.model_dump(),
            "concrete_model_type": self.concrete_model_type,
            "steel_model_type": self.steel_model_type,
            "use_accidental": self.use_accidental,
        }
        if self.concrete_model_override is not None:
            snapshot["concrete_override_key"] = getattr(
                self.concrete_model_override, "cache_key", id(self.concrete_model_override)
            )
        if self.steel_models_override is not None:
            snapshot["steel_override_keys"] = [
                getattr(sm, "cache_key", id(sm)) for sm in self.steel_models_override
            ]
        return snapshot

    def _get_diagram(self, ignore_compression_steel: bool = False) -> MNInteractionDiagram:
        """Get the cached diagram, rebuilding if inputs have changed."""
        snapshot = self._take_snapshot()

        if ignore_compression_steel:
            if self._diagram_no_comp_steel is None or snapshot != self._diagram_no_comp_snapshot:
                self._diagram_no_comp_steel = MNInteractionDiagram(
                    section=self.section,
                    concrete=self.concrete,
                    concrete_model_type=self.concrete_model_type,
                    steel_model_type=self.steel_model_type,
                    use_characteristic=False,
                    use_accidental=self.use_accidental,
                    ignore_compression_steel=True,
                    concrete_model_override=self.concrete_model_override,
                    steel_models_override=self.steel_models_override,
                )
                self._diagram_no_comp_snapshot = snapshot
            return self._diagram_no_comp_steel
        else:
            if self._diagram is None or snapshot != self._diagram_snapshot:
                self._diagram = MNInteractionDiagram(
                    section=self.section,
                    concrete=self.concrete,
                    concrete_model_type=self.concrete_model_type,
                    steel_model_type=self.steel_model_type,
                    use_characteristic=False,
                    use_accidental=self.use_accidental,
                    ignore_compression_steel=False,
                    concrete_model_override=self.concrete_model_override,
                    steel_models_override=self.steel_models_override,
                )
                self._diagram_snapshot = snapshot
            return self._diagram

    @property
    def _A_transformed(self) -> float:
        """Transformed area (mm²)."""
        return self.section.get_transformed_area(self.concrete.E_cm)

    @property
    def _A_gross(self) -> float:
        """Gross area (mm²)."""
        return self.section.get_area()

    # ===============================================
    # Properties (immutable - don't depend on loads)
    # ===============================================

    @property
    def breadth(self) -> float:
        """
        Section breadth b_w for shear design (mm).

        If ``breadth_override`` is set, that value is used directly.
        Otherwise, computed automatically from section geometry using
        ``breadth_policy``, ``breadth_average_height_ratio``, and
        ``breadth_shear_direction``.

        The computed breadth is cached and automatically recomputed if the
        section or breadth-policy snapshot changes.
        """
        if self.breadth_override is not None:
            return self.breadth_override

        breadth_snapshot_now = self._take_breadth_snapshot()
        try:
            policy = self.breadth_policy
            average_height_ratio = self.breadth_average_height_ratio
            shear_direction = self.breadth_shear_direction
        except AttributeError:
            # Compatibility for tests that instantiate via object.__new__
            policy = "minimum"
            average_height_ratio = 1.0
            shear_direction = (0.0, 1.0)

        breadth_cache: Optional[float]
        breadth_snapshot: Optional[dict[str, Any]]
        try:
            breadth_cache = self._breadth_cache
            breadth_snapshot = self._breadth_snapshot
        except AttributeError:
            # Compatibility for tests that instantiate via object.__new__
            breadth_cache = None
            breadth_snapshot = None

        if breadth_cache is None or breadth_snapshot != breadth_snapshot_now:
            breadth_cache = calculate_section_breadth(
                self.section,
                policy=policy,
                average_height_ratio=average_height_ratio,
                shear_direction=shear_direction,
            )
            try:
                self._breadth_cache = breadth_cache
                self._breadth_snapshot = breadth_snapshot_now
            except AttributeError:
                # Compatibility for tests that instantiate via object.__new__
                object.__setattr__(self, "_breadth_cache", breadth_cache)
                object.__setattr__(self, "_breadth_snapshot", breadth_snapshot_now)
        assert breadth_cache is not None
        return breadth_cache

    @property
    def f_cd_design(self) -> float:
        """Design concrete compressive strength for shear (accidental or persistent) in MPa."""
        if self.use_accidental:
            return self.concrete.f_cd_shear_accidental
        return self.concrete.f_cd_shear

    @property
    def f_ctd_design(self) -> float:
        """Design concrete tensile strength (accidental or persistent) in MPa."""
        return self.concrete.f_ctd_accidental if self.use_accidental else self.concrete.f_ctd

    @property
    def gamma_c_design(self) -> float:
        """Partial factor for concrete (accidental or persistent)."""
        return self.concrete.gamma_c_accidental if self.use_accidental else self.concrete.gamma_c

    @property
    def f_ywd_design(self) -> float:
        """Design yield strength of shear reinforcement (accidental or persistent) in MPa."""
        if self.shear_reinforcement is None:
            return 0.0
        return (
            self.shear_reinforcement.f_yd_accidental
            if self.use_accidental
            else self.shear_reinforcement.f_yd
        )


    # ===========================
    # Load-dependent methods
    # ===========================

    def find_effective_depth(
        self,
        M_Ed: float,
        N_Ed: float,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
        *,
        m_tol: float = 1e-6,
        strain_tol: float = 1e-15,
        warn_on_fallback: bool = True,
        ignore_compression_steel: bool = False,
    ) -> float:
        """
        Effective depth d (mm) measured from the governing compression face.

        Delegates to ``find_effective_depth_for_flexure`` (the single source of truth).

        When the strain state is ambiguous (net compression, net tension, pure axial),
        the ``d_fallback`` policy on this check instance controls the result.
        """
        # Only build diagram if needed (strains not provided and M_Ed non-zero)
        need_diagram = (eps_top is None or eps_bottom is None) and abs(M_Ed) > m_tol
        diagram = self._get_diagram(ignore_compression_steel) if need_diagram else None
        return find_effective_depth_for_flexure(
            section=self.section,
            diagram=diagram,
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            m_tol=m_tol,
            strain_tol=strain_tol,
            warn_on_fallback=warn_on_fallback,
            d_fallback=self.d_fallback,
            d_ratio=self.d_ratio,
            _stacklevel=3,
        )


    def find_lever_arm(
        self,
        M_Ed: float,
        N_Ed: float,
        d: float,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
        ignore_compression_steel: bool = False,
        force_virtual: bool = False,
    ) -> tuple[float, Optional[float]]:
        """
        Lever arm for this load case.

        Behaviour:
            If use_mechanical_lever_arm=True: computes from force resultant centroids, clamped
                to [z_d_ratio_lower * d, z_d_ratio_upper * d]
            If use_mechanical_lever_arm=False: uses z_d_ratio * d approximation

        Returns:
            (z_design, z_mech)
        """
        # No diagram available (or user opted out) => always use configured ratio
        if not self.use_mechanical_lever_arm:
            return (self.z_d_ratio * d, None)

        # Delegate to MNInteractionDiagram with configured bounds
        return self._get_diagram(ignore_compression_steel).get_lever_arm(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            d=d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            use_mechanical_lever_arm=True,
            z_d_upper=self.z_d_ratio_upper,
            z_d_lower=self.z_d_ratio_lower,
            z_d_approx=self.z_d_ratio,
            force_virtual=force_virtual,
        )


    def _find_rho_l(
        self,
        M_Ed: float,
        N_Ed: float,
        d: float,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
        ignore_compression_steel: bool = False,
    ) -> float:
        """
        Longitudinal reinforcement ratio (§6.2.2(1)).

        ρ_l = A_sl / (b_w·d) ≤ 0.02

        If rigorous: uses actual neutral axis from strain field
        If approximate: uses centroid approximation

        Args:
            M_Ed: Design moment in kN·m
            N_Ed: Design axial force in kN
            d: Effective depth in mm
            eps_top: Pre-computed top strain (optional, avoids re-solving)
            eps_bottom: Pre-computed bottom strain (optional, avoids re-solving)

        Returns:
            ρ_l (dimensionless)
        """
        if not self.use_mechanical_lever_arm:
            # Approximate mode: if we have strain information, use it to determine tension side
            # This handles hogging/sagging and N-M interaction correctly
            if eps_top is not None and eps_bottom is not None:
                # Use strain-based approach (same as rigorous, just without diagram solver)
                return self._compute_rho_l_from_strains(eps_top, eps_bottom, d)

            # Fallback for truly approximate case (no strain info): centroid-based
            # This assumes sagging (bottom in tension), which may be wrong for hogging
            centroid_y = self.section.get_centroid()[1]

            A_sl = 0.0
            for group in self.section.rebar_groups:
                for pos in group.positions:
                    if pos.y < centroid_y:  # Below centroid = tension (sagging assumption)
                        A_sl += group.rebar.area

            if A_sl == 0:
                return 0.0

            b_w = self.breadth
            rho_l = A_sl / (b_w * d)
            return min(rho_l, 0.02)

        # Rigorous: use actual NA from strain state
        if eps_top is None or eps_bottom is None:
            eps_top, eps_bottom = self._get_diagram(ignore_compression_steel).find_strains_for_MN(M_Ed, N_Ed)
        return self._compute_rho_l_from_strains(eps_top, eps_bottom, d)


    def _find_sigma_cp(
            self,
            N_Ed: float,
        ) -> float:
        """
        Compressive stress in concrete due to axial force (§6.2.2(1)).

        σ_cp = N_Ed / A_c,transformed, limited to 0.2·f_cd

        Uses transformed area (concrete + n·steel) for more accurate stress calculation.

        Args:
            N_Ed: Design axial force in kN (compression positive)

        Returns:
            Stress in MPa
        """
        # 1. Check policy for tension
        if N_Ed <= 0 and not self.allow_negative_sigma_cp:
            return 0.0
        
        # 2. Check policy for area calculation
        A_eff = self._A_transformed if self.use_transformed_area_for_sigma_cp else self._A_gross

        sigma_cp_uncapped = sigma_cp_from_N_and_area(N_Ed, A_eff)
        return cap_sigma_cp_upper(sigma_cp_uncapped, self.f_cd_design)


    # =================================
    # Helper methods for rigorous mode
    # =================================

    def _compute_rho_l_from_strains(
        self,
        eps_top: float,
        eps_bottom: float,
        d: float
    ) -> float:
        """
        Compute rho_l using actual neutral axis from strain profile.

        Args:
            eps_top: Strain at top fibre
            eps_bottom: Strain at bottom fibre
            d: Effective depth in mm

        Returns:
            ρ_l (dimensionless)
        """
        return find_rho_l_from_strains(
            section=self.section,
            b_w=self.breadth,
            d=d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            rho_l_max=0.02,
        )


    # ===========================
    # EC2 calculation methods
    # ===========================

    def find_V_Rd_c(self, d: float, rho_l: float, sigma_cp: float) -> float:
        """
        Design shear resistance without shear reinforcement (§6.2.2, Eq. 6.2).

        Public method - takes computed parameters.

        Args:
            d: Effective depth in mm
            rho_l: Longitudinal reinforcement ratio
            sigma_cp: Compressive stress from axial force in MPa

        Returns:
            V_Rd,c in kN
        """
        return find_V_Rd_c_cracked(
            b_w=self.breadth, d=d, rho_l=rho_l, sigma_cp=sigma_cp,
            f_ck=self.concrete.f_ck, gamma_c=self.gamma_c_design,
        )

    def find_V_Rd_c_uncracked(self, sigma_cp: float, alpha_I: float = 1.0) -> float:
        """
        Uncracked shear resistance based on principal tensile stress (§6.2.2(2)).

        V_Rd,c,uncracked = (I * b_w / S) * sqrt(f_ctd^2 + alpha_I * sigma_cp * f_ctd)

        Notes:
            - This is reported for reference in standard runs.
            - It should only govern design checks when explicitly requested.

        Args:
            sigma_cp: Axial stress in concrete (MPa)
            alpha_I: Coefficient for prestress contribution (default 1.0)

        Returns:
            V_Rd,c,uncracked in kN
        """
        b_w = self.breadth
        if b_w <= 0.0:
            return 0.0

        I_xx, _, _ = self.section.get_second_moment_area()
        if I_xx <= 0.0:
            return 0.0

        _, cy = self.section.get_centroid()
        min_x, min_y, max_x, max_y = self.section.outline.bounds
        span = max(max_x - min_x, max_y - min_y, 1.0)
        pad = span

        # First moment S of area above centroidal axis about that axis.
        # For centroidal axes, top/bottom first moments have equal magnitude.
        from shapely.geometry import box

        top_region = self.section.outline.intersection(
            box(min_x - pad, cy, max_x + pad, max_y + pad)
        )
        A_top = float(top_region.area)
        if A_top <= 0.0:
            return 0.0

        y_top = float(top_region.centroid.y)
        S = A_top * (y_top - cy)
        if S <= 0.0:
            return 0.0

        f_ctd = self.f_ctd_design
        inner = f_ctd**2 + alpha_I * sigma_cp * f_ctd
        if inner <= 0.0:
            return 0.0

        V_Rd_c_N = (I_xx * b_w / S) * sqrt(inner)
        return to_kn(V_Rd_c_N, ForceUnit.N)


    def find_V_Rd_c_max_unreinforced(self, d: float) -> float:
        """
        Maximum shear force for members without shear reinforcement (§6.2.2(6), Eq. 6.5).

        V_Rd,c,max = 0.5·b_w·d·ν·f_cd

        This limit applies when no shear reinforcement is provided.
        It ensures diagonal compression failure does not occur.

        Args:
            d: Effective depth in mm

        Returns:
            V_Rd,c,max in kN
        """
        return find_V_Rd_c_max_unreinforced(
            b_w=self.breadth, d=d, f_ck=self.concrete.f_ck, f_cd=self.f_cd_design,
        )


    def find_V_Rd_s(
        self, cot_theta: float, z: float, use_note_2: bool = False
    ) -> float:
        """
        Shear resistance of shear reinforcement (§6.2.3(3), Eq. 6.8).

        Public method - takes computed parameters.

        Args:
            cot_theta: Cotangent of strut angle (pre-clamped)
            z: Lever arm in mm
            use_note_2: If True, use reduced f_ywd = 0.8·f_ywk per Note 2 requirement

        Returns:
            V_Rd,s in kN

        Note:
            Per EC2 §6.2.3(3) Note 2: If the increased ν₁ from Note 2 is used,
            f_ywd should be reduced to 0.8·f_ywk for V_Rd,s calculation.
        """
        if self.shear_reinforcement is None:
            raise ValueError("V_Rd_s cannot be found without providing shear reinforcement.")

        A_sw_over_s = self.shear_reinforcement.area_per_unit_length
        link_angle_rads = radians(self.shear_reinforcement.angle)

        # Per Note 2: when using increased ν₁, f_ywd is reduced to 0.8·f_ywk
        if use_note_2:
            f_ywd = 0.8 * self.shear_reinforcement.f_yk
        else:
            f_ywd = self.f_ywd_design

        V_Rd_s = A_sw_over_s * z * f_ywd * (cot_theta + cot(link_angle_rads)) * sin(link_angle_rads)
        return to_kn(V_Rd_s, ForceUnit.N)


    def find_V_Rd_max(
        self, cot_theta: float, z: float, sigma_cp: float, use_note_2: bool = False
    ) -> float:
        """
        Maximum shear resistance limited by crushing of compression struts (§6.2.3, Eq. 6.9).

        Args:
            cot_theta: Cotangent of strut angle (pre-clamped)
            z: Lever arm in mm
            sigma_cp: Compressive stress from axial force in MPa
            use_note_2: If True, use increased ν₁ from Note 2 (default: False)

        Returns:
            V_Rd,max in kN
        """
        if self.shear_reinforcement is None:
            raise ValueError("V_Rd_max cannot be found without providing shear reinforcement.")

        f_cd = self.f_cd_design
        alpha_cw = find_alpha_cw(
            f_cd,
            sigma_cp,
            use_sigma_cp_for_alpha_cw=self.use_sigma_cp_for_alpha_cw,
        )
        b_w = self.breadth

        # Select appropriate nu_1 factor
        if use_note_2:
            nu_1 = find_nu_1_factor_note_2(self.concrete.f_ck, self.shear_reinforcement.angle)
        else:
            nu_1 = find_nu_1_factor(self.concrete.f_ck, self.shear_reinforcement.angle)

        link_angle_rads = radians(self.shear_reinforcement.angle)
        V_Rd_max = (alpha_cw * b_w * z * nu_1 * f_cd) * (cot_theta + cot(link_angle_rads)) / (1 + cot_theta**2)
        return to_kn(V_Rd_max, ForceUnit.N)


    def _calculate_K(self, z: float, sigma_cp: float, use_note_2: bool = False) -> float:
        """
        Calculate K parameter for cot(θ) determination.

        K = α_cw · b_w · z · ν₁ · f_cd

        This value is used in find_cot_theta_for_V_Ed_from_V_Rd_max.
        When using Note 2 iteration,
        K must be recalculated because ν₁ changes.

        Args:
            z: Lever arm in mm
            sigma_cp: Compressive stress from axial force in MPa
            use_note_2: If True, use increased ν₁ from Note 2

        Returns:
            K in N (not kN)
        """
        if self.shear_reinforcement is None:
            raise ValueError("K cannot be calculated without shear reinforcement.")

        f_cd = self.f_cd_design
        alpha_cw = find_alpha_cw(
            f_cd,
            sigma_cp,
            use_sigma_cp_for_alpha_cw=self.use_sigma_cp_for_alpha_cw,
        )
        b_w = self.breadth

        if use_note_2:
            nu_1 = find_nu_1_factor_note_2(self.concrete.f_ck, self.shear_reinforcement.angle)
        else:
            nu_1 = find_nu_1_factor(self.concrete.f_ck, self.shear_reinforcement.angle)

        return alpha_cw * b_w * z * nu_1 * f_cd


    def _find_cot_theta_limits(
        self,
        sigma_cp: float,
        z: float,
        V_Ed: float,
    ) -> tuple[float, float]:
        """
        Compute cot(θ) limits for this load case.

        For EU/EU_UK: constants (1.0 to 2.5)
        For EU_DE: max limit depends on stress state per German NA formula.

        Args:
            sigma_cp: Axial compressive stress in MPa
            z: Lever arm in mm
            V_Ed: Design shear force in kN (absolute value)

        Returns:
            (cot_theta_min, cot_theta_max)
        """
        # Minimum cot(theta) - typically constant
        min_val = get_ndp("cot_theta_lower_lim")
        cot_min = float(min_val() if callable(min_val) else min_val)

        # Maximum cot(theta) - may be callable for German NA
        max_val = get_ndp("cot_theta_upper_lim")
        if callable(max_val):
            # German NA formula requires these parameters
            V_Ed_N = from_kn(V_Ed, ForceUnit.N) if V_Ed > 0 else 1.0  # Avoid div/0
            cot_max = float(max_val(
                f_ck=self.concrete.f_ck,
                f_cd=self.f_cd_design,
                sigma_cp=sigma_cp,
                b_w=self.breadth,
                z=z,
                V_Ed=V_Ed_N,
            ))
        else:
            cot_max = float(max_val)

        # Apply reduced upper limit for shear with externally applied tension
        if self.apply_tension_cot_theta_limit and sigma_cp < 0:
            tension_lim = get_ndp("cot_theta_upper_lim_tension")
            if tension_lim is not None and not callable(tension_lim):
                cot_max = min(cot_max, float(tension_lim))

        # Ensure valid range (protect against edge cases in formulas)
        cot_max = max(cot_min, cot_max)

        return cot_min, cot_max

    def _find_cot_theta_for_V_Ed(
        self,
        *,
        V_Ed: float,
        z: float,
        sigma_cp: float,
        cot_min: float,
        cot_max: float,
        use_note_2: bool = False,
        use_v_rd_s_for_cot_theta: bool = False,
    ) -> float:
        """
        Determine cot(θ) from either V_Rd,max (Eq. 6.14) or V_Rd,s (Eq. 6.13).

        Args:
            V_Ed: Design shear force in kN (absolute value)
            z: Lever arm in mm
            sigma_cp: Axial stress in MPa
            cot_min: Lower cot(θ) limit
            cot_max: Upper cot(θ) limit
            use_note_2: If True, use Note 2 modifiers (ν₁ and f_ywd reduction)
            use_v_rd_s_for_cot_theta: If True, solve from V_Rd,s = V_Ed (Eq. 6.13).
                If False, solve from V_Rd,max = V_Ed (Eq. 6.14).
        """
        if self.shear_reinforcement is None:
            raise ValueError("cot(theta) cannot be determined without shear reinforcement.")

        if use_v_rd_s_for_cot_theta:
            f_ywd = 0.8 * self.shear_reinforcement.f_yk if use_note_2 else self.f_ywd_design
            return find_cot_theta_for_V_Ed_from_V_Rd_s(
                V_Ed=V_Ed,
                A_sw_over_s=self.shear_reinforcement.area_per_unit_length,
                z=z,
                f_ywd=f_ywd,
                link_angle_degrees=self.shear_reinforcement.angle,
                cot_min=cot_min,
                cot_max=cot_max,
            )

        K = self._calculate_K(z, sigma_cp, use_note_2=use_note_2)
        return find_cot_theta_for_V_Ed_from_V_Rd_max(
            V_Ed=V_Ed,
            K=K,
            link_angle_degrees=self.shear_reinforcement.angle,
            cot_min=cot_min,
            cot_max=cot_max,
        )


    def _find_V_Rd_max_with_note_2_iteration(
        self,
        V_Ed: float,
        z: float,
        sigma_cp: float,
        use_v_rd_s_for_cot_theta: bool = False,
        suppress_warnings: bool = False,
    ) -> tuple[float, bool]:
        """
        Calculate V_Rd,max with ν₁ Note 2 iteration per EC2 §6.2.3(3) Note 2.

        Iterates to check if σ_s < 0.8·f_yk, allowing increased ν₁ factor.
        Detects oscillation and reverts to ν₁ Note 1 if needed.

        Note: K is recalculated internally for each iteration because ν₁ changes
        between Note 1 and Note 2, which affects the cot(θ) determination.

        Args:
            V_Ed: Design shear force in kN (for stress calculation)
            z: Lever arm in mm
            sigma_cp: Compressive stress from axial force in MPa
            use_v_rd_s_for_cot_theta: If True, determine cot(θ) from
                rearranged Eq. 6.13 (V_Rd,s = V_Ed). If False, use
                rearranged Eq. 6.14 / V_Rd,max.

        Returns:
            Tuple of (V_Rd,max in kN, used_note_2: bool)
        """
        if self.shear_reinforcement is None:
            raise ValueError("Cannot iterate V_Rd,max without shear reinforcement.")

        f_yk = self.shear_reinforcement.f_yk
        threshold = 0.8 * f_yk  # as per Note 2
        f_ywd = self.f_ywd_design

        # Get cot(theta) limits for this load case
        cot_min, cot_max = self._find_cot_theta_limits(sigma_cp, z, V_Ed)

        # Iteration 1: Calculate with Note 1
        cot_theta_1 = self._find_cot_theta_for_V_Ed(
            V_Ed=V_Ed,
            z=z,
            sigma_cp=sigma_cp,
            cot_min=cot_min,
            cot_max=cot_max,
            use_note_2=False,
            use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
        )
        V_Rd_max_1 = self.find_V_Rd_max(cot_min, z, sigma_cp, use_note_2=False)
        V_Rd_s_1 = self.find_V_Rd_s(cot_theta_1, z)

        # Calculate stress in reinforcement: σ_s = f_ywd · (V_Ed / V_Rd_s)
        sigma_s_1 = f_ywd * (V_Ed / V_Rd_s_1) if V_Rd_s_1 > 0 else f_yk

        # Check if Note 2 is applicable
        if sigma_s_1 >= threshold:
            # Stress too high, use Note 1
            return V_Rd_max_1, False

        # Iteration 2: Try Note 2 (recalculate K with Note 2's ν₁)
        # Per Note 2: f_ywd is reduced to 0.8·f_ywk for V_Rd_s calculation
        cot_theta_2 = self._find_cot_theta_for_V_Ed(
            V_Ed=V_Ed,
            z=z,
            sigma_cp=sigma_cp,
            cot_min=cot_min,
            cot_max=cot_max,
            use_note_2=True,
            use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
        )
        V_Rd_max_2 = self.find_V_Rd_max(cot_min, z, sigma_cp, use_note_2=True)
        V_Rd_s_2 = self.find_V_Rd_s(cot_theta_2, z, use_note_2=True)
        f_ywd_note_2 = 0.8 * f_yk  # Reduced f_ywd per Note 2
        sigma_s_2 = f_ywd_note_2 * (V_Ed / V_Rd_s_2) if V_Rd_s_2 > 0 else f_yk

        # Check for oscillation: Note 2 pushes stress above threshold
        if sigma_s_2 >= threshold:
            # Oscillation detected - revert to Note 1
            if not suppress_warnings:
                warnings.warn(
                f"EC2 §6.2.3(3) Note 2: Oscillation detected. "
                f"Note 1: σ_s={sigma_s_1:.1f} MPa < {threshold:.1f} MPa, "
                f"Note 2: σ_s={sigma_s_2:.1f} MPa >= {threshold:.1f} MPa. "
                f"Reverting to Note 1 (conservative).",
                    stacklevel=3,
                )
            return V_Rd_max_1, False

        # Converged with Note 2
        return V_Rd_max_2, True


    # ===========================
    # Main check method
    # ===========================

    def perform_check(
        self,
        *,
        load_case: ShearLoadCase,
        cot_theta_override: Optional[float] = None,
        use_v_rd_s_for_cot_theta: bool = False,
        use_uncracked_V_Rd_c: bool = False,
        warning_threshold: float = 0.95,
        suppress_warnings: bool = False,
        ignore_compression_steel: bool = False,
        **kwargs,
    ) -> CheckResult:
        """
        Perform shear check per EC2 §6.2 for a single load case.

        Forces (V_Ed, M_Ed, N_Ed) are parameters via ShearLoadCase, not fields.
        This enables efficient checking of multiple load cases against
        the same section.

        Checks:
        1. V_Ed ≤ V_Rd,c (if no shear reinforcement)
        2. V_Ed ≤ V_Rd,s (if shear reinforcement provided)
        3. V_Ed ≤ V_Rd,max (crushing of struts)

        Args:
            load_case: Single load case to check
            warning_threshold: Utilization threshold for warning
            suppress_warnings:
                If True, suppress warnings emitted during this check.
            cot_theta_override: User provided cot theta value to use
            use_v_rd_s_for_cot_theta: If True, solve cot(θ) from rearranged EC2
                Eq. 6.13 (V_Rd,s = V_Ed). If False (default), solve cot(θ) from
                rearranged EC2 Eq. 6.14 / V_Rd,max.
            use_uncracked_V_Rd_c:
                If True, use V_Rd,c,uncracked (§6.2.2(2)) in place of the cracked
                Eq. 6.2 V_Rd,c for design checks. Default False (recommended).

        Returns:
            CheckResult with status, utilization, and details

        Examples:
            >>> # Single load case
            >>> result = check.perform_check(
            ...     load_case=ShearLoadCase(V_Ed=150, M_Ed=50, N_Ed=100)
            ... )
            >>>
            >>> # Multiple load cases - use list comprehension
            >>> results = [
            ...     check.perform_check(load_case=case)
            ...     for case in load_cases
            ... ]
        """
        return self._check_single_case(
            V_Ed=load_case.V_Ed,
            M_Ed=load_case.M_Ed,
            N_Ed=load_case.N_Ed,
            cot_theta_override=cot_theta_override,
            use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
            use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
            warning_threshold=warning_threshold,
            suppress_warnings=suppress_warnings,
            ignore_compression_steel=ignore_compression_steel,
        )


    def _check_single_case(
        self,
        V_Ed: float,
        M_Ed: float,
        N_Ed: float,
        cot_theta_override: Optional[float],
        use_v_rd_s_for_cot_theta: bool,
        use_uncracked_V_Rd_c: bool,
        warning_threshold: float,
        suppress_warnings: bool = False,
        ignore_compression_steel: bool = False,
    ) -> CheckResult:
        """Perform check for single load case (internal)."""        
        # Treat shear as magnitude (absolute value)
        # Negative shear from FEA sign conventions should not give negative utilization
        V_Ed = abs(V_Ed)

        # Solve for strains once (both modes need for compression face detection)
        # This avoids redundant solves and ensures consistency
        # Only solve if M_Ed is non-zero (moment determines compression face, not axial load)
        if abs(M_Ed) > 1e-6:
            eps_top, eps_bottom = self._get_diagram(ignore_compression_steel).find_strains_for_MN(M_Ed, N_Ed)
        else:
            eps_top, eps_bottom = None, None

        # Compute load-dependent geometric parameters (pass strains to avoid re-solving)
        d = self.find_effective_depth(M_Ed, N_Ed, eps_top, eps_bottom, ignore_compression_steel=ignore_compression_steel)
        sigma_cp = self._find_sigma_cp(N_Ed)
        rho_l = self._find_rho_l(M_Ed, N_Ed, d, eps_top, eps_bottom, ignore_compression_steel=ignore_compression_steel)

        # 1. Initialize variables that might not be reached
        V_Rd_s: Optional[float] = None
        V_Rd_max: Optional[float] = None
        V_Rd_c_max: Optional[float] = None
        cot_theta: Optional[float] = None
        z_ec2: Optional[float] = None
        z_mech: Optional[float] = None
        K: Optional[float] = None
        used_note_2: bool = False
        link_spacing_max_allowable: Optional[float] = None
        link_spacing_satisfied: Optional[bool] = None
        leg_spacing_max_allowable: Optional[float] = None
        leg_spacing_satisfied: Optional[bool] = None
        governing_component: Optional[str] = None

        # Compute capacities (use z_ec2 for design checks per EC2)

        # Compute both V_Rd,c variants and choose the one used for design.
        V_Rd_c_cracked = self.find_V_Rd_c(d, rho_l, sigma_cp)
        V_Rd_c_uncracked = self.find_V_Rd_c_uncracked(sigma_cp=sigma_cp)
        V_Rd_c = V_Rd_c_uncracked if use_uncracked_V_Rd_c else V_Rd_c_cracked

        reinforcement = self.shear_reinforcement

        # Determine governing capacity
        if reinforcement is None:  # only unreinforced checks reported
            # For unreinforced members, also check V_Ed limit from Eq. 6.5
            V_Rd_c_max = self.find_V_Rd_c_max_unreinforced(d)

            if V_Rd_c <= V_Rd_c_max:
                V_Rd = V_Rd_c
                if use_uncracked_V_Rd_c:
                    governing_mode = "concrete shear (V_Rd,c,uncracked; no shear reinforcement)"
                    governing_component = "V_Rd_c_uncracked"
                    code_ref = "EC2 §6.2.2(2)"
                else:
                    governing_mode = "concrete shear (V_Rd,c,cracked incl. ρ_l; no shear reinforcement)"
                    governing_component = "V_Rd_c_cracked"
                    code_ref = "EC2 §6.2.2 (Eq. 6.2)"
            else:
                V_Rd = V_Rd_c_max
                governing_mode = "diagonal compression (V_Rd,c,max; no shear reinforcement)"
                governing_component = "V_Rd_c_max"
                code_ref = "EC2 §6.2.2 (Eq. 6.5)"
        else:
            z_ec2, z_mech = self.find_lever_arm(
                M_Ed=M_Ed,
                N_Ed=N_Ed,
                d=d,
                eps_top=eps_top,
                eps_bottom=eps_bottom,
                ignore_compression_steel=ignore_compression_steel
                )

            # Apply German NA z_cap if applicable: z_cap = max(d - 2·d_2, d - d_2 - 30)
            z_cap_ndp = get_ndp("z_cap")
            if z_cap_ndp is not None and callable(z_cap_ndp):
                # Determine compression face from strains
                if eps_top is not None and eps_bottom is not None:
                    compression_face = "top" if eps_top >= eps_bottom else "bottom"
                else:
                    # Conservative: assume top compression (typical for sagging)
                    compression_face = "top"

                d_2 = self.section.get_compression_rebar_depth(compression_face)
                if d_2 is None:
                    d_2 = 0.0  # Safe fallback: z_cap = max(d, d - 30) = d

                z_cap = z_cap_ndp(d, d_2)
                if z_ec2 > z_cap:
                    z_ec2 = z_cap

            # Get cot(theta) limits for this load case
            cot_min, cot_max = self._find_cot_theta_limits(sigma_cp, z_ec2, V_Ed)

            # Calculate K for cot_theta determination (uses Note 1 ν₁) when relevant
            if not use_v_rd_s_for_cot_theta:
                K = self._calculate_K(z_ec2, sigma_cp, use_note_2=False)
            else:
                K = None

            if cot_theta_override is not None:
                cot_theta = cot_theta_override

                if cot_theta_override > cot_max:
                    if not suppress_warnings:
                        warnings.warn(
                        f"Cot theta value provided, cot(θ) = {cot_theta_override}, is greater than max value: {cot_max:.2f}.",
                            stacklevel=2,
                        )
                elif cot_theta_override < cot_min:
                    if not suppress_warnings:
                        warnings.warn(
                        f"Cot theta value provided, cot(θ) = {cot_theta_override}, is smaller than min value: {cot_min:.2f}.",
                            stacklevel=2,
                        )
            else:
                # Calculate cot_theta from selected equation (already clamped by function)
                cot_theta = self._find_cot_theta_for_V_Ed(
                    V_Ed=V_Ed,
                    z=z_ec2,
                    sigma_cp=sigma_cp,
                    cot_min=cot_min,
                    cot_max=cot_max,
                    use_note_2=False,
                    use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
                )

            # the maximum capacity of the concrete strut is found using the largest theta (smallest cot_theta)
            used_note_2 = False
            if self.use_increased_nu_1:
                # Use EC2 §6.2.3(3) Note 2 iteration to potentially increase nu_1 if stress allows
                V_Rd_max, used_note_2 = self._find_V_Rd_max_with_note_2_iteration(
                    V_Ed,
                    z_ec2,
                    sigma_cp,
                    use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
                    suppress_warnings=suppress_warnings,
                )
            else:
                V_Rd_max = self.find_V_Rd_max(cot_min, z_ec2, sigma_cp)

            # Calculate V_Rd_s with appropriate f_ywd (reduced if Note 2 is used)
            V_Rd_s = self.find_V_Rd_s(cot_theta, z_ec2, use_note_2=used_note_2)

            # Maximum allowable longitudinal link spacing (NDP-dependent)
            _, min_y, _, max_y = self.section.get_bounding_box()
            section_depth = max_y - min_y
            link_spacing_max_allowable = find_max_allowable_link_spacing(
                effective_depth=d,
                section_depth=section_depth,
                f_ck=self.concrete.f_ck,
                V_Ed=V_Ed,
                V_Rd_max=V_Rd_max,
                V_Rd_c=V_Rd_c,
                link_angle_degrees=reinforcement.angle,
            )
            link_spacing_satisfied = reinforcement.link_spacing <= link_spacing_max_allowable + 1e-9
            if not link_spacing_satisfied and not suppress_warnings:
                warnings.warn(
                    "Provided shear link spacing exceeds the maximum allowable spacing: "
                    f"s={reinforcement.link_spacing:.1f} mm > s_max={link_spacing_max_allowable:.1f} mm.",
                    stacklevel=2,
                )

            # Maximum allowable transverse leg spacing (only when provided by user)
            if reinforcement.leg_spacing is not None:
                leg_spacing_max_allowable = find_max_allowable_leg_spacing(
                    effective_depth=d,
                    section_depth=section_depth,
                    f_ck=self.concrete.f_ck,
                    V_Ed=V_Ed,
                    V_Rd_max=V_Rd_max,
                    V_Rd_c=V_Rd_c,
                    link_angle_degrees=reinforcement.angle,
                )
                leg_spacing_satisfied = reinforcement.leg_spacing <= leg_spacing_max_allowable + 1e-9
                if not leg_spacing_satisfied and not suppress_warnings:
                    warnings.warn(
                        "Provided shear leg spacing exceeds the maximum allowable spacing: "
                        f"s_t={reinforcement.leg_spacing:.1f} mm > s_t,max={leg_spacing_max_allowable:.1f} mm.",
                        stacklevel=2,
                    )

            assert V_Rd_s is not None and V_Rd_max is not None

            if V_Ed > V_Rd_c:
                # Shear reinforcement is engaged
                V_Rd = min(V_Rd_s, V_Rd_max)

                if V_Rd_s < V_Rd_max:
                    governing_mode = "shear reinforcement (V_Rd,s)"
                    governing_component = "V_Rd_s"
                    code_ref = "EC2 §6.2.3 (Eq. 6.8)"
                else:
                    governing_mode = "compression strut (V_Rd,max)"
                    governing_component = "V_Rd_max"
                    code_ref = "EC2 §6.2.3 (Eq. 6.9)"
            else:
                V_Rd = min(V_Rd_c, V_Rd_max)
                if V_Rd_max < V_Rd_c:
                    governing_mode = "compression strut (V_Rd,max)"
                    governing_component = "V_Rd_max"
                    code_ref = "EC2 §6.2.3 (Eq. 6.9)"
                else:
                    if use_uncracked_V_Rd_c:
                        governing_mode = "concrete shear (V_Rd,c,uncracked)"
                        governing_component = "V_Rd_c_uncracked"
                        code_ref = "EC2 §6.2.2(2)"
                    else:
                        governing_mode = "concrete shear (V_Rd,c,cracked)"
                        governing_component = "V_Rd_c_cracked"
                        code_ref = "EC2 §6.2.2 (Eq. 6.2)"

        # Create message
        utilization = V_Ed / V_Rd if V_Rd > 0 else float('inf')

        if utilization <= 1.0:
            if utilization >= warning_threshold:
                message = f"High shear utilization ({utilization:.1%}) - governed by {governing_mode}"
            else:
                message = f"Shear check satisfied - governed by {governing_mode}"
        else:
            if governing_component == "V_Rd_c_max":
                message = (
                    "Shear capacity exceeded: diagonal compression limit V_Rd,c,max reached "
                    "(member without shear reinforcement)."
                )
            elif governing_component == "V_Rd_c_cracked":
                if reinforcement is None:
                    message = (
                        "Shear capacity exceeded: cracked concrete shear resistance "
                        "V_Rd,c reached (member without shear reinforcement)."
                    )
                else:
                    message = "Shear capacity exceeded: cracked concrete shear resistance V_Rd,c reached."
            elif governing_component == "V_Rd_c_uncracked":
                if reinforcement is None:
                    message = (
                        "Shear capacity exceeded: uncracked concrete shear resistance "
                        "V_Rd,c reached (member without shear reinforcement)."
                    )
                else:
                    message = "Shear capacity exceeded: uncracked concrete shear resistance V_Rd,c reached."
            elif governing_component == "V_Rd_max":
                message = "Shear capacity exceeded: compression strut limit V_Rd,max reached."
            elif governing_component == "V_Rd_s":
                message = "Shear capacity exceeded: shear reinforcement limit V_Rd,s reached."
            else:  # pragma: no cover - defensive fallback for unforeseen future governing modes
                message = "Shear capacity exceeded."

        # Details — common keys match CircularSectionCheck for consistency
        details = {
            "V_Ed": V_Ed,
            "M_Ed": M_Ed,
            "N_Ed": N_Ed,
            "V_Rd": V_Rd,
            "V_Rd_c": V_Rd_c,
            "V_Rd_c_cracked": V_Rd_c_cracked,
            "V_Rd_c_uncracked": V_Rd_c_uncracked,
            "use_uncracked_V_Rd_c": use_uncracked_V_Rd_c,
            "V_Rd_c_max_unreinforced": V_Rd_c_max if not reinforcement else None,
            "V_Rd_s": V_Rd_s if reinforcement else None,
            "V_Rd_max": V_Rd_max if reinforcement else None,
            "governing_mode": governing_mode,
            "governing_component": governing_component,
            "cot_theta": cot_theta if reinforcement else None,
            "theta_deg": degrees(atan(1 / cot_theta)) if cot_theta else None,
            "section_name": self.section.section_name or "unnamed",
            "d": d,
            "z": z_ec2 if reinforcement else None,
            "z_mech": z_mech if reinforcement else None,
            "b_w": self.breadth,
            "sigma_cp": sigma_cp,
            "alpha_cw": (
                find_alpha_cw(
                    self.f_cd_design,
                    sigma_cp,
                    use_sigma_cp_for_alpha_cw=self.use_sigma_cp_for_alpha_cw,
                )
                if reinforcement
                else None
            ),
            "nu_1": (
                find_nu_1_factor_note_2(self.concrete.f_ck, reinforcement.angle)
                if reinforcement and used_note_2
                else find_nu_1_factor(self.concrete.f_ck, reinforcement.angle)
                if reinforcement
                else None
            ),
            "K": K if reinforcement else None,
            "f_ywd": (
                0.8 * reinforcement.f_yk
                if reinforcement and used_note_2
                else self.f_ywd_design
                if reinforcement
                else None
            ),
            "used_note_2": used_note_2 if reinforcement else None,
            "cot_theta_from_v_rd_s": use_v_rd_s_for_cot_theta if reinforcement else None,
            "link_spacing_satisfied": link_spacing_satisfied if reinforcement else None,
            "link_spacing_provided": reinforcement.link_spacing if reinforcement else None,
            "link_spacing_max_allowable": link_spacing_max_allowable if reinforcement else None,
            "leg_spacing_satisfied": leg_spacing_satisfied if reinforcement else None,
            "leg_spacing_provided": (
                reinforcement.leg_spacing if reinforcement and reinforcement.leg_spacing is not None else None
            ),
            "leg_spacing_max_allowable": leg_spacing_max_allowable if reinforcement else None,
            # ShearCheck-specific
            "rho_l": rho_l,
            "z_mode": "rigorous" if self.use_mechanical_lever_arm else "approximate",
            "z_d_ratio": self.z_d_ratio,
            "z_d_ratio_upper": self.z_d_ratio_upper,
            "z_d_ratio_lower": self.z_d_ratio_lower,
        }

        return self._create_result(
            check_name="Shear check (EC2 §6.2)",
            demand=V_Ed,
            capacity=V_Rd,
            units="kN",
            code_reference=code_ref,
            warning_threshold=warning_threshold,
            message=message,
            details=details,
        )


    # ===============================================
    # Plotting convenience methods
    # ===============================================

    def plot_cot_theta_study(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        n_points: int = 60,
        cot_theta_min: Optional[float] = None,
        cot_theta_max: Optional[float] = None,
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1000,
        height: int = 560,
    ) -> Any:
        """
        Plot shear-capacity components over a cot(theta) sweep.

        The figure contains demand and capacity references (`V_Ed`, `V_Rd,c`),
        variable capacities (`V_Rd,s`, `V_Rd,max`) and fixed design reference lines
        at the governing code limits for cot(theta).

        Args:
            load_case: Shear demand definition as either ``ShearLoadCase`` or a
                ``dict`` with keys ``V_Ed`` and optional ``M_Ed``/``N_Ed`` (kN, kN·m).
            n_points: Number of cot(theta) samples in the sweep.
            cot_theta_min: Optional lower bound for cot(theta). If ``None``,
                the EC2-based minimum from the current check context is used.
            cot_theta_max: Optional upper bound for cot(theta). If ``None``,
                the EC2-based maximum from the current check context is used.
            use_uncracked_V_Rd_c: If ``True``, use uncracked concrete shear capacity
                ``V_Rd,c,uncracked`` as the concrete reference.
            use_note_2: If ``True``, apply EC2 6.2.3(3) Note 2 variants for
                ``nu_1`` and reinforcement yield stress assumptions.
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure: Plotly figure instance for further
            customization or export.

        Raises:
            ValueError: If the ``ShearCheck`` has no shear reinforcement.
            TypeError: If ``load_case`` is not a ``ShearLoadCase`` or compatible dict.
            ImportError: If Plotly is not installed.
        """
        from materials.reinforced_concrete.analysis.shear_viewer import ShearViewer
        return ShearViewer(self).plot_cot_theta_study(
            load_case=load_case,
            n_points=n_points,
            cot_theta_min=cot_theta_min,
            cot_theta_max=cot_theta_max,
            use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
            use_note_2=use_note_2,
            save_path=save_path,
            show=show,
            title=title,
            width=width,
            height=height
        )

    def plot_cot_theta_moment_shift_study(
        self,
        *,
        load_case: ShearLoadCase,
        **kwargs,
    ) -> Any:
        """
        Plot utilization and tension-shift add-on versus cot(theta).

        This is a convenience wrapper around
        ``materials.reinforced_concrete.analysis.shear_viewer.ShearViewer``.

        Args:
            load_case: Design load case (``V_Ed``, ``M_Ed``, ``N_Ed``) used for
                the sweep.
            **kwargs: Additional plotting parameters forwarded to
                ``ShearViewer.plot_cot_theta_moment_shift_study`` (for example:
                ``n_points``, ``cot_theta_min``, ``cot_theta_max``,
                ``use_uncracked_V_Rd_c``, ``use_note_2``, ``save_path``,
                ``show``, ``title``, ``width``, ``height``).

        Returns:
            plotly.graph_objects.Figure: Plotly figure generated by the viewer.

        Raises:
            ValueError: If required shear reinforcement is missing.
            ImportError: If Plotly is not installed.
            TypeError: Propagated from the viewer for invalid inputs.
        """
        from materials.reinforced_concrete.analysis.shear_viewer import ShearViewer
        return ShearViewer(self).plot_cot_theta_moment_shift_study(load_case=load_case, **kwargs)

    def plot_link_angle_study(
        self,
        *,
        load_case: ShearLoadCase,
        **kwargs,
    ) -> Any:
        """
        Plot shear-capacity components over a link-angle sweep.

        This is a convenience wrapper around
        ``materials.reinforced_concrete.analysis.shear_viewer.ShearViewer``.

        Args:
            load_case: Design load case (``V_Ed``, ``M_Ed``, ``N_Ed``) used for
                the sweep.
            **kwargs: Additional plotting parameters forwarded to
                ``ShearViewer.plot_link_angle_study`` (for example:
                ``cot_theta_min``, ``cot_theta_max``, ``n_cot``,
                ``angle_min``, ``angle_max``, ``n_points``,
                ``use_uncracked_V_Rd_c``, ``use_note_2``, ``save_path``,
                ``show``, ``title``, ``width``, ``height``).

        Returns:
            plotly.graph_objects.Figure: Plotly figure generated by the viewer.

        Raises:
            ValueError: If required shear reinforcement is missing.
            ImportError: If Plotly is not installed.
            TypeError: Propagated from the viewer for invalid inputs.
        """
        from materials.reinforced_concrete.analysis.shear_viewer import ShearViewer
        return ShearViewer(self).plot_link_angle_study(load_case=load_case, **kwargs)

    def plot_link_angle_moment_shift_study(
        self,
        *,
        load_case: ShearLoadCase,
        **kwargs,
    ) -> Any:
        """
        Plot utilization and tension-shift add-on versus link angle.

        This is a convenience wrapper around
        ``materials.reinforced_concrete.analysis.shear_viewer.ShearViewer``.

        Args:
            load_case: Design load case (``V_Ed``, ``M_Ed``, ``N_Ed``) used for
                the sweep.
            **kwargs: Additional plotting parameters forwarded to
                ``ShearViewer.plot_link_angle_moment_shift_study`` (for example:
                ``cot_theta_min``, ``cot_theta_max``, ``n_cot``,
                ``angle_min``, ``angle_max``, ``n_points``,
                ``use_uncracked_V_Rd_c``, ``use_note_2``, ``save_path``,
                ``show``, ``title``, ``width``, ``height``).

        Returns:
            plotly.graph_objects.Figure: Plotly figure generated by the viewer.

        Raises:
            ValueError: If required shear reinforcement is missing.
            ImportError: If Plotly is not installed.
            TypeError: Propagated from the viewer for invalid inputs.
        """
        from materials.reinforced_concrete.analysis.shear_viewer import ShearViewer
        return ShearViewer(self).plot_link_angle_moment_shift_study(load_case=load_case, **kwargs)

    def plot_cot_theta_link_angle_heatmap(
        self,
        *,
        load_case: ShearLoadCase,
        **kwargs,
    ) -> Any:
        """
        Plot a cot(theta)-vs-link-angle heatmap for shear metrics.

        This is a convenience wrapper around
        ``materials.reinforced_concrete.analysis.shear_viewer.ShearViewer``.

        metric: Response quantity on the color axis. Supported values are:
                ``"utilization"``, ``"capacity"``, ``"v_rd_s"``, and ``"v_rd_max"``.

        Args:
            load_case: Design load case (``V_Ed``, ``M_Ed``, ``N_Ed``) used for
                the heatmap.
            **kwargs: Additional plotting parameters forwarded to
                ``ShearViewer.plot_cot_theta_link_angle_heatmap`` (for example:
                ``cot_theta_min``, ``cot_theta_max``, ``angle_min``, ``angle_max``,
                ``n_cot``, ``n_angles``, ``metric``, ``use_uncracked_V_Rd_c``,
                ``use_note_2``, ``save_path``, ``show``, ``title``, ``width``,
                ``height``).

        Returns:
            plotly.graph_objects.Figure: Plotly heatmap generated by the viewer.

        Raises:
            ValueError: If required shear reinforcement is missing, or metric is invalid.
            ImportError: If Plotly is not installed.
            TypeError: Propagated from the viewer for invalid inputs.
        """
        from materials.reinforced_concrete.analysis.shear_viewer import ShearViewer
        return ShearViewer(self).plot_cot_theta_link_angle_heatmap(load_case=load_case, **kwargs)

    def plot_force_cot_theta_contour(
        self,
        *,
        load_case: ShearLoadCase,
        **kwargs,
    ) -> Any:
        """
        Plot a cot(theta)-vs-force heatmap with a slider for the other force.

        Depending on ``moment_on_y_axis``, the y-axis is either ``M_Ed`` or
        ``N_Ed``. The slider controls the other quantity.

        Args:
            load_case: Base shear load case (``V_Ed`` is kept fixed). Can be
                ``ShearLoadCase`` or a ``dict`` with keys ``V_Ed`` and optional
                ``M_Ed``/``N_Ed``.
            n_min: Minimum axial force ``N_Ed`` in kN.
            n_max: Maximum axial force ``N_Ed`` in kN.
            m_min: Minimum moment ``M_Ed`` in kN·m.
            m_max: Maximum moment ``M_Ed`` in kN·m.
            n_axial: Number of axial-force samples.
            n_moment: Number of moment samples.
            moment_on_y_axis: If ``True``, the y-axis is moment and the slider
                controls axial force. If ``False``, the y-axis is axial force and
                the slider controls moment.
            cot_theta_min: Optional lower bound for cot(theta). If ``None``,
                the EC2-based minimum from the current check context is used.
            cot_theta_max: Optional upper bound for cot(theta). If ``None``,
                the EC2-based maximum from the current check context is used.
            n_cot: Number of cot(theta) samples.
            metric: Response quantity on the color axis. Supported values are:
                ``"utilization"``, ``"capacity"``, ``"v_rd_s"``, and ``"v_rd_max"``.
            use_uncracked_V_Rd_c: If ``True``, use uncracked concrete shear capacity
                ``V_Rd,c,uncracked`` when forming governing capacity/utilization.
            use_note_2: If ``True``, apply EC2 6.2.3(3) Note 2 variants for
                ``nu_1`` and reinforcement yield stress assumptions.
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure: Plotly heatmap figure.
        """
        from materials.reinforced_concrete.analysis.shear_viewer import ShearViewer
        return ShearViewer(self).plot_force_cot_theta_contour(load_case=load_case, **kwargs)


    # ===========================
    # Utility methods
    # ===========================

    def get_required_shear_reinforcement(
        self,
        V_Ed: float,
        M_Ed: float,
        N_Ed: float,
        cot_theta: float = 2.5,
        f_ywd: Optional[float] = None,
    ) -> float:
        """
        Calculate required A_sw/s for given shear force.

        Args:
            V_Ed: Design shear force in kN
            M_Ed: Design moment in kN·m
            N_Ed: Design axial force in kN
            cot_theta: Cotangent of strut angle
            f_ywd: Design yield strength (uses rebar f_yd if None)

        Returns:
            Required A_sw/s in mm²/mm
        """
        # Treat shear as magnitude (absolute value)
        # Negative shear from FEA sign conventions should not give negative reinforcement
        V_Ed = abs(V_Ed)

        d = self.find_effective_depth(M_Ed, N_Ed)
        sigma_cp = self._find_sigma_cp(N_Ed)
        rho_l = self._find_rho_l(M_Ed, N_Ed, d)

        V_Rd_c = self.find_V_Rd_c(d, rho_l, sigma_cp)

        if V_Ed <= V_Rd_c:
            return 0.0

        z_ec2, _ = self.find_lever_arm(M_Ed, N_Ed, d)

        if f_ywd is None and self.shear_reinforcement is not None:
            f_ywd = self.f_ywd_design
        elif f_ywd is None:
            if self.use_accidental:
                f_ywd = ShearRebar.f_yd_accidental_for()
            else:
                f_ywd = ShearRebar.f_yd_for()

        # Get cot(theta) limits for this load case
        cot_min, cot_max = self._find_cot_theta_limits(sigma_cp, z_ec2, V_Ed)

        cot_theta_limited = clamp_cot_theta(
            cot_theta=cot_theta,
            cot_min=cot_min,
            cot_max=cot_max
        )
        
        A_sw_over_s_min = self._find_min_a_sw_over_s()
        A_sw_over_s = from_kn(V_Ed, ForceUnit.N) / (z_ec2 * f_ywd * cot_theta_limited)
        return max(A_sw_over_s, A_sw_over_s_min)


    def _find_min_a_sw_over_s(self, use_defaults: bool = False) -> float:
        '''Minimum shear reinforcement area per unit length times sin α (§9.2.2(5)).
        
        Args:
            use_defaults: If True, assumes vertical links and grade 500 (α=90°, f_yk=500 MPa). 
                          If False, uses angle and f_yk from provided shear reinforcement.
                          To be used if shear reinforcement is not provided.

        Returns:
            A_sw / s in mm²/mm
        '''
        if use_defaults:
            f_yk = ShearRebar.f_yk_for()
            link_angle_deg = 90.0  # Default vertical links
        elif self.shear_reinforcement is None:
            raise ValueError("Shear reinforcement must be provided to compute minimum shear reinforcement.")
        else:
            f_yk = self.shear_reinforcement.f_yk
            link_angle_deg = self.shear_reinforcement.angle

        rho_w_min = find_minimum_ratio_of_shear_reinforcement(self.concrete.f_ck, f_yk, self.concrete.f_ctm)
        reinforcement_angle_rads = radians(link_angle_deg)
        return rho_w_min * self.breadth * sin(reinforcement_angle_rads)

