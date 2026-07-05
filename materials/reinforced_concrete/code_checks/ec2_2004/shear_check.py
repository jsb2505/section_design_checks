"""
Shear check using codified EC2 stress approach.

This is a CODIFIED check with business logic - uses EC2 formulas directly
rather than first principles. Implements §6.2 Variable Strut Inclination Method.

N_Ed, M_Ed, and V_Ed are now parameters to perform_check(),not fields.
This enables checking multiple load cases against the same section efficiently.
"""

from functools import cached_property
from typing import Optional, ClassVar
from math import atan, degrees, radians, sin, sqrt
import warnings
from pydantic import BaseModel, Field, PrivateAttr

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
    find_cot_theta_for_V_Ed,
    find_alpha_cw,
    find_nu_factor,
    find_k_factor,
    find_v_min,
    sigma_cp_from_N_and_area,
    cap_sigma_cp_upper,
    clamp_cot_theta,
    find_minimum_ratio_of_shear_reinforcement
)
from materials.reinforced_concrete.analysis.interaction_diagram import (
    MNInteractionDiagram,
)


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
    - **Rigorous mode** (use_rigorous=True, default): Uses M-N interaction solver for
      accurate neutral axis, compression face detection, and lever arm computation from
      force resultant centroids. Most accurate. Initialization: ~100ms.
    - **Approximate mode** (use_rigorous=False): Uses M-N solver only for compression
      face detection (when M_Ed or N_Ed provided), but always uses z=0.9d for lever arm.
      Faster check time but less accurate lever arm for eccentric loading.

    N_Ed, M_Ed, V_Ed are  parameters to perform_check(), not fields.
    This allows efficiently checking many load cases against the same section.

    This is a CODIFIED approach with business logic:
    - Uses EC2 empirical formulas (§6.2.2, §6.2.3)
    - Concrete shear resistance V_Rd,c (Eq. 6.2)
    - Shear reinforcement resistance V_Rd,s (Eq. 6.8)
    - Compression strut resistance V_Rd,max (Eq. 6.9, 6.14)
    - Variable strut angle 21.8° ≤ θ ≤ 45° (cot θ = 1.0 to 2.5)

    Attributes:
        section: RC section geometry
        concrete: Concrete material
        shear_reinforcement: Shear links/stirrups (optional)
        use_accidental: Use accidental limit state partial factors (default: False)
        use_rigorous: Use solver-based approach for NA and lever arm (default: True)
        cap_lever_arm: Cap lever arm to 0.9d per EC2 (default: True, rigorous mode only)
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
        >>> shear_rebar = ShearRebar(diameter=10, spacing=200, n_legs=2, grade="B500B")
        >>>
        >>> # Create check once (diagram created on init if use_rigorous=True)
        >>> check = ShearCheck(
        ...     section=section,
        ...     concrete=concrete,
        ...     shear_reinforcement=shear_rebar,
        ...     use_rigorous=True,  # Default - accurate NA and lever arm
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

    use_rigorous: bool = Field(
        default=True,
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

    cap_lever_arm: bool = Field(
        default=True,
        description=(
            "Cap computed lever arm to 0.9d per EC2 codified simplification (default: True). "
            "When True, lever arm z is limited to z <= 0.9d to match EC2 truss model assumptions. "
            "The uncapped mechanical lever arm z_mech is still stored in details for reference. "
            "Only affects rigorous mode - approximate mode always uses z=0.9d."
        ),
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


    # ==================================
    # Material models for rigorous mode
    # ==================================

    concrete_model_type: ConcreteModelType = Field(
        default=ConcreteModelType.PARABOLA_RECTANGLE,
        description="Concrete stress-strain model type (used if use_rigorous=True)",
    )

    steel_model_type: SteelModelType = Field(
        default=SteelModelType.INCLINED,
        description="Steel stress-strain branch type (used if use_rigorous=True)",
    )


    # =========================
    # Internal state (private)
    # =========================

    _diagram: MNInteractionDiagram = PrivateAttr()

    # Constants from EC2
    MIN_COT_THETA: ClassVar[float] = 1.0  # θ = 45°
    MAX_COT_THETA: ClassVar[float] = 2.5  # θ = 21.8°

    def model_post_init(self, __context):
        """
        Create M-N diagram for both modes (but don't generate curve points).

        Both rigorous and approximate modes need the diagram for compression face detection.
        The difference:
        - Rigorous: also computes accurate lever arm from force centroids
        - Approximate: only uses diagram for compression face, always z=0.9d
        """
        super().model_post_init(__context)

        # Create diagram (mesh + models) but DON'T generate curve points
        # This saves ~900ms initialization time - we only need the forward model
        # for the inverse solver
        self._diagram = MNInteractionDiagram(
            section=self.section,
            concrete=self.concrete,
            concrete_model_type=self.concrete_model_type,
            steel_model_type=self.steel_model_type,
            use_characteristic=False,  # Use design strengths
            use_accidental=self.use_accidental,  # Match limit state
        )

        # cached properties to save time later
        self._A_transformed = self.section.get_transformed_area(self.concrete.E_cm)  # mm²
        self._A_gross = self.section.get_area()  # mm²

    # ===============================================
    # Properties (immutable - don't depend on loads)
    # ===============================================

    @cached_property
    def breadth(self) -> float:
        """
        Minimum web breadth b_w for shear design (mm).

        If ``breadth_override`` is set, that value is used directly.
        Otherwise, computed automatically per EC2 §6.2 as the minimum width
        between tension and compression chords.
        """
        if self.breadth_override is not None:
            return self.breadth_override
        return calculate_section_breadth(self.section)

    @cached_property
    def f_cd_design(self) -> float:
        """Design concrete strength (accidental or persistent) in MPa."""
        return self.concrete.f_cd_accidental if self.use_accidental else self.concrete.f_cd

    @cached_property
    def gamma_c_design(self) -> float:
        """Partial factor for concrete (accidental or persistent)."""
        return self.concrete.gamma_c_accidental if self.use_accidental else self.concrete.gamma_c

    @cached_property
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
    ) -> float:
        """
        Effective depth d (mm) measured from the governing compression face.

        If strains are provided, compression face is taken as the face with the larger
        (more positive) strain (compression is positive in this codebase).

        If strains are not provided:
        - If a diagram/solver is available and |M_Ed| is significant, strains are solved.
        - Otherwise, fallback returns min(d_top, d_bottom) for conservatism (useful for shear).

        Notes:
        - If both faces are in tension (eps_top<=0 and eps_bottom<=0), compression face is
            physically undefined; fallback is used.
        """
        # Get effective depths for each compression face assumption
        # Handle case where no rebar exists in the tension zone for one face
        d_top: Optional[float] = None
        d_bot: Optional[float] = None

        try:
            d_top = float(self.section.get_effective_depth(compression_face="top"))
        except ValueError:
            pass  # No rebar in bottom tension zone

        try:
            d_bot = float(self.section.get_effective_depth(compression_face="bottom"))
        except ValueError:
            pass  # No rebar in top tension zone

        # If neither worked, we have a problem
        if d_top is None and d_bot is None:
            raise ValueError("Cannot compute effective depth: no rebars found in either tension zone")

        # Helper to get conservative depth (handles one being None)
        def _get_conservative_d() -> float:
            if d_top is not None and d_bot is not None:
                return min(d_top, d_bot)
            elif d_top is not None:
                return d_top
            else:
                assert d_bot is not None  # Can't be None, checked above
                return d_bot

        # Pure shear / pure axial / no clear bending => conservative depth
        if abs(M_Ed) <= m_tol:
            return _get_conservative_d()

        # If strains missing, try to solve if you can (robust helper)
        if (eps_top is None or eps_bottom is None) and self._diagram:
            try:
                eps_top, eps_bottom = self._diagram.find_strains_for_MN(M_Ed, N_Ed)
            except Exception:
                eps_top, eps_bottom = None, None

        # Still missing -> fallback conservative
        if eps_top is None or eps_bottom is None:
            if warn_on_fallback:
                warnings.warn(
                    "Effective depth fallback used (strain state unavailable). "
                    "Returning conservative min(d_top, d_bottom).",
                    stacklevel=2,
                )
            return _get_conservative_d()

        # If there is no compression anywhere, compression face is undefined -> fallback
        if eps_top <= strain_tol and eps_bottom <= strain_tol:
            if warn_on_fallback:
                warnings.warn(
                    "Effective depth fallback used (both faces in tension; compression face undefined). "
                    "Returning conservative min(d_top, d_bottom).",
                    stacklevel=2,
                )
            return _get_conservative_d()

        # Otherwise: choose the more compressive face (bigger + strain)
        compression_face = "top" if eps_top >= eps_bottom else "bottom"

        if compression_face == "top":
            if d_top is not None:
                return d_top
            # Compression at top but no rebar in bottom (tension) zone - use fallback
            if warn_on_fallback:
                warnings.warn(
                    "Effective depth fallback used (no rebar in tension zone for this compression face).",
                    stacklevel=2,
                )
            return _get_conservative_d()
        else:
            if d_bot is not None:
                return d_bot
            # Compression at bottom but no rebar in top (tension) zone - use fallback
            if warn_on_fallback:
                warnings.warn(
                    "Effective depth fallback used (no rebar in tension zone for this compression face).",
                    stacklevel=2,
                )
            return _get_conservative_d()


    def find_lever_arm(
        self,
        M_Ed: float,
        N_Ed: float,
        d: float,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
    ) -> tuple[float, Optional[float]]:
        """
        Lever arm for this load case.

        Behaviour:
            If use_rigorous=True: computes from force resultant centroids (with sensible fallback)
            If use_rigorous=False: uses 0.9d approximation

        Returns:
            (z_ec2, z_mech)
        """
        # No diagram available (or user opted out) => always use EC2 approx
        if (self._diagram is None) or (not self.use_rigorous):
            return (0.9 * d, None)

        # Delegate to MNInteractionDiagram; it should:
        # - compute z_mech if possible else None
        # - fallback to 0.9d when z_mech is None / suspicious
        # - optionally cap to 0.9d
        # - emit warnings when fallback/cap occurs
        return self._diagram.get_lever_arm(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            d=d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            prefer_rigorous=True,
            cap_to_09d=self.cap_lever_arm,
        )


    def _find_rho_l(
        self,
        M_Ed: float,
        N_Ed: float,
        d: float,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
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
        if not self.use_rigorous or self._diagram is None:
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
            eps_top, eps_bottom = self._diagram.find_strains_for_MN(M_Ed, N_Ed)
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
        y_top = float(self.section.outline.bounds[3])
        y_bot = float(self.section.outline.bounds[1])
        h = y_top - y_bot

        # Sum steel in tension (strain < 0, negative = tension)
        A_sl = 0.0
        for group in self.section.rebar_groups:
            for pos in group.positions:
                # Linear strain field
                strain_at_bar = eps_bottom + (eps_top - eps_bottom) * (pos.y - y_bot) / h
                if strain_at_bar < 0:  # Tension
                    A_sl += group.rebar.area

        if A_sl == 0:
            return 0.0

        b_w = self.breadth
        rho_l = A_sl / (b_w * d)
        return min(rho_l, 0.02)


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
        C_Rd_c = 0.18 / self.gamma_c_design
        k = find_k_factor(d)
        f_ck = self.concrete.f_ck
        k_1 = 0.15
        b_w = self.breadth

        # Main formula (Eq. 6.2a)
        V_Rd_c = (C_Rd_c * k * ((100 * rho_l * f_ck) ** (1/3)) + k_1 * sigma_cp) * b_w * d

        # Minimum value (Eq. 6.2b)
        v_min = find_v_min(f_ck, k)
        V_Rd_c_min = (v_min + k_1 * sigma_cp) * b_w * d
        # TODO use unit conversion here
        V_Rd_c_kN = max(V_Rd_c, V_Rd_c_min) / 1000  # Convert to kN

        return max(V_Rd_c_kN, 0)  # Prevents negative values if sigma_cp is large negative


    def find_V_Rd_s(self, cot_theta: float, z: float) -> float:
        """
        Shear resistance of shear reinforcement (§6.2.3(3), Eq. 6.8).

        Public method - takes computed parameters.

        Args:
            cot_theta: Cotangent of strut angle (pre-clamped)
            z: Lever arm in mm

        Returns:
            V_Rd,s in kN
        """
        if self.shear_reinforcement is None:
            raise ValueError("V_Rd_s cannot be found without providing shear reinforcement.")

        A_sw_over_s = self.shear_reinforcement.area_per_unit_length
        f_ywd = self.f_ywd_design
        link_angle_rads = radians(self.shear_reinforcement.angle)

        V_Rd_s = A_sw_over_s * z * f_ywd * (cot_theta + cot(link_angle_rads)) * sin(link_angle_rads)
        return V_Rd_s / 1000  # TODO use unit conversion here


    def find_V_Rd_max(self, cot_theta: float, z: float, sigma_cp: float) -> float:
        """
        Maximum shear resistance limited by crushing of compression struts (§6.2.3, Eq. 6.9).

        Public method - takes computed parameters.

        Args:
            cot_theta: Cotangent of strut angle (pre-clamped)
            z: Lever arm in mm
            sigma_cp: Compressive stress from axial force in MPa

        Returns:
            V_Rd,max in kN
        """
        if self.shear_reinforcement is None:
            raise ValueError("V_Rd_max cannot be found without providing shear reinforcement.")
        
        f_cd = self.f_cd_design
        alpha_cw = find_alpha_cw(f_cd, sigma_cp)

        b_w = self.breadth
        nu = find_nu_factor(self.concrete.f_ck)

        link_angle_rads = radians(self.shear_reinforcement.angle)
        V_Rd_max = (alpha_cw * b_w * z * nu * f_cd) * (cot_theta + cot(link_angle_rads)) / (1 + cot_theta**2)
        return V_Rd_max / 1000  # TODO use unit conversion here


    # ===========================
    # Main check method
    # ===========================

    def perform_check(
        self,
        *,
        load_case: ShearLoadCase,
        cot_theta_override: Optional[float] = None,
        warning_threshold: float = 0.95,
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
            cot_theta_override: User provided cot theta value to use

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
            warning_threshold=warning_threshold,
        )


    def _check_single_case(
        self,
        V_Ed: float,
        M_Ed: float,
        N_Ed: float,
        cot_theta_override: Optional[float],
        warning_threshold: float,
    ) -> CheckResult:
        """Perform check for single load case (internal)."""
        # Treat shear as magnitude (absolute value)
        # Negative shear from FEA sign conventions should not give negative utilization
        V_Ed = abs(V_Ed)

        # Solve for strains once (both modes need for compression face detection)
        # This avoids redundant solves and ensures consistency
        # Only solve if M_Ed is non-zero (moment determines compression face, not axial load)
        if self._diagram is not None and abs(M_Ed) > 1e-6:
            eps_top, eps_bottom = self._diagram.find_strains_for_MN(M_Ed, N_Ed)
        else:
            eps_top, eps_bottom = None, None

        # Compute load-dependent geometric parameters (pass strains to avoid re-solving)
        d = self.find_effective_depth(M_Ed, N_Ed, eps_top, eps_bottom)
        sigma_cp = self._find_sigma_cp(N_Ed)
        rho_l = self._find_rho_l(M_Ed, N_Ed, d, eps_top, eps_bottom)

        # 1. Initialize variables that might not be reached
        V_Rd_s: Optional[float] = None
        V_Rd_max: Optional[float] = None
        cot_theta: Optional[float] = None
        z_ec2: Optional[float] = None
        z_mech: Optional[float] = None

        # Compute capacities (use z_ec2 for design checks per EC2)
        V_Rd_c = self.find_V_Rd_c(d, rho_l, sigma_cp)

        reinforcement = self.shear_reinforcement

        if reinforcement:

            z_ec2, z_mech = self.find_lever_arm(M_Ed, N_Ed, d, eps_top, eps_bottom)

            if cot_theta_override is not None:
                cot_theta = cot_theta_override

                if cot_theta_override > self.MAX_COT_THETA:
                    warnings.warn(
                        f"Cot theta value provided, cot(θ) = {cot_theta_override}, is greater than stored max value: {self.MAX_COT_THETA}.",
                        stacklevel=2,
                    )
                elif cot_theta_override < self.MIN_COT_THETA:
                    warnings.warn(
                        f"Cot theta value provided, cot(θ) = {cot_theta_override}, is smaller than stored min value: {self.MIN_COT_THETA}.",
                        stacklevel=2,
                    )
            else:
                # Calculate cot_theta based on V_Ed = V_Rd,max
                f_cd = self.f_cd_design
                alpha_cw = find_alpha_cw(f_cd, sigma_cp)
                b_w = self.breadth
                nu = find_nu_factor(self.concrete.f_ck)

                K = alpha_cw * b_w * z_ec2 * nu * f_cd

                # already clamped by function
                cot_theta = find_cot_theta_for_V_Ed(
                    V_Ed=V_Ed,
                    K=K,
                    link_angle_degrees=reinforcement.angle,
                    cot_min=self.MIN_COT_THETA,
                    cot_max=self.MAX_COT_THETA,
                )

            V_Rd_s = self.find_V_Rd_s(cot_theta, z_ec2)

            # the maximum capacity of the concrete strut is found using the largest theta (smallest cot_theta)
            V_Rd_max = self.find_V_Rd_max(self.MIN_COT_THETA, z_ec2, sigma_cp)

        # Determine governing capacity
        if self.shear_reinforcement is None:
            V_Rd = V_Rd_c
            governing_mode = "concrete (no shear reinforcement)"
            code_ref = "EC2 §6.2.2"
        else:

            assert V_Rd_s is not None and V_Rd_max is not None

            if V_Ed > V_Rd_c:
                # Shear reinforcement is engaged
                V_Rd = min(V_Rd_s, V_Rd_max)

                if V_Rd_s < V_Rd_max:
                    governing_mode = "shear reinforcement"
                    code_ref = "EC2 §6.2.3 (Eq. 6.8)"
                else:
                    governing_mode = "compression strut"
                    code_ref = "EC2 §6.2.3 (Eq. 6.9)"
            else:
                V_Rd = min(V_Rd_c, V_Rd_max)
                if V_Rd_max < V_Rd_c:
                    governing_mode = "compression strut"
                    code_ref = "EC2 §6.2.3 (Eq. 6.9)"
                else:
                    governing_mode = "concrete"
                    code_ref = "EC2 §6.2.2"

        # Create message
        utilization = V_Ed / V_Rd if V_Rd > 0 else float('inf')

        if utilization <= 1.0:
            if utilization >= warning_threshold:
                message = f"High shear utilization - governed by {governing_mode}"
            else:
                message = f"Shear capacity adequate - governed by {governing_mode}"
        else:
            if self.shear_reinforcement is None:
                message = "Shear capacity exceeded - provide shear reinforcement"
            elif governing_mode == "compression strut":
                message = "Compression strut capacity exceeded - increase section size"
            else:
                message = "Shear reinforcement capacity exceeded - reduce spacing or increase diameter"

        # Details
        details = {
            "V_Ed": V_Ed,
            "M_Ed": M_Ed,
            "N_Ed": N_Ed,
            "V_Rd": V_Rd,
            "V_Rd_c": V_Rd_c,
            "V_Rd_s": V_Rd_s if self.shear_reinforcement else None,
            "V_Rd_max": V_Rd_max if self.shear_reinforcement else None,
            "governing_mode": governing_mode,
            "cot_theta": cot_theta if self.shear_reinforcement else None,
            "theta_deg": degrees(atan(1 / cot_theta)) if cot_theta else None,
            "section_name": self.section.section_name or "unnamed",
            "d": d,
            "z": z_ec2 if self.shear_reinforcement else None,  # Lever arm used in EC2 check (capped if cap_lever_arm=True)
            "z_mech": z_mech if self.shear_reinforcement else None,  # Mechanical lever arm from force centroids (uncapped)
            "b_w": self.breadth,
            "rho_l": rho_l,
            "sigma_cp": sigma_cp,
            "z_mode": "rigorous" if self.use_rigorous else "approximate",
            "cap_lever_arm": self.cap_lever_arm,  # Document if capping was applied
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

        cot_theta_limited = clamp_cot_theta(
            cot_theta=cot_theta,
            cot_min=self.MIN_COT_THETA,
            cot_max=self.MAX_COT_THETA
        )
        
        A_sw_over_s_min = self._find_min_a_sw_over_s()
        # TODO use unit conversion here
        A_sw_over_s = (V_Ed * 1000) / (z_ec2 * f_ywd * cot_theta_limited)
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

        rho_w_min = find_minimum_ratio_of_shear_reinforcement(self.concrete.f_ck, f_yk)
        reinforcement_angle_rads = radians(link_angle_deg)
        return rho_w_min * self.breadth * sin(reinforcement_angle_rads)
