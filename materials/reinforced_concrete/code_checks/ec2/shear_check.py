"""
Shear check using codified EC2 stress approach.

This is a CODIFIED check with business logic - uses EC2 formulas directly
rather than first principles. Implements §6.2 Variable Strut Inclination Method.

**BREAKING CHANGE (v2.0):** N_Ed, M_Ed, and V_Ed are now parameters to perform_check(),
not fields. This enables checking multiple load cases against the same section efficiently.
"""

from typing import Optional, ClassVar
from math import sqrt, atan, degrees
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, computed_field

from materials.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
from materials.reinforced_concrete.analysis.interaction_diagram import (
    MNInteractionDiagram,
    ConcreteModelType,
    SteelBranchType,
)


class ShearLoadCase(BaseModel):
    """
    Single shear load case for checking.

    Attributes:
        V_Ed: Design shear force in kN
        M_Ed: Design moment in kN·m (optional, defaults to 0.0)
              - In rigorous mode: used for accurate NA and lever arm via M-N solver
              - In approximate mode: if non-zero, used to determine compression face (still uses z=0.9d)
        N_Ed: Design axial force in kN (compression positive, default 0)
    """
    V_Ed: float = Field(..., description="Design shear force in kN")
    M_Ed: float = Field(default=0.0, description="Design moment in kN·m (optional for simple checks)")
    N_Ed: float = Field(default=0.0, description="Design axial force in kN (compression positive)")


class ShearCheck(BaseCodeCheck):
    """
    EC2 shear check using Variable Strut Inclination Method (§6.2).

    **NEW API (v2.0):** Supports two modes:
    - **Rigorous mode** (use_rigorous=True, default): Uses M-N interaction solver for
      accurate neutral axis, compression face detection, and lever arm computation from
      force resultant centroids. Most accurate. Initialization: ~100ms.
    - **Approximate mode** (use_rigorous=False): Uses M-N solver only for compression
      face detection (when M_Ed or N_Ed provided), but always uses z=0.9d for lever arm.
      Faster check time but less accurate lever arm for eccentric loading.

    **BREAKING CHANGE:** N_Ed, M_Ed, V_Ed are now parameters to perform_check(), not fields.
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
        concrete_model_type: Concrete stress-strain model (for rigorous mode)
        steel_branch_type: Steel stress-strain branch (for rigorous mode)

    Example (new API):
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

    # ==================================
    # Material models for rigorous mode
    # ==================================

    concrete_model_type: ConcreteModelType = Field(
        default="parabola-rectangle",
        description="Concrete stress-strain model type (used if use_rigorous=True)",
    )

    steel_branch_type: SteelBranchType = Field(
        default="inclined",
        description="Steel stress-strain branch type (used if use_rigorous=True)",
    )

    # ===========================
    # Internal state (private)
    # ===========================

    _diagram: Optional[MNInteractionDiagram] = None

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
            steel_branch_type=self.steel_branch_type,
            use_characteristic=False,  # Use design strengths
            use_accidental=self.use_accidental,  # Match limit state
        )

    # ===========================
    # Properties (immutable - don't depend on loads)
    # ===========================

    @computed_field
    @property
    def breadth(self) -> float:
        """Web breadth in mm."""
        # TODO FOR WEB BEAMS FIND THE BREADTH USED FOR SHEAR AREA
        # For simple sections, use width. For T-beams, would use web width
        return self.section.outline.bounds[2] - self.section.outline.bounds[0]

    @property
    def f_cd_design(self) -> float:
        """Design concrete strength (accidental or persistent) in MPa."""
        return self.concrete.f_cd_accidental if self.use_accidental else self.concrete.f_cd

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
    ) -> float:
        """
        Effective depth from compression face.

        Both modes use M-N strain solver to determine compression face when M_Ed or N_Ed provided.
        This accounts for M-N interaction (e.g., large N_Ed can overpower small hogging moment).

        Fallback: If M_Ed and N_Ed both zero (or solver unavailable), assumes top compression.

        Args:
            M_Ed: Design moment in kN·m
            N_Ed: Design axial force in kN
            eps_top: Pre-computed top strain (optional, avoids re-solving)
            eps_bottom: Pre-computed bottom strain (optional, avoids re-solving)

        Returns:
            Effective depth in mm
        """
        # If solver unavailable, fall back to default assumption
        if self._diagram is None:
            return self.section.get_effective_depth(compression_face="top")

        # If M_Ed is negligible (pure shear or pure axial), use default (top compression)
        # No need to solve - moment determines which face is compressed, not axial load
        if abs(M_Ed) < 1e-6:
            return self.section.get_effective_depth(compression_face="top")

        # Use M-N solver to determine compression face (both modes)
        # Only needed when M_Ed is non-zero (accounts for M-N interaction)
        if eps_top is None or eps_bottom is None:
            eps_top, eps_bottom = self._diagram.find_strains_for_MN(M_Ed, N_Ed)

        # Compression strain is POSITIVE in interaction diagram sign convention
        # Larger positive strain = more compressed
        # Use >= to handle pure axial case (eps_top == eps_bottom) deterministically
        compression_face = "top" if eps_top >= eps_bottom else "bottom"

        return self.section.get_effective_depth(compression_face=compression_face)

    def find_sigma_cp(self, N_Ed: float) -> float:
        """
        Compressive stress in concrete due to axial force (§6.2.2(1)).

        σ_cp = N_Ed / A_c,transformed, limited to 0.2·f_cd

        Uses transformed area (concrete + n·steel) for more accurate stress calculation.

        Args:
            N_Ed: Design axial force in kN (compression positive)

        Returns:
            Stress in MPa
        """
        if N_Ed <= 0:
            return 0.0

        # Transformed area: A_c,tr = A_concrete + n·A_steel
        A_concrete = self.section.get_area()  # mm² (gross concrete area)

        # Calculate weighted average E_s for all steel
        total_steel_stiffness = 0.0
        total_steel_area = 0.0

        for group in self.section.rebar_groups:
            group_area = len(group.positions) * group.rebar.area
            group_E_s = group.rebar.E_s
            total_steel_stiffness += group_area * group_E_s
            total_steel_area += group_area

        if total_steel_area > 0:
            E_s_avg = total_steel_stiffness / total_steel_area
            E_c = self.concrete.E_cm
            n = E_s_avg / E_c
            A_c_transformed = A_concrete + n * total_steel_area
        else:
            A_c_transformed = A_concrete

        sigma_cp_uncapped = (N_Ed * 1000) / A_c_transformed
        return min(sigma_cp_uncapped, 0.2 * self.f_cd_design)

    def find_lever_arm(
        self,
        M_Ed: float,
        N_Ed: float,
        d: float,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
    ) -> float:
        """
        Lever arm for this load case.

        If use_rigorous=True: computes from force resultant centroids
        If use_rigorous=False: uses 0.9d approximation

        Args:
            M_Ed: Design moment in kN·m
            N_Ed: Design axial force in kN
            d: Effective depth in mm
            eps_top: Pre-computed top strain (optional, avoids re-solving)
            eps_bottom: Pre-computed bottom strain (optional, avoids re-solving)

        Returns:
            Lever arm in mm
        """
        if not self.use_rigorous or self._diagram is None:
            return 0.9 * d  # EC2 approximation

        # Rigorous: compute from strain state
        if eps_top is None or eps_bottom is None:
            eps_top, eps_bottom = self._diagram.find_strains_for_MN(M_Ed, N_Ed)
        return self._compute_lever_arm_from_strains(eps_top, eps_bottom, d)

    def find_rho_l(
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
            # Approximate: centroid-based
            centroid_y = self.section.get_centroid()[1]

            # Assume bottom in tension
            A_sl = 0.0
            for group in self.section.rebar_groups:
                for pos in group.positions:
                    if pos.y < centroid_y:  # Below centroid = tension
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
            eps_top: Strain at top fiber
            eps_bottom: Strain at bottom fiber
            d: Effective depth in mm

        Returns:
            ρ_l (dimensionless)
        """
        y_top = float(self.section.outline.bounds[3])
        y_bot = float(self.section.outline.bounds[1])
        h = y_top - y_bot

        # Find neutral axis y-coordinate
        if abs(eps_top - eps_bottom) < 1e-18:
            # Pure axial - use centroid
            na_y = self.section.get_centroid()[1]
        else:
            # NA at zero strain: eps_bot + (eps_top - eps_bot) * (na_y - y_bot) / h = 0
            na_y = y_bot - h * (eps_bottom / (eps_top - eps_bottom))

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

    def _compute_lever_arm_from_strains(
        self,
        eps_top: float,
        eps_bottom: float,
        d: float
    ) -> float:
        """
        Compute lever arm from force resultant centroids.

        z = |y_T - y_C| where y_T, y_C are centroids of tension/compression forces

        Args:
            eps_top: Strain at top fiber
            eps_bottom: Strain at bottom fiber
            d: Effective depth in mm (already computed with correct compression face)

        Returns:
            Lever arm in mm
        """
        # This method should only be called when diagram exists (rigorous mode)
        if self._diagram is None:
            raise RuntimeError("_compute_lever_arm_from_strains called without diagram")

        # Get fiber data (order: x, y, area, material_type, material_index)
        _, y_coords, areas, mat_type, mat_idx = self._diagram.mesh.get_fiber_arrays()

        # Compute strains at all fibers
        y_bot = float(self.section.outline.bounds[1])
        y_top = float(self.section.outline.bounds[3])
        h = y_top - y_bot
        strains = eps_bottom + (eps_top - eps_bottom) * (y_coords - y_bot) / h

        # Compute stresses
        stresses = np.zeros_like(strains)

        # Concrete fibers
        conc_mask = mat_type == "concrete"
        if np.any(conc_mask):
            stresses[conc_mask] = self._diagram._concrete_stress_with_options(strains[conc_mask])

        # Steel fibers
        steel_mask = mat_type == "steel"
        if np.any(steel_mask):
            for gi, sm in enumerate(self._diagram.steel_models):
                m = (mat_idx == gi) & steel_mask
                if np.any(m):
                    stresses[m] = sm.get_stress_array(strains[m])

        # Forces per fiber (compression positive)
        forces = stresses * areas

        # Separate tension and compression
        tension_mask = forces < 0
        compression_mask = forces > 0

        # Compute centroids
        if np.any(tension_mask):
            T_total = np.sum(-forces[tension_mask])  # Make positive
            y_T = np.sum(-forces[tension_mask] * y_coords[tension_mask]) / T_total
        else:
            y_T = y_bot

        if np.any(compression_mask):
            C_total = np.sum(forces[compression_mask])
            y_C = np.sum(forces[compression_mask] * y_coords[compression_mask]) / C_total
        else:
            y_C = y_top

        lever_arm = abs(y_T - y_C)

        # Fallback: if lever arm is too small (numerical issue or pure axial),
        # use 0.9d approximation
        if lever_arm < 1.0:  # Less than 1mm is suspicious
            # Use the already-computed d (which has correct compression face)
            lever_arm = 0.9 * d

        return lever_arm

    # ===========================
    # EC2 calculation methods
    # ===========================

    def find_k_factor(self, d: float) -> float:
        """
        Size effect factor (§6.2.2(1)).

        k = 1 + √(200/d) ≤ 2.0

        Args:
            d: Effective depth in mm

        Returns:
            k factor (dimensionless)
        """
        if d <= 0:
            raise ValueError(f"Effective depth must be > 0, got {d} mm")
        return min(2.0, 1.0 + sqrt(200 / d))

    def find_v_min(self, d: float) -> float:
        """
        Minimum shear strength coefficient (§6.2.2(1), Eq. 6.3N).

        v_min = 0.035·k^(3/2)·√f_ck

        Args:
            d: Effective depth in mm

        Returns:
            v_min in MPa
        """
        k = self.find_k_factor(d)
        f_ck = self.concrete.f_ck
        return 0.035 * (k ** 1.5) * sqrt(f_ck)

    def find_nu_factor(self) -> float:
        """
        Strength reduction factor for concrete cracked in shear (§6.2.2(6), Eq. 6.6N).

        ν = 0.6·(1 - f_ck/250)

        Returns:
            ν factor (dimensionless)
        """
        f_ck = self.concrete.f_ck
        return 0.6 * (1 - f_ck / 250)

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
        k = self.find_k_factor(d)
        f_ck = self.concrete.f_ck
        k_1 = 0.15
        b_w = self.breadth

        # Main formula (Eq. 6.2a)
        V_Rd_c = (C_Rd_c * k * ((100 * rho_l * f_ck) ** (1/3)) + k_1 * sigma_cp) * b_w * d

        # Minimum value (Eq. 6.2b)
        v_min = self.find_v_min(d)
        V_Rd_c_min = (v_min + k_1 * sigma_cp) * b_w * d

        return max(V_Rd_c, V_Rd_c_min) / 1000  # Convert to kN

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
            return 0.0

        A_sw_over_s = self.shear_reinforcement.area_per_unit_length
        f_ywd = self.f_ywd_design

        V_Rd_s = A_sw_over_s * z * f_ywd * cot_theta
        return V_Rd_s / 1000

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
        f_cd = self.f_cd_design

        # Coefficient α_cw (§6.2.3(3))
        if sigma_cp == 0:
            alpha_cw = 1.0
        elif sigma_cp <= 0.25 * f_cd:
            alpha_cw = 1.0 + sigma_cp / f_cd
        elif sigma_cp <= 0.5 * f_cd:
            alpha_cw = 1.25
        else:
            alpha_cw = 2.5 * (1 - sigma_cp / f_cd)

        b_w = self.breadth
        nu = self.find_nu_factor()
        tan_theta = 1 / cot_theta

        V_Rd_max = (alpha_cw * b_w * z * nu * f_cd) / (cot_theta + tan_theta)
        return V_Rd_max / 1000

    # ===========================
    # Main check method
    # ===========================

    def perform_check(
        self,
        *,
        load_case: ShearLoadCase,
        cot_theta: float = 2.5,
        warning_threshold: float = 0.95,
        **kwargs,
    ) -> CheckResult:
        """
        Perform shear check per EC2 §6.2 for a single load case.

        **NEW API:** Forces (V_Ed, M_Ed, N_Ed) are now parameters via ShearLoadCase,
        not fields. This enables efficient checking of multiple load cases against
        the same section.

        Checks:
        1. V_Ed ≤ V_Rd,c (if no shear reinforcement)
        2. V_Ed ≤ V_Rd,s (if shear reinforcement provided)
        3. V_Ed ≤ V_Rd,max (crushing of struts)

        Args:
            load_case: Single load case to check
            cot_theta: Cotangent of strut angle (1.0 to 2.5, default 2.5)
            warning_threshold: Utilization threshold for warning

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
            cot_theta=cot_theta,
            warning_threshold=warning_threshold,
        )

    def _check_single_case(
        self,
        V_Ed: float,
        M_Ed: float,
        N_Ed: float,
        cot_theta: float,
        warning_threshold: float,
    ) -> CheckResult:
        """Perform check for single load case (internal)."""
        # Validate and clamp cot_theta
        if cot_theta <= 0:
            raise ValueError(f"cot_theta must be > 0, got {cot_theta}")
        cot_theta_used = max(self.MIN_COT_THETA, min(self.MAX_COT_THETA, cot_theta))

        # Solve for strains once (both modes need for compression face detection)
        # This avoids redundant solves and ensures consistency
        # Only solve if M_Ed is non-zero (moment determines compression face, not axial load)
        if self._diagram is not None and abs(M_Ed) > 1e-6:
            eps_top, eps_bottom = self._diagram.find_strains_for_MN(M_Ed, N_Ed)
        else:
            eps_top, eps_bottom = None, None

        # Compute load-dependent geometric parameters (pass strains to avoid re-solving)
        d = self.find_effective_depth(M_Ed, N_Ed, eps_top, eps_bottom)
        z = self.find_lever_arm(M_Ed, N_Ed, d, eps_top, eps_bottom)
        sigma_cp = self.find_sigma_cp(N_Ed)
        rho_l = self.find_rho_l(M_Ed, N_Ed, d, eps_top, eps_bottom)

        # Compute capacities
        V_Rd_c = self.find_V_Rd_c(d, rho_l, sigma_cp)
        V_Rd_s = self.find_V_Rd_s(cot_theta_used, z)
        V_Rd_max = self.find_V_Rd_max(cot_theta_used, z, sigma_cp)

        # Determine governing capacity
        if self.shear_reinforcement is None:
            V_Rd = V_Rd_c
            governing_mode = "concrete (no shear reinforcement)"
            code_ref = "EC2 §6.2.2"
        else:
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
            "cot_theta": cot_theta_used,
            "theta_deg": degrees(atan(1 / cot_theta_used)),
            "section_name": self.section.section_name or "unnamed",
            "d": d,
            "z": z,
            "b_w": self.breadth,
            "rho_l": rho_l,
            "sigma_cp": sigma_cp,
            "mode": "rigorous" if self.use_rigorous else "approximate",
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
        d = self.find_effective_depth(M_Ed, N_Ed)
        sigma_cp = self.find_sigma_cp(N_Ed)
        rho_l = self.find_rho_l(M_Ed, N_Ed, d)

        V_Rd_c = self.find_V_Rd_c(d, rho_l, sigma_cp)

        if V_Ed <= V_Rd_c:
            return 0.0

        z = self.find_lever_arm(M_Ed, N_Ed, d)

        if f_ywd is None and self.shear_reinforcement is not None:
            f_ywd = self.f_ywd_design
        elif f_ywd is None:
            if self.use_accidental:
                f_ywd = 500.0
            else:
                f_ywd = 434.8

        cot_theta_limited = max(self.MIN_COT_THETA, min(self.MAX_COT_THETA, cot_theta))

        A_sw_over_s = (V_Ed * 1000) / (z * f_ywd * cot_theta_limited)
        return A_sw_over_s
