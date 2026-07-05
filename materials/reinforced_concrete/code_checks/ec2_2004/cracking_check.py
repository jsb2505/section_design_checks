"""
Cracking check for reinforced concrete sections according to EC2 §7.3.

This is a SERVICEABILITY check using characteristic material properties and
elastic/cracked section analysis to calculate crack widths.
"""

from math import exp
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional, Tuple, cast
import warnings

from pydantic import Field, PrivateAttr, computed_field

from materials.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from materials.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial
from materials.reinforced_concrete.analysis import create_interaction_diagram
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from materials.reinforced_concrete.code_checks.ec2_2004 import flexure_utils
from materials.core.units import ForceUnit, MomentUnit, from_kn, to_knm
from materials.reinforced_concrete.ndp import get_ndp


class LoadDuration(StrEnum):
    """
    Load duration for k_t factor in crack width calculation (EC2 §7.3.4(2)).
    
    Attributes:
        SHORT_TERM
        LONG_TERM
    """
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"

    @property
    def k_t(self) -> float:
        """Factor for load duration (EC2 §7.3.4(2)): 0.6 short-term, 0.4 long-term."""
        return {
            LoadDuration.SHORT_TERM: 0.6,
            LoadDuration.LONG_TERM: 0.4,
        }[self]


class SLSCombination(StrEnum):
    """
    SLS load combination type for stress limitation checks (EC2 §7.2).
    
    Attributes:
        CHARACTERISTIC
        FREQUENT
        QUASI_PERMANENT
    """
    CHARACTERISTIC = "characteristic"
    FREQUENT = "frequent"
    QUASI_PERMANENT = "quasi_permanent"


@dataclass
class CrackingResult:
    """Detailed results from crack width calculation."""
    w_k: float  # Calculated crack width (mm)
    w_k_limit: float  # Allowable crack width (mm)
    s_r_max: float  # Maximum crack spacing (mm)
    eps_sm_minus_eps_cm: float  # Difference in mean strains (dimensionless)
    sigma_s: float  # Steel stress in tension rebar (MPa)
    rho_p_eff: float  # Effective reinforcement ratio (dimensionless)
    h_c_ef: float  # Effective height of concrete in tension (mm)
    x: Optional[float]  # Neutral axis depth from compression face (mm)
    is_cracked: bool  # Whether section is cracked
    phi_eq: float  # Equivalent bar diameter (mm)
    cover: float  # Concrete cover to tension rebar (mm)
    sigma_c_peak: float = 0.0  # Peak concrete compressive stress (MPa)
    nonlinear_creep_applied: bool = False  # Whether non-linear creep adjustment was applied
    creep_coefficient_used: float = 0.0  # Actual creep coefficient used (may be φ_NL)
    steel_yielded: bool = False  # Whether σ_s > f_yk (EC2 §7.2(4)P inelastic strain)


class CrackingCheck(BaseCodeCheck):
    """
    EC2 2004 cracking check for reinforced concrete sections (§7.3).

    Calculates crack widths using EC2 formula (Eq. 7.8):
        w_k = s_r,max × (ε_sm - ε_cm)

    The check process:
    1. Determine if section is cracked (compare M_Ed to cracking moment)
    2. If cracked, solve for strain state using M-N interaction diagram
    3. Calculate steel stress from strain state
    4. Calculate h_c,ef, ρ_p,eff, and s_r,max
    5. Calculate crack width and compare to limit

    Attributes:
        section: RC section geometry with reinforcement
        concrete: Concrete material (characteristic properties for SLS)
        w_k_limit: Allowable crack width (default 0.3mm for XC2/XC3)
        load_duration: SHORT_TERM (k_t=0.6) or LONG_TERM (k_t=0.4)
        creep_coefficient:
            Linear creep coefficient φ. Modifies E_cm to Ec,eff.
            Set to 0.0 to use E_cm. (default 1.5)
        sls_combination: SLS load combination type (affects stress checks)
        apply_nonlinear_creep: Auto-adjust E_cm,eff when σ_c > k_2·f_ck

    Example:
        >>> from materials.reinforced_concrete.geometry import create_rectangular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial
        >>>
        >>> section = create_rectangular_section(width=300, height=500)
        >>> # ... add reinforcement ...
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>>
        >>> check = CrackingCheck(section=section, concrete=concrete)
        >>> result = check.perform_check(M_Ed=50.0, N_Ed=0.0)  # SLS moments in kN·m
        >>>
        >>> # With creep coefficient φ = 2.0:
        >>> check_lt = CrackingCheck(section=section, concrete=concrete, creep_coefficient=2.0)
    """

    section: RCSection = Field(
        ...,
        description="RC section with reinforcement",
    )

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material properties",
    )

    w_k_limit: float = Field(
        default=0.3,
        description="Allowable crack width in mm (EC2 Table 7.1N)",
        gt=0.0,
    )

    load_duration: LoadDuration = Field(
        default=LoadDuration.LONG_TERM,
        description="Load duration: SHORT_TERM (k_t=0.6) or LONG_TERM (k_t=0.4)",
    )

    concrete_model_type: ConcreteModelType = Field(
        default=ConcreteModelType.LINEAR_ELASTIC,
        description="EC2 concrete stress-strain model",
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

    is_high_bond_bar: bool = Field(
        default=True,
        description="True for ribbed bars (k_1=0.8), False for plain bars (k_1=1.6)",
    )

    creep_coefficient: float = Field(
        default=1.5,
        description="Linear creep coefficient φ for long-term SLS. "
                    "E_cm,eff = E_cm / (1 + φ). Default 1.5 for typical long-term loading."
                    "Set to 0.0 to use short-term E_cm.",
        ge=0.0,
    )

    sls_combination: SLSCombination = Field(
        default=SLSCombination.QUASI_PERMANENT,
        description="SLS load combination type. Affects stress limitation checks (EC2 §7.2).",
    )

    apply_nonlinear_creep: bool = Field(
        default=True,
        description="If True, automatically adjust E_cm,eff when σ_c > k_2·f_ck (EC2 §3.1.4(4)).",
    )

    iterate_nonlinear_creep: bool = Field(
        default=False,
        description="If True, iterate non-linear creep adjustment until convergence (max 5 iterations).",
    )

    # ===========================
    # Internal state (private)
    # ===========================

    _diagram: Optional[MNInteractionDiagram] = PrivateAttr(default=None)
    _diagram_no_comp_steel: Optional[MNInteractionDiagram] = PrivateAttr(default=None)
    _diagram_snapshot: Optional[dict] = PrivateAttr(default=None)
    _diagram_no_comp_snapshot: Optional[dict] = PrivateAttr(default=None)

    def _take_snapshot(self) -> dict:
        """Capture current state of inputs that affect the interaction diagram."""
        return {
            "section": self.section.model_dump(),
            "concrete": self.concrete.model_dump(),
            "concrete_model_type": self.concrete_model_type,
            "steel_model_type": self.steel_model_type,
            "n_fibres_width": self.n_fibres_width,
            "n_fibres_height": self.n_fibres_height,
            "E_cm_eff": self.E_cm_eff,
        }

    def _get_diagram(self, ignore_compression_steel: bool = False) -> MNInteractionDiagram:
        """Get the cached diagram, rebuilding if inputs have changed."""
        snapshot = self._take_snapshot()

        if ignore_compression_steel:
            if self._diagram_no_comp_steel is None or snapshot != self._diagram_no_comp_snapshot:
                self._diagram_no_comp_steel = create_interaction_diagram(
                    section=self.section,
                    concrete=self.concrete,
                    concrete_model_type=self.concrete_model_type,
                    steel_model_type=self.steel_model_type,
                    n_fibres_width=self.n_fibres_width,
                    n_fibres_height=self.n_fibres_height,
                    use_characteristic=True,
                    ignore_compression_steel=True,
                    elastic_modulus=self.E_cm_eff,
                )
                self._diagram_no_comp_snapshot = snapshot
            return self._diagram_no_comp_steel
        else:
            if self._diagram is None or snapshot != self._diagram_snapshot:
                self._diagram = create_interaction_diagram(
                    section=self.section,
                    concrete=self.concrete,
                    concrete_model_type=self.concrete_model_type,
                    steel_model_type=self.steel_model_type,
                    n_fibres_width=self.n_fibres_width,
                    n_fibres_height=self.n_fibres_height,
                    use_characteristic=True,
                    ignore_compression_steel=False,
                    elastic_modulus=self.E_cm_eff,
                )
                self._diagram_snapshot = snapshot
            return self._diagram

    # ===============================================
    # Properties (immutable - don't depend on loads)
    # ===============================================

    @computed_field
    @property
    def height(self) -> float:
        """Section height in mm."""
        return flexure_utils.calculate_section_height(self.section)

    @computed_field
    @property
    def breadth(self) -> float:
        """Section breadth (width) in mm."""
        return flexure_utils.calculate_section_breadth(self.section)

    @property
    def k_t(self) -> float:
        """Factor for load duration (EC2 §7.3.4(2))."""
        return self.load_duration.k_t

    @property
    def k_1(self) -> float:
        """Bond coefficient (EC2 §7.3.4(3)): 0.8 for high bond, 1.6 for plain."""
        return 0.8 if self.is_high_bond_bar else 1.6

    @property
    def k_3(self) -> float:
        """NDP coefficient k_3 for crack spacing (EC2 §7.3.4(3))."""
        return cast(float, get_ndp("k_3_crack"))

    @property
    def k_4(self) -> float:
        """NDP coefficient k_4 for crack spacing (EC2 §7.3.4(3))."""
        return cast(float, get_ndp("k_4_crack"))

    @property
    def k_1_stress(self) -> float:
        """
        NDP characteristic stress limit factor k_1 (EC2 §7.2(2)).

        Longitudinal cracking risk if σ_c > k_1·f_ck under characteristic loads.
        """
        return cast(float, get_ndp("k_1_stress"))

    @property
    def k_2_stress(self) -> float:
        """
        NDP quasi-permanent stress limit factor k_2 (EC2 §7.2(3)).

        Non-linear creep threshold if σ_c > k_2·f_ck under quasi-permanent loads.
        """
        return cast(float, get_ndp("k_2_stress"))

    @property
    def k_3_stress(self) -> float:
        """
        NDP reinforcement stress limit factor k_3 (EC2 §7.2(5)).

        Yield risk if σ_s > k_3·f_yk.
        """
        return cast(float, get_ndp("k_3_stress"))

    @property
    def effective_modulus_ratio(self) -> float:
        """Effective modulus ratio (1 + φ). Derived from creep_coefficient."""
        return 1.0 + self.creep_coefficient

    @property
    def E_cm_eff(self) -> float:
        """
        Effective concrete modulus accounting for creep (EC2 §7.4.3).

        E_cm,eff = E_cm / (1 + φ)

        Returns:
            Effective modulus in MPa
        """
        return self.concrete.get_elastic_modulus() / self.effective_modulus_ratio

    @property
    def alpha_e(self) -> float:
        """
        Modular ratio E_s / E_cm,eff (EC2 §7.3.4(2)).

        Uses area-weighted average E_s when multiple rebar groups have different
        elastic moduli. This is appropriate since alpha_e multiplies rho_p_eff
        (which sums tension steel areas).

        Returns:
            Modular ratio (dimensionless)
        """
        # TODO strictly speaking it is only the bars that are in tension
        # that should be considered in the weighting.
        # Can remove need for heavy compute by first checking if E_s in bars differ or not.
        if not self.section.rebar_groups:
            E_s = 200000.0  # Default E_s = 200 GPa
        else:
            # Area-weighted average E_s across all rebar groups
            total_area = 0.0
            weighted_E_s = 0.0
            for group in self.section.rebar_groups:
                group_area = group.rebar.area * len(group.positions)
                total_area += group_area
                weighted_E_s += group.rebar.E_s * group_area

            E_s = weighted_E_s / total_area if total_area > 0 else 200000.0

        return E_s / self.E_cm_eff

    # ===============================================
    # Cracking moment calculation
    # ===============================================

    def find_cracking_moment(self) -> float:
        """
        Cracking moment M_cr (kN·m) - moment at which section first cracks.

        M_cr = f_ctm,fl × W_el / 10^6

        where:
        - f_ctm,fl = mean flexural tensile strength (accounts for size effect)
        - W_el = elastic section modulus to tension face

        Returns:
            Cracking moment in kN·m
        """
        # Flexural tensile strength (EC2 §3.1.8)
        f_ctm_fl = self.concrete.find_mean_flexural_tensile_strength(self.height)

        # Elastic section modulus (uncracked transformed section)
        I_yy, _, _ = self.section.get_transformed_second_moment_area(self.E_cm_eff)
        _, c_y, _ = self.section.get_transformed_centroid(self.E_cm_eff)
        bounds = self.section.outline.bounds
        y_tension = c_y - bounds[1]  # Distance to bottom (tension) face

        W_el = I_yy / y_tension if y_tension > 0 else I_yy / (self.height / 2)

        # M_cr in kN·m (W_el in mm³, f_ctm_fl in MPa → result in N·mm)
        return to_knm(f_ctm_fl * W_el, MomentUnit.NMM)

    # ===============================================
    # h_c,ef calculation (EC2 §7.3.2(3), Fig 7.1)
    # ===============================================

    def find_h_c_ef(
        self,
        d: float,
        x: Optional[float] = None,
    ) -> float:
        """
        Effective height of concrete in tension zone h_c,ef (EC2 §7.3.2(3), Fig 7.1).

        h_c,ef = min(2.5(h-d), (h-x)/3, h/2)

        where:
        - h = section depth
        - d = effective depth to tension steel
        - x = neutral axis depth from compression face

        Note: For sections fully in tension (both faces), use find_h_c_ef_tension_member()
        instead, which calculates separate h_c,ef values for each face.

        Args:
            d: Effective depth to tension reinforcement (mm)
            x: Neutral axis depth from compression face (mm), or None for uncracked

        Returns:
            Effective concrete height in tension zone (mm)
        """
        h = self.height

        candidates = [
            2.5 * (h - d),
            h / 2,
        ]

        # Only include (h-x)/3 if we have a valid NA depth
        if x is not None and x > 0:
            candidates.append((h - x) / 3)

        return min(candidates)

    def find_h_c_ef_tension_member(
        self,
        d_top: float,
        d_bottom: float,
    ) -> Tuple[float, float]:
        """
        Effective heights for fully tensioned sections (EC2 Fig 7.1, case c).

        When both faces are in tension (no neutral axis within section),
        h_c,ef must be calculated separately for each face.

        Args:
            d_top: Depth from top face to centroid of top tension reinforcement (mm)
            d_bottom: Depth from bottom face to centroid of bottom tension reinforcement (mm)

        Returns:
            Tuple of (h_c_ef_top, h_c_ef_bottom) in mm
        """
        h = self.height

        # Top half: tension face at top
        h_c_ef_top = min(2.5 * d_top, h / 2)

        # Bottom half: tension face at bottom
        h_c_ef_bottom = min(2.5 * d_bottom, h / 2)

        return h_c_ef_top, h_c_ef_bottom

    # ===============================================
    # Reinforcement ratio ρ_p,eff
    # ===============================================

    def find_rho_p_eff(
        self,
        A_s_tension: float,
        h_c_ef: float,
        xi_1: float = 0.0,
        A_p: float = 0.0,
    ) -> float:
        """
        Effective reinforcement ratio ρ_p,eff (EC2 §7.3.4(2)).

        ρ_p,eff = (A_s + ξ₁ × A_p') / A_c,eff

        where A_c,eff = h_c,ef × b

        Args:
            A_s_tension: Area of tension reinforcement (mm²)
            h_c_ef: Effective height of concrete in tension (mm)
            xi_1: Adjusted ratio of bond strengths (ξ₁), default 0 for no prestress
            A_p: Area of prestressing tendons (mm²), default 0

        Returns:
            Effective reinforcement ratio (dimensionless)
        """
        A_c_eff = h_c_ef * self.breadth
        if A_c_eff <= 0:
            raise ValueError("Effective concrete area A_c,eff must be > 0")

        return (A_s_tension + xi_1 * A_p) / A_c_eff

    # ===============================================
    # Maximum crack spacing s_r,max
    # ===============================================

    def find_k_2(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> float:
        """
        Strain distribution coefficient k_2 (EC2 §7.3.4(3)).

        k_2 = (ε₁ + ε₂) / (2 × ε₁)

        where ε₁ is the greater and ε₂ the lesser tensile strain at the
        section boundaries.

        Returns:
            - 0.5 for pure bending (one face in compression)
            - 1.0 for pure tension (uniform tension)
            - Intermediate values for eccentric tension

        Args:
            eps_top: Strain at top fibre (compression positive, tension negative)
            eps_bottom: Strain at bottom fibre (compression positive, tension negative)

        Returns:
            k_2 coefficient (0.5 to 1.0)
        """
        # Check if either face is in compression (strain >= 0)
        if eps_top >= 0 or eps_bottom >= 0:
            # At least one face in compression -> bending dominated
            return 0.5

        # Both faces in tension (both strains negative)
        # ε₁ = greater tensile strain (more negative = larger absolute tension)
        # ε₂ = lesser tensile strain
        eps_1 = min(eps_top, eps_bottom)  # More negative = greater tension
        eps_2 = max(eps_top, eps_bottom)  # Less negative = lesser tension

        # Both are negative, so abs values for the formula
        abs_eps_1 = abs(eps_1)
        abs_eps_2 = abs(eps_2)

        if abs_eps_1 < 1e-12:
            return 0.5  # Guard against division by zero

        k_2 = (abs_eps_1 + abs_eps_2) / (2 * abs_eps_1)

        # Clamp to valid range [0.5, 1.0]
        return max(0.5, min(1.0, k_2))

    def find_maximum_crack_spacing(
        self,
        cover: float,
        phi_eq: float,
        rho_p_eff: float,
        k_2: float,
        x: Optional[float] = None,
        has_tension_reinforcement: bool = True,
    ) -> float:
        """
        Maximum crack spacing s_r,max (EC2 §7.3.4(3), Eq. 7.11).

        Standard formula (spacing ≤ 5(c + φ/2)):
            s_r,max = k_3·c + k_1·k_2·k_4·φ / ρ_p,eff

        If spacing > 5(c + φ/2) or no bonded reinforcement (Eq. 7.14):
            s_r_max = 1.3(h - x)

        Args:
            cover: Concrete cover to tension reinforcement (mm)
            phi_eq: Equivalent bar diameter (mm)
            rho_p_eff: Effective reinforcement ratio (dimensionless)
            k_2: Strain distribution coefficient (0.5 for bending, 1.0 for tension)
            x: Neutral axis depth (mm), or None for uncracked/fully cracked
            has_tension_reinforcement: True if bonded reinforcement exists in tension zone

        Returns:
            Maximum crack spacing in mm
        """
        # Standard formula (Eq. 7.11)
        if rho_p_eff > 0:
            s_r_max = self.k_3 * cover + (self.k_1 * k_2 * self.k_4 * phi_eq / rho_p_eff)
        else:
            # No reinforcement in tension zone - use upper bound
            s_r_max = float('inf')

        # Check if spacing exceeds 5(c + φ/2) -> use upper bound formula (Eq. 7.14)
        spacing_limit = 5 * (cover + phi_eq / 2)

        # Use upper bound if spacing too large OR no bonded reinforcement in tension zone
        if s_r_max > spacing_limit or not has_tension_reinforcement:
            # Upper bound formula
            h = self.height
            if x is not None and x > 0:
                s_r_max = 1.3 * (h - x)
            else:
                # If no NA (uncracked or fully cracked), use full height
                s_r_max = 1.3 * h

        return s_r_max

    # ===============================================
    # Mean strain difference (ε_sm - ε_cm)
    # ===============================================

    def find_strain_difference(
        self,
        sigma_s: float,
        rho_p_eff: float,
        E_s: float,
    ) -> float:
        """
        Mean strain difference (ε_sm - ε_cm) (EC2 §7.3.4(2), Eq. 7.9).

        (ε_sm - ε_cm) = [σ_s - k_t × f_ct,eff × (1 + α_e × ρ_p,eff) / ρ_p,eff] / E_s
                      ≥ 0.6 × σ_s / E_s

        where:
        - σ_s = stress in tension reinforcement (MPa, positive for tension)
        - k_t = load duration factor (0.6 short, 0.4 long)
        - f_ct,eff = mean tensile strength of concrete (MPa)
        - α_e = E_s / E_cm
        - ρ_p,eff = effective reinforcement ratio

        Args:
            sigma_s: Absolute steel stress in tension reinforcement (MPa, positive)
            rho_p_eff: Effective reinforcement ratio
            E_s: Steel elastic modulus (MPa)

        Returns:
            Mean strain difference (dimensionless, always positive)
        """
        if sigma_s <= 0:
            return 0.0  # No tension, no cracking

        f_ct_eff = self.concrete.f_ctm  # Could be f_ctm(t) for early age

        # Full formula
        if rho_p_eff > 0:
            tension_stiffening = self.k_t * f_ct_eff * (1 + self.alpha_e * rho_p_eff) / rho_p_eff
            eps_diff = (sigma_s - tension_stiffening) / E_s
        else:
            eps_diff = sigma_s / E_s

        # Minimum value (Eq. 7.9 lower bound)
        eps_min = 0.6 * sigma_s / E_s

        return max(eps_diff, eps_min)

    # ===============================================
    # Crack width calculation
    # ===============================================

    def calculate_crack_width(
        self,
        s_r_max: float,
        eps_sm_minus_eps_cm: float,
    ) -> float:
        """
        Characteristic crack width w_k (EC2 §7.3.4(1), Eq. 7.8).

        w_k = s_r,max × (ε_sm - ε_cm)

        Args:
            s_r_max: Maximum crack spacing (mm)
            eps_sm_minus_eps_cm: Mean strain difference (dimensionless)

        Returns:
            Crack width in mm
        """
        return s_r_max * eps_sm_minus_eps_cm

    # ===============================================
    # Minimum reinforcement (EC2 §7.3.2(2))
    # ===============================================

    def find_minimum_crack_reinforcement(
        self,
        steel_stress: float = 500.0,
        k_c: Optional[float] = None,
        N_Ed: float = 0.0,
        is_in_bending: bool = True,
    ) -> float:
        """
        Minimum reinforcement to control cracking A_s,min (EC2 §7.3.2(2), Eq. 7.1).

        A_s,min × σ_s = k_c × k × f_ct,eff × A_ct

        Args:
            steel_stress:
                The absolute value of the maximum permitted stress in the reinforcement.
                (default 500 MPa)
            k_c: Stress distribution coefficient (calculated if None)
            N_Ed: Axial force for k_c calculation (kN, compression positive)
            is_in_bending: True for bending, False for pure tension

        Returns:
            Minimum reinforcement area in mm²
        """
        if k_c is None:
            k_c = self.find_k_c(N_Ed, is_in_bending)

        # k factor for non-uniform self-equilibrating stresses
        h = self.height
        b = self.breadth
        min_dim = min(h, b)

        if min_dim <= 300:
            k = 1.0
        elif min_dim >= 800:
            k = 0.65
        else:
            # Linear interpolation
            k = 1.0 - 0.35 * (min_dim - 300) / 500

        # A_ct: area of concrete within tensile zone just before first crack
        # For uncracked section, tension zone is below the elastic neutral axis
        # For rectangular section in bending: A_ct ≈ h/2 × b
        # For tension members, more of section is in tension
        if N_Ed >= 0:  # Compression or pure bending
            A_ct = 0.5 * h * b
        else:
            # Tension - use transformed section centroid to estimate tension zone
            _, c_y, _ = self.section.get_transformed_centroid(self.E_cm_eff)
            bounds = self.section.outline.bounds
            y_from_bottom = c_y - bounds[1]
            A_ct = y_from_bottom * b

        # f_ct,eff (could be f_ctm(t) for early age)
        f_ct_eff = self.concrete.f_ctm

        A_s_min = k_c * k * f_ct_eff * A_ct / abs(steel_stress)
        return A_s_min

    def find_k_c(
        self,
        N_Ed: float = 0.0,
        is_in_bending: bool = True,
    ) -> float:
        """
        Stress distribution coefficient k_c (EC2 §7.3.2(2)).

        For bending (rectangular stress block):
            k_c = 0.4 × [1 - σ_c / (k_1 × (h/h*) × f_ct,eff)] ≤ 1.0

        For pure tension:
            k_c = 1.0

        Args:
            N_Ed: Axial force (kN, compression positive)
            is_in_bending: True for bending, False for pure tension

        Returns:
            k_c coefficient (dimensionless)
        """
        if not is_in_bending:
            return 1.0

        h = self.height
        h_star = min(h, 1000)

        # k_1 depends on axial force
        if N_Ed >= 0:
            k_1 = 1.5  # Compression or zero axial
        else:
            k_1 = (2 * h_star) / (3 * h)  # Tension

        # Concrete stress from axial force
        A_eff = self.section.get_transformed_area(self.concrete.E_cm)
        sigma_c = from_kn(N_Ed, ForceUnit.N) / A_eff  # MPa

        f_ct_eff = self.concrete.f_ctm

        k_c = 0.4 * (1 - sigma_c / (k_1 * (h / h_star) * f_ct_eff))
        return min(1.0, max(0.0, k_c))

    # ===============================================
    # Helper methods for rebar analysis
    # ===============================================

    def _get_tension_rebar_info(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> Tuple[float, float, List[Tuple[float, int]]]:
        """
        Get tension reinforcement information from strain state.

        Returns:
            Tuple of (total_area, mean_cover, bar_sizes) where:
            - total_area: Total area of tension reinforcement (mm²)
            - mean_cover: Area-weighted mean cover to tension bars (mm)
            - bar_sizes: List of (diameter, count) for equivalent diameter calc
        """
        bounds = self.section.outline.bounds
        h = bounds[3] - bounds[1]
        y_min = bounds[1]
        y_max = bounds[3]

        tension_bars: List[Tuple[float, int]] = []
        total_area = 0.0
        cover_sum = 0.0

        # Determine tension face
        comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)
        if comp_face is None:
            # Both faces in tension - use bottom as reference
            comp_face = "top"

        for group in self.section.rebar_groups:
            diameter = float(group.rebar.diameter)
            bar_area = float(group.rebar.area)
            bar_count = 0

            for pos in group.positions:
                # Calculate strain at bar location
                y_rel = (pos.y - y_min) / h
                strain_at_bar = eps_bottom + (eps_top - eps_bottom) * y_rel

                # Tension is negative strain
                if strain_at_bar < 0:
                    bar_count += 1
                    total_area += bar_area

                    # Calculate cover based on tension face
                    if comp_face == "top":
                        # Tension at bottom
                        cover = pos.y - y_min - diameter / 2
                    else:
                        # Tension at top
                        cover = y_max - pos.y - diameter / 2

                    cover_sum += bar_area * max(0, cover)

            if bar_count > 0:
                tension_bars.append((diameter, bar_count))

        mean_cover = cover_sum / total_area if total_area > 0 else 0.0

        return total_area, mean_cover, tension_bars

    def _get_steel_stress(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> float:
        """
        Get maximum steel stress in tension zone from strain state.

        Note: Returns the absolute value of stress (always positive),
        even though tension strains are negative by convention.

        Args:
            eps_top: Top fibre strain (compression positive)
            eps_bottom: Bottom fibre strain (compression positive)

        Returns:
            Maximum tensile stress in reinforcement (MPa, always positive)
        """
        bounds = self.section.outline.bounds
        h = bounds[3] - bounds[1]
        y_min = bounds[1]

        max_tension_stress = 0.0

        for group in self.section.rebar_groups:
            E_s = group.rebar.E_s
            f_yk = group.rebar.f_yk
            epsilon_uk = group.rebar.epsilon_uk
            k_ratio = group.rebar.grade.ft_ratio_min  # Hardening ratio
            
            for pos in group.positions:
                # Strain at bar location
                y_rel = (pos.y - y_min) / h
                strain_at_bar = eps_bottom + (eps_top - eps_bottom) * y_rel

                # Only consider tension (negative strain)
                if strain_at_bar < 0:
                    stress = flexure_utils.calculate_rebar_characteristic_stress_from_strain(
                        strain=strain_at_bar,
                        steel_model_type=self.steel_model_type,
                        E_s=E_s,
                        f_yk=f_yk,
                        k=k_ratio,
                        epsilon_uk=epsilon_uk,
                    )
                    max_tension_stress = max(max_tension_stress, abs(stress))

        # Return absolute value (always positive for tension)
        return max_tension_stress

    def _get_tension_zone_E_s(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> float:
        """
        Get E_s from the outermost tension rebar layer.

        Optimized to return early if all rebar groups have the same E_s.

        Args:
            eps_top: Top fibre strain (compression positive)
            eps_bottom: Bottom fibre strain (compression positive)

        Returns:
            Elastic modulus of outermost tension rebar (MPa)
        """
        if not self.section.rebar_groups:
            return 200000.0  # Default

        # Early return optimization: check if all E_s values are the same
        first_E_s = self.section.rebar_groups[0].rebar.E_s
        all_same = all(g.rebar.E_s == first_E_s for g in self.section.rebar_groups)
        if all_same:
            return first_E_s

        # Different E_s values - find outermost tension bar
        bounds = self.section.outline.bounds
        h = bounds[3] - bounds[1]
        y_min = bounds[1]

        comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)

        outermost_y: Optional[float] = None
        outermost_E_s = first_E_s

        for group in self.section.rebar_groups:
            for pos in group.positions:
                # Check if bar is in tension
                y_rel = (pos.y - y_min) / h
                strain = eps_bottom + (eps_top - eps_bottom) * y_rel
                if strain >= 0:  # Compression, skip
                    continue

                # Track outermost tension bar
                if comp_face == "top":
                    # Tension at bottom - find lowest bar
                    if outermost_y is None or pos.y < outermost_y:
                        outermost_y = pos.y
                        outermost_E_s = group.rebar.E_s
                else:
                    # Tension at top - find highest bar
                    if outermost_y is None or pos.y > outermost_y:
                        outermost_y = pos.y
                        outermost_E_s = group.rebar.E_s

        return outermost_E_s

    # ===============================================
    # Stress limitation helpers (EC2 §7.2)
    # ===============================================

    def _get_peak_concrete_stress(
        self,
        eps_top: float,
        eps_bottom: float,
        diagram: Optional[MNInteractionDiagram] = None,
    ) -> float:
        """
        Peak compressive stress in concrete from fibre integration.

        Args:
            eps_top: Top fibre strain (compression positive)
            eps_bottom: Bottom fibre strain (compression positive)
            diagram: Diagram to use (defaults to self._get_diagram())

        Returns:
            Peak compressive stress in MPa (positive)
        """
        diag = diagram or self._get_diagram()
        forces, y, areas = diag.get_fibre_forces_from_end_strains(eps_top, eps_bottom)

        # Identify concrete fibres
        conc_mask = diag._fibre_mat == "concrete"

        # Stresses = forces / areas (guard against zero-area fibres)
        conc_forces = forces[conc_mask]
        conc_areas = areas[conc_mask]
        nonzero = conc_areas > 0.0
        if not nonzero.any():
            return 0.0

        conc_stresses = conc_forces[nonzero] / conc_areas[nonzero]

        # Peak compressive stress (compression positive)
        peak = float(conc_stresses.max()) if len(conc_stresses) > 0 else 0.0
        return max(0.0, peak)

    def _compute_nonlinear_creep_coefficient(self, sigma_c: float) -> float:
        """
        Non-linear creep coefficient per EC2 §3.1.4(4), Eq. 3.7.

        φ_NL = φ · exp(1.5 · (k_σ − 0.45))

        where k_σ = σ_c / f_cm (stress to mean strength ratio).

        Args:
            sigma_c: Peak concrete compressive stress (MPa)

        Returns:
            Non-linear creep coefficient φ_NL
        """
        k_sigma = sigma_c / self.concrete.f_cm
        return self.creep_coefficient * exp(1.5 * (k_sigma - 0.45))

    def _build_diagram_with_E_cm_eff(
        self, E_cm_eff: float, ignore_compression_steel: bool = False,
    ) -> MNInteractionDiagram:
        """Build a temporary interaction diagram with a specific E_cm,eff."""
        return create_interaction_diagram(
            section=self.section,
            concrete=self.concrete,
            concrete_model_type=self.concrete_model_type,
            steel_model_type=self.steel_model_type,
            n_fibres_width=self.n_fibres_width,
            n_fibres_height=self.n_fibres_height,
            use_characteristic=True,
            ignore_compression_steel=ignore_compression_steel,
            elastic_modulus=E_cm_eff,
        )

    def _get_f_yk_max(self) -> float:
        """Maximum f_yk across all rebar groups."""
        if not self.section.rebar_groups:
            return 500.0
        return max(g.rebar.f_yk for g in self.section.rebar_groups)

    # ===============================================
    # Main check method
    # ===============================================

    def perform_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        warning_threshold: float = 0.95,
        ignore_compression_steel: bool = False,
        **kwargs,
    ) -> CheckResult:
        """
        Perform crack width check for applied serviceability loads.

        Args:
            M_Ed: Design moment at SLS (kN·m)
            N_Ed: Design axial force at SLS (kN, compression positive)
            warning_threshold: Utilization threshold for warnings

        Returns:
            CheckResult with crack width utilization
        """
        return self._check_single_case(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            warning_threshold=warning_threshold,
            ignore_compression_steel=ignore_compression_steel,
        )

    def _check_single_case(
        self,
        *,
        M_Ed: float,
        N_Ed: float,
        warning_threshold: float,
        ignore_compression_steel: bool = False,
    ) -> CheckResult:
        """Internal implementation of crack check."""

        # Step 1: Check if section is cracked
        #TODO this logic doesn't work if N_Ed is provided as the comparison
        # of moment to cracking moment should consider axial force as well.
        # Really it is a stress comparison of the most tensile concrete fibre to f_ctm,fl
        # Provide an override to allow the user to force a crack width check.
        M_cr = self.find_cracking_moment()
        is_cracked = abs(M_Ed) > abs(M_cr)

        if not is_cracked:
            # Section uncracked - no crack width to check
            return self._create_result(
                check_name="Cracking check (EC2 §7.3)",
                code_reference="EC2 §7.3",
                warning_threshold=warning_threshold,
                utilization=0.0,
                demand_components={"M_Ed": float(M_Ed), "N_Ed": float(N_Ed)},
                capacity_components={"w_k_limit": self.w_k_limit, "M_cr": float(M_cr)},
                units_components={"M_Ed": "kN·m", "N_Ed": "kN", "w_k_limit": "mm", "M_cr": "kN·m"},
                message="Section uncracked (M_Ed < M_cr)",
                details={
                    "M_Ed": float(M_Ed),
                    "N_Ed": float(N_Ed),
                    "M_cr": float(M_cr),
                    "is_cracked": False,
                    "w_k": 0.0,
                    "w_k_limit": self.w_k_limit,
                },
            )

        # Step 2: Solve for strain state (cracked section)
        try:
            eps_top, eps_bottom = self._get_diagram(ignore_compression_steel).find_strains_for_MN(
                M_target=M_Ed,
                N_target=N_Ed,
            )
        except ValueError as e:
            # Load point outside capacity - section fails
            return self._create_result(
                check_name="Cracking check (EC2 §7.3)",
                code_reference="EC2 §7.3",
                warning_threshold=warning_threshold,
                utilization=float("inf"),
                demand_components={"M_Ed": float(M_Ed), "N_Ed": float(N_Ed)},
                capacity_components={"w_k_limit": self.w_k_limit},
                units_components={"M_Ed": "kN·m", "N_Ed": "kN", "w_k_limit": "mm"},
                message=f"Failed to solve strain state: {e}",
                details={"error": str(e)},
            )

        # Step 2.5: Stress limitation checks (EC2 §7.2) and non-linear creep
        sigma_c_peak = self._get_peak_concrete_stress(eps_top, eps_bottom)
        nonlinear_creep_applied = False
        creep_coefficient_used = self.creep_coefficient
        diagram_for_check = self._get_diagram(ignore_compression_steel)

        # EC2 §7.2(2): Characteristic stress limit (longitudinal cracking risk)
        if self.sls_combination == SLSCombination.CHARACTERISTIC:
            limit_char = self.k_1_stress * self.concrete.f_ck
            if sigma_c_peak > limit_char:
                warnings.warn(
                    f"EC2 §7.2(2): σ_c = {sigma_c_peak:.1f} MPa > "
                    f"{self.k_1_stress}·f_ck = {limit_char:.1f} MPa under characteristic loads. "
                    f"Longitudinal cracking risk for XD/XF/XS exposure classes.",
                    stacklevel=3,
                )

        # EC2 §7.2(3): Quasi-permanent stress limit (non-linear creep threshold)
        limit_qp = self.k_2_stress * self.concrete.f_ck
        if sigma_c_peak > limit_qp:
            warnings.warn(
                f"EC2 §7.2(3): σ_c = {sigma_c_peak:.1f} MPa > "
                f"{self.k_2_stress}·f_ck = {limit_qp:.1f} MPa. "
                f"Non-linear creep threshold exceeded.",
                stacklevel=3,
            )

            if self.apply_nonlinear_creep and self.sls_combination == SLSCombination.QUASI_PERMANENT:
                max_iterations = 5 if self.iterate_nonlinear_creep else 1
                for _ in range(max_iterations):
                    phi_NL = self._compute_nonlinear_creep_coefficient(sigma_c_peak)
                    E_cm_eff_NL = self.concrete.get_elastic_modulus() / (1.0 + phi_NL)

                    if abs(E_cm_eff_NL - (self.concrete.get_elastic_modulus() / (1.0 + creep_coefficient_used))) < 1.0:
                        break  # Converged (within 1 MPa)

                    creep_coefficient_used = phi_NL
                    diagram_for_check = self._build_diagram_with_E_cm_eff(E_cm_eff_NL, ignore_compression_steel)
                    eps_top, eps_bottom = diagram_for_check.find_strains_for_MN(
                        M_target=M_Ed, N_target=N_Ed,
                    )
                    sigma_c_peak = self._get_peak_concrete_stress(eps_top, eps_bottom, diagram_for_check)
                    nonlinear_creep_applied = True

        # Step 3: Calculate neutral axis depth
        x = flexure_utils.calculate_neutral_axis_depth_from_strains(
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            section_height=self.height,
        )

        # Step 4: Get tension reinforcement info
        A_s_tension, mean_cover, bar_sizes = self._get_tension_rebar_info(eps_top, eps_bottom)

        if A_s_tension <= 0:
            warnings.warn(
                "No tension reinforcement found - cannot calculate crack width",
                stacklevel=2,
            )
            return self._create_result(
                check_name="Cracking check (EC2 §7.3)",
                code_reference="EC2 §7.3",
                warning_threshold=warning_threshold,
                utilization=float("inf"),
                demand_components={"M_Ed": float(M_Ed), "N_Ed": float(N_Ed)},
                capacity_components={"w_k_limit": self.w_k_limit},
                units_components={"M_Ed": "kN·m", "N_Ed": "kN", "w_k_limit": "mm"},
                message="No tension reinforcement found",
                details={"is_cracked": True, "A_s_tension": 0.0},
            )

        # Step 5: Calculate equivalent diameter
        phi_eq = flexure_utils.find_equivalent_diameter(bar_sizes)

        # Step 6: Get effective depth and h_c,ef
        comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)
        if comp_face == "top":
            d = self.section.get_effective_depth(compression_face="top")
        else:
            d = self.section.get_effective_depth(compression_face="bottom")

        h_c_ef = self.find_h_c_ef(d=d, x=x)

        # Step 7: Calculate ρ_p,eff
        rho_p_eff = self.find_rho_p_eff(A_s_tension=A_s_tension, h_c_ef=h_c_ef)

        # Step 8: Get steel stress (returned as absolute value, always positive)
        sigma_s = self._get_steel_stress(eps_top, eps_bottom)

        # EC2 §7.2(5): Reinforcement stress limit (serviceability stress limit)
        # TODO  this check is only valid if the limit state is SLSCombination.CHARACTERISTIC
        f_yk = self._get_f_yk_max()
        limit_steel = self.k_3_stress * f_yk
        if sigma_s > limit_steel:
            warnings.warn(
                f"EC2 §7.2(5): σ_s = {sigma_s:.1f} MPa > "
                f"{self.k_3_stress}·f_yk = {limit_steel:.1f} MPa. "
                f"Reinforcement stress limit exceeded.",
                stacklevel=3,
            )

        # EC2 §7.2(4)P: Check for inelastic strain (yielding)
        # "Tensile stresses in the reinforcement shall be limited in order to
        # avoid inelastic strain, unacceptable cracking or deformation."
        if sigma_s > f_yk:
            warnings.warn(
                f"EC2 §7.2(4)P: σ_s = {sigma_s:.1f} MPa > f_yk = {f_yk:.1f} MPa. "
                f"Reinforcement has yielded - inelastic strain occurring. "
                f"SLS crack width calculation may be unreliable.",
                stacklevel=3,
            )

        # Step 9: Get cover (use calculated mean or from section)
        try:
            cover = self.section.get_concrete_cover(
                reference="bottom" if comp_face == "top" else "top"
            )
        except ValueError:
            cover = mean_cover

        # Step 10: Calculate k_2 and maximum crack spacing
        k_2 = self.find_k_2(eps_top, eps_bottom)
        s_r_max = self.find_maximum_crack_spacing(
            cover=cover,
            phi_eq=phi_eq,
            rho_p_eff=rho_p_eff,
            k_2=k_2,
            x=x,
            has_tension_reinforcement=A_s_tension > 0,
        )

        # Step 11: Calculate strain difference
        E_s = self._get_tension_zone_E_s(eps_top, eps_bottom)
        eps_sm_minus_eps_cm = self.find_strain_difference(
            sigma_s=sigma_s,
            rho_p_eff=rho_p_eff,
            E_s=E_s,
        )

        # Step 12: Calculate crack width
        w_k = self.calculate_crack_width(s_r_max, eps_sm_minus_eps_cm)

        # Step 13: Calculate utilization
        utilization = w_k / self.w_k_limit if self.w_k_limit > 0 else float("inf")

        # Build detailed results
        details = {
            "M_Ed": float(M_Ed),
            "N_Ed": float(N_Ed),
            "M_cr": float(M_cr),
            "is_cracked": True,
            "eps_top": float(eps_top),
            "eps_bottom": float(eps_bottom),
            "x": float(x) if x is not None else None,
            "d": float(d),
            "h_c_ef": float(h_c_ef),
            "A_s_tension": float(A_s_tension),
            "phi_eq": float(phi_eq),
            "cover": float(cover),
            "rho_p_eff": float(rho_p_eff),
            "sigma_s": float(sigma_s),
            "s_r_max": float(s_r_max),
            "eps_sm_minus_eps_cm": float(eps_sm_minus_eps_cm),
            "w_k": float(w_k),
            "w_k_limit": float(self.w_k_limit),
            "k_t": float(self.k_t),
            "k_1": float(self.k_1),
            "k_2": float(k_2),
            "k_3": float(self.k_3),
            "k_4": float(self.k_4),
            "sigma_c_peak": float(sigma_c_peak),
            "k_1_stress_limit": float(self.k_1_stress * self.concrete.f_ck),
            "k_2_stress_limit": float(self.k_2_stress * self.concrete.f_ck),
            "k_3_stress_limit": float(self.k_3_stress * f_yk),
            "f_yk": float(f_yk),
            "steel_yielded": sigma_s > f_yk,
            "nonlinear_creep_applied": nonlinear_creep_applied,
            "creep_coefficient_used": float(creep_coefficient_used),
        }

        # Create result
        is_pass = w_k <= self.w_k_limit
        message = f"w_k = {w_k:.3f} mm {'<=' if is_pass else '>'} {self.w_k_limit:.2f} mm limit"

        return self._create_result(
            check_name="Cracking check (EC2 §7.3)",
            code_reference="EC2 §7.3",
            warning_threshold=warning_threshold,
            utilization=utilization,
            demand_components={"w_k": float(w_k)},
            capacity_components={"w_k_limit": self.w_k_limit},
            units_components={"w_k": "mm", "w_k_limit": "mm"},
            message=message,
            details=details,
        )

    def calculate_detailed(
        self,
        M_Ed: float,
        N_Ed: float = 0.0,
        ignore_compression_steel: bool = False,
    ) -> CrackingResult:
        """
        Calculate detailed cracking results without creating CheckResult.

        Useful for parametric studies or when you need the raw values.

        Args:
            M_Ed: Design moment at SLS (kN·m)
            N_Ed: Design axial force at SLS (kN, compression positive)

        Returns:
            CrackingResult dataclass with all intermediate values
        """
        # Check if cracked
        #TODO this logic doesn't work if N_Ed is provided as the comparison
        # of moment to cracking moment should consider axial force as well.
        # Really it is a stress comparison of the most tensile concrete fibre to f_ctm,fl
        # Provide an override to allow the user to force a crack width check
        M_cr = self.find_cracking_moment()
        is_cracked = abs(M_Ed) > abs(M_cr)

        if not is_cracked:
            return CrackingResult(
                w_k=0.0,
                w_k_limit=self.w_k_limit,
                s_r_max=0.0,
                eps_sm_minus_eps_cm=0.0,
                sigma_s=0.0,
                rho_p_eff=0.0,
                h_c_ef=0.0,
                x=None,
                is_cracked=False,
                phi_eq=0.0,
                cover=0.0,
            )

        # Solve strain state
        eps_top, eps_bottom = self._get_diagram(ignore_compression_steel).find_strains_for_MN(M_Ed, N_Ed)

        # Stress limitation and non-linear creep (same logic as _check_single_case)
        sigma_c_peak = self._get_peak_concrete_stress(eps_top, eps_bottom)
        nonlinear_creep_applied = False
        creep_coefficient_used = self.creep_coefficient

        limit_qp = self.k_2_stress * self.concrete.f_ck
        if sigma_c_peak > limit_qp and self.apply_nonlinear_creep and self.sls_combination == SLSCombination.QUASI_PERMANENT:
            max_iterations = 5 if self.iterate_nonlinear_creep else 1
            for _ in range(max_iterations):
                phi_NL = self._compute_nonlinear_creep_coefficient(sigma_c_peak)
                E_cm_eff_NL = self.concrete.get_elastic_modulus() / (1.0 + phi_NL)
                if abs(E_cm_eff_NL - (self.concrete.get_elastic_modulus() / (1.0 + creep_coefficient_used))) < 1.0:
                    break
                creep_coefficient_used = phi_NL
                diagram_nl = self._build_diagram_with_E_cm_eff(E_cm_eff_NL, ignore_compression_steel)
                eps_top, eps_bottom = diagram_nl.find_strains_for_MN(M_Ed, N_Ed)
                sigma_c_peak = self._get_peak_concrete_stress(eps_top, eps_bottom, diagram_nl)
                nonlinear_creep_applied = True

        # Calculate all values
        x = flexure_utils.calculate_neutral_axis_depth_from_strains(
            eps_top, eps_bottom, self.height
        )

        A_s_tension, mean_cover, bar_sizes = self._get_tension_rebar_info(eps_top, eps_bottom)
        phi_eq = flexure_utils.find_equivalent_diameter(bar_sizes) if bar_sizes else 0.0

        comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)
        d = self.section.get_effective_depth(
            compression_face=comp_face if comp_face else "top"
        )

        h_c_ef = self.find_h_c_ef(d=d, x=x)
        rho_p_eff = self.find_rho_p_eff(A_s_tension, h_c_ef) if A_s_tension > 0 else 0.0
        sigma_s = self._get_steel_stress(eps_top, eps_bottom)

        try:
            cover = self.section.get_concrete_cover(
                reference="bottom" if comp_face == "top" else "top"
            )
        except ValueError:
            cover = mean_cover

        k_2 = self.find_k_2(eps_top, eps_bottom)
        if rho_p_eff > 0:
            s_r_max = self.find_maximum_crack_spacing(
                cover=cover,
                phi_eq=phi_eq,
                rho_p_eff=rho_p_eff,
                k_2=k_2,
                x=x,
                has_tension_reinforcement=A_s_tension > 0,
            )
        else:
            s_r_max = 0.0

        E_s = self._get_tension_zone_E_s(eps_top, eps_bottom)
        eps_diff = self.find_strain_difference(sigma_s, rho_p_eff, E_s) if rho_p_eff > 0 else 0.0

        w_k = self.calculate_crack_width(s_r_max, eps_diff)

        # Check for steel yielding (EC2 §7.2(4)P)
        f_yk = self._get_f_yk_max()
        steel_yielded = sigma_s > f_yk

        return CrackingResult(
            w_k=w_k,
            w_k_limit=self.w_k_limit,
            s_r_max=s_r_max,
            eps_sm_minus_eps_cm=eps_diff,
            sigma_s=sigma_s,
            rho_p_eff=rho_p_eff,
            h_c_ef=h_c_ef,
            x=x,
            is_cracked=True,
            phi_eq=phi_eq,
            cover=cover,
            sigma_c_peak=sigma_c_peak,
            nonlinear_creep_applied=nonlinear_creep_applied,
            creep_coefficient_used=creep_coefficient_used,
            steel_yielded=steel_yielded,
        )
