"""
Cracking check for reinforced concrete sections according to EC2 §7.3.

This is a SERVICEABILITY check using characteristic material properties and
elastic/cracked section analysis to calculate crack widths.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, cast
import math
import re
import warnings

if TYPE_CHECKING:
    from materials.reinforced_concrete.analysis.strain_state import StrainState

import numpy as np
from pydantic import Field, PrivateAttr, computed_field

from materials.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from materials.reinforced_concrete.code_checks.ec2_2004.stress_limits_check import (
    check_characteristic_concrete_stress,
    check_quasi_permanent_concrete_stress,
    check_characteristic_reinforcement_stress,
    check_reinforcement_yielding,
    check_imposed_deformation_stress,
    compute_nonlinear_creep_coefficient,
)
from materials.reinforced_concrete.constitutive import (
    ConcreteModelType,
    SteelModelType,
    ConcreteStressStrainLinearElastic,
)
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial
from materials.reinforced_concrete.analysis import create_interaction_diagram
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from materials.reinforced_concrete.code_checks.ec2_2004 import flexure_utils
from materials.core.geometry import Point2D
from materials.core.units import ForceUnit, MomentUnit, from_kn, to_knm
from materials.reinforced_concrete.ndp import get_ndp, get_ndp_callable


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



class CrackFacePolicy(StrEnum):
    """
    Policy for selecting which face(s) to check for crack width.

    Attributes:
        TOP_BOTTOM: Legacy behaviour — uses eps_top/eps_bottom to pick top or bottom face.
        NA_NORMAL: Uses the neutral axis normal (compression direction) to identify the
                   tension region and selects the face perpendicular to it. Required for
                   biaxial bending with rotated NA.
        ALL_FACES: Check all bounding faces, worst governs.
    """
    TOP_BOTTOM = "top_bottom"
    NA_NORMAL = "na_normal"
    ALL_FACES = "all_faces"


@dataclass
class CrackingResult:
    """Detailed results from crack width calculation."""
    w_k: Optional[float]  # Calculated crack width (mm)
    w_k_limit: float  # Allowable crack width (mm)
    s_r_max: Optional[float]  # Maximum crack spacing (mm)
    eps_sm_minus_eps_cm: Optional[float]  # Difference in mean strains (dimensionless)
    sigma_s: Optional[float]  # Steel stress in tension rebar (MPa)
    rho_p_eff: Optional[float]  # Effective reinforcement ratio (dimensionless)
    h_c_ef: Optional[float]  # Effective height of concrete in tension (mm)
    x: Optional[float]  # Neutral axis depth from compression face (mm)
    is_cracked: bool  # Whether section is cracked
    phi_eq: Optional[float]  # Equivalent bar diameter (mm)
    cover: Optional[float]  # Concrete cover to tension rebar (mm)
    solved: bool = True  # Whether full equilibrium-based result was solved
    solver_stage: Optional[str] = None  # Stage where solve failed (if any)
    solver_error: Optional[str] = None  # Solver failure message
    solver_residual_N: Optional[float] = None  # dN (kN) from solver, if available
    solver_residual_M: Optional[float] = None  # dM (kN.m) from solver, if available
    sigma_c_peak: Optional[float] = None  # Peak concrete compressive stress (MPa)
    nonlinear_creep_applied: bool = False  # Whether non-linear creep adjustment was applied
    creep_coefficient_used: float = 0.0  # Actual creep coefficient used (may be φ_NL)
    steel_yielded: bool = False  # Whether σ_s > f_yk (EC2 §7.2(4)P inelastic strain)
    governing_face: Optional[str] = None  # "top" or "bottom" — which face governed w_k
    # Bar diameter correction fields (for equivalent-area bar substitution)
    actual_bar_diameter: Optional[float] = None  # User-supplied actual bar diameter (mm)
    s_r_max_uncorrected: Optional[float] = None  # s_r,max before diameter correction (mm)
    phi_correction_factor: Optional[float] = None  # φ_actual / φ_eq ratio applied


class CrackingCheck(BaseCodeCheck):
    """
    EC2 2004 cracking check for reinforced concrete sections (§7.3).

    Calculates crack widths using EC2 formula (Eq. 7.8):
        w_k = s_r,max × (ε_sm - ε_cm)

    The check process:
    1. Determine if section is cracked using an uncracked M-N solver probe
       (LINEAR_ELASTIC concrete with include_tension=True)
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
        check_k1_stress: Enable EC2 §7.2(2) characteristic concrete stress limit
        check_k2_stress: Enable EC2 §7.2(3) quasi-permanent concrete stress limit
        check_k3_stress: Enable EC2 §7.2(5) reinforcement stress limit
        check_yielding: Enable EC2 §7.2(4)P inelastic strain check
        check_k4_stress: Enable EC2 §7.2(5) imposed deformation stress limit
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

    check_k1_stress: bool = Field(
        default=False,
        description="EC2 §7.2(2) characteristic concrete stress limit.",
    )

    check_k2_stress: bool = Field(
        default=True,
        description="EC2 §7.2(3) quasi-permanent concrete stress limit. Triggers non-linear creep when exceeded.",
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

    apply_nonlinear_creep: bool = Field(
        default=True,
        description="If True, automatically adjust E_cm,eff when σ_c > k_2·f_ck (EC2 §3.1.4(4)).",
    )

    iterate_nonlinear_creep: bool = Field(
        default=False,
        description="If True, iterate non-linear creep adjustment until convergence (max 5 iterations).",
    )

    free_neutral_axis: bool = Field(
        default=False,
        description=(
            "Allow the neutral axis to rotate to satisfy biaxial equilibrium. "
            "Note: for SLS checks (linear-elastic with tension), the biaxial "
            "solver is not used even when True; the 2D solver with horizontal "
            "NA is used instead. The flag is accepted for API consistency."
        ),
    )

    net_tension_face: Optional[Literal["top", "bottom"]] = Field(
        default=None,
        description=(
            "Face-checking policy for net tension (both faces in tension). "
            "None (default): check both faces independently and report the worst. "
            "'top' or 'bottom': only check the specified face."
        ),
    )

    crack_face_policy: CrackFacePolicy = Field(
        default=CrackFacePolicy.TOP_BOTTOM,
        description=(
            "Policy for selecting which face(s) to check for crack width. "
            "TOP_BOTTOM (default): legacy behaviour using eps_top/eps_bottom. "
            "NA_NORMAL: uses the neutral axis normal to identify the tension region "
            "(required for biaxial bending with rotated NA). "
            "ALL_FACES: check all bounding faces, worst governs."
        ),
    )

    bar_spacing_policy: Literal["x_axis", "perpendicular_to_na"] = Field(
        default="perpendicular_to_na",
        description=(
            "Policy for measuring bar spacing in _compute_max_bar_spacing. "
            "'perpendicular_to_na' (default): sort bars by their projection along the "
            "tension face (perpendicular to compression_direction). Correct for biaxial; "
            "equivalent to x-axis for horizontal NA. "
            "'x_axis': sort bars by x-coordinate (legacy behaviour)."
        ),
    )

    # =========================
    # Internal state (private)
    # =========================

    _diagram: Optional[MNInteractionDiagram] = PrivateAttr(default=None)
    _diagram_no_comp_steel: Optional[MNInteractionDiagram] = PrivateAttr(default=None)
    _diagram_uncracked: Optional[MNInteractionDiagram] = PrivateAttr(default=None)
    _diagram_uncracked_no_comp_steel: Optional[MNInteractionDiagram] = PrivateAttr(default=None)
    _diagram_snapshot: Optional[dict] = PrivateAttr(default=None)
    _diagram_no_comp_snapshot: Optional[dict] = PrivateAttr(default=None)
    _diagram_uncracked_snapshot: Optional[dict] = PrivateAttr(default=None)
    _diagram_uncracked_no_comp_snapshot: Optional[dict] = PrivateAttr(default=None)

    def _take_snapshot(self) -> dict:
        """Capture current state of inputs that affect the interaction diagram."""
        return {
            "section": self.section.model_dump(),
            "concrete": self.concrete.model_dump(),
            "concrete_model_type": self.concrete_model_type,
            "steel_model_type": self.steel_model_type,
            "include_tension": True,
            "crack_to_neutral_axis_on_first_tension_failure": True,
            "n_fibres_width": self.n_fibres_width,
            "n_fibres_height": self.n_fibres_height,
            "E_c_eff": self.E_c_eff,
        }

    def _get_diagram(self, ignore_compression_steel: bool = False) -> MNInteractionDiagram:
        """Get the cached diagram, rebuilding if inputs have changed."""
        snapshot = self._take_snapshot()

        if ignore_compression_steel:
            if self._diagram_no_comp_steel is None or snapshot != self._diagram_no_comp_snapshot:
                self._diagram_no_comp_steel = create_interaction_diagram(
                    section=self.section,
                    concrete=self.concrete,
                    free_neutral_axis=self.free_neutral_axis,
                    concrete_model_type=self.concrete_model_type,
                    steel_model_type=self.steel_model_type,
                    n_fibres_width=self.n_fibres_width,
                    n_fibres_height=self.n_fibres_height,
                    use_characteristic=True,
                    ignore_compression_steel=True,
                    elastic_modulus=self.E_c_eff,
                    include_tension=True,
                    crack_to_neutral_axis_on_first_tension_failure=True,
                )
                self._diagram_no_comp_snapshot = snapshot
            return self._diagram_no_comp_steel
        else:
            if self._diagram is None or snapshot != self._diagram_snapshot:
                self._diagram = create_interaction_diagram(
                    section=self.section,
                    concrete=self.concrete,
                    free_neutral_axis=self.free_neutral_axis,
                    concrete_model_type=self.concrete_model_type,
                    steel_model_type=self.steel_model_type,
                    n_fibres_width=self.n_fibres_width,
                    n_fibres_height=self.n_fibres_height,
                    use_characteristic=True,
                    ignore_compression_steel=False,
                    elastic_modulus=self.E_c_eff,
                    include_tension=True,
                    crack_to_neutral_axis_on_first_tension_failure=True,
                )
                self._diagram_snapshot = snapshot
            return self._diagram

    def _get_uncracked_diagram(self, ignore_compression_steel: bool = False) -> MNInteractionDiagram:
        """
        Get cached uncracked-state probe diagram.

        This probe uses linear-elastic concrete with tension enabled so cracking can
        be identified from concrete tensile strain exceeding the cracking strain.
        """
        snapshot = self._take_snapshot()

        if ignore_compression_steel:
            if (
                self._diagram_uncracked_no_comp_steel is None
                or snapshot != self._diagram_uncracked_no_comp_snapshot
            ):
                self._diagram_uncracked_no_comp_steel = create_interaction_diagram(
                    section=self.section,
                    concrete=self.concrete,
                    concrete_model_type=ConcreteModelType.LINEAR_ELASTIC,
                    steel_model_type=self.steel_model_type,
                    n_fibres_width=self.n_fibres_width,
                    n_fibres_height=self.n_fibres_height,
                    use_characteristic=True,
                    ignore_compression_steel=True,
                    elastic_modulus=self.E_c_eff,
                    include_tension=True,
                    crack_to_neutral_axis_on_first_tension_failure=True,
                )
                self._diagram_uncracked_no_comp_snapshot = snapshot
            return self._diagram_uncracked_no_comp_steel

        if self._diagram_uncracked is None or snapshot != self._diagram_uncracked_snapshot:
            self._diagram_uncracked = create_interaction_diagram(
                section=self.section,
                concrete=self.concrete,
                concrete_model_type=ConcreteModelType.LINEAR_ELASTIC,
                steel_model_type=self.steel_model_type,
                n_fibres_width=self.n_fibres_width,
                n_fibres_height=self.n_fibres_height,
                use_characteristic=True,
                ignore_compression_steel=False,
                elastic_modulus=self.E_c_eff,
                include_tension=True,
                crack_to_neutral_axis_on_first_tension_failure=True,
            )
            self._diagram_uncracked_snapshot = snapshot

        return self._diagram_uncracked

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

    def find_k_1(self, k_2: Optional[float] = None) -> float:
        """Bond coefficient (EC2 §7.3.4(3)), via NDP k_1_crack.

        Base EC2: 0.8 for high bond, 1.6 for plain (independent of k_2).
        DIN EN 1992-1-1/NA: 1/k_2 (so that k_1·k_2 = 1.0).

        Args:
            k_2: Strain distribution coefficient. Required by some National
                Annexes (e.g. DIN), optional for base EC2.
        """
        k_1_fn = get_ndp_callable("k_1_crack")
        return k_1_fn(self.is_high_bond_bar, k_2)

    @property
    def k_3(self) -> float:
        """NDP coefficient k_3 for crack spacing (EC2 §7.3.4(3))."""
        return cast(float, get_ndp("k_3_crack"))

    @property
    def k_4(self) -> float:
        """NDP coefficient k_4 for crack spacing (EC2 §7.3.4(3))."""
        return cast(float, get_ndp("k_4_crack"))

    @property
    def effective_modulus_ratio(self) -> float:
        """Effective modulus ratio (1 + φ). Derived from creep_coefficient."""
        return 1.0 + self.creep_coefficient

    @property
    def E_c_eff(self) -> float:
        """
        Effective concrete modulus accounting for creep (EC2 §7.4.3).

        E_cm,eff = E_cm / (1 + φ)

        Returns:
            Effective modulus in MPa
        """
        return self.concrete.get_elastic_modulus() / self.effective_modulus_ratio

    # ===============================================
    # Cracking moment calculation
    # ===============================================

    def find_cracking_moment(
        self,
        N_Ed: float = 0.0,
        use_f_ctm_fl: bool = False,
        na_angle_deg: Optional[float] = None,
        ) -> float:
        """
        Cracking moment M_cr (kN·m) - moment at which section first cracks.

        M_cr = (f_ct,eff + σ_N) × W_el / 10^6

        where:
        - f_ctm,fl = mean flexural tensile strength (accounts for size effect)
        - f_ctm = mean tensile strength
        - W_el = elastic section modulus to tension face
        - σ_N = N_Ed / A_transformed (axial stress, compression positive)

        For biaxial bending (``na_angle_deg`` provided and non-zero), the
        second moment of area about the rotated NA direction is approximated
        as ``I_eff = I_yy * cos²θ + I_xx * sin²θ``, and the tension face
        distance is measured along the compression direction.

        Compressive axial load increases M_cr (delays cracking).
        Tensile axial load decreases M_cr (promotes cracking).

        Args:
            N_Ed: Design axial force (kN, compression positive). (Default 0).
            use_f_ctm_fl:
                Whether to use the mean flexural tensile strength (f_ctm,fl)
                or mean tensile strength (f_ctm).
            na_angle_deg: Angle of the neutral axis from horizontal (degrees).
                When provided and non-zero, uses the biaxial I_eff
                approximation instead of I_yy only.

        Returns:
            Cracking moment in kN·m
        """
        if use_f_ctm_fl:
            # Flexural tensile strength (EC2 §3.1.8)
            f_ct_eff = self.concrete.find_mean_flexural_tensile_strength(self.height)
        else:
            # Basic tensile strength without size effect
            f_ct_eff = self.concrete.f_ctm

        # Elastic section modulus (uncracked transformed section)
        I_xx, I_yy, _ = self.section.get_transformed_second_moment_area(self.E_c_eff)
        _, c_y, _ = self.section.get_transformed_centroid(self.E_c_eff)
        bounds = self.section.outline.bounds

        is_biaxial = na_angle_deg is not None and abs(na_angle_deg) > 1e-6

        if is_biaxial:
            # Biaxial approximation: I about the rotated NA direction
            # I_eff = I_xx * cos²θ + I_yy * sin²θ  (Mohr's circle rotation)
            # where θ is the NA angle from horizontal
            theta = math.radians(na_angle_deg)  # type: ignore[arg-type]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            I_eff = I_xx * cos_t ** 2 + I_yy * sin_t ** 2

            # Tension face distance along compression direction
            # Compression direction is perpendicular to NA.
            # Use section corners projected along the compression direction.
            dx = math.sin(theta)   # compression_direction x component
            dy = math.cos(theta)   # compression_direction y component
            corners = [
                (bounds[0], bounds[1]),  # min_x, min_y
                (bounds[2], bounds[1]),  # max_x, min_y
                (bounds[0], bounds[3]),  # min_x, max_y
                (bounds[2], bounds[3]),  # max_x, max_y
            ]
            # Project corners relative to centroid
            cx, _, _ = self.section.get_transformed_centroid(self.E_c_eff)
            projs = [dx * (x - cx) + dy * (y - c_y) for x, y in corners]
            y_tension = abs(min(projs))  # Distance from centroid to tension face

            W_el = I_eff / y_tension if y_tension > 0 else I_eff / (self.height / 2)
        else:
            # Uniaxial: I_xx about horizontal axis
            y_tension = c_y - bounds[1]  # Distance to bottom (tension) face
            W_el = I_xx / y_tension if y_tension > 0 else I_xx / (self.height / 2)

        # Axial stress contribution (compression positive increases M_cr)
        sigma_N = 0.0
        if N_Ed != 0.0:
            A_tr = self.section.get_transformed_area(self.E_c_eff)
            sigma_N = N_Ed * 1000 / A_tr  # kN → N, then N/mm² = MPa

        # M_cr in kN·m (W_el in mm³, stresses in MPa → result in N·mm)
        return to_knm((f_ct_eff + sigma_N) * W_el, MomentUnit.NMM)


    # ===============================================
    # h_c,ef calculation (EC2 §7.3.2(3), Fig 7.1)
    # ===============================================

    def find_h_c_ef(
        self,
        d: float,
        x: Optional[float] = None,
        *,
        h_override: Optional[float] = None,
    ) -> float:
        """
        Effective height of concrete in tension zone h_c,ef (EC2 §7.3.2(3), Fig 7.1).

        h_c,ef = min(f(h,d)·(h-d), (h-x)/3, h/2)

        where:
        - h = section depth
        - d = effective depth to tension steel
        - x = neutral axis depth from compression face
        - f(h,d) = 2.5 (base EC2) or NDP-dependent multiplier (e.g. German NA bilinear)

        Note: For sections fully in tension (both faces), use find_h_c_ef_tension_member()
        instead, which calculates separate h_c,ef values for each face.

        Args:
            d: Effective depth to tension reinforcement (mm)
            x: Neutral axis depth from compression face (mm), or None for uncracked
            h_override: Optional section height override (for biaxial NA, use ``h_perp``).

        Returns:
            Effective concrete height in tension zone (mm)
        """
        h = h_override if h_override is not None else self.height

        # NDP-dependent multiplier on (h - d)
        multiplier_func = get_ndp("h_c_ef_multiplier")
        if multiplier_func is not None and callable(multiplier_func):
            d_1_term = multiplier_func(h, d) * (h - d)
        else:
            d_1_term = 2.5 * (h - d)

        candidates = [
            d_1_term,
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

        multiplier_func = get_ndp("h_c_ef_multiplier")

        if multiplier_func is not None and callable(multiplier_func):
            # d_top is the distance from top face to bars (d_1 for top face).
            # multiplier_func(h, d) computes from d_1 = h - d, so pass d = h - d_top.
            h_c_ef_top = multiplier_func(h, h - d_top) * d_top
            h_c_ef_bottom = multiplier_func(h, h - d_bottom) * d_bottom
        else:
            h_c_ef_top = min(2.5 * d_top, h / 2)
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
        A_c_eff: Optional[float] = None,
    ) -> float:
        """
        Effective reinforcement ratio ρ_p,eff (EC2 §7.3.4(2)).

        ρ_p,eff = (A_s + ξ₁ × A_p') / A_c,eff

        where A_c,eff = h_c,ef × b (or a pre-computed value for widely-spaced bars)

        Args:
            A_s_tension: Area of tension reinforcement (mm²)
            h_c_ef: Effective height of concrete in tension (mm)
            xi_1: Adjusted ratio of bond strengths (ξ₁), default 0 for no prestress
            A_p: Area of prestressing tendons (mm²), default 0
            A_c_eff: Pre-computed effective concrete area (mm²). When provided,
                overrides h_c,ef × b. Used for widely-spaced bars where per-bar
                zones reduce the effective area (EC2 Figure 7.2).

        Returns:
            Effective reinforcement ratio (dimensionless)
        """
        if A_c_eff is None:
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

        .. seealso::
            :meth:`find_k_2_from_strain_state` for biaxial bending, which
            projects extreme fibre strains along
            ``strain_state.compression_direction``.

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


    def find_k_2_from_strain_state(
        self,
        strain_state: "StrainState",
    ) -> float:
        """
        Strain distribution coefficient k_2 from a 2D strain state (EC2 §7.3.4(3)).

        For biaxial bending the extreme fibre strains are projected along the
        ``compression_direction`` of the strain state rather than taken from
        ``eps_top``/``eps_bottom``.  The same k₂ formula applies:

        k_2 = (ε₁ + ε₂) / (2 × ε₁)

        where ε₁ is the greater and ε₂ the lesser tensile strain at the
        extreme fibres along the compression direction.

        Falls back to ``find_k_2(eps_top, eps_bottom)`` when the strain state
        is uniaxial.

        Args:
            strain_state: Full 2D strain state from the solver.

        Returns:
            k_2 coefficient (0.5 to 1.0)
        """
        if not strain_state.is_biaxial:
            # Uniaxial: fall back to existing method using eps_top/eps_bottom
            bounds = self.section.outline.bounds  # (minx, miny, maxx, maxy)
            _, cy, _ = self.section.get_transformed_centroid(self.E_c_eff)
            eps_top = strain_state.strain_at(0.0, bounds[3] - cy)
            eps_bottom = strain_state.strain_at(0.0, bounds[1] - cy)
            return self.find_k_2(eps_top, eps_bottom)

        # Biaxial: project all concrete fibres along compression_direction
        # to find extreme fibre strains
        diag = self._get_diagram()
        conc_mask = diag._fibre_mat == "concrete"
        fibre_x = diag._fibre_x[conc_mask]
        fibre_y = diag._fibre_y[conc_mask]
        cx, cy = diag._section_cx, diag._section_cy

        min_proj, max_proj = strain_state.perpendicular_extent(
            fibre_x, fibre_y, cx, cy,
        )

        # Strain at extreme fibres along compression direction.
        # Because the strain field is planar, all points at the same
        # projection along the gradient have identical strain.
        dx, dy = strain_state.compression_direction
        eps_tension = strain_state.strain_at(dx * min_proj, dy * min_proj)
        eps_compression = strain_state.strain_at(dx * max_proj, dy * max_proj)

        # Reuse existing k_2 formula (eps_compression maps to "top",
        # eps_tension maps to "bottom" in the projected sense)
        return self.find_k_2(eps_compression, eps_tension)


    def find_maximum_crack_spacing(
        self,
        cover: float,
        phi_eq: float,
        rho_p_eff: float,
        k_2: float,
        x: Optional[float] = None,
        has_tension_reinforcement: bool = True,
        sigma_s: float = 0.0,
        bar_spacing: float = 0.0,
    ) -> float:
        """
        Maximum crack spacing s_r,max (EC2 §7.3.4(3), Eq. 7.11).

        Standard formula (spacing ≤ 5(c + φ/2)):
            s_r,max = k_3·c + k_1·k_2·k_4·φ / ρ_p,eff

        If bar_spacing > 5(c + φ/2) or no bonded reinforcement (Eq. 7.14):
            s_r_max = 1.3(h - x)

        An additional NDP upper limit s_r_max_lim may cap the Eq. 7.11 result
        (e.g. DIN EN 1992-1-1/NA: σ_s·φ / (3.6·f_ct,eff)).

        Args:
            cover: Concrete cover to tension reinforcement (mm)
            phi_eq: Equivalent bar diameter (mm)
            rho_p_eff: Effective reinforcement ratio (dimensionless)
            k_2: Strain distribution coefficient (0.5 for bending, 1.0 for tension)
            x: Neutral axis depth (mm), or None for uncracked/fully cracked
            has_tension_reinforcement: True if bonded reinforcement exists in tension zone
            sigma_s: Steel stress in tension reinforcement (MPa, positive).
                Required by some National Annexes for s_r,max upper limit.
            bar_spacing: Maximum centre-to-centre spacing of tension bars (mm).
                Used to trigger Eq. 7.14 when bar_spacing > 5(c + φ/2).

        Returns:
            Maximum crack spacing in mm
        """
        k_1 = self.find_k_1(k_2)

        # Standard formula (Eq. 7.11)
        if rho_p_eff > 0:
            s_r_max = self.k_3 * cover + (k_1 * k_2 * self.k_4 * phi_eq / rho_p_eff)
        else:
            # No reinforcement in tension zone - use upper bound
            s_r_max = float('inf')

        # NDP upper limit on s_r,max (e.g. DIN NA), applied before spacing check
        s_r_max_lim = get_ndp("s_r_max_lim")
        if s_r_max_lim is not None and callable(s_r_max_lim) and sigma_s > 0:
            f_ct_eff = self.concrete.f_ctm
            s_r_max = min(s_r_max, s_r_max_lim(sigma_s, phi_eq, f_ct_eff))

        # Eq. 7.14: upper bound crack width estimate when spacing > 5(c + φ/2)
        # or no bonded reinforcement in h_c,ef zone.
        # Per IDEA StatiCa interpretation: Eq. 7.14 bounds w_k, not s_r,max.
        # Therefore take max(Eq.7.11, Eq.7.14) to be conservative.
        spacing_limit = 5 * (cover + phi_eq / 2)
        if bar_spacing > spacing_limit or not has_tension_reinforcement:
            h = self.height
            if x is not None and x > 0:
                s_r_max_7_14 = 1.3 * (h - x)
            else:
                s_r_max_7_14 = 1.3 * h
            s_r_max = max(s_r_max, s_r_max_7_14)

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
        - α_e = E_s / E_cm (using the passed E_s for the tension zone)
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
        alpha_e = flexure_utils.calculate_modular_ratio(
            E_s=E_s,
            E_cm=self.concrete.get_elastic_modulus(),
        )

        # Full formula
        if rho_p_eff > 0:
            stiffening_from_tension = self.k_t * f_ct_eff * (1 + alpha_e * rho_p_eff) / rho_p_eff
            eps_diff = (sigma_s - stiffening_from_tension) / E_s
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
            _, c_y, _ = self.section.get_transformed_centroid(self.E_c_eff)
            bounds = self.section.outline.bounds
            y_from_bottom = c_y - bounds[1]
            A_ct = y_from_bottom * b

        # f_ct,eff (could be f_ctm(t) for early age)
        # NDP lower bound (e.g. DIN EN 1992-1-1/NA: f_ct,eff ≥ 3.0 MPa)
        f_ct_eff = self.concrete.f_ctm
        f_ct_eff_min = get_ndp("f_ct_eff_min")
        if f_ct_eff_min is not None:
            f_ct_eff = max(f_ct_eff, cast(float, f_ct_eff_min))

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
            #! EC2 §7.3.2(2) doesn't state what to take if N_Ed = 0
            k_1 = 1.5  # Compression or zero axial
        else:
            k_1 = (2 * h_star) / (3 * h)  # Tension

        # Concrete stress from axial force
        A_eff = self.section.get_transformed_area(self.concrete.E_cm)
        sigma_c = from_kn(N_Ed, ForceUnit.N) / A_eff  # MPa

        f_ct_eff = self.concrete.f_ctm

        k_c = 0.4 * (1 - sigma_c / (k_1 * (h / h_star) * f_ct_eff))
        return min(1.0, max(0.0, k_c))


    def _compute_uncracked_na_depth(self, compression_face: str = "top") -> float:
        """Neutral axis depth in State I (uncracked transformed section).

        Uses the section's transformed centroid (concrete + modular ratio steel)
        to determine the elastic NA depth from the compression face.

        Args:
            compression_face: "top" or "bottom"

        Returns:
            x_I in mm (depth from compression face to transformed centroid)
        """
        E_cm = self.concrete.E_cm
        _, _, cy_tr = self.section.get_transformed_centroid(E_cm)
        bounds = self.section.outline.bounds
        y_min, y_max = bounds[1], bounds[3]

        if compression_face == "top":
            return y_max - cy_tr
        else:
            return cy_tr - y_min


    # ===============================================
    # Helper methods for rebar analysis
    # ===============================================
    def _get_tension_rebar_info(
        self,
        eps_top: float,
        eps_bottom: float,
        face: Optional[str] = None,
        h_c_ef_limit: Optional[float] = None,
        strain_state: Optional[StrainState] = None,
    ) -> Tuple[float, float, List[Tuple[float, int]]]:
        """
        Get tension reinforcement information from strain state.

        Args:
            eps_top: Top fibre strain (compression positive)
            eps_bottom: Bottom fibre strain (compression positive)
            face: For net tension, restrict to bars near this face ("top" or "bottom").
                Bars are assigned to the nearest bounding face (bottom takes ties).
                When None, all tension bars are included.
            h_c_ef_limit: When provided, only include bars within this distance of
                the tension face. Used by the iterative h_c,ef process.
            strain_state: Optional full 2D strain state for biaxial evaluation.

        Returns:
            Tuple of (total_area, mean_cover, bar_sizes) where:
            - total_area: Total area of tension reinforcement (mm²)
            - mean_cover: Area-weighted mean cover to tension bars (mm)
            - bar_sizes: List of (diameter, count) for equivalent diameter calc
        """
        use_biaxial = strain_state is not None and strain_state.is_biaxial

        bounds = self.section.outline.bounds
        h = bounds[3] - bounds[1]
        y_min = bounds[1]
        y_max = bounds[3]
        tension_bars: List[Tuple[float, int]] = []
        total_area = 0.0
        cover_sum = 0.0

        # 1D path: determine which face is the tension reference for cover calculation
        if not use_biaxial:
            comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)
            cover_ref = (face or "bottom") if comp_face is None else ("bottom" if comp_face == "top" else "top")
            cx = cy = 0.0

        for group in self.section.rebar_groups:
            diameter = float(group.rebar.diameter)
            bar_area = float(group.rebar.area)
            bar_count = 0

            for pos in group.positions:
                # --- Strain at bar location ---
                if use_biaxial:
                    cx, cy = self.section.get_centroid()
                    strain_at_bar = strain_state.strain_at(  # type: ignore[union-attr]
                        float(pos.x) - cx, float(pos.y) - cy,
                    )
                else:
                    y_rel = (pos.y - y_min) / h
                    strain_at_bar = eps_bottom + (eps_top - eps_bottom) * y_rel

                # Only tension bars (negative strain)
                if strain_at_bar >= 0:
                    continue

                if use_biaxial:
                    # --- 2D path: projection-based distance and cover ---
                    dist_from_face = self._compute_biaxial_bar_face_distance(pos, strain_state)  # type: ignore[arg-type]
                    bar_cover = self._compute_biaxial_cover(pos, diameter)
                    # Net-tension face filter: skip (face concept ill-defined for rotated NA;
                    # all tension bars contribute regardless of face label)
                    if h_c_ef_limit is not None and dist_from_face > h_c_ef_limit:
                        continue
                else:
                    # --- 1D path: Y-axis face filter, h_c,ef filter, cover ---
                    if face is not None and comp_face is None:  # type: ignore[possibly-undefined]
                        dist_to_bottom = pos.y - y_min
                        dist_to_top = y_max - pos.y
                        if face == "bottom" and dist_to_bottom > dist_to_top:
                            continue
                        if face == "top" and dist_to_bottom <= dist_to_top:
                            continue
                    if h_c_ef_limit is not None:
                        dist_from_face = pos.y - y_min if cover_ref == "bottom" else y_max - pos.y  # type: ignore[possibly-undefined]
                        if dist_from_face > h_c_ef_limit:
                            continue
                    bar_cover = (pos.y - y_min if cover_ref == "bottom" else y_max - pos.y) - diameter / 2  # type: ignore[possibly-undefined]

                bar_count += 1
                total_area += bar_area
                cover_sum += bar_area * max(0.0, bar_cover)

            if bar_count > 0:
                tension_bars.append((diameter, bar_count))

        mean_cover = cover_sum / total_area if total_area > 0 else 0.0

        return total_area, mean_cover, tension_bars


    def _get_steel_stress(
        self,
        eps_top: float,
        eps_bottom: float,
        face: Optional[str] = None,
        h_c_ef_limit: Optional[float] = None,
        strain_state: Optional[StrainState] = None,
    ) -> float:
        """
        Get maximum steel stress in tension zone from strain state.

        Note: Returns the absolute value of stress (always positive),
        even though tension strains are negative by convention.

        Args:
            eps_top: Top fibre strain (compression positive)
            eps_bottom: Bottom fibre strain (compression positive)
            face: For net tension, restrict to bars near this face.
            h_c_ef_limit: Only consider bars within this distance of the tension face.

        Returns:
            Maximum tensile stress in reinforcement (MPa, always positive)
        """
        use_biaxial = strain_state is not None and strain_state.is_biaxial

        bounds = self.section.outline.bounds
        h = bounds[3] - bounds[1]
        y_min = bounds[1]
        y_max = bounds[3]

        # 1D path only: determine tension face reference
        if not use_biaxial:
            comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)
            cover_ref = (face or "bottom") if comp_face is None else ("bottom" if comp_face == "top" else "top")
            cx = cy = 0.0

        max_tension_stress = 0.0

        for group in self.section.rebar_groups:
            E_s = group.rebar.E_s
            f_yk = group.rebar.f_yk
            epsilon_uk = group.rebar.epsilon_uk
            k_ratio = group.rebar.grade.ft_ratio_min  # Hardening ratio

            for pos in group.positions:
                # --- Strain at bar location ---
                if use_biaxial:
                    cx, cy = self.section.get_centroid()
                    strain_at_bar = strain_state.strain_at(  # type: ignore[union-attr]
                        float(pos.x) - cx, float(pos.y) - cy,
                    )
                else:
                    y_rel = (pos.y - y_min) / h
                    strain_at_bar = eps_bottom + (eps_top - eps_bottom) * y_rel

                # Only consider tension (negative strain)
                if strain_at_bar >= 0:
                    continue

                if use_biaxial:
                    # --- 2D path: projection-based distance filter ---
                    dist_from_face = self._compute_biaxial_bar_face_distance(pos, strain_state)  # type: ignore[arg-type]
                    if h_c_ef_limit is not None and dist_from_face > h_c_ef_limit:
                        continue
                    # Net-tension face filter: all tension bars included for biaxial
                else:
                    # --- 1D path: Y-axis face and h_c,ef filters ---
                    if face is not None and comp_face is None:  # type: ignore[possibly-undefined]
                        dist_to_bottom = pos.y - y_min
                        dist_to_top = y_max - pos.y
                        if face == "bottom" and dist_to_bottom > dist_to_top:
                            continue
                        if face == "top" and dist_to_bottom <= dist_to_top:
                            continue
                    if h_c_ef_limit is not None:
                        dist_from_face = pos.y - y_min if cover_ref == "bottom" else y_max - pos.y  # type: ignore[possibly-undefined]
                        if dist_from_face > h_c_ef_limit:
                            continue

                stress = flexure_utils.calculate_rebar_characteristic_stress_from_strain(
                    strain=strain_at_bar,
                    steel_model_type=self.steel_model_type,
                    E_s=E_s,
                    f_yk=f_yk,
                    k=k_ratio,
                    epsilon_uk=epsilon_uk,
                )
                max_tension_stress = max(max_tension_stress, abs(stress))

        return max_tension_stress


    def _get_tension_zone_E_s(
        self,
        eps_top: float,
        eps_bottom: float,
        strain_state: Optional[StrainState] = None,
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
        use_biaxial = strain_state is not None and strain_state.is_biaxial

        bounds = self.section.outline.bounds
        h = bounds[3] - bounds[1]
        y_min = bounds[1]

        # 1D path: compression face to determine tension direction
        if not use_biaxial:
            comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)
            cx = cy = 0.0

        # Track the outermost tension bar: by Y-coordinate (1D) or by projection distance (2D)
        outermost_y: Optional[float] = None
        outermost_dist: float = -1.0
        outermost_E_s = first_E_s

        for group in self.section.rebar_groups:
            for pos in group.positions:
                # --- Strain at bar location ---
                if use_biaxial:
                    cx, cy = self.section.get_centroid()
                    strain = strain_state.strain_at(  # type: ignore[union-attr]
                        float(pos.x) - cx, float(pos.y) - cy,
                    )
                else:
                    y_rel = (pos.y - y_min) / h
                    strain = eps_bottom + (eps_top - eps_bottom) * y_rel
                if strain >= 0:  # Compression, skip
                    continue

                if use_biaxial:
                    # --- 2D path: outermost = furthest from compression face along compression_direction ---
                    dist = self._compute_biaxial_bar_face_distance(pos, strain_state)  # type: ignore[arg-type]
                    if dist > outermost_dist:
                        outermost_dist = dist
                        outermost_E_s = group.rebar.E_s
                else:
                    # --- 1D path: outermost = lowest (tension at bottom) or highest (tension at top) ---
                    if comp_face == "top":  # type: ignore[possibly-undefined]
                        if outermost_y is None or pos.y < outermost_y:
                            outermost_y = pos.y
                            outermost_E_s = group.rebar.E_s
                    else:
                        if outermost_y is None or pos.y > outermost_y:
                            outermost_y = pos.y
                            outermost_E_s = group.rebar.E_s

        return outermost_E_s


    def _compute_max_bar_spacing(
        self,
        eps_top: float,
        eps_bottom: float,
        face: Optional[str] = None,
        h_c_ef_limit: Optional[float] = None,
        strain_state: Optional[StrainState] = None,
    ) -> float:
        """
        Maximum centre-to-centre spacing between adjacent tension bars.

        Collects qualifying bar positions using the same filters as
        ``_get_tension_rebar_info`` (strain, face, h_c,ef), sorts by
        x-coordinate, and returns the max gap between consecutive bars.

        Args:
            eps_top: Top fibre strain (compression positive)
            eps_bottom: Bottom fibre strain (compression positive)
            face: For net tension, restrict to bars near this face.
            h_c_ef_limit: Only consider bars within this distance of the
                tension face.

        Returns:
            Maximum centre-to-centre bar spacing (mm), or 0.0 if < 2 bars.
        """
        use_biaxial = strain_state is not None and strain_state.is_biaxial

        bounds = self.section.outline.bounds
        h = bounds[3] - bounds[1]
        y_min = bounds[1]
        y_max = bounds[3]

        # 1D path only: tension face reference
        if not use_biaxial:
            comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)
            cover_ref = (face or "bottom") if comp_face is None else ("bottom" if comp_face == "top" else "top")
            cx = cy = 0.0

        qualifying: List[Point2D] = []

        for group in self.section.rebar_groups:
            for pos in group.positions:
                # --- Strain filter — tension only ---
                if use_biaxial:
                    cx, cy = self.section.get_centroid()
                    strain_at_bar = strain_state.strain_at(  # type: ignore[union-attr]
                        float(pos.x) - cx, float(pos.y) - cy,
                    )
                else:
                    y_rel = (pos.y - y_min) / h
                    strain_at_bar = eps_bottom + (eps_top - eps_bottom) * y_rel
                if strain_at_bar >= 0:
                    continue

                if use_biaxial:
                    # --- 2D path: projection-based h_c,ef filter ---
                    dist_from_face = self._compute_biaxial_bar_face_distance(pos, strain_state)  # type: ignore[arg-type]
                    if h_c_ef_limit is not None and dist_from_face > h_c_ef_limit:
                        continue
                    # Net-tension face filter: all tension bars included
                else:
                    # --- 1D path: Y-axis face and h_c,ef filters ---
                    if face is not None and comp_face is None:  # type: ignore[possibly-undefined]
                        dist_to_bottom = pos.y - y_min
                        dist_to_top = y_max - pos.y
                        if face == "bottom" and dist_to_bottom > dist_to_top:
                            continue
                        if face == "top" and dist_to_bottom <= dist_to_top:
                            continue
                    if h_c_ef_limit is not None:
                        dist_from_face = pos.y - y_min if cover_ref == "bottom" else y_max - pos.y  # type: ignore[possibly-undefined]
                        if dist_from_face > h_c_ef_limit:
                            continue

                qualifying.append(pos)

        if len(qualifying) < 2:
            return 0.0

        # Sort bars and measure spacing
        if use_biaxial and self.bar_spacing_policy == "perpendicular_to_na":
            # Sort along the tension face: perpendicular to compression_direction
            dx, dy = strain_state.compression_direction  # type: ignore[union-attr]
            qualifying.sort(key=lambda p: -dy * float(p.x) + dx * float(p.y))
        else:
            qualifying.sort(key=lambda p: p.x)

        max_spacing = 0.0
        for i in range(len(qualifying) - 1):
            spacing = qualifying[i].distance_to(qualifying[i + 1])
            max_spacing = max(max_spacing, spacing)

        return max_spacing


    def _compute_A_c_eff(
        self,
        eps_top: float,
        eps_bottom: float,
        h_c_ef: float,
        cover: float,
        phi_eq: float,
        bar_spacing: float,
        face: Optional[str] = None,
        h_c_ef_limit: Optional[float] = None,
        strain_state: Optional["StrainState"] = None,
        breadth_override: Optional[float] = None,
    ) -> float:
        """
        Effective concrete area in tension A_c,eff (EC2 §7.3.4, Figure 7.2).

        When bar spacing <= 5(c + phi/2), uses full section width:
            A_c,eff = h_c,ef * b

        When bar spacing > 5(c + phi/2), each bar has its own effective zone
        of width 5(c_i + phi_i/2) using per-bar cover and diameter, capped at
        the section edges:
            A_c,eff = h_c,ef * sum(zone widths)

        Args:
            eps_top: Top fibre strain (compression positive)
            eps_bottom: Bottom fibre strain (compression positive)
            h_c_ef: Effective height of concrete in tension (mm)
            cover: Global concrete cover (mm) — used for spacing limit check
            phi_eq: Equivalent bar diameter (mm) — used for spacing limit check
            bar_spacing: Maximum bar spacing (mm), from _compute_max_bar_spacing
            face: For net tension, restrict to bars near this face
            h_c_ef_limit: Only consider bars within this distance of the face
            strain_state: Optional full 2D strain state for biaxial evaluation.
            breadth_override: Optional breadth override (for biaxial NA, use ``b_perp``).

        Returns:
            Effective concrete area in tension (mm²)
        """
        b_eff = breadth_override if breadth_override is not None else self.breadth
        spacing_limit = 5 * (cover + phi_eq / 2)
        if bar_spacing <= spacing_limit:
            return h_c_ef * b_eff

        # --- Collect qualifying bars with per-bar properties ---
        use_biaxial = strain_state is not None and strain_state.is_biaxial

        bounds = self.section.outline.bounds
        x_min = bounds[0]
        x_max = bounds[2]
        h = bounds[3] - bounds[1]
        y_min = bounds[1]
        y_max = bounds[3]

        # 1D path only: tension face reference
        if not use_biaxial:
            comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)
            cover_ref = (face or "bottom") if comp_face is None else ("bottom" if comp_face == "top" else "top")
            cx = cy = 0.0

        # Each entry: (bar_x, bar_diameter, bar_cover_to_tension_face)
        qualifying: List[Tuple[float, float, float]] = []

        for group in self.section.rebar_groups:
            diameter = float(group.rebar.diameter)

            for pos in group.positions:
                # --- Strain filter — tension only ---
                if use_biaxial:
                    cx, cy = self.section.get_centroid()
                    strain_at_bar = strain_state.strain_at(  # type: ignore[union-attr]
                        float(pos.x) - cx, float(pos.y) - cy,
                    )
                else:
                    y_rel = (pos.y - y_min) / h
                    strain_at_bar = eps_bottom + (eps_top - eps_bottom) * y_rel
                if strain_at_bar >= 0:
                    continue

                if use_biaxial:
                    # --- 2D path: projection-based distance and Euclidean cover ---
                    dist_from_face = self._compute_biaxial_bar_face_distance(pos, strain_state)  # type: ignore[arg-type]
                    bar_cover = max(0.0, self._compute_biaxial_cover(pos, diameter))
                    if h_c_ef_limit is not None and dist_from_face > h_c_ef_limit:
                        continue
                    # Net-tension face filter: all tension bars included for biaxial
                else:
                    # --- 1D path: Y-axis face and h_c,ef filters ---
                    if face is not None and comp_face is None:  # type: ignore[possibly-undefined]
                        dist_to_bottom = pos.y - y_min
                        dist_to_top = y_max - pos.y
                        if face == "bottom" and dist_to_bottom > dist_to_top:
                            continue
                        if face == "top" and dist_to_bottom <= dist_to_top:
                            continue
                    if h_c_ef_limit is not None:
                        dist_from_face = pos.y - y_min if cover_ref == "bottom" else y_max - pos.y  # type: ignore[possibly-undefined]
                        if dist_from_face > h_c_ef_limit:
                            continue
                    bar_cover = max(0.0, (pos.y - y_min if cover_ref == "bottom" else y_max - pos.y) - diameter / 2)  # type: ignore[possibly-undefined]

                qualifying.append((float(pos.x), diameter, bar_cover))

        if not qualifying:
            return h_c_ef * b_eff

        # Sum per-bar effective zone widths, capped at section edges
        total_width = 0.0
        for bar_x, bar_dia, bar_cover in qualifying:
            half_zone = 2.5 * (bar_cover + bar_dia / 2)
            left = max(x_min, bar_x - half_zone)
            right = min(x_max, bar_x + half_zone)
            total_width += max(0.0, right - left)

        # Cap at full breadth
        total_width = min(total_width, b_eff)

        return h_c_ef * total_width


    # ===============================================
    # Stress limitation helpers (EC2 §7.2)
    # ===============================================

    def _get_peak_concrete_stress(
        self,
        eps_top: float,
        eps_bottom: float,
        diagram: Optional[MNInteractionDiagram] = None,
        strain_state: Optional[StrainState] = None,
    ) -> float:
        """
        Peak compressive stress in concrete from fibre integration.

        Args:
            eps_top: Top fibre strain (compression positive)
            eps_bottom: Bottom fibre strain (compression positive)
            diagram: Diagram to use (defaults to self._get_diagram())
            strain_state: Optional full 2D strain state for biaxial evaluation.

        Returns:
            Peak compressive stress in MPa (positive)
        """
        diag = diagram or self._get_diagram()
        if strain_state is not None and strain_state.is_biaxial:
            forces, _x, _y, areas = diag.get_fibre_forces_from_strain_state(strain_state)
        else:
            forces, _y, areas = diag.get_fibre_forces_from_end_strains(eps_top, eps_bottom)

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
        # Use Python max() to avoid NumPy max() default-sentinel edge cases seen
        # when NumPy is reloaded in some test environments.
        peak = float(max(float(s) for s in conc_stresses)) if len(conc_stresses) > 0 else 0.0
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
        return compute_nonlinear_creep_coefficient(
            sigma_c, self.concrete.f_ck, self.creep_coefficient
        )


    def _build_diagram_with_E_c_eff(
        self, E_c_eff: float, ignore_compression_steel: bool = False,
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
            elastic_modulus=E_c_eff,
            include_tension=True,
            crack_to_neutral_axis_on_first_tension_failure=True,
        )

    @staticmethod
    def _extract_solver_residuals(error_message: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract dN and dM residuals from inverse solver error text."""
        match = re.search(
            r"dN=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*kN,\s*dM=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*kN\.m",
            error_message,
        )
        if not match:
            return (None, None)
        try:
            return (float(match.group(1)), float(match.group(2)))
        except ValueError:
            return (None, None)

    def _build_unsolved_cracking_result(
        self,
        *,
        is_cracked: bool,
        solver_stage: str,
        solver_error: str,
        sigma_c_peak: Optional[float] = None,
        nonlinear_creep_applied: bool = False,
        creep_coefficient_used: Optional[float] = None,
    ) -> CrackingResult:
        """Create a detailed result payload for non-solvable load cases."""
        dN, dM = self._extract_solver_residuals(solver_error)
        return CrackingResult(
            w_k=None,
            w_k_limit=self.w_k_limit,
            s_r_max=None,
            eps_sm_minus_eps_cm=None,
            sigma_s=None,
            rho_p_eff=None,
            h_c_ef=None,
            x=None,
            is_cracked=is_cracked,
            phi_eq=None,
            cover=None,
            solved=False,
            solver_stage=solver_stage,
            solver_error=solver_error,
            solver_residual_N=dN,
            solver_residual_M=dM,
            sigma_c_peak=sigma_c_peak,
            nonlinear_creep_applied=nonlinear_creep_applied,
            creep_coefficient_used=(
                float(creep_coefficient_used)
                if creep_coefficient_used is not None
                else float(self.creep_coefficient)
            ),
        )

    @staticmethod
    def _require_result_value(value: Optional[float], field_name: str) -> float:
        """Return a float from an optional result field, or raise a clear error."""
        if value is None:
            raise RuntimeError(
                f"Expected solved cracking result to define '{field_name}', but got None."
            )
        return float(value)

    def _is_cracked_by_solver(
        self,
        *,
        M_Ed: float,
        N_Ed: float,
        ignore_compression_steel: bool = False,
        _mz_kw: Optional[dict] = None,
    ) -> Tuple[bool, float, float, Optional[float], Optional[float]]:
        """
        Determine cracked state using an uncracked solver pass.

        Returns:
            (is_cracked, eps_top, eps_bottom, min_tension_concrete_strain, cracking_strain)
        """
        probe_diagram = self._get_uncracked_diagram(ignore_compression_steel)
        eps_top, eps_bottom = probe_diagram.find_strains_for_MN(
            My_target=M_Ed,
            N_target=N_Ed,
            strict=True,
            **(_mz_kw or {}),
        )

        # No concrete tension zone -> uncracked for cracking-width purposes
        if eps_top >= 0.0 and eps_bottom >= 0.0:
            return False, float(eps_top), float(eps_bottom), None, None

        conc_mask = probe_diagram._fibre_mat == "concrete"
        if not np.any(conc_mask):
            return False, float(eps_top), float(eps_bottom), None, None

        concrete_strains = probe_diagram._strain_field_from_end_strains(
            eps_top=float(eps_top),
            eps_bottom=float(eps_bottom),
        )[conc_mask]
        concrete_strains_real = np.real(concrete_strains)
        tension_mask = concrete_strains_real < 0.0
        if not np.any(tension_mask):
            return False, float(eps_top), float(eps_bottom), None, None

        min_tension_concrete_strain = float(np.min(concrete_strains_real[tension_mask]))

        cracking_strain: Optional[float] = None
        if isinstance(probe_diagram.concrete_model, ConcreteStressStrainLinearElastic):
            cracking_strain = float(probe_diagram.concrete_model.cracking_strain)
            return (
                bool(min_tension_concrete_strain < cracking_strain),
                float(eps_top),
                float(eps_bottom),
                min_tension_concrete_strain,
                cracking_strain,
            )

        # Generic fallback for non-linear concrete models.
        # (For current probe setup this branch is not expected.)
        return (
            bool(min_tension_concrete_strain < 0.0),
            float(eps_top),
            float(eps_bottom),
            min_tension_concrete_strain,
            cracking_strain,
        )


    @staticmethod
    def _compute_bar_diameter_correction(
        s_r_max: float,
        phi_eq: float,
        actual_bar_diameter: float,
        cover: float,
        k_3: float,
    ) -> Tuple[float, float]:
        """
        Correct s_r,max for equivalent-area bar substitution.

        When a non-integer bar count is rounded and the diameter adjusted to
        preserve total area, the φ-dependent term of s_r,max (Eq. 7.11) must
        be corrected to use the real bar diameter.

        s_r,max = k_3·c + (k_1·k_2·k_4·φ / ρ_p,eff)
                  ^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^
                  cover     φ-dependent term
                  term

        The cover term is unchanged. The φ-dependent term is scaled by
        φ_actual / φ_eq.

        Args:
            s_r_max: Original s_r,max computed with model φ_eq (mm)
            phi_eq: Equivalent bar diameter from the model (mm)
            actual_bar_diameter: Real bar diameter (mm)
            cover: Concrete cover used in s_r,max (mm)
            k_3: NDP coefficient k_3

        Returns:
            Tuple of (corrected s_r,max, correction factor φ_actual/φ_eq)
        """
        if phi_eq <= 0:
            return s_r_max, 1.0

        factor = actual_bar_diameter / phi_eq
        cover_term = k_3 * cover
        phi_term = s_r_max - cover_term
        s_r_max_corrected = cover_term + phi_term * factor
        return s_r_max_corrected, factor

    def _get_f_yk_max(self) -> float:
        """Maximum f_yk across all rebar groups."""
        if not self.section.rebar_groups:
            return 500.0
        return max(g.rebar.f_yk for g in self.section.rebar_groups)


    # ===============================================
    # Biaxial geometry helpers
    # ===============================================

    def _compute_biaxial_section_extents(
        self,
        strain_state: "StrainState",
    ) -> tuple[float, float]:
        """
        Compute section height and breadth for a rotated NA.

        Returns:
            ``(h_perp, b_perp)`` where ``h_perp`` is the section extent along
            the compression direction (replaces ``self.height``) and ``b_perp``
            is the extent perpendicular to it (replaces ``self.breadth``).
        """
        import numpy as np
        from shapely.geometry import Polygon

        dx, dy = strain_state.compression_direction
        if dx == 0.0 and dy == 0.0:
            return (self.height, self.breadth)

        # Get section boundary vertices
        outline: Polygon = self.section.outline
        coords = np.array(outline.exterior.coords)
        cx, cy = self.section.get_centroid()

        # Project onto compression direction => h_perp
        proj_comp = dx * (coords[:, 0] - cx) + dy * (coords[:, 1] - cy)
        h_perp = float(np.max(proj_comp) - np.min(proj_comp))

        # Project onto perpendicular direction => b_perp
        # Perpendicular to (dx, dy) is (-dy, dx)
        proj_perp = -dy * (coords[:, 0] - cx) + dx * (coords[:, 1] - cy)
        b_perp = float(np.max(proj_perp) - np.min(proj_perp))

        return (h_perp, b_perp)

    def _compute_biaxial_bar_face_distance(
        self,
        pos: Point2D,
        strain_state: "StrainState",
    ) -> float:
        """
        Distance from bar to section face along compression direction (tension side).

        For biaxial NA, measures how far the bar is from the tension-side face
        along the compression direction (replaces ``pos.y - y_min`` / ``y_max - pos.y``).
        """
        import numpy as np
        from shapely.geometry import Polygon

        dx, dy = strain_state.compression_direction
        cx, cy = self.section.get_centroid()

        outline: Polygon = self.section.outline
        coords = np.array(outline.exterior.coords)

        # All projections along compression direction
        proj_comp = dx * (coords[:, 0] - cx) + dy * (coords[:, 1] - cy)
        min_proj = float(np.min(proj_comp))  # tension extreme

        bar_proj = dx * (float(pos.x) - cx) + dy * (float(pos.y) - cy)
        return bar_proj - min_proj

    def _compute_biaxial_cover(
        self,
        pos: Point2D,
        diameter: float,
    ) -> float:
        """
        Geometric cover from bar to nearest section boundary (Euclidean).

        Uses Shapely distance from bar centre to section outline minus bar radius.
        """
        from shapely.geometry import Point

        bar_centre = Point(float(pos.x), float(pos.y))
        dist_to_boundary = self.section.outline.exterior.distance(bar_centre)
        return max(0.0, dist_to_boundary - diameter / 2.0)

    # ===============================================
    # Face-based crack width calculation
    # ===============================================

    def _calculate_face_crack_width(
        self,
        eps_top: float,
        eps_bottom: float,
        face: Literal["top", "bottom"],
        x: Optional[float],
        is_net_tension: bool,
        suppress_warnings: bool = False,
        actual_bar_diameter: Optional[float] = None,
        cover_override: Optional[float] = None,
        strain_state: Optional["StrainState"] = None,
        h_override: Optional[float] = None,
        breadth_override: Optional[float] = None,
        d_override: Optional[float] = None,
    ) -> CrackingResult:
        """
        Calculate crack width for a single face using iterative h_c,ef.

        Uses an iterative process (based on IDEA StatiCa RCS) to determine
        which bars lie within the effective concrete tension zone:

        1. Start with all tension bars → compute d and h_c,ef
        2. Filter to bars within h_c,ef of the tension face
        3. If any bars excluded, recompute d and h_c,ef from remaining bars
        4. Repeat until stable (max 3 iterations)

        Args:
            eps_top: Top fibre strain (compression positive)
            eps_bottom: Bottom fibre strain (compression positive)
            face: The tension face to compute w_k for ("top" or "bottom")
            x: Neutral axis depth from compression face (mm), or None
            is_net_tension: True if both faces are in tension
            suppress_warnings: If True, suppress warnings (used by viewer)
            h_override: Optional section height override (for biaxial NA, use ``h_perp``).
            breadth_override: Optional breadth override (for biaxial NA, use ``b_perp``).

        Returns:
            CrackingResult with all intermediate values
        """
        h = h_override if h_override is not None else self.height

        # --- Step 1: Initial bar set (all tension bars for this face) ---
        A_s, mean_cover, bar_sizes = self._get_tension_rebar_info(
            eps_top, eps_bottom,
            face=face if is_net_tension else None,
            strain_state=strain_state,
        )

        if A_s <= 0 or not bar_sizes:
            if not suppress_warnings:
                warnings.warn(
                    "No tension reinforcement found - cannot calculate crack width",
                    stacklevel=3,
                )
            return CrackingResult(
                w_k=0.0, w_k_limit=self.w_k_limit, s_r_max=0.0,
                eps_sm_minus_eps_cm=0.0, sigma_s=0.0, rho_p_eff=0.0,
                h_c_ef=0.0, x=x, is_cracked=True, phi_eq=0.0, cover=0.0,
            )

        # --- Step 2: Iterative h_c,ef determination ---
        # Compute initial d
        if d_override is not None:
            d = d_override
        elif is_net_tension:
            # For net tension: d from the face to the bar centroid
            comp_face_for_d = "bottom" if face == "top" else "top"
            d = self.section.get_effective_depth(
                compression_face=comp_face_for_d, zone_fraction=0.5,
            )
        else:
            comp_face = "top" if face == "bottom" else "bottom"
            d = self.section.get_effective_depth(compression_face=comp_face)

        # Iterative refinement (max 3 iterations)
        h_c_ef: float = h / 2  # Initial conservative estimate, refined below
        prev_bar_count = sum(cnt for _, cnt in bar_sizes)

        for _ in range(3):
            # Compute h_c,ef
            if is_net_tension:
                d_face = h - d  # Distance from face to bar centroid
                multiplier_func = get_ndp("h_c_ef_multiplier")
                if multiplier_func is not None and callable(multiplier_func):
                    h_c_ef = multiplier_func(h, d) * d_face
                else:
                    h_c_ef = min(2.5 * d_face, h / 2)
            else:
                h_c_ef = self.find_h_c_ef(d=d, x=x, h_override=h_override)

            # Filter bars to those within h_c,ef of the tension face
            A_s_filtered, mean_cover_filtered, bar_sizes_filtered = (
                self._get_tension_rebar_info(
                    eps_top, eps_bottom,
                    face=face if is_net_tension else None,
                    h_c_ef_limit=h_c_ef,
                    strain_state=strain_state,
                )
            )

            new_bar_count = sum(cnt for _, cnt in bar_sizes_filtered)

            if new_bar_count == 0:
                # No bars in h_c,ef zone — edge case (high compression, thin zone)
                if not is_net_tension and x is not None and x > 0:
                    relaxed_factor = get_ndp("h_c_ef_relaxed_na_factor")

                    if relaxed_factor is not None:
                        # German NA: first try (h - x_I) × factor
                        use_biaxial = strain_state is not None and strain_state.is_biaxial
                        if use_biaxial:
                            # For biaxial: approximate x_I as half the perpendicular extent
                            # (h already incorporates h_override = h_perp when set by caller)
                            x_I = h / 2.0
                        else:
                            comp_face_name = "top" if face == "bottom" else "bottom"
                            x_I = self._compute_uncracked_na_depth(comp_face_name)
                        h_c_ef_relaxed = (h - x_I) * cast(float, relaxed_factor)
                        if h_c_ef_relaxed > h_c_ef:
                            if not suppress_warnings:
                                warnings.warn(
                                    f"No bars within h_c,ef = {h_c_ef:.1f} mm "
                                    f"(governed by (h-x)/3). Relaxing to "
                                    f"(h-x_I)×{relaxed_factor} = {h_c_ef_relaxed:.1f} mm "
                                    f"(x_I = {x_I:.1f} mm, State I).",
                                    stacklevel=3,
                                )
                            h_c_ef = h_c_ef_relaxed
                            A_s_filtered, mean_cover_filtered, bar_sizes_filtered = (
                                self._get_tension_rebar_info(
                                    eps_top, eps_bottom,
                                    face=face if is_net_tension else None,
                                    h_c_ef_limit=h_c_ef,
                                    strain_state=strain_state,
                                )
                            )
                            new_bar_count = sum(cnt for _, cnt in bar_sizes_filtered)

                    # Fallback: drop NA term, use NDP multiplier × (h-d)
                    if new_bar_count == 0:
                        multiplier_func = get_ndp("h_c_ef_multiplier")
                        if multiplier_func is not None and callable(multiplier_func):
                            h_c_ef_no_na = multiplier_func(h, d) * (h - d)
                        else:
                            h_c_ef_no_na = min(2.5 * (h - d), h / 2)
                        if h_c_ef_no_na > h_c_ef:
                            if not suppress_warnings:
                                warnings.warn(
                                    f"No bars within h_c,ef = {h_c_ef:.1f} mm "
                                    f"(governed by NA term). Relaxing to "
                                    f"{h_c_ef_no_na:.1f} mm (NA term dropped).",
                                    stacklevel=3,
                                )
                            h_c_ef = h_c_ef_no_na
                            A_s_filtered, mean_cover_filtered, bar_sizes_filtered = (
                                self._get_tension_rebar_info(
                                    eps_top, eps_bottom,
                                    face=face if is_net_tension else None,
                                    h_c_ef_limit=h_c_ef,
                                    strain_state=strain_state,
                                )
                            )
                            new_bar_count = sum(cnt for _, cnt in bar_sizes_filtered)

                if new_bar_count == 0:
                    # Still no bars — use all tension bars with relaxed h_c,ef
                    break

            # Update bar set
            A_s, mean_cover, bar_sizes = A_s_filtered, mean_cover_filtered, bar_sizes_filtered

            # Check convergence
            if new_bar_count == prev_bar_count:
                break
            prev_bar_count = new_bar_count

            # Recompute d from the remaining bars
            # (approximate: use mean_cover + estimated bar radius)
            # Better: use section's get_effective_depth with zone_fraction
            # For now, use the original d (stable enough for iteration)

        if A_s <= 0 or not bar_sizes:
            return CrackingResult(
                w_k=0.0, w_k_limit=self.w_k_limit, s_r_max=0.0,
                eps_sm_minus_eps_cm=0.0, sigma_s=0.0, rho_p_eff=0.0,
                h_c_ef=h_c_ef or 0.0, x=x, is_cracked=True,
                phi_eq=0.0, cover=0.0,
            )

        # --- Step 3: Compute crack width components ---
        phi_eq = flexure_utils.find_equivalent_diameter(bar_sizes)

        # Cover (computed early — needed for A_c,eff)
        if cover_override is not None:
            cover = cover_override
        else:
            try:
                cover = self.section.get_concrete_cover(reference=face)
            except ValueError:
                cover = mean_cover

        # Max bar spacing (computed early — needed for A_c,eff)
        bar_spacing = self._compute_max_bar_spacing(
            eps_top, eps_bottom,
            face=face if is_net_tension else None,
            h_c_ef_limit=h_c_ef,
            strain_state=strain_state,
        )

        # Effective concrete area — per-bar zones when widely spaced (EC2 Fig 7.2)
        A_c_eff = self._compute_A_c_eff(
            eps_top, eps_bottom, h_c_ef=h_c_ef,
            cover=cover, phi_eq=phi_eq, bar_spacing=bar_spacing,
            face=face if is_net_tension else None,
            h_c_ef_limit=h_c_ef,
            breadth_override=breadth_override,
            strain_state=strain_state,
        )

        rho_p_eff = self.find_rho_p_eff(A_s_tension=A_s, h_c_ef=h_c_ef, A_c_eff=A_c_eff)

        # Steel stress (peak in the bar set for this face)
        sigma_s = self._get_steel_stress(
            eps_top, eps_bottom,
            face=face if is_net_tension else None,
            h_c_ef_limit=h_c_ef,
            strain_state=strain_state,
        )

        # k_2 (strain distribution coefficient)
        if strain_state is not None and strain_state.is_biaxial:
            k_2 = self.find_k_2_from_strain_state(strain_state)
        else:
            k_2 = self.find_k_2(eps_top, eps_bottom)

        # s_r,max
        has_tension_reinforcement = A_s > 0
        s_r_max = self.find_maximum_crack_spacing(
            cover=cover, phi_eq=phi_eq, rho_p_eff=rho_p_eff, k_2=k_2,
            x=x, has_tension_reinforcement=has_tension_reinforcement,
            sigma_s=sigma_s,
            bar_spacing=bar_spacing,
        )

        # Bar diameter correction for equivalent-area substitution
        s_r_max_uncorrected: Optional[float] = None
        phi_correction_factor: Optional[float] = None
        if actual_bar_diameter is not None and phi_eq > 0:
            s_r_max_uncorrected = s_r_max
            s_r_max, phi_correction_factor = self._compute_bar_diameter_correction(
                s_r_max=s_r_max,
                phi_eq=phi_eq,
                actual_bar_diameter=actual_bar_diameter,
                cover=cover,
                k_3=self.k_3,
            )

        # Strain difference (ε_sm - ε_cm)
        E_s = self._get_tension_zone_E_s(eps_top, eps_bottom, strain_state=strain_state)
        eps_diff = self.find_strain_difference(sigma_s, rho_p_eff, E_s)

        # Crack width
        w_k = self.calculate_crack_width(s_r_max, eps_diff)

        # Steel yielding check
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
            steel_yielded=steel_yielded,
            actual_bar_diameter=actual_bar_diameter,
            s_r_max_uncorrected=s_r_max_uncorrected,
            phi_correction_factor=phi_correction_factor,
        )

    # ===============================================
    # Main check method
    # ===============================================

    def perform_check(
        self,
        *,
        My_Ed: Optional[float] = None,
        N_Ed: float = 0.0,
        Mz_Ed: float = 0.0,
        warning_threshold: float = 0.95,
        ignore_compression_steel: bool = False,
        force_cracked: bool = False,
        suppress_warnings: bool = False,
        actual_bar_diameter: Optional[float] = None,
        cover_override: Optional[float] = None,
        **kwargs,
    ) -> CheckResult:
        """
        Perform crack width check for applied serviceability loads.

        Args:
            My_Ed: Design major-axis moment at SLS (kN·m).
                   Formerly named M_Ed — passing M_Ed=... is still accepted
                   but will raise a DeprecationWarning.
            Mz_Ed: Design minor-axis moment at SLS (kN·m, default 0).
                   Requires free_neutral_axis=True when non-zero.
            N_Ed: Design axial force at SLS (kN, compression positive)
            warning_threshold: Utilization threshold for warnings
            ignore_compression_steel: If True, ignore compression reinforcement
            force_cracked: If True, skip the uncracked solver probe and proceed
                directly to cracked analysis.
            suppress_warnings: If True, suppresses warnings
            actual_bar_diameter: If provided, corrects s_r,max for equivalent-area
                bar substitution. When bars are modelled with a modified diameter
                to achieve equivalent area (e.g. rounding a non-integer bar count),
                supply the real bar diameter here. The φ-dependent term of s_r,max
                is scaled by φ_actual / φ_model.
            cover_override: If provided, use this value (mm) as the concrete cover
                to tension reinforcement instead of auto-computing it from the
                section geometry. Useful for asymmetric sections, minor-axis
                bending, or rotated neutral axes where the automatic cover
                detection may not apply.

        Returns:
            CheckResult with crack width utilization
        """
        # Backwards-compatibility shim: accept legacy M_Ed keyword argument.
        if "M_Ed" in kwargs:
            if My_Ed is not None:
                raise TypeError("Cannot pass both 'M_Ed' and 'My_Ed' to perform_check()")
            warnings.warn(
                "The 'M_Ed' parameter of CrackingCheck.perform_check() has been renamed to "
                "'My_Ed'. Please update your call sites. 'M_Ed' will be removed in a future "
                "version.",
                DeprecationWarning,
                stacklevel=2,
            )
            My_Ed = kwargs.pop("M_Ed")
        if My_Ed is None:
            raise TypeError("perform_check() missing required keyword argument: 'My_Ed'")

        return self._check_single_case(
            My_Ed=My_Ed,
            Mz_Ed=Mz_Ed,
            N_Ed=N_Ed,
            warning_threshold=warning_threshold,
            ignore_compression_steel=ignore_compression_steel,
            force_cracked=force_cracked,
            suppress_warnings=suppress_warnings,
            actual_bar_diameter=actual_bar_diameter,
            cover_override=cover_override,
        )


    def _check_single_case(
        self,
        *,
        My_Ed: float,
        Mz_Ed: float = 0.0,
        N_Ed: float,
        warning_threshold: float,
        ignore_compression_steel: bool = False,
        force_cracked: bool = False,
        suppress_warnings: bool = False,
        actual_bar_diameter: Optional[float] = None,
        cover_override: Optional[float] = None,
    ) -> CheckResult:
        """Internal implementation of crack check."""
        # Validate Mz_Ed requires free neutral axis
        if abs(Mz_Ed) > 1e-9 and not self.free_neutral_axis:
            raise ValueError(
                "Mz_Ed != 0 requires free_neutral_axis=True. "
                "Set free_neutral_axis=True on the CrackingCheck to enable biaxial bending."
            )

        # Local alias so the rest of the method body can use M_Ed
        M_Ed = My_Ed

        # Build keyword dict for Mz_target (only passed when non-zero)
        _mz_kw: Dict[str, Any] = {"Mz_target": Mz_Ed} if abs(Mz_Ed) > 1e-9 else {}

        if (
            not self.section.rebar_groups
            or sum(len(group.positions) for group in self.section.rebar_groups) == 0
        ):
            raise ValueError(
                "CrackingCheck is invalid for unreinforced sections. "
                "Provide longitudinal reinforcement before calling perform_check()."
            )

        # Step 1: Determine cracked state.
        crack_detection_method = "forced_cracked" if force_cracked else "solver_uncracked_tension_threshold"
        probe_solver_error: Optional[str] = None
        probe_eps_top: Optional[float] = None
        probe_eps_bottom: Optional[float] = None
        probe_min_tension_concrete_strain: Optional[float] = None
        probe_cracking_strain: Optional[float] = None

        if force_cracked:
            is_cracked = True
        else:
            try:
                (
                    is_cracked,
                    probe_eps_top,
                    probe_eps_bottom,
                    probe_min_tension_concrete_strain,
                    probe_cracking_strain,
                ) = self._is_cracked_by_solver(
                    M_Ed=M_Ed,
                    N_Ed=N_Ed,
                    ignore_compression_steel=ignore_compression_steel,
                    _mz_kw=_mz_kw,
                )
            except ValueError as e:
                # If uncracked-state equilibrium cannot be solved, continue with
                # cracked analysis rather than failing immediately.
                is_cracked = True
                probe_solver_error = str(e)
                crack_detection_method = "solver_failed_assumed_cracked"

        if not is_cracked:
            # Section uncracked - no crack width to check
            return self._create_result(
                check_name="Cracking check (EC2 §7.3)",
                code_reference="EC2 §7.3",
                warning_threshold=warning_threshold,
                utilization=0.0,
                demand_components={"My": float(M_Ed), "Mz": float(Mz_Ed), "N": float(N_Ed)},
                capacity_components={"w_k_limit": self.w_k_limit},
                units_components={"My": "kN·m", "Mz": "kN·m", "N": "kN", "w_k_limit": "mm"},
                message="Section uncracked (solver: concrete tension <= cracking limit)",
                details={
                    "M_Ed": float(M_Ed),
                    "N_Ed": float(N_Ed),
                    "is_cracked": False,
                    "solved": True,
                    "solver_stage": None,
                    "solver_error": None,
                    "solver_residual_N": None,
                    "solver_residual_M": None,
                    "w_k": 0.0,
                    "w_k_limit": self.w_k_limit,
                    "crack_detection_method": crack_detection_method,
                    "probe_solver_error": probe_solver_error,
                    "probe_eps_top": float(probe_eps_top) if probe_eps_top is not None else None,
                    "probe_eps_bottom": float(probe_eps_bottom) if probe_eps_bottom is not None else None,
                    "probe_min_tension_concrete_strain": (
                        float(probe_min_tension_concrete_strain)
                        if probe_min_tension_concrete_strain is not None
                        else None
                    ),
                    "probe_cracking_strain": (
                        float(probe_cracking_strain)
                        if probe_cracking_strain is not None
                        else None
                    ),
                },
            )

        # Step 2: Solve for strain state (cracked section).
        # Capture the diagram instance here so that stress extraction in Step 2.5
        # uses the same model (important when ignore_compression_steel=True).
        diagram_for_check = self._get_diagram(ignore_compression_steel)
        try:
            eps_top, eps_bottom = diagram_for_check.find_strains_for_MN(
                My_target=M_Ed,
                N_target=N_Ed,
                strict=True,
                **_mz_kw,
            )
        except ValueError as e:
            solver_error = str(e)
            dN, dM = self._extract_solver_residuals(solver_error)
            # Load point outside capacity - section fails
            return self._create_result(
                check_name="Cracking check (EC2 §7.3)",
                code_reference="EC2 §7.3",
                warning_threshold=warning_threshold,
                utilization=float("inf"),
                demand_components={"My": float(M_Ed), "Mz": float(Mz_Ed), "N": float(N_Ed)},
                capacity_components={"w_k_limit": self.w_k_limit},
                units_components={"My": "kN·m", "Mz": "kN·m", "N": "kN", "w_k_limit": "mm"},
                message=f"Failed to solve strain state: {e}",
                details={
                    "error": solver_error,
                    "solver_error": solver_error,
                    "solved": False,
                    "solver_stage": "cracked_state",
                    "solver_residual_N": dN,
                    "solver_residual_M": dM,
                },
            )

        strain_state_local = diagram_for_check.find_strain_state_for_MN(
            My_target=M_Ed, N_target=N_Ed, **_mz_kw,
        )

        # Step 2.5: Stress limitation checks (EC2 §7.2) and non-linear creep
        sigma_c_peak = self._get_peak_concrete_stress(
            eps_top, eps_bottom, diagram=diagram_for_check,
            strain_state=strain_state_local,
        )
        nonlinear_creep_applied = False
        creep_coefficient_used = self.creep_coefficient

        # EC2 §7.2(2): Characteristic stress limit (longitudinal cracking risk)
        if self.check_k1_stress:
            exceeded, msg = check_characteristic_concrete_stress(sigma_c_peak, self.concrete.f_ck)
            if exceeded:
                warnings.warn(msg, stacklevel=3)

        # EC2 §7.2(3): Quasi-permanent stress limit (non-linear creep threshold)
        if self.check_k2_stress:
            exceeded, msg = check_quasi_permanent_concrete_stress(sigma_c_peak, self.concrete.f_ck)
            if exceeded:
                warnings.warn(msg, stacklevel=3)

            if exceeded and self.apply_nonlinear_creep:
                max_iterations = 5 if self.iterate_nonlinear_creep else 1
                for _ in range(max_iterations):
                    phi_NL = self._compute_nonlinear_creep_coefficient(sigma_c_peak)
                    E_c_eff_NL = self.concrete.get_elastic_modulus() / (1.0 + phi_NL)

                    if abs(E_c_eff_NL - (self.concrete.get_elastic_modulus() / (1.0 + creep_coefficient_used))) < 1.0:
                        break  # Converged (within 1 MPa)

                    diagram_nl = self._build_diagram_with_E_c_eff(E_c_eff_NL, ignore_compression_steel)
                    try:
                        eps_top, eps_bottom = diagram_nl.find_strains_for_MN(
                            My_target=M_Ed,
                            N_target=N_Ed,
                            strict=True,
                            **_mz_kw,
                        )
                    except ValueError as exc:
                        solver_error = str(exc)
                        dN, dM = self._extract_solver_residuals(solver_error)
                        return self._create_result(
                            check_name="Cracking check (EC2 §7.3)",
                            code_reference="EC2 §7.3",
                            warning_threshold=warning_threshold,
                            utilization=float("inf"),
                            demand_components={"My": float(M_Ed), "Mz": float(Mz_Ed), "N": float(N_Ed)},
                            capacity_components={"w_k_limit": self.w_k_limit},
                            units_components={"My": "kN·m", "Mz": "kN·m", "N": "kN", "w_k_limit": "mm"},
                            message="Failed to solve strain state after non-linear creep update",
                            details={
                                "error": solver_error,
                                "solver_error": solver_error,
                                "solved": False,
                                "solver_stage": "nonlinear_creep",
                                "solver_residual_N": dN,
                                "solver_residual_M": dM,
                                "eps_top_pre_nl": float(eps_top),
                                "eps_bottom_pre_nl": float(eps_bottom),
                                "sigma_c_peak_pre_nl": float(sigma_c_peak),
                                "creep_coefficient_used_pre_nl": float(creep_coefficient_used),
                            },
                        )

                    creep_coefficient_used = phi_NL
                    diagram_for_check = diagram_nl
                    strain_state_local = diagram_nl.find_strain_state_for_MN(
                        My_target=M_Ed, N_target=N_Ed, **_mz_kw,
                    )
                    sigma_c_peak = self._get_peak_concrete_stress(
                        eps_top, eps_bottom, diagram_for_check,
                        strain_state=strain_state_local,
                    )
                    nonlinear_creep_applied = True

        # --- Net compression: both faces in compression → w_k = 0 ---
        if eps_top >= 0 and eps_bottom >= 0:
            return self._create_result(
                check_name="Cracking check (EC2 §7.3)",
                code_reference="EC2 §7.3",
                warning_threshold=warning_threshold,
                utilization=0.0,
                demand_components={"My": float(M_Ed), "Mz": float(Mz_Ed), "N": float(N_Ed)},
                capacity_components={"w_k_limit": self.w_k_limit},
                units_components={"My": "kN·m", "Mz": "kN·m", "N": "kN", "w_k_limit": "mm"},
                message="Net compression — no cracking possible",
                details={
                    "M_Ed": float(M_Ed),
                    "N_Ed": float(N_Ed),
                    "is_cracked": False,
                    "w_k": 0.0,
                    "w_k_limit": self.w_k_limit,
                    "eps_top": float(eps_top),
                    "eps_bottom": float(eps_bottom),
                    "sigma_c_peak": float(sigma_c_peak),
                    "nonlinear_creep_applied": nonlinear_creep_applied,
                    "creep_coefficient_used": float(creep_coefficient_used),
                    "solved": True,
                    "solver_stage": None,
                    "solver_error": None,
                    "solver_residual_N": None,
                    "solver_residual_M": None,
                    "crack_detection_method": crack_detection_method,
                    "probe_solver_error": probe_solver_error,
                    "probe_eps_top": float(probe_eps_top) if probe_eps_top is not None else None,
                    "probe_eps_bottom": float(probe_eps_bottom) if probe_eps_bottom is not None else None,
                    "probe_min_tension_concrete_strain": (
                        float(probe_min_tension_concrete_strain)
                        if probe_min_tension_concrete_strain is not None
                        else None
                    ),
                    "probe_cracking_strain": (
                        float(probe_cracking_strain)
                        if probe_cracking_strain is not None
                        else None
                    ),
                },
            )

        # --- Determine strain regime and delegate to face-based helper ---
        comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)
        is_net_tension = comp_face is None

        # Biaxial face policy: auto-upgrade TOP_BOTTOM → NA_NORMAL when biaxial
        _h_override: Optional[float] = None
        _b_override: Optional[float] = None
        effective_policy = self.crack_face_policy
        if (
            strain_state_local is not None
            and strain_state_local.is_biaxial
            and effective_policy == CrackFacePolicy.TOP_BOTTOM
        ):
            effective_policy = CrackFacePolicy.NA_NORMAL
            if not suppress_warnings:
                warnings.warn(
                    "Biaxial strain state detected: auto-upgrading crack_face_policy "
                    "from TOP_BOTTOM to NA_NORMAL for rotated neutral axis.",
                    stacklevel=3,
                )

        if (
            effective_policy == CrackFacePolicy.NA_NORMAL
            and strain_state_local is not None
            and strain_state_local.is_biaxial
        ):
            h_perp, b_perp = self._compute_biaxial_section_extents(strain_state_local)
            _h_override = h_perp
            _b_override = b_perp

        section_h = _h_override if _h_override is not None else self.height
        x = flexure_utils.calculate_neutral_axis_depth_from_strains(
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            section_height=section_h,
        )

        # Pre-compute biaxial effective depth so _calculate_face_crack_width can use it.
        # For 1D, d is face-dependent and computed inside _calculate_face_crack_width.
        _d_override: Optional[float] = None
        if strain_state_local is not None and strain_state_local.is_biaxial:
            _d_override = flexure_utils.find_effective_depth_for_flexure(
                section=self.section,
                diagram=diagram_for_check,
                M_Ed=M_Ed,
                N_Ed=N_Ed,
                eps_top=eps_top,
                eps_bottom=eps_bottom,
                strain_state=strain_state_local,
                warn_on_fallback=not suppress_warnings,
                _stacklevel=4,
            )

        if is_net_tension:
            # Both faces in tension (EC2 Fig 7.1, case c)
            # Check both faces independently and report the worst crack width.
            if self.net_tension_face is not None:
                faces_to_check: list[Literal["top", "bottom"]] = [self.net_tension_face]
            else:
                faces_to_check = ["bottom", "top"]

            best_cr: Optional[CrackingResult] = None
            governing_face_result: Literal["top", "bottom"] = "bottom"

            for face_candidate in faces_to_check:
                crack_width_kwargs: dict[str, Any] = dict(
                    suppress_warnings=suppress_warnings,
                    actual_bar_diameter=actual_bar_diameter,
                )
                if cover_override is not None:
                    crack_width_kwargs["cover_override"] = cover_override
                if _d_override is not None:
                    crack_width_kwargs["d_override"] = _d_override
                cr_candidate = self._calculate_face_crack_width(
                    eps_top, eps_bottom, face=face_candidate,
                    x=x, is_net_tension=True,
                    strain_state=strain_state_local,
                    h_override=_h_override,
                    breadth_override=_b_override,
                    **crack_width_kwargs,
                )
                candidate_w_k = self._require_result_value(cr_candidate.w_k, "w_k")
                if best_cr is None:
                    best_cr = cr_candidate
                    governing_face_result = face_candidate
                else:
                    best_w_k = self._require_result_value(best_cr.w_k, "w_k")
                    if candidate_w_k > best_w_k:
                        best_cr = cr_candidate
                        governing_face_result = face_candidate

            assert best_cr is not None  # at least one face always checked
            cr = best_cr
            cr.governing_face = governing_face_result
        else:
            # Bending: one compression face, one tension face
            tension_face: Literal["top", "bottom"] = "bottom" if comp_face == "top" else "top"
            crack_width_kwargs: dict[str, Any] = dict(
                suppress_warnings=suppress_warnings,
                actual_bar_diameter=actual_bar_diameter,
            )
            if cover_override is not None:
                crack_width_kwargs["cover_override"] = cover_override
            if _d_override is not None:
                crack_width_kwargs["d_override"] = _d_override
            cr = self._calculate_face_crack_width(
                eps_top, eps_bottom, face=tension_face,
                x=x, is_net_tension=False,
                strain_state=strain_state_local,
                h_override=_h_override,
                breadth_override=_b_override,
                **crack_width_kwargs,
            )
            cr.governing_face = tension_face

        # Attach non-linear creep metadata to CrackingResult
        cr.sigma_c_peak = sigma_c_peak
        cr.nonlinear_creep_applied = nonlinear_creep_applied
        cr.creep_coefficient_used = creep_coefficient_used

        # From this point we are in a solved branch; required fields must exist.
        w_k = self._require_result_value(cr.w_k, "w_k")
        sigma_s = self._require_result_value(cr.sigma_s, "sigma_s")
        h_c_ef = self._require_result_value(cr.h_c_ef, "h_c_ef")
        phi_eq = self._require_result_value(cr.phi_eq, "phi_eq")
        cover = self._require_result_value(cr.cover, "cover")
        rho_p_eff = self._require_result_value(cr.rho_p_eff, "rho_p_eff")
        s_r_max = self._require_result_value(cr.s_r_max, "s_r_max")
        eps_sm_minus_eps_cm = self._require_result_value(
            cr.eps_sm_minus_eps_cm, "eps_sm_minus_eps_cm"
        )

        # EC2 §7.2(5): Reinforcement stress limit
        f_yk = self._get_f_yk_max()
        if self.check_k3_stress:
            exceeded, msg = check_characteristic_reinforcement_stress(sigma_s, f_yk)
            if exceeded:
                warnings.warn(msg, stacklevel=3)

        # EC2 §7.2(4)P: Check for inelastic strain (yielding)
        if self.check_yielding:
            exceeded, msg = check_reinforcement_yielding(sigma_s, f_yk)
            if exceeded:
                cr.steel_yielded = True
                warnings.warn(msg, stacklevel=3)

        # EC2 §7.2(5): Imposed deformation stress limit
        if self.check_k4_stress:
            exceeded, msg = check_imposed_deformation_stress(sigma_s, f_yk)
            if exceeded:
                warnings.warn(msg, stacklevel=3)

        # Build utilization and result
        utilization = w_k / self.w_k_limit if self.w_k_limit > 0 else float("inf")
        if strain_state_local is not None and strain_state_local.is_biaxial:
            k_2 = self.find_k_2_from_strain_state(strain_state_local)
        else:
            k_2 = self.find_k_2(eps_top, eps_bottom)

        details = {
            "M_Ed": float(M_Ed),
            "N_Ed": float(N_Ed),
            "is_cracked": True,
            "eps_top": float(eps_top),
            "eps_bottom": float(eps_bottom),
            "x": float(cr.x) if cr.x is not None else None,
            "h_c_ef": h_c_ef,
            "phi_eq": phi_eq,
            "cover": cover,
            "rho_p_eff": rho_p_eff,
            "sigma_s": sigma_s,
            "s_r_max": s_r_max,
            "eps_sm_minus_eps_cm": eps_sm_minus_eps_cm,
            "w_k": float(w_k),
            "w_k_limit": float(self.w_k_limit),
            "k_t": float(self.k_t),
            "k_1": float(self.find_k_1(k_2)),
            "k_2": float(k_2),
            "k_3": float(self.k_3),
            "k_4": float(self.k_4),
            "sigma_c_peak": float(sigma_c_peak),
            "f_yk": float(f_yk),
            "steel_yielded": cr.steel_yielded,
            "nonlinear_creep_applied": nonlinear_creep_applied,
            "creep_coefficient_used": float(creep_coefficient_used),
            "solved": True,
            "solver_stage": None,
            "solver_error": None,
            "solver_residual_N": None,
            "solver_residual_M": None,
            "is_net_tension": is_net_tension,
            "governing_face": cr.governing_face,
            "crack_detection_method": crack_detection_method,
            "probe_solver_error": probe_solver_error,
            "probe_eps_top": float(probe_eps_top) if probe_eps_top is not None else None,
            "probe_eps_bottom": float(probe_eps_bottom) if probe_eps_bottom is not None else None,
            "probe_min_tension_concrete_strain": (
                float(probe_min_tension_concrete_strain)
                if probe_min_tension_concrete_strain is not None
                else None
            ),
            "probe_cracking_strain": (
                float(probe_cracking_strain)
                if probe_cracking_strain is not None
                else None
            ),
        }

        # Bar diameter correction reporting
        if cr.actual_bar_diameter is not None:
            details["actual_bar_diameter"] = float(cr.actual_bar_diameter)
            details["phi_correction_factor"] = float(cr.phi_correction_factor) if cr.phi_correction_factor is not None else None
            details["s_r_max_uncorrected"] = float(cr.s_r_max_uncorrected) if cr.s_r_max_uncorrected is not None else None

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
        My_Ed: Optional[float] = None,
        N_Ed: float = 0.0,
        ignore_compression_steel: bool = False,
        force_cracked: bool = False,
        suppress_warnings: bool = False,
        actual_bar_diameter: Optional[float] = None,
        cover_override: Optional[float] = None,
        skip_stress_checks: bool = False,
        Mz_Ed: float = 0.0,
        **kwargs,
    ) -> CrackingResult:
        """
        Calculate detailed cracking results without creating CheckResult.

        Useful for parametric studies or when you need the raw values.

        Args:
            My_Ed: Design major-axis moment at SLS (kN·m).
                   Formerly named M_Ed — passing M_Ed=... is still accepted.
            N_Ed: Design axial force at SLS (kN, compression positive)
            Mz_Ed: Design minor-axis moment at SLS (kN·m, default 0).
            ignore_compression_steel: If True, ignore compression reinforcement
            force_cracked: If True, skip the uncracked solver probe and proceed
                directly to cracked analysis.
            suppress_warnings: If True, suppresses warnings
            actual_bar_diameter: If provided, corrects s_r,max for equivalent-area
                bar substitution. See ``perform_check`` for details.
            cover_override: If provided, use this value (mm) as the concrete cover
                instead of auto-computing from section geometry.
            skip_stress_checks: If True, skip stress limitation checks and
                nonlinear creep iteration. Only the crack width is computed.
                Useful for parametric sweeps where only w_k is needed.

        Returns:
            CrackingResult dataclass with all intermediate values.
            If equilibrium cannot be solved (strict mode), returns
            ``solved=False`` with ``solver_error``/residuals populated and
            undefined response quantities set to ``None``.
        """
        # Backwards-compatibility shim: accept legacy M_Ed positional/keyword argument.
        if "M_Ed" in kwargs:
            if My_Ed is not None:
                raise TypeError("Cannot pass both 'M_Ed' and 'My_Ed' to calculate_detailed()")
            warnings.warn(
                "The 'M_Ed' parameter of CrackingCheck.calculate_detailed() has been renamed to "
                "'My_Ed'. Please update your call sites.",
                DeprecationWarning,
                stacklevel=2,
            )
            My_Ed = kwargs.pop("M_Ed")
        if My_Ed is None:
            raise TypeError("calculate_detailed() missing required argument: 'My_Ed'")

        # Validate Mz_Ed requires free neutral axis
        if abs(Mz_Ed) > 1e-9 and not self.free_neutral_axis:
            raise ValueError(
                "Mz_Ed != 0 requires free_neutral_axis=True. "
                "Set free_neutral_axis=True on the CrackingCheck to enable biaxial bending."
            )

        # Local alias
        M_Ed = My_Ed
        _mz_kw: Dict[str, Any] = {"Mz_target": Mz_Ed} if abs(Mz_Ed) > 1e-9 else {}

        # Determine cracked state using an uncracked solver probe.
        if force_cracked:
            is_cracked = True
        else:
            try:
                is_cracked, *_ = self._is_cracked_by_solver(
                    M_Ed=M_Ed,
                    N_Ed=N_Ed,
                    ignore_compression_steel=ignore_compression_steel,
                    _mz_kw=_mz_kw,
                )
            except ValueError:
                # If uncracked-state equilibrium cannot be solved, continue with
                # cracked analysis rather than failing immediately.
                is_cracked = True

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

        # Solve strain state on cracked-analysis diagram
        diagram_for_check = self._get_diagram(ignore_compression_steel)
        try:
            eps_top, eps_bottom = diagram_for_check.find_strains_for_MN(
                M_Ed, N_Ed, strict=True, **_mz_kw,
            )
        except ValueError as exc:
            return self._build_unsolved_cracking_result(
                is_cracked=True,
                solver_stage="cracked_state",
                solver_error=str(exc),
                nonlinear_creep_applied=False,
                creep_coefficient_used=self.creep_coefficient,
            )

        strain_state_local = diagram_for_check.find_strain_state_for_MN(
            My_target=M_Ed, N_target=N_Ed, **_mz_kw,
        )

        # Stress limitation and non-linear creep (same logic as _check_single_case)
        nonlinear_creep_applied = False
        creep_coefficient_used = self.creep_coefficient

        if skip_stress_checks:
            sigma_c_peak = 0.0
        else:
            sigma_c_peak = self._get_peak_concrete_stress(
                eps_top, eps_bottom, diagram_for_check,
                strain_state=strain_state_local,
            )

        if not skip_stress_checks and self.check_k2_stress and self.apply_nonlinear_creep:
            exceeded_qp, _ = check_quasi_permanent_concrete_stress(sigma_c_peak, self.concrete.f_ck)
            if exceeded_qp:
                max_iterations = 5 if self.iterate_nonlinear_creep else 1
                for _ in range(max_iterations):
                    phi_NL = self._compute_nonlinear_creep_coefficient(sigma_c_peak)
                    E_c_eff_NL = self.concrete.get_elastic_modulus() / (1.0 + phi_NL)
                    if abs(E_c_eff_NL - (self.concrete.get_elastic_modulus() / (1.0 + creep_coefficient_used))) < 1.0:
                        break
                    diagram_nl = self._build_diagram_with_E_c_eff(E_c_eff_NL, ignore_compression_steel)
                    try:
                        eps_top, eps_bottom = diagram_nl.find_strains_for_MN(M_Ed, N_Ed, strict=True, **_mz_kw)
                    except ValueError as exc:
                        return self._build_unsolved_cracking_result(
                            is_cracked=True,
                            solver_stage="nonlinear_creep",
                            solver_error=str(exc),
                            sigma_c_peak=float(sigma_c_peak),
                            nonlinear_creep_applied=nonlinear_creep_applied,
                            creep_coefficient_used=creep_coefficient_used,
                        )

                    creep_coefficient_used = phi_NL
                    diagram_for_check = diagram_nl
                    strain_state_local = diagram_nl.find_strain_state_for_MN(
                        My_target=M_Ed, N_target=N_Ed, **_mz_kw,
                    )
                    sigma_c_peak = self._get_peak_concrete_stress(
                        eps_top, eps_bottom, diagram_nl,
                        strain_state=strain_state_local,
                    )
                    nonlinear_creep_applied = True

        # --- Net compression: both faces in compression → w_k = 0 ---
        if eps_top >= 0 and eps_bottom >= 0:
            return CrackingResult(
                w_k=0.0, w_k_limit=self.w_k_limit, s_r_max=0.0,
                eps_sm_minus_eps_cm=0.0, sigma_s=0.0, rho_p_eff=0.0,
                h_c_ef=0.0, x=None, is_cracked=False, phi_eq=0.0, cover=0.0,
                sigma_c_peak=sigma_c_peak,
                nonlinear_creep_applied=nonlinear_creep_applied,
                creep_coefficient_used=creep_coefficient_used,
            )

        # --- Determine strain regime and delegate to face-based helper ---
        comp_face = flexure_utils.calculate_compression_face_from_strains(eps_top, eps_bottom)
        is_net_tension = comp_face is None

        # Biaxial face policy: auto-upgrade TOP_BOTTOM → NA_NORMAL when biaxial
        _h_override_d: Optional[float] = None
        _b_override_d: Optional[float] = None
        effective_policy_d = self.crack_face_policy
        if (
            strain_state_local is not None
            and strain_state_local.is_biaxial
            and effective_policy_d == CrackFacePolicy.TOP_BOTTOM
        ):
            effective_policy_d = CrackFacePolicy.NA_NORMAL
        if (
            effective_policy_d == CrackFacePolicy.NA_NORMAL
            and strain_state_local is not None
            and strain_state_local.is_biaxial
        ):
            h_perp_d, b_perp_d = self._compute_biaxial_section_extents(strain_state_local)
            _h_override_d = h_perp_d
            _b_override_d = b_perp_d

        section_h_d = _h_override_d if _h_override_d is not None else self.height
        x = flexure_utils.calculate_neutral_axis_depth_from_strains(
            eps_top, eps_bottom, section_h_d,
        )

        if is_net_tension:
            # Both faces in tension (EC2 Fig 7.1, case c)
            # Check both faces independently and report the worst crack width.
            if self.net_tension_face is not None:
                faces_to_check: list[Literal["top", "bottom"]] = [self.net_tension_face]
            else:
                faces_to_check = ["bottom", "top"]

            best_result: Optional[CrackingResult] = None
            governing_face_result: Literal["top", "bottom"] = "bottom"

            for face_candidate in faces_to_check:
                crack_width_kwargs: dict[str, Any] = dict(
                    suppress_warnings=suppress_warnings,
                    actual_bar_diameter=actual_bar_diameter,
                )
                if cover_override is not None:
                    crack_width_kwargs["cover_override"] = cover_override
                result_candidate = self._calculate_face_crack_width(
                    eps_top, eps_bottom, face=face_candidate,
                    x=x, is_net_tension=True,
                    strain_state=strain_state_local,
                    h_override=_h_override_d,
                    breadth_override=_b_override_d,
                    **crack_width_kwargs,
                )
                candidate_w_k = self._require_result_value(result_candidate.w_k, "w_k")
                if best_result is None:
                    best_result = result_candidate
                    governing_face_result = face_candidate
                else:
                    best_w_k = self._require_result_value(best_result.w_k, "w_k")
                    if candidate_w_k > best_w_k:
                        best_result = result_candidate
                        governing_face_result = face_candidate

            assert best_result is not None  # at least one face always checked
            result = best_result
            result.governing_face = governing_face_result
        else:
            # Bending: one compression face, one tension face
            tension_face: Literal["top", "bottom"] = "bottom" if comp_face == "top" else "top"
            crack_width_kwargs: dict[str, Any] = dict(
                suppress_warnings=suppress_warnings,
                actual_bar_diameter=actual_bar_diameter,
            )
            if cover_override is not None:
                crack_width_kwargs["cover_override"] = cover_override
            result = self._calculate_face_crack_width(
                eps_top, eps_bottom, face=tension_face,
                x=x, is_net_tension=False,
                strain_state=strain_state_local,
                h_override=_h_override_d,
                breadth_override=_b_override_d,
                **crack_width_kwargs,
            )
            result.governing_face = tension_face

        # Attach non-linear creep metadata
        result.sigma_c_peak = sigma_c_peak
        result.nonlinear_creep_applied = nonlinear_creep_applied
        result.creep_coefficient_used = creep_coefficient_used

        return result

    # ===============================================
    # Plotting convenience methods
    # ===============================================

    def plot_load_cases(
        self,
        load_cases: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> Any:
        """
        3D stem plot of crack widths at discrete M-N load cases.

        Convenience wrapper around ``CrackWidthViewer.plot_load_cases``.
        See that method for full argument documentation.

        Args:
            load_cases: Sequence of dicts with ``M_Ed``, ``N_Ed``, and
                optionally ``name`` keys.
            **kwargs: Forwarded to ``CrackWidthViewer.plot_load_cases``.

        Returns:
            Plotly Figure object.
        """
        from materials.reinforced_concrete.analysis.crack_width_viewer import CrackWidthViewer
        return CrackWidthViewer(self).plot_load_cases(load_cases, **kwargs)

    def plot_crack_width_contours(
        self,
        **kwargs,
    ) -> Any:
        """
        2D contour map of crack width across the M-N domain.

        Convenience wrapper around ``CrackWidthViewer.plot_contours``.
        See that method for full argument documentation.

        Args:
            **kwargs: Forwarded to ``CrackWidthViewer.plot_contours``
                (e.g. ``load_cases``, ``n_grid``, ``show``).

        Returns:
            Plotly Figure object.
        """
        from materials.reinforced_concrete.analysis.crack_width_viewer import CrackWidthViewer
        return CrackWidthViewer(self).plot_contours(**kwargs)
