"""
SLS stress limitation checks for reinforced concrete sections (EC2 §7.2).

Provides:
- **Pure functions** for each stress limit check — importable by CrackingCheck
  or any other module without needing a class instance.
- **StressLimitsCheck** (BaseCodeCheck) — standalone check class with its own
  cached M-N diagram for running all §7.2 stress checks as a single
  ``perform_check()`` call.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import exp
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from section_design_checks.reinforced_concrete.analysis.strain_state import StrainState

from pydantic import Field, PrivateAttr

from section_design_checks.reinforced_concrete.analysis import create_interaction_diagram
from section_design_checks.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from section_design_checks.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from section_design_checks.reinforced_concrete.code_checks.ec2_2004 import flexure_utils
from section_design_checks.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType
from section_design_checks.reinforced_concrete.geometry import RCSection
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial
from section_design_checks.reinforced_concrete.ndp import get_ndp

# =====================================================================
# Pure functions — used by both StressLimitsCheck and CrackingCheck
# =====================================================================


def check_characteristic_concrete_stress(
    sigma_c: float,
    f_ck: float,
) -> tuple[bool, str]:
    """EC2 §7.2(2): characteristic concrete stress limit.

    Uses k₁ which is a NDP stress limit factor.

    Longitudinal cracking risk if σ_c > k₁·f_ck under characteristic loads.
    Relevant for XD, XF, and XS exposure classes.

    Args:
        sigma_c: Peak concrete compressive stress (MPa, positive).
        f_ck: Characteristic cylinder compressive strength (MPa).

    Returns:
        (exceeded, message) — exceeded is True when the limit is breached.
    """
    k_1 = cast(float, get_ndp("k_1_stress"))
    limit = k_1 * f_ck
    exceeded = sigma_c > limit
    msg = (
        f"EC2 §7.2(2): σ_c = {sigma_c:.1f} MPa > "
        f"{k_1}·f_ck = {limit:.1f} MPa under characteristic loads. "
        f"Longitudinal cracking risk for XD/XF/XS exposure classes."
    ) if exceeded else ""
    return exceeded, msg


def check_quasi_permanent_concrete_stress(
    sigma_c: float,
    f_ck: float,
) -> tuple[bool, str]:
    """EC2 §7.2(3): quasi-permanent concrete stress limit.

    Uses k₂ which is a NDP stress limit factor.

    Non-linear creep threshold if σ_c > k₂·f_ck.

    Args:
        sigma_c: Peak concrete compressive stress (MPa, positive).
        f_ck: Characteristic cylinder compressive strength (MPa).

    Returns:
        (exceeded, message).
    """
    k_2 = cast(float, get_ndp("k_2_stress"))
    limit = k_2 * f_ck
    exceeded = sigma_c > limit
    msg = (
        f"EC2 §7.2(3): σ_c = {sigma_c:.1f} MPa > "
        f"{k_2}·f_ck = {limit:.1f} MPa. "
        f"Non-linear creep threshold exceeded."
    ) if exceeded else ""
    return exceeded, msg


def check_characteristic_reinforcement_stress(
    sigma_s: float,
    f_yk: float,
) -> tuple[bool, str]:
    """EC2 §7.2(5): SLS characteristic stress limit in reinforcement for
    for which the appearance of cracking or deformations may be deemed acceptable.

    Uses k₃ which is a NDP stress limit factor.

    Args:
        sigma_s: Maximum tensile stress in reinforcement (MPa, positive).
        f_yk: Characteristic yield strength of reinforcement (MPa).

    Returns:
        (exceeded, message).
    """
    k_3 = cast(float, get_ndp("k_3_stress"))
    limit = k_3 * f_yk
    exceeded = sigma_s > limit
    msg = (
        f"EC2 §7.2(5): σ_s = {sigma_s:.1f} MPa > "
        f"{k_3}·f_yk = {limit:.1f} MPa. "
        f"Reinforcement stress limit exceeded."
    ) if exceeded else ""
    return exceeded, msg


def check_reinforcement_yielding(
    sigma_s: float,
    f_yk: float,
) -> tuple[bool, str]:
    """EC2 §7.2(4)P: inelastic strain check.

    Args:
        sigma_s: Maximum tensile stress in reinforcement (MPa, positive).
        f_yk: Characteristic yield strength of reinforcement (MPa).

    Returns:
        (exceeded, message).
    """
    exceeded = sigma_s > f_yk
    msg = (
        f"EC2 §7.2(4)P: σ_s = {sigma_s:.1f} MPa > f_yk = {f_yk:.1f} MPa. "
        f"Reinforcement has yielded - inelastic strain occurring. "
        f"SLS crack width calculation may be unreliable."
    ) if exceeded else ""
    return exceeded, msg


def check_imposed_deformation_stress(
    sigma_s: float,
    f_yk: float,
) -> tuple[bool, str]:
    """EC2 §7.2(5): imposed deformation SLS Characteristic stress limit.

    Uses k₄ which is a NDP stress limit factor.

    Applicable when tensile stress arises from imposed deformations
    (e.g. restraint, settlement, temperature).

    Args:
        sigma_s: Maximum tensile stress in reinforcement (MPa, positive).
        f_yk: Characteristic yield strength of reinforcement (MPa).

    Returns:
        (exceeded, message).
    """
    k_4 = cast(float, get_ndp("k_4_stress"))
    limit = k_4 * f_yk
    exceeded = sigma_s > limit
    msg = (
        f"EC2 §7.2(5): σ_s = {sigma_s:.1f} MPa > "
        f"{k_4}·f_yk = {limit:.1f} MPa. "
        f"Imposed deformation stress limit exceeded."
    ) if exceeded else ""
    return exceeded, msg


def compute_nonlinear_creep_coefficient(
    sigma_c: float,
    f_ck: float,
    creep_coefficient: float,
) -> float:
    """Non-linear creep coefficient per EC2 §3.1.4(4), Eq. 3.7.

    φ_NL = φ · exp(1.5 · (k_σ − 0.45))  where  k_σ = σ_c / f_ck.

    Args:
        sigma_c: Peak concrete compressive stress (MPa, positive).
        f_ck: Characteristic cylinder compressive strength (MPa).
        creep_coefficient: Linear (basic) creep coefficient φ.

    Returns:
        Non-linear creep coefficient φ_NL.
    """
    if f_ck <= 0:
        raise ValueError("f_ck cannot be equal to or less than zero.")
    k_sigma = sigma_c / f_ck
    return creep_coefficient * exp(1.5 * (k_sigma - 0.45))


# =====================================================================
# Result dataclass
# =====================================================================


@dataclass
class StressLimitResult:
    """Detailed results from stress limitation checks (EC2 §7.2)."""

    sigma_c_peak: float  # Peak concrete compressive stress (MPa)
    sigma_s_max: float   # Max steel tension stress (MPa)
    f_yk: float          # Steel yield strength (MPa)
    k1_exceeded: bool = False
    k2_exceeded: bool = False
    k3_exceeded: bool = False
    yielding: bool = False
    k4_exceeded: bool = False
    nonlinear_creep_applied: bool = False
    creep_coefficient_used: float = 0.0
    messages: list[str] = field(default_factory=list)


# =====================================================================
# StressLimitsCheck class
# =====================================================================


class StressLimitsCheck(BaseCodeCheck):
    """
    EC2 §7.2 stress limitation checks for reinforced concrete sections.

    Standalone check class that verifies concrete and reinforcement stresses
    are within SLS limits. Uses a linear-elastic M-N interaction diagram
    with characteristic material properties (same SLS conventions as
    CrackingCheck).

    Note: Applies a default creep coefficient of 1.5 to E_cm to get E_c,eff.
    If short-term only is required set creep_coefficient to 1.0

    Example:
        >>> check = StressLimitsCheck(
        ...     section=section, concrete=concrete,
        ...     check_k1_stress=True,  # characteristic concrete stress
        ... )
        >>> result = check.perform_check(M_Ed=200.0, N_Ed=0.0)
    """

    section: RCSection = Field(
        ...,
        description="RC section with reinforcement",
    )

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material properties",
    )

    creep_coefficient: float = Field(
        default=1.5,
        description="Linear creep coefficient φ. E_cm,eff = E_cm / (1 + φ).",
        ge=0.0,
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
        ge=10, le=500,
    )

    n_fibres_height: int = Field(
        default=30,
        description="Number of concrete fibres across height",
        ge=10, le=500,
    )

    # --- Stress check flags ---

    check_k1_stress: bool = Field(
        default=False,
        description="EC2 §7.2(2) characteristic concrete stress limit.",
    )

    check_k2_stress: bool = Field(
        default=True,
        description="EC2 §7.2(3) quasi-permanent concrete stress limit. "
                    "Triggers non-linear creep adjustment when exceeded.",
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
        description="Auto-adjust E_cm,eff when σ_c > k_2·f_ck (EC2 §3.1.4(4)).",
    )

    iterate_nonlinear_creep: bool = Field(
        default=False,
        description="Iterate non-linear creep adjustment until convergence.",
    )

    free_neutral_axis: bool = Field(
        default=False,
        description=(
            "Allow the neutral axis to rotate to satisfy biaxial equilibrium. "
            "Note: for SLS checks (linear-elastic), the biaxial solver is not "
            "used even when True; the 2D solver is used instead. "
            "The flag is accepted for API consistency."
        ),
    )

    # --- Internal state ---

    _diagram: MNInteractionDiagram | None = PrivateAttr(default=None)
    _diagram_no_comp_steel: MNInteractionDiagram | None = PrivateAttr(default=None)
    _diagram_snapshot: dict | None = PrivateAttr(default=None)
    _diagram_no_comp_snapshot: dict | None = PrivateAttr(default=None)

    # --- Properties ---

    @property
    def E_cm_eff(self) -> float:
        """Effective concrete modulus accounting for creep: E_cm / (1 + φ)."""
        return self.concrete.get_elastic_modulus() / (1.0 + self.creep_coefficient)

    @property
    def k_1_stress(self) -> float:
        """NDP characteristic stress limit factor (EC2 §7.2(2))."""
        return cast(float, get_ndp("k_1_stress"))

    @property
    def k_2_stress(self) -> float:
        """NDP quasi-permanent stress limit factor (EC2 §7.2(3))."""
        return cast(float, get_ndp("k_2_stress"))

    @property
    def k_3_stress(self) -> float:
        """NDP reinforcement characteristic stress limit factor (EC2 §7.2(5))."""
        return cast(float, get_ndp("k_3_stress"))

    @property
    def k_4_stress(self) -> float:
        """NDP imposed deformation characteristic stress limit factor (EC2 §7.2(5))."""
        return cast(float, get_ndp("k_4_stress"))

    # --- Diagram caching ---

    def _take_snapshot(self) -> dict:
        return {
            "section": self.section.model_dump(),
            "concrete": self.concrete.model_dump(),
            "concrete_model_type": self.concrete_model_type,
            "steel_model_type": self.steel_model_type,
            "n_fibres_width": self.n_fibres_width,
            "n_fibres_height": self.n_fibres_height,
            "E_cm_eff": self.E_cm_eff,
        }

    def _get_diagram(
        self, ignore_compression_steel: bool = False,
    ) -> MNInteractionDiagram:
        snapshot = self._take_snapshot()

        if ignore_compression_steel:
            if (
                self._diagram_no_comp_steel is None
                or snapshot != self._diagram_no_comp_snapshot
            ):
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
                    elastic_modulus=self.E_cm_eff,
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
                    elastic_modulus=self.E_cm_eff,
                )
                self._diagram_snapshot = snapshot
            return self._diagram

    def _build_diagram_with_E_cm_eff(
        self, E_cm_eff: float, ignore_compression_steel: bool = False,
    ) -> MNInteractionDiagram:
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

    # --- Stress extraction helpers ---

    def _get_peak_concrete_stress(
        self,
        eps_top: float,
        eps_bottom: float,
        diagram: MNInteractionDiagram | None = None,
        strain_state: StrainState | None = None,
    ) -> float:
        """Peak compressive stress in concrete from fibre integration."""
        diag = diagram or self._get_diagram()

        use_biaxial = strain_state is not None and strain_state.is_biaxial
        if use_biaxial:
            assert strain_state is not None  # implied by use_biaxial
            forces, _x, _y, areas = diag.get_fibre_forces_from_strain_state(strain_state)
        else:
            forces, _y, areas = diag.get_fibre_forces_from_end_strains(eps_top, eps_bottom)

        conc_mask = diag._fibre_mat == "concrete"
        conc_forces = forces[conc_mask]
        conc_areas = areas[conc_mask]
        nonzero = conc_areas > 0.0
        if not nonzero.any():
            return 0.0

        conc_stresses = conc_forces[nonzero] / conc_areas[nonzero]
        # Use Python max() to avoid NumPy max() default-sentinel edge cases seen
        # when NumPy is reloaded in some test environments.
        peak = float(max(float(s) for s in conc_stresses)) if len(conc_stresses) > 0 else 0.0
        return max(0.0, peak)

    def _get_max_steel_stress(
        self,
        eps_top: float,
        eps_bottom: float,
        strain_state: StrainState | None = None,
    ) -> float:
        """Maximum tensile stress across all bars (MPa, converted to positive stress).

        Assumes negative strains are tensile.
        """
        bounds = self.section.outline.bounds
        h = bounds[3] - bounds[1]
        y_min = bounds[1]

        if h <= 0:
            raise ValueError(f"Height, h: {h} mm cannot be equal to or less than zero.")

        use_biaxial = strain_state is not None and strain_state.is_biaxial
        cx = (bounds[0] + bounds[2]) / 2.0 if use_biaxial else 0.0
        cy = (bounds[1] + bounds[3]) / 2.0 if use_biaxial else 0.0

        max_stress = 0.0
        for group in self.section.rebar_groups:
            E_s = group.rebar.E_s
            f_yk = group.rebar.f_yk
            epsilon_uk = group.rebar.epsilon_uk
            k_ratio = group.rebar.grade.ft_ratio_min

            for pos in group.positions:
                if use_biaxial:
                    assert strain_state is not None  # implied by use_biaxial
                    strain = strain_state.strain_at(pos.x - cx, pos.y - cy)
                else:
                    y_rel = (pos.y - y_min) / h
                    strain = eps_bottom + (eps_top - eps_bottom) * y_rel

                if strain >= 0:
                    continue

                stress = flexure_utils.calculate_rebar_characteristic_stress_from_strain(
                    strain=strain,
                    steel_model_type=self.steel_model_type,
                    E_s=E_s,
                    f_yk=f_yk,
                    k=k_ratio,
                    epsilon_uk=epsilon_uk,
                )
                max_stress = max(max_stress, abs(stress))

        return max_stress

    def _get_f_yk_max(self) -> float:
        """Maximum f_yk across all rebar groups."""
        if not self.section.rebar_groups:
            return 500.0
        return max(g.rebar.f_yk for g in self.section.rebar_groups)

    # --- Public API ---

    def calculate_detailed(
        self,
        My_Ed: float,
        N_Ed: float = 0.0,
        ignore_compression_steel: bool = False,
        Mz_Ed: float = 0.0,
    ) -> StressLimitResult:
        """
        Run all enabled stress limitation checks and return detailed results.

        If apply_nonlinear_creep = True the returned stresses for concrete and steel
        will be calculated for the non-linear creep coefficient derived if the k_2
        limit is exceed. All other limits however will use stress derived with Ec
        used in the given the concrete_model_type (default E_cm).

        Args:
            My_Ed: Design moment at SLS (kN·m).
            N_Ed: Design axial force at SLS (kN, compression positive).
            ignore_compression_steel: If True, use conservative diagram.
            Mz_Ed: Design minor-axis moment at SLS (kN·m, default 0.0).

        Returns:
            StressLimitResult with all intermediate values.
        """
        M_Ed = My_Ed
        diagram = self._get_diagram(ignore_compression_steel)
        eps_top, eps_bottom = diagram.find_strains_for_MN(M_Ed, N_Ed, strict=True, Mz_target=Mz_Ed)
        strain_state_local = diagram.find_strain_state_for_MN(
            My_target=M_Ed, N_target=N_Ed, Mz_target=Mz_Ed,
        )

        sigma_c = self._get_peak_concrete_stress(
            eps_top, eps_bottom, diagram, strain_state=strain_state_local,
        )
        sigma_s = self._get_max_steel_stress(
            eps_top, eps_bottom, strain_state=strain_state_local,
        )
        f_yk = self._get_f_yk_max()

        nonlinear_creep_applied = False
        creep_coefficient_used = self.creep_coefficient
        messages: list[str] = []

        result = StressLimitResult(
            sigma_c_peak=sigma_c,
            sigma_s_max=sigma_s,
            f_yk=f_yk,
            creep_coefficient_used=creep_coefficient_used,
        )

        # EC2 §7.2(2): Characteristic concrete stress
        if self.check_k1_stress:
            exceeded, msg = check_characteristic_concrete_stress(sigma_c, self.concrete.f_ck)
            result.k1_exceeded = exceeded
            if exceeded:
                messages.append(msg)

        # EC2 §7.2(3): Quasi-permanent concrete stress + non-linear creep
        # Non-linear derived stresses do no affect downstream checks
        # (which are for different Serviceability limit states)
        if self.check_k2_stress:
            exceeded, msg = check_quasi_permanent_concrete_stress(sigma_c, self.concrete.f_ck)
            result.k2_exceeded = exceeded
            if exceeded:
                messages.append(msg)
                if self.apply_nonlinear_creep:
                    max_iterations = 5 if self.iterate_nonlinear_creep else 1
                    sigma_c_nl = sigma_c
                    sigma_s_nl = sigma_s
                    for _ in range(max_iterations):
                        phi_NL = compute_nonlinear_creep_coefficient(
                            sigma_c_nl, self.concrete.f_ck, self.creep_coefficient,
                        )
                        E_cm_eff_NL = self.concrete.get_elastic_modulus() / (1.0 + phi_NL)
                        if abs(E_cm_eff_NL - (self.concrete.get_elastic_modulus() / (1.0 + creep_coefficient_used))) < 1.0:
                            break
                        creep_coefficient_used = phi_NL
                        diagram_nl = self._build_diagram_with_E_cm_eff(
                            E_cm_eff_NL, ignore_compression_steel,
                        )
                        eps_top, eps_bottom = diagram_nl.find_strains_for_MN(
                            M_Ed, N_Ed, strict=True, Mz_target=Mz_Ed
                        )
                        strain_state_local = diagram_nl.find_strain_state_for_MN(
                            My_target=M_Ed, N_target=N_Ed, Mz_target=Mz_Ed,
                        )
                        sigma_c_nl = self._get_peak_concrete_stress(
                            eps_top, eps_bottom, diagram_nl,
                            strain_state=strain_state_local,
                        )
                        sigma_s_nl = self._get_max_steel_stress(
                            eps_top, eps_bottom, strain_state=strain_state_local,
                        )
                        nonlinear_creep_applied = True

                    result.sigma_c_peak = sigma_c_nl
                    result.sigma_s_max = sigma_s_nl
                    result.nonlinear_creep_applied = nonlinear_creep_applied
                    result.creep_coefficient_used = creep_coefficient_used

        # EC2 §7.2(5): Reinforcement stress
        if self.check_k3_stress:
            exceeded, msg = check_characteristic_reinforcement_stress(sigma_s, f_yk)
            result.k3_exceeded = exceeded
            if exceeded:
                messages.append(msg)

        # EC2 §7.2(4)P: Yielding
        if self.check_yielding:
            exceeded, msg = check_reinforcement_yielding(sigma_s, f_yk)
            result.yielding = exceeded
            if exceeded:
                messages.append(msg)

        # EC2 §7.2(5): Imposed deformation
        if self.check_k4_stress:
            exceeded, msg = check_imposed_deformation_stress(sigma_s, f_yk)
            result.k4_exceeded = exceeded
            if exceeded:
                messages.append(msg)

        result.messages = messages
        return result

    def perform_check(
        self,
        *,
        My_Ed: float | None = None,
        N_Ed: float = 0.0,
        Mz_Ed: float = 0.0,
        ignore_compression_steel: bool = False,
        warning_threshold: float = 0.95,
        suppress_warnings: bool = False,
        **kwargs: Any,
    ) -> CheckResult:
        """
        Perform EC2 §7.2 stress limitation checks.

        Args:
            My_Ed: Design major-axis moment at SLS (kN·m).
            N_Ed: Design axial force at SLS (kN, compression positive).
            Mz_Ed: Design minor-axis moment at SLS (kN·m, default 0.0).
            ignore_compression_steel: If True, use conservative diagram.
            warning_threshold: Utilization ratio triggering warnings.
            suppress_warnings: If True, suppress warnings emitted during this check.

        Returns:
            CheckResult with governing stress check utilization.
        """
        # Legacy support: remap M_Ed kwarg to My_Ed
        if "M_Ed" in kwargs:
            warnings.warn(
                "M_Ed is deprecated; use My_Ed instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            My_Ed = kwargs.pop("M_Ed")
        if My_Ed is None:
            raise TypeError("perform_check() missing required keyword argument: 'My_Ed'")
        try:
            r = self.calculate_detailed(My_Ed, N_Ed, ignore_compression_steel, Mz_Ed=Mz_Ed)
        except ValueError as e:
            return self._create_result(
                check_name="Stress limitation check (EC2 §7.2)",
                code_reference="EC2 §7.2",
                warning_threshold=warning_threshold,
                utilization=float("inf"),
                demand_components={"My_Ed": float(My_Ed), "Mz_Ed": float(Mz_Ed), "N_Ed": float(N_Ed)},
                capacity_components={},
                units_components={"My_Ed": "kN·m", "Mz_Ed": "kN·m", "N_Ed": "kN"},
                message=f"Failed to solve strain state: {e}",
                details={"error": str(e)},
            )

        # Compute utilization for each enabled check
        utilizations: list[tuple[float, str]] = []

        if self.check_k1_stress:
            limit = self.k_1_stress * self.concrete.f_ck
            util = r.sigma_c_peak / limit if limit > 0 else 0.0
            utilizations.append((util, "k1_concrete_char"))

        if self.check_k2_stress:
            limit = self.k_2_stress * self.concrete.f_ck
            util = r.sigma_c_peak / limit if limit > 0 else 0.0
            utilizations.append((util, "k2_concrete_qp"))

        if self.check_k3_stress:
            limit = self.k_3_stress * r.f_yk
            util = r.sigma_s_max / limit if limit > 0 else 0.0
            utilizations.append((util, "k3_reinforcement"))

        if self.check_yielding:
            util = r.sigma_s_max / r.f_yk if r.f_yk > 0 else 0.0
            utilizations.append((util, "yielding"))

        if self.check_k4_stress:
            limit = self.k_4_stress * r.f_yk
            util = r.sigma_s_max / limit if limit > 0 else 0.0
            utilizations.append((util, "k4_imposed_deformation"))

        if not utilizations:
            governing_util = 0.0
            governing_check = "none"
        else:
            governing_util, governing_check = max(utilizations, key=lambda x: x[0])

        # Emit warnings for exceeded checks
        if not suppress_warnings:
            for msg in r.messages:
                warnings.warn(msg, stacklevel=2)

        # Build details
        details: dict[str, Any] = {
            "My_Ed": float(My_Ed),
            "Mz_Ed": float(Mz_Ed),
            "N_Ed": float(N_Ed),
            "sigma_c_peak": r.sigma_c_peak,
            "sigma_s_max": r.sigma_s_max,
            "f_yk": r.f_yk,
            "governing_check": governing_check,
            "k1_exceeded": r.k1_exceeded,
            "k2_exceeded": r.k2_exceeded,
            "k3_exceeded": r.k3_exceeded,
            "yielding": r.yielding,
            "k4_exceeded": r.k4_exceeded,
            "nonlinear_creep_applied": r.nonlinear_creep_applied,
            "creep_coefficient_used": r.creep_coefficient_used,
        }

        is_pass = governing_util <= 1.0
        message = (
            f"Stress limits {'OK' if is_pass else 'EXCEEDED'} "
            f"(governing: {governing_check}, util={governing_util:.2f})"
        )

        return self._create_result(
            check_name="Stress limitation check (EC2 §7.2)",
            code_reference="EC2 §7.2",
            warning_threshold=warning_threshold,
            utilization=governing_util,
            demand_components={"My": float(My_Ed), "Mz": float(Mz_Ed), "N": float(N_Ed)},
            capacity_components={},
            units_components={"My": "kN·m", "Mz": "kN·m", "N": "kN"},
            message=message,
            details=details,
        )
