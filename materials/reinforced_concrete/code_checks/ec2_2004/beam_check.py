"""
Beam wrapper check for EC2 2004.

This module provides a light orchestration class that groups the four main
beam-related checks:
- BendingCheck (ULS)
- ShearCheck (ULS)
- CrackingCheck (SLS)
- StressLimitsCheck (SLS)

Unlike circular-section checks, this wrapper applies no beam-specific formula
overrides and delegates directly to the underlying check classes.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from materials.reinforced_concrete.code_checks.base_check import CheckResult
from materials.reinforced_concrete.code_checks.ec2_2004.bending_check import BendingCheck
from materials.reinforced_concrete.code_checks.ec2_2004.cracking_check import (
    CrackingCheck,
    LoadDuration,
)
from materials.reinforced_concrete.code_checks.ec2_2004.flexure_utils import (
    EffectiveDepthFallback,
)
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import (
    ShearCheck,
    ShearLoadCase,
)
from materials.reinforced_concrete.code_checks.ec2_2004.stress_limits_check import (
    StressLimitsCheck,
)
from materials.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
from materials.reinforced_concrete.ndp import get_ndp_context


class BeamCheck(BaseModel):
    """
    Composite EC2 beam check wrapper.

    This class encapsulates the four principal checks for reinforced concrete
    beams and forwards calls directly to each sub-check implementation.
    """

    model_config = ConfigDict(frozen=True)

    # ===========================
    # Core inputs
    # ===========================

    section: RCSection = Field(
        ...,
        description="Beam RC section geometry with reinforcement",
    )

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material properties",
    )

    shear_reinforcement: Optional[ShearRebar] = Field(
        default=None,
        description="Shear links/stirrups (None if unreinforced)",
    )

    # ===========================
    # Forwarded to ULS sub-checks
    # ===========================

    use_accidental: bool = Field(
        default=False,
        description="Use accidental limit state partial factors",
    )

    use_mechanical_lever_arm: bool = Field(
        default=True,
        description="Use rigorous solver mode in ShearCheck",
    )

    allow_negative_sigma_cp: bool = Field(
        default=True,
        description="Allow negative sigma_cp from axial tension in ShearCheck",
    )

    use_transformed_area_for_sigma_cp: bool = Field(
        default=True,
        description="Use transformed area for sigma_cp in ShearCheck",
    )

    use_sigma_cp_for_alpha_cw: bool = Field(
        default=False,
        description="Include sigma_cp in alpha_cw for ShearCheck V_Rd,max",
    )

    z_d_ratio: float = Field(
        default=0.9,
        description="Lever arm ratio z/d for approximate mode in ShearCheck",
        gt=0.0, le=1.0,
    )

    z_d_ratio_upper: float = Field(
        default=0.95,
        description="Upper bound lever arm ratio z/d for rigorous mode in ShearCheck",
        gt=0.0, le=1.0,
    )

    z_d_ratio_lower: float = Field(
        default=0.65,
        description="Lower bound lever arm ratio z/d for rigorous mode in ShearCheck",
        gt=0.0, le=1.0,
    )

    breadth_override: Optional[float] = Field(
        default=None,
        description="Optional manual web breadth override for ShearCheck",
        gt=0.0,
    )

    use_increased_nu_1: bool = Field(
        default=False,
        description="Enable EC2 Note 2 increased nu_1 policy in ShearCheck",
    )

    apply_tension_cot_theta_limit: bool = Field(
        default=True,
        description="Apply tension-specific cot(theta) upper limit where available",
    )

    d_fallback: EffectiveDepthFallback = Field(
        default="ratio_of_h",
        description="Fallback policy for effective depth in ambiguous strain states",
    )

    d_ratio: float = Field(
        default=0.9,
        description="Depth ratio for d_fallback policy",
        gt=0.0,
        le=1.0,
    )

    concrete_model_type: ConcreteModelType = Field(
        default=ConcreteModelType.PARABOLA_RECTANGLE,
        description="Concrete stress-strain model for ULS checks",
    )

    steel_model_type: SteelModelType = Field(
        default=SteelModelType.INCLINED,
        description="Steel post-yield behaviour model",
    )

    n_fibres_width: int = Field(
        default=20,
        description="Number of concrete fibres across section width",
        ge=10,
        le=500,
    )

    n_fibres_height: int = Field(
        default=30,
        description="Number of concrete fibres across section height",
        ge=10,
        le=500,
    )

    # ===========================
    # Forwarded to SLS sub-checks
    # ===========================

    w_k_limit: float = Field(
        default=0.3,
        description="Allowable crack width in mm",
        gt=0.0,
    )

    load_duration: LoadDuration = Field(
        default=LoadDuration.LONG_TERM,
        description="Load duration category for crack-width calculations",
    )

    creep_coefficient: float = Field(
        default=1.5,
        description="Linear creep coefficient for SLS checks",
        ge=0.0,
    )

    is_high_bond_bar: bool = Field(
        default=True,
        description="True for ribbed bars (k_1=0.8), False for plain bars (k_1=1.6)",
    )

    check_k1_stress: bool = Field(
        default=False,
        description="EC2 7.2(2) characteristic concrete stress limit",
    )

    check_k2_stress: bool = Field(
        default=True,
        description="EC2 7.2(3) quasi-permanent concrete stress limit",
    )

    check_k3_stress: bool = Field(
        default=False,
        description="EC2 7.2(5) reinforcement stress limit",
    )

    check_yielding: bool = Field(
        default=True,
        description="EC2 7.2(4)P inelastic strain/yielding check",
    )

    check_k4_stress: bool = Field(
        default=False,
        description="EC2 7.2(5) imposed deformation stress limit",
    )

    apply_nonlinear_creep: bool = Field(
        default=True,
        description="Enable non-linear creep adjustment for SLS stress checks",
    )

    iterate_nonlinear_creep: bool = Field(
        default=False,
        description="Iterate non-linear creep adjustment to convergence",
    )

    net_tension_face: Optional[Literal["top", "bottom"]] = Field(
        default=None,
        description="Optional face selection policy for net-tension crack checks",
    )

    # ===========================
    # Private sub-check instances
    # ===========================

    _bending_check: Optional[BendingCheck] = PrivateAttr(default=None)
    _shear_check: Optional[ShearCheck] = PrivateAttr(default=None)
    _cracking_check: Optional[CrackingCheck] = PrivateAttr(default=None)
    _stress_limits_check: Optional[StressLimitsCheck] = PrivateAttr(default=None)
    _ndp_snapshot: tuple = PrivateAttr(default=())

    @model_validator(mode="after")
    def _post_init(self) -> "BeamCheck":
        self._bending_check = BendingCheck(
            section=self.section,
            concrete=self.concrete,
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
            concrete=self.concrete,
            shear_reinforcement=self.shear_reinforcement,
            use_accidental=self.use_accidental,
            use_mechanical_lever_arm=self.use_mechanical_lever_arm,
            allow_negative_sigma_cp=self.allow_negative_sigma_cp,
            use_transformed_area_for_sigma_cp=self.use_transformed_area_for_sigma_cp,
            use_sigma_cp_for_alpha_cw=self.use_sigma_cp_for_alpha_cw,
            z_d_ratio=self.z_d_ratio,
            z_d_ratio_upper=self.z_d_ratio_upper,
            z_d_ratio_lower=self.z_d_ratio_lower,
            breadth_override=self.breadth_override,
            use_increased_nu_1=self.use_increased_nu_1,
            apply_tension_cot_theta_limit=self.apply_tension_cot_theta_limit,
            concrete_model_type=self.concrete_model_type,
            steel_model_type=self.steel_model_type,
            d_fallback=self.d_fallback,
            d_ratio=self.d_ratio,
        )

        self._cracking_check = CrackingCheck(
            section=self.section,
            concrete=self.concrete,
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
            apply_nonlinear_creep=self.apply_nonlinear_creep,
            iterate_nonlinear_creep=self.iterate_nonlinear_creep,
            net_tension_face=self.net_tension_face,
        )

        self._stress_limits_check = StressLimitsCheck(
            section=self.section,
            concrete=self.concrete,
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
            apply_nonlinear_creep=self.apply_nonlinear_creep,
            iterate_nonlinear_creep=self.iterate_nonlinear_creep,
        )

        self._ndp_snapshot = get_ndp_context()
        return self

    def _check_ndp_context(self) -> None:
        """Warn if active NDP context changed since wrapper construction."""
        current = get_ndp_context()
        if current != self._ndp_snapshot:
            warnings.warn(
                f"NDP context has changed since this BeamCheck was constructed "
                f"(was {self._ndp_snapshot}, now {current}). "
                f"Sub-check settings may reflect the original context.",
                UserWarning,
                stacklevel=3,
            )

    def with_updates(self, **changes: Any) -> "BeamCheck":
        """
        Return a new BeamCheck with updated constructor fields.

        This rebuilds all sub-check delegates from scratch to avoid stale state.
        """
        current = {name: getattr(self, name) for name in type(self).model_fields}
        current.update(changes)
        return type(self)(**current)

    @property
    def bending(self) -> BendingCheck:
        """Access the internal BendingCheck."""
        assert self._bending_check is not None
        return self._bending_check

    @property
    def shear(self) -> ShearCheck:
        """Access the internal ShearCheck."""
        assert self._shear_check is not None
        return self._shear_check

    @property
    def cracking(self) -> CrackingCheck:
        """Access the internal CrackingCheck."""
        assert self._cracking_check is not None
        return self._cracking_check

    @property
    def stress_limits(self) -> StressLimitsCheck:
        """Access the internal StressLimitsCheck."""
        assert self._stress_limits_check is not None
        return self._stress_limits_check

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
        iterate_z: bool = False,
        **kwargs: Any,
    ) -> CheckResult:
        """
        Forward a bending check to the internal BendingCheck delegate.
        """
        self._check_ndp_context()
        shear_reinf = (
            shear_reinforcement
            if shear_reinforcement is not None
            else self.shear_reinforcement
        )
        assert self._bending_check is not None
        return self._bending_check.perform_check(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            V_Ed=V_Ed,
            M_cap=M_cap,
            shear_reinforcement=shear_reinf,
            cot_theta_override=cot_theta_override,
            use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
            warning_threshold=warning_threshold,
            suppress_warnings=suppress_warnings,
            ignore_compression_steel=ignore_compression_steel,
            iterate_z=iterate_z,
            **kwargs,
        )

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
        **kwargs: Any,
    ) -> CheckResult:
        """
        Forward a shear check to the internal ShearCheck delegate.
        """
        self._check_ndp_context()
        assert self._shear_check is not None
        return self._shear_check.perform_check(
            load_case=load_case,
            cot_theta_override=cot_theta_override,
            use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
            use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
            warning_threshold=warning_threshold,
            suppress_warnings=suppress_warnings,
            ignore_compression_steel=ignore_compression_steel,
            **kwargs,
        )

    def perform_cracking_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        warning_threshold: float = 0.95,
        ignore_compression_steel: bool = False,
        force_cracked: bool = False,
        suppress_warnings: bool = False,
        actual_bar_diameter: Optional[float] = None,
        **kwargs: Any,
    ) -> CheckResult:
        """
        Forward a cracking check to the internal CrackingCheck delegate.
        """
        self._check_ndp_context()
        assert self._cracking_check is not None
        return self._cracking_check.perform_check(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            warning_threshold=warning_threshold,
            ignore_compression_steel=ignore_compression_steel,
            force_cracked=force_cracked,
            suppress_warnings=suppress_warnings,
            actual_bar_diameter=actual_bar_diameter,
            **kwargs,
        )

    def perform_stress_limits_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        warning_threshold: float = 0.95,
        ignore_compression_steel: bool = False,
        suppress_warnings: bool = False,
        **kwargs: Any,
    ) -> CheckResult:
        """
        Forward a stress-limitation check to the internal StressLimitsCheck delegate.
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
