"""Time-dependent concrete material properties according to Eurocode 2.

This module provides ConcreteAge which extends ConcreteMaterial with time-dependent
properties for early-age concrete behaviour and strength development.
"""

from enum import StrEnum
from math import exp, sqrt

from pydantic import BaseModel, ConfigDict, Field

from section_design_checks.reinforced_concrete.materials import ConcreteMaterial
from section_design_checks.reinforced_concrete.materials.concrete import (
    find_mean_flexural_tensile_strength,
)


class CementClass(StrEnum):
    '''Cement class for strength development rate.

    Attributes:
        R: = Rapid (Class R)
        N: = Normal (Class N)
        S: = Slow (Class S)
    '''
    R = "R"
    N = "N"
    S = "S"

    @property
    def s_coefficient(self) -> float:
        """
        Coefficient 's' depends on cement type (§3.1.2(6)).
        R = 0.20, N = 0.25, S = 0.38
        """
        return {
            CementClass.R: 0.20,
            CementClass.N: 0.25,
            CementClass.S: 0.38,
        }[self]


class ConcreteAge(BaseModel):
    """
    Time-dependent concrete properties at a specific age.

    Uses composition to extend ConcreteMaterial with age-dependent properties
    according to EC2 §3.1.2 for strength development over time.

    Note:
        Valid for ages strictly greater than 3 days (t > 3).
    """

    # Mirror your BaseMaterial style config for consistency across the repo
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=False,
    )

    concrete: ConcreteMaterial = Field(
        ...,
        description="Base concrete material with 28-day properties",
    )

    age: float = Field(
        ...,
        gt=3.0,
        description="Age of concrete in days (must be > 3 days per EC2 §3.1.2)",
    )

    cement_class: CementClass = Field(
        ...,
        description="Cement strength class: R (rapid), N (normal), S (slow)",
    )

    @property
    def beta_cc_t(self) -> float:
        """
        Coefficient for strength development over time (§3.1.2(6), Eq. 3.2).

        β_cc(t) = exp[s · (1 - √(28/t))]
        """
        s = self.cement_class.s_coefficient
        return exp(s * (1.0 - sqrt(28.0 / self.age)))

    @property
    def f_cm_t(self) -> float:
        """
        Mean cylinder compressive strength at age t (§3.1.2(6), Eq. 3.1).

        f_cm(t) = β_cc(t) · f_cm
        """
        return self.beta_cc_t * self.concrete.f_cm

    @property
    def f_ck_t(self) -> float:
        """
        Characteristic cylinder compressive strength at age t (§3.1.2(5)).

        Uncapped form (applies for all t > 3):
            f_ck(t) = f_cm(t) - 8 MPa
        """
        return self.f_cm_t - 8.0

    @property
    def f_ctm_t(self) -> float:
        """
        Mean tensile strength at age t (§3.1.2(9), Eq. 3.4).

        f_ctm(t) = f_ctm · β_cc(t)^α

        where:
        - α = 1 for t < 28 days
        - α = 2/3 for t ≥ 28 days
        """
        alpha = 1.0 if self.age < 28.0 else (2.0 / 3.0)
        return float(self.concrete.f_ctm * (self.beta_cc_t ** alpha))

    @property
    def f_ctd_t(self) -> float:
        """
        Design tensile strength at age t.

        Uses 0.7 · f_ctm(t) as characteristic value:
            f_ctd(t) = α_ct · [0.7 f_ctm(t)] / γ_c
        """
        f_ctk_t = 0.7 * self.f_ctm_t
        return self.concrete.alpha_ct * f_ctk_t / self.concrete.gamma_c

    @property
    def E_cm_t(self) -> float:
        """
        Secant modulus of elasticity at age t (§3.1.3(3), Eq. 3.5).

        E_cm(t) = E_cm · [f_cm(t) / f_cm]^0.3
        """
        return float(self.concrete.E_cm * ((self.f_cm_t / self.concrete.f_cm) ** 0.3))

    def find_mean_flexural_tensile_strength(self, section_height: float) -> float:
        """
        Mean flexural tensile strength at age t (§3.1.8(1), Eq. 3.23).

        Args:
            section_height: Geometrical height of section (mm)

        Returns:
            f_ctm,fl,t in MPa
        """
        return find_mean_flexural_tensile_strength(self.f_ctm_t, section_height)

    def to_material(self) -> ConcreteMaterial:
        """Create a ConcreteMaterial with age-adjusted properties.

        The returned object has the same grade, partial factors, and alpha
        coefficients as the base concrete, but f_ck, f_cm, and E_cm are
        overridden with age-adjusted values from EC2 §3.1.2.

        All dependent design values (f_cd, f_ctm, f_ctd, strain limits, etc.)
        auto-recompute from the overridden strengths.

        Returns:
            ConcreteMaterial with age-adjusted properties.
        """
        return self.concrete.model_copy(
            update={
                "name": f"{self.concrete.grade} at {self.age:.0f} days ({self.cement_class})",
                "f_ck_override": self.f_ck_t,
                "f_cm_override": self.f_cm_t,
                "E_cm_override": self.E_cm_t,
            }
        )

    def __str__(self) -> str:
        return (
            f"{self.concrete.grade} at {self.age:.1f} days "
            f"(cement class {self.cement_class}, "
            f"f_cm={self.f_cm_t:.1f} MPa, f_ck={self.f_ck_t:.1f} MPa, "
            f"E_cm={self.E_cm_t:.0f} MPa)"
        )

    def __repr__(self) -> str:
        return (
            f"ConcreteAge(concrete={self.concrete.grade}, "
            f"age={self.age:.1f} days, cement_class='{self.cement_class}')"
        )
