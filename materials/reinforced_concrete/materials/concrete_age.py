"""Time-dependent concrete material properties according to Eurocode 2.

This module provides ConcreteAge which extends ConcreteMaterial with time-dependent
properties for early-age concrete behaviour and strength development.
"""

from typing import Literal
from math import exp, sqrt
from pydantic import BaseModel, Field, computed_field

from .concrete import ConcreteMaterial


# Cement class for strength development rate
CementClass = Literal["R", "N", "S"]


class ConcreteAge(BaseModel):
    """
    Time-dependent concrete properties at a specific age.

    Uses composition to extend ConcreteMaterial with age-dependent properties
    according to EC2 §3.1.2 for strength development over time.

    The minimum valid age is 3 days as EC2 formulas for strength development
    require t > 3 days. For earlier ages, testing is required.

    Attributes:
        concrete: Base concrete material (28-day properties)
        age: Age of concrete in days (must be > 3 days)
        cement_class: Cement strength class affecting development rate
            - 'R': Rapid hardening (s=0.20)
            - 'N': Normal hardening (s=0.25)
            - 'S': Slow hardening (s=0.38)

    Time-dependent properties (computed):
        - beta_cc_t: Strength development coefficient
        - f_cm_t: Mean cylinder strength at age t
        - f_ck_t: Characteristic cylinder strength at age t
        - f_ctm_t: Mean tensile strength at age t
        - f_ctd_t: Design tensile strength at age t
        - E_cm_t: Secant modulus of elasticity at age t

    Examples:
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial, ConcreteAge
        >>>
        >>> # Create base concrete
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>>
        >>> # Create concrete at 7 days with normal hardening cement
        >>> concrete_7d = ConcreteAge(
        ...     concrete=concrete,
        ...     age=7.0,
        ...     cement_class="N"
        ... )
        >>>
        >>> # Access time-dependent properties
        >>> print(concrete_7d.f_cm_t)  # Mean strength at 7 days
        >>> print(concrete_7d.E_cm_t)  # Modulus at 7 days
    """

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

    @computed_field
    @property
    def beta_cc_t(self) -> float:
        """
        Coefficient for strength development over time (§3.1.2(6), Eq. 3.2).

        β_cc(t) = exp[s · (1 - √(28/t))]

        where s depends on cement class:
        - R (rapid): s = 0.20
        - N (normal): s = 0.25
        - S (slow): s = 0.38

        Returns:
            β_cc(t) (dimensionless)
        """
        # Cement class exponent
        s_values = {
            "R": 0.2,
            "N": 0.25,
            "S": 0.38,
        }
        s = s_values[self.cement_class]

        return exp(s * (1 - sqrt(28 / self.age)))

    @computed_field
    @property
    def f_cm_t(self) -> float:
        """
        Mean cylinder compressive strength at age t (§3.1.2(6), Eq. 3.1).

        f_cm(t) = β_cc(t) · f_cm

        Returns:
            f_cm(t) in MPa
        """
        return self.beta_cc_t * self.concrete.f_cm

    @computed_field
    @property
    def f_ck_t(self) -> float:
        """
        Characteristic cylinder compressive strength at age t (§3.1.2(5)).

        For t ≥ 28 days: f_ck(t) = f_ck (28-day value)
        For 3 < t < 28 days: f_ck(t) = f_cm(t) - 8 MPa

        Returns:
            f_ck(t) in MPa
        """
        if self.age >= 28:
            return self.concrete.f_ck
        else:
            return self.f_cm_t - 8.0

    @computed_field
    @property
    def f_ctm_t(self) -> float:
        """
        Mean tensile strength at age t (§3.1.2(9), Eq. 3.4).

        f_ctm(t) = f_ctm · β_cc(t)^α

        where:
        - α = 1 for t < 28 days
        - α = 2/3 for t ≥ 28 days

        Returns:
            f_ctm(t) in MPa
        """
        alpha = 1.0 if self.age < 28 else 2.0 / 3.0
        return self.concrete.f_ctm * (self.beta_cc_t ** alpha)

    @computed_field
    @property
    def f_ctd_t(self) -> float:
        """
        Design tensile strength at age t (§8.10.2.2(1)).

        f_ctd(t) = α_ct · f_ctm(t) / γ_c

        Uses 0.7 · f_ctm(t) as characteristic value.
        Uses alpha_ct from the composed concrete material.

        Returns:
            f_ctd(t) in MPa
        """
        # Use 0.7 factor for characteristic tensile strength
        f_ctk_t = 0.7 * self.f_ctm_t
        return self.concrete.alpha_ct * f_ctk_t / self.concrete.gamma_c

    @computed_field
    @property
    def E_cm_t(self) -> float:
        """
        Secant modulus of elasticity at age t (§3.1.3(3), Eq. 3.5).

        E_cm(t) = E_cm · [f_cm(t) / f_cm]^0.3

        Returns:
            E_cm(t) in MPa
        """
        return self.concrete.E_cm * ((self.f_cm_t / self.concrete.f_cm) ** 0.3)

    def get_flexural_tensile_strength(self, section_height_mm: float) -> float:
        """
        Mean flexural tensile strength at age t (§3.1.8(1), Eq. 3.23).

        Depends on section height:
        f_ctm,fl = f_ctm(t) · max(1.6 - h/1000, 1.0)

        Args:
            section_height_mm: Height of the section in mm

        Returns:
            f_ctm,fl in MPa
        """
        h_m = section_height_mm / 1000.0  # Convert to meters
        return self.f_ctm_t * max(1.6 - h_m, 1.0)

    def __str__(self) -> str:
        """User-friendly representation."""
        return (
            f"{self.concrete.grade} at {self.age:.1f} days "
            f"(cement class {self.cement_class}, "
            f"f_cm={self.f_cm_t:.1f} MPa, E_cm={self.E_cm_t:.0f} MPa)"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"ConcreteAge(concrete={self.concrete.grade}, "
            f"age={self.age:.1f} days, cement_class='{self.cement_class}')"
        )
