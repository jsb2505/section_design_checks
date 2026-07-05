"""
Reinforcing steel material properties according to Eurocode 2.

Implements characteristic and design strengths for reinforcing bar grades.
"""

from typing import Literal, Optional
from pydantic import Field, field_validator, computed_field
from materials.core.base_material import BaseMaterial


# Steel grades according to EC2 §C.1
ReinforcingSteelGrade = Literal["B500A", "B500B", "B500C"]


class ReinforcingSteel(BaseMaterial):
    """
    Reinforcing steel properties per Eurocode 2.

    Properties from EC2 Annex C (UK National Annex).

    Units:
        - Strength: MPa
        - Modulus: MPa (GPa × 1000)
        - Strain: dimensionless
    """

    name: str = Field(default="Reinforcing Steel", description="Material name")
    grade: ReinforcingSteelGrade = Field(
        ...,
        description="Steel grade per EC2 Annex C"
    )

    gamma_s: float = Field(
        default=1.15,
        description="Partial factor for steel - ULS (§2.4.2.4)",
        gt=0,
    )

    gamma_s_accidental: float = Field(
        default=1.0,
        description="Partial factor for steel - accidental (§2.4.2.4)",
        gt=0,
    )

    density: Optional[float] = Field(
        default=7850.0,
        description="Steel density in kg/m³",
        ge=7800,
        le=7900,
    )

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, v: str) -> str:
        """Validate steel grade."""
        valid_grades = ["B500A", "B500B", "B500C"]
        if v not in valid_grades:
            raise ValueError(
                f"Invalid steel grade: {v}. "
                f"Must be one of {valid_grades}"
            )
        return v

    @computed_field
    @property
    def f_yk(self) -> float:
        """
        Characteristic yield strength (§C.1).

        All B500 grades: f_yk = 500 MPa

        Returns:
            f_yk in MPa
        """
        return 500.0

    @computed_field
    @property
    def f_yd(self) -> float:
        """
        Design yield strength (§2.4.2.4).

        f_yd = f_yk / γ_s

        Returns:
            f_yd in MPa
        """
        return self.f_yk / self.gamma_s

    @computed_field
    @property
    def f_yd_accidental(self) -> float:
        """
        Design yield strength for accidental load combinations.

        f_yd = f_yk / γ_s,accidental (typically 1.0)

        Returns:
            f_yd in MPa
        """
        return self.f_yk / self.gamma_s_accidental

    @computed_field
    @property
    def E_s(self) -> float:
        """
        Modulus of elasticity for reinforcing steel (§3.2.7).

        E_s = 200 GPa = 200,000 MPa

        Returns:
            E_s in MPa
        """
        return 200_000.0

    def get_elastic_modulus(self) -> float:
        """
        Return elastic modulus (implements BaseMaterial abstract method).

        Returns:
            E_s in MPa
        """
        return self.E_s

    @computed_field
    @property
    def f_t(self) -> float:
        """
        Characteristic tensile strength (§C.1).

        Grade-dependent:
        - B500A: f_t/f_yk ≥ 1.05
        - B500B: f_t/f_yk ≥ 1.08
        - B500C: f_t/f_yk ≥ 1.15

        Uses minimum ratio × f_yk.

        Returns:
            f_t in MPa
        """
        ratios = {
            "B500A": 1.05,
            "B500B": 1.08,
            "B500C": 1.15,
        }
        return ratios[self.grade] * self.f_yk

    @computed_field
    @property
    def epsilon_yk(self) -> float:
        """
        Characteristic yield strain (§3.2.7).

        ε_yk = f_yk / E_s

        Returns:
            ε_yk (dimensionless)
        """
        return self.f_yk / self.E_s

    @computed_field
    @property
    def epsilon_yd(self) -> float:
        """
        Design yield strain.

        ε_yd = f_yd / E_s

        Returns:
            ε_yd (dimensionless)
        """
        return self.f_yd / self.E_s

    @computed_field
    @property
    def epsilon_uk(self) -> float:
        """
        Characteristic strain at maximum load (§C.1).

        Grade-dependent:
        - B500A: ε_uk ≥ 2.5%
        - B500B: ε_uk ≥ 5.0%
        - B500C: ε_uk ≥ 7.5%

        Returns:
            ε_uk (dimensionless)
        """
        strains = {
            "B500A": 0.025,
            "B500B": 0.050,
            "B500C": 0.075,
        }
        return strains[self.grade]

    @computed_field
    @property
    def epsilon_ud(self) -> float:
        """
        Design ultimate strain (§3.2.7).

        ε_ud = 0.9 · ε_uk

        Returns:
            ε_ud (dimensionless)
        """
        return 0.9 * self.epsilon_uk

    @computed_field
    @property
    def k_ratio(self) -> float:
        """
        Ratio f_t/f_yk for ductility classification (§C.1).

        Returns:
            k = f_t / f_yk
        """
        return self.f_t / self.f_yk

    @computed_field
    @property
    def ductility_class(self) -> Literal["A", "B", "C"]:
        """
        Ductility class based on grade.

        Returns:
            Ductility class (A, B, or C)
        """
        classes = {
            "B500A": "A",
            "B500B": "B",
            "B500C": "C",
        }
        return classes[self.grade]

    def __str__(self) -> str:
        """User-friendly representation."""
        return (
            f"{self.grade} "
            f"(f_yk={self.f_yk} MPa, f_yd={self.f_yd:.1f} MPa, "
            f"Class {self.ductility_class})"
        )
