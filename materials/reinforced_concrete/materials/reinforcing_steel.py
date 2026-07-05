"""
Reinforcing steel material properties according to Eurocode 2.

Implements characteristic and design strengths for reinforcing bar grades.
"""

from typing import Literal, TypedDict
from pydantic import Field
from materials.core.base_material import BaseMaterial


# Steel grades according to EC2 §C.1 (single source of truth)
ReinforcingSteelGrade = Literal["B500A", "B500B", "B500C"]


class SteelGradeData(TypedDict):
    ft_ratio_min: float
    epsilon_uk_min: float
    ductility_class: Literal["A", "B", "C"]


STEEL_GRADE_DATA: dict[ReinforcingSteelGrade, SteelGradeData] = {
    "B500A": {"ft_ratio_min": 1.05, "epsilon_uk_min": 0.025, "ductility_class": "A"},
    "B500B": {"ft_ratio_min": 1.08, "epsilon_uk_min": 0.050, "ductility_class": "B"},
    "B500C": {"ft_ratio_min": 1.15, "epsilon_uk_min": 0.075, "ductility_class": "C"},
}


class ReinforcingSteel(BaseMaterial):
    """
    Reinforcing steel properties per Eurocode 2.

    Properties from EC2 Annex C.

    Units:
        - Strength: MPa
        - Modulus: MPa (GPa × 1000)
        - Strain: dimensionless
    """

    name: str = Field(default="Reinforcing Steel", description="Material name")

    grade: ReinforcingSteelGrade = Field(
        default="B500B",
        description="Steel grade per EC2 Annex C (defaults to B500B)",
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

    # Overrides BaseMaterial.density (Optional[float]) with a fixed default.
    # Relax constraint to allow 0 or greater as requested.
    density: float = Field(
        default=7850.0,
        description="Steel density in kg/m³",
        ge=0,
    )

    # ---- Derived properties (NOT included in model_dump) ----

    @property
    def f_yk(self) -> float:
        """Characteristic yield strength (§C.1). All B500 grades: 500 MPa."""
        return 500.0

    @property
    def f_yd(self) -> float:
        """Design yield strength (§2.4.2.4): f_yd = f_yk / γ_s."""
        return self.f_yk / self.gamma_s

    @property
    def f_yd_accidental(self) -> float:
        """Accidental design yield strength: f_yd = f_yk / γ_s,accidental."""
        return self.f_yk / self.gamma_s_accidental

    @property
    def E_s(self) -> float:
        """Elastic modulus (§3.2.7): E_s = 200 GPa = 200,000 MPa."""
        return 200_000.0

    def get_elastic_modulus(self) -> float:
        """
        Return elastic modulus (implements BaseMaterial abstract method).

        Returns:
            E_s in MPa
        """
        return self.E_s

    @property
    def f_t(self) -> float:
        """
        Characteristic tensile strength (§C.1).

        Uses minimum ratio × f_yk.
        """
        ft_ratio = float(STEEL_GRADE_DATA[self.grade]["ft_ratio_min"])
        return ft_ratio * self.f_yk

    @property
    def epsilon_yk(self) -> float:
        """Characteristic yield strain (§3.2.7): ε_yk = f_yk / E_s."""
        return self.f_yk / self.E_s

    @property
    def epsilon_yd(self) -> float:
        """Design yield strain: ε_yd = f_yd / E_s."""
        return self.f_yd / self.E_s

    @property
    def epsilon_uk(self) -> float:
        """
        Characteristic strain at maximum load (§C.1).

        Returns:
            ε_uk (dimensionless)
        """
        return float(STEEL_GRADE_DATA[self.grade]["epsilon_uk_min"])

    @property
    def epsilon_ud(self) -> float:
        """Design ultimate strain (§3.2.7): ε_ud = 0.9 · ε_uk."""
        return 0.9 * self.epsilon_uk

    @property
    def k_ratio(self) -> float:
        """Ratio f_t/f_yk for ductility classification (§C.1)."""
        return self.f_t / self.f_yk

    @property
    def ductility_class(self) -> Literal["A", "B", "C"]:
        """Ductility class based on grade."""
        return STEEL_GRADE_DATA[self.grade]["ductility_class"]  # type: ignore[return-value]

    def __str__(self) -> str:
        return (
            f"{self.grade} "
            f"(f_yk={self.f_yk} MPa, f_yd={self.f_yd:.1f} MPa, "
            f"Class {self.ductility_class})"
        )
