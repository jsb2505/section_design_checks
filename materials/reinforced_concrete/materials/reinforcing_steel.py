"""
Reinforcing steel material properties according to Eurocode 2.

Implements characteristic and design strengths for reinforcing bar grades.
"""

from enum import StrEnum
from functools import cached_property
from typing import Literal
from pydantic import Field, ConfigDict
from materials.core.base_material import BaseMaterial


# Define a Type Alias for clarity
DuctilityClass = Literal["A", "B", "C"]

# Steel grades according to EC2 §C.1 (single source of truth)
class ReinforcingSteelGrade(StrEnum):
    '''
    Reinforcing steel grades supported as per ECC2.

    Attributes:
        B500A
        B500B
        B500C
    '''
    B500A = "B500A"
    B500B = "B500B"
    B500C = "B500C"

    @property
    def _data(self) -> tuple[float, float, float, DuctilityClass]:
        """
        Returns (f_yk, ft_ratio_min, epsilon_uk_min, ductility_class)
        Explicitly typed to satisfy Pylance.
        """
        mapping: dict[ReinforcingSteelGrade, tuple[float, float, float, DuctilityClass]] = {
            ReinforcingSteelGrade.B500A: (500.0, 1.05, 0.025, "A"),
            ReinforcingSteelGrade.B500B: (500.0, 1.08, 0.050, "B"),
            ReinforcingSteelGrade.B500C: (500.0, 1.15, 0.075, "C"),
        }
        return mapping[self]
    
    @property
    def f_yk(self) -> float: return self._data[0]

    @property
    def ft_ratio_min(self) -> float: return self._data[1]

    @property
    def epsilon_uk_min(self) -> float: return self._data[2]

    @property
    def ductility_class(self) -> DuctilityClass:
        return self._data[3]


class ReinforcingSteel(BaseMaterial):
    """
    Reinforcing steel properties per Eurocode 2.

    Properties from EC2 Annex C.

    Units:
        - Strength: MPa
        - Modulus: MPa (GPa × 1000)
        - Strain: dimensionless
    """

    model_config = ConfigDict(
        ignored_types=(cached_property,),  # Allow cached_property to work
    )

    name: str = Field(default="Reinforcing Steel", description="Material name")

    grade: ReinforcingSteelGrade = Field(
        default=ReinforcingSteelGrade.B500B,
        description="Steel grade per EC2 Annex C (defaults to B500B)",
    )

    E_s: float = Field(
        default=200_000.0,
        description="Elastic modulus (§3.2.7): default 200 GPa = 200,000 MPa",
        gt=0,
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

    density: float = Field(
        default=7850.0,
        description="Steel density in kg/m³",
        ge=0,
    )

    # ---- Derived properties (NOT included in model_dump) ----

    @property
    def f_yk(self) -> float:
        """Characteristic yield strength (§C.1). All B500 grades: 500 MPa."""
        return self.grade.f_yk

    @cached_property
    def f_yd(self) -> float:
        """Design yield strength (§2.4.2.4): f_yd = f_yk / γ_s."""
        return self.f_yk / self.gamma_s

    @cached_property
    def f_yd_accidental(self) -> float:
        """Accidental design yield strength: f_yd = f_yk / γ_s,accidental."""
        return self.f_yk / self.gamma_s_accidental

    @cached_property
    def f_t(self) -> float:
        """
        Characteristic tensile strength f_tk (§C.1).

        Uses minimum ratio × f_yk.
        """
        return self.grade.ft_ratio_min * self.f_yk

    @cached_property
    def f_td(self) -> float:
        """
        Design tensile strength f_td = f_t / γ_s.

        Used for inclined branch of stress-strain curve at ULS.
        """
        return self.f_t / self.gamma_s

    @cached_property
    def f_td_accidental(self) -> float:
        """
        Accidental design tensile strength f_td,acc = f_t / γ_s,acc.

        Used for inclined branch of stress-strain curve at accidental ULS.
        """
        return self.f_t / self.gamma_s_accidental

    @cached_property
    def epsilon_yk(self) -> float:
        """Characteristic yield strain (§3.2.7): ε_yk = f_yk / E_s."""
        return self.f_yk / self.E_s

    @cached_property
    def epsilon_yd(self) -> float:
        """Design yield strain: ε_yd = f_yd / E_s."""
        return self.f_yd / self.E_s

    @cached_property
    def epsilon_uk(self) -> float:
        """
        Characteristic strain at maximum load (§C.1).

        Returns:
            ε_uk (dimensionless)
        """
        return self.grade.epsilon_uk_min

    @cached_property
    def epsilon_ud(self) -> float:
        """Design ultimate strain (§3.2.7): ε_ud = 0.9 · ε_uk."""
        return 0.9 * self.epsilon_uk

    @cached_property
    def k_ratio(self) -> float:
        """Ratio f_t/f_yk for ductility classification (§C.1)."""
        return self.f_t / self.f_yk

    @property
    def ductility_class(self) -> Literal["A", "B", "C"]:
        """Ductility class based on grade."""
        return self.grade.ductility_class
    
    def get_elastic_modulus(self) -> float:
        """
        Return elastic modulus (implements BaseMaterial abstract method).

        Returns:
            E_s in MPa
        """
        return self.E_s

    @classmethod
    def f_yk_for(cls, grade: ReinforcingSteelGrade | str | None = None) -> float:
        """Characteristic yield strength (MPa) for a given grade (defaults to B500B)."""
        # Convert string to Enum if needed, default to B500B
        g = ReinforcingSteelGrade(grade) if grade else ReinforcingSteelGrade.B500B
        return g.f_yk

    @classmethod
    def f_yd_for(
        cls,
        *,
        grade: ReinforcingSteelGrade | None = None,
        gamma_s: float = 1.15,
    ) -> float:
        """Design yield strength (MPa): f_yd = f_yk / gamma_s."""
        return cls.f_yk_for(grade) / gamma_s

    @classmethod
    def f_yd_accidental_for(
        cls,
        *,
        grade: ReinforcingSteelGrade | None = None,
        gamma_s_accidental: float = 1.0,
    ) -> float:
        """Accidental design yield strength (MPa): f_yd,acc = f_yk / gamma_s_accidental."""
        return cls.f_yk_for(grade) / gamma_s_accidental

    def __str__(self) -> str:
        return (
            f"{self.grade} "
            f"(f_yk={self.f_yk} MPa, f_yd={self.f_yd:.1f} MPa, "
            f"Class {self.ductility_class})"
        )
