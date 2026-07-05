"""
Concrete material properties according to Eurocode 2.

Implements characteristic strengths, design strengths, elastic modulus,
and stress-strain parameters for concrete grades C12/15 to C90/105.
"""

from typing import Literal
import math
from pydantic import Field
from materials.core.base_material import BaseMaterial


# Concrete grades according to EC2 Table 3.1 (single source of truth)
ConcreteGrade = Literal[
    "C12/15", "C16/20", "C20/25", "C25/30", "C30/37", "C35/45", "C40/50",
    "C45/55", "C50/60", "C55/67", "C60/75", "C70/85", "C80/95", "C90/105"
]

AggregateType = Literal["quartzite", "limestone", "sandstone", "basalt"]

_AGGREGATE_ECM_FACTORS: dict[AggregateType, float] = {
    "basalt": 1.2,
    "quartzite": 1.0,
    "limestone": 0.9,
    "sandstone": 0.7,
}


class ConcreteMaterial(BaseMaterial):
    """
    Concrete material properties per Eurocode 2.

    All properties calculated from concrete grade.
    Uses standard material factors: γ_c = 1.5 for ULS.

    Units:
        - Strength: MPa
        - Modulus: MPa (GPa in some references, converted to MPa)
        - Strain: dimensionless
    """

    name: str = Field(default="Concrete", description="Material name")
    grade: ConcreteGrade = Field(..., description="Concrete grade per EC2 Table 3.1")

    gamma_c: float = Field(
        default=1.5,
        description="Partial factor for concrete for ULS persistent/transient (§2.4.2.4)",
        gt=0,
    )

    gamma_c_accidental: float = Field(
        default=1.2,
        description="Partial factor for concrete for ULS accidental (§2.4.2.4, National Annex)",
        gt=0,
    )

    alpha_cc: float = Field(
        default=0.85,
        description="Coefficient for long-term effects on strength (§3.1.6(1)P)",
        gt=0,
        le=1.0,
    )

    alpha_ct: float = Field(
        default=1.0,
        description="Coefficient for long-term effects on tensile strength",
        gt=0,
        le=1.0,
    )

    aggregate_type: AggregateType = Field(
        default="quartzite",
        description="Aggregate type for elastic modulus adjustment (§3.1.3)",
    )

    density: float = Field(
        default=2400.0,
        description="Concrete density in kg/m³",
        ge=0,
    )

    # ---- Derived properties (NOT included in model_dump) ----

    @property
    def f_ck(self) -> float:
        """Characteristic cylinder compressive strength at 28 days (§3.1.2), MPa."""
        f_ck_int = int(self.grade.split("/")[0].replace("C", ""))
        if f_ck_int > 90:
            raise ValueError("Concrete cylinder strengths above 90 are not supported as per (§3.1.2(2)P)")
        return float(f_ck_int)

    @property
    def f_ck_cube(self) -> float:
        """Characteristic cube compressive strength at 28 days (§3.1.2), MPa."""
        return float(self.grade.split("/")[1])

    @property
    def f_cm(self) -> float:
        """Mean cylinder compressive strength (§Table 3.1): f_cm = f_ck + 8 (MPa)."""
        return self.f_ck + 8.0

    @property
    def f_cd(self) -> float:
        """Design compressive strength (§3.1.6): f_cd = α_cc · f_ck / γ_c (MPa)."""
        return self.alpha_cc * self.f_ck / self.gamma_c

    @property
    def f_cd_accidental(self) -> float:
        """Accidental design compressive strength: f_cd,acc = α_cc · f_ck / γ_c,acc (MPa)."""
        return self.alpha_cc * self.f_ck / self.gamma_c_accidental

    @property
    def f_ctm(self) -> float:
        """
        Mean tensile strength (§Table 3.1), MPa.

        f_ctm = 0.30 · f_ck^(2/3) for f_ck ≤ 50 MPa
        f_ctm = 2.12 · ln(1 + f_cm/10) for f_ck > 50 MPa
        """
        if self.f_ck <= 50:
            return 0.30 * (self.f_ck ** (2.0 / 3.0))
        return 2.12 * math.log(1.0 + self.f_cm / 10.0)

    @property
    def f_ctk_005(self) -> float:
        """5% fractile characteristic tensile strength (§Table 3.1): 0.7 · f_ctm (MPa)."""
        return 0.7 * self.f_ctm

    @property
    def f_ctk_095(self) -> float:
        """95% fractile characteristic tensile strength (§Table 3.1): 1.3 · f_ctm (MPa)."""
        return 1.3 * self.f_ctm

    @property
    def f_ctd(self) -> float:
        """Design tensile strength (§3.1.6): f_ctd = α_ct · f_ctk,0.05 / γ_c (MPa)."""
        return self.alpha_ct * self.f_ctk_005 / self.gamma_c

    @property
    def E_cm(self) -> float:
        """
        Secant modulus of elasticity (§Table 3.1), MPa.

        E_cm = 22 · (f_cm / 10)^0.3 GPa, converted to MPa,
        adjusted for aggregate type per §3.1.3(2).
        """
        e_base_gpa = 22.0 * ((self.f_cm / 10.0) ** 0.3)
        factor = _AGGREGATE_ECM_FACTORS[self.aggregate_type]
        return e_base_gpa * 1000.0 * factor

    def get_elastic_modulus(self) -> float:
        """Implements BaseMaterial abstract method: return E_cm (MPa)."""
        return self.E_cm

    @property
    def epsilon_c1(self) -> float:
        """
        Strain at peak stress for parabola-rectangle diagram (§Table 3.1).

        ε_c1 = 0.7 · f_cm^0.31 ≤ 2.8‰

        Returns dimensionless (e.g., 0.0022 for 2.2‰)
        """
        return min(0.7 * (self.f_cm ** 0.31) / 1000.0, 0.0028)

    @property
    def epsilon_cu1(self) -> float:
        """Ultimate strain for parabola-rectangle (§Table 3.1), dimensionless."""
        if self.f_ck <= 50:
            return 0.0035
        return (2.8 + 27.0 * (((98.0 - self.f_cm) / 100.0) ** 4.0)) / 1000.0

    @property
    def epsilon_c2(self) -> float:
        """Strain at reaching f_ck for parabola-rectangle (§Table 3.1), dimensionless."""
        if self.f_ck <= 50:
            return 0.0020
        return (2.0 + 0.085 * ((self.f_ck - 50.0) ** 0.53)) / 1000.0

    @property
    def epsilon_cu2(self) -> float:
        """Ultimate strain for parabola-rectangle (§Table 3.1), dimensionless."""
        if self.f_ck <= 50:
            return 0.0035
        return (2.6 + 35.0 * (((90.0 - self.f_ck) / 100.0) ** 4.0)) / 1000.0

    @property
    def n(self) -> float:
        """Exponent for parabola-rectangle (§Table 3.1), dimensionless."""
        if self.f_ck <= 50:
            return 2.0
        return 1.4 + 23.4 * (((90.0 - self.f_ck) / 100.0) ** 4.0)

    @property
    def epsilon_c3(self) -> float:
        """Strain at reaching f_ck for bilinear (§Table 3.1), dimensionless."""
        if self.f_ck <= 50:
            return 0.00175
        return (1.75 + 0.55 * ((self.f_ck - 50.0) / 40.0)) / 1000.0

    @property
    def epsilon_cu3(self) -> float:
        """Ultimate strain for bilinear (§Table 3.1), dimensionless."""
        if self.f_ck <= 50:
            return 0.0035
        return (2.6 + 35.0 * (((90.0 - self.f_ck) / 100.0) ** 4.0)) / 1000.0

    def __str__(self) -> str:
        return f"{self.grade} (f_ck={self.f_ck} MPa, f_cd={self.f_cd:.1f} MPa)"
