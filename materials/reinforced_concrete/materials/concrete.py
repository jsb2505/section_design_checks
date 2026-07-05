"""
Concrete material properties according to Eurocode 2.

Implements characteristic strengths, design strengths, elastic modulus,
and stress-strain parameters for concrete grades C12/15 to C90/105.
"""

from enum import StrEnum
from math import log
from typing import cast
from pydantic import Field, ConfigDict

from materials.core.base_material import BaseMaterial
from materials.core.units import StressUnit, LengthUnit, to_mpa, from_mm
from materials.reinforced_concrete.ndp import get_ndp


# Concrete grades according to EC2 Table 3.1 (single source of truth)
# Table data: (f_ck, f_ck_cube, f_cm) per EC2 Table 3.1, units: MPa
_GRADE_TABLE: dict[str, tuple[float, float, float]] = {
    "C12/15":  (12.0, 15.0, 20.0),
    "C16/20":  (16.0, 20.0, 24.0),
    "C20/25":  (20.0, 25.0, 28.0),
    "C25/30":  (25.0, 30.0, 33.0),
    "C30/37":  (30.0, 37.0, 38.0),
    "C35/45":  (35.0, 45.0, 43.0),
    "C40/50":  (40.0, 50.0, 48.0),
    "C45/55":  (45.0, 55.0, 53.0),
    "C50/60":  (50.0, 60.0, 58.0),
    "C55/67":  (55.0, 67.0, 63.0),
    "C60/75":  (60.0, 75.0, 68.0),
    "C70/85":  (70.0, 85.0, 78.0),
    "C80/95":  (80.0, 95.0, 88.0),
    "C90/105": (90.0, 105.0, 98.0),
}


class ConcreteGrade(StrEnum):
    '''
    Concrete grades supported as per EC2.

    Attributes:
        C12_15: C12/15
        C16_20: C16/20
        C20_25: C20/25
        C25_30: C25/30
        C30_37: C30/37
        C35_45: C35/45
        C40_50: C40/50
        C45_55: C45/55
        C50_60: C50/60
        C55_67: C55/67
        C60_75: C60/75
        C70_85: C70/85
        C80_95: C80/95
        C90_105: C90/105
    '''
    C12_15 = "C12/15"
    C16_20 = "C16/20"
    C20_25 = "C20/25"
    C25_30 = "C25/30"
    C30_37 = "C30/37"
    C35_45 = "C35/45"
    C40_50 = "C40/50"
    C45_55 = "C45/55"
    C50_60 = "C50/60"
    C55_67 = "C55/67"
    C60_75 = "C60/75"
    C70_85 = "C70/85"
    C80_95 = "C80/95"
    C90_105 = "C90/105"

    @property
    def f_ck(self) -> float:
        return _GRADE_TABLE[self][0]

    @property
    def f_ck_cube(self) -> float:
        return _GRADE_TABLE[self][1]

    @property
    def f_cm(self) -> float:
        return _GRADE_TABLE[self][2]


_AGGREGATE_FACTORS: dict[str, float] = {
    "basalt": 1.2,
    "quartzite": 1.0,
    "limestone": 0.9,
    "sandstone": 0.7,
}


class AggregateType(StrEnum):
    BASALT = "basalt"
    QUARTZITE = "quartzite"
    LIMESTONE = "limestone"
    SANDSTONE = "sandstone"

    @property
    def e_cm_factor(self) -> float:
        """Correction factor for E_cm based on aggregate type (§3.1.3(2))."""
        return _AGGREGATE_FACTORS[self]


def find_mean_flexural_tensile_strength(f_ctm: float, section_height: float) -> float:
    """Mean flexural tensile strength of concrete (§3.1.8(1), Eq. 3.23).

    f_ctm,fl = f_ctm · max(1.6 − h, 1.0)  where h is in metres.

    Args:
        f_ctm: Mean axial tensile strength (MPa)
        section_height: Geometrical height of section (mm)

    Returns:
        f_ctm,fl in MPa
    """
    h = from_mm(section_height, LengthUnit.M)
    return f_ctm * max(1.6 - h, 1.0)


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

    model_config = ConfigDict()

    name: str = Field(default="Concrete", description="Material name")
    grade: ConcreteGrade = Field(..., description="Concrete grade per EC2 Table 3.1")

    gamma_c: float = Field(
        default_factory=lambda: cast(float, get_ndp("gamma_c")),
        description="Partial factor for concrete for ULS persistent/transient (§2.4.2.4, NDP)",
        gt=0,
    )

    gamma_c_accidental: float = Field(
        default_factory=lambda: cast(float, get_ndp("gamma_c_accidental")),
        description="Partial factor for concrete for ULS accidental (§2.4.2.4, NDP)",
        gt=0,
    )

    alpha_cc: float = Field(
        default_factory=lambda: cast(float, get_ndp("alpha_cc")),
        description="Coefficient for long-term effects on strength (§3.1.6(1)P, NDP)",
        gt=0,
        le=1.0,
    )

    alpha_ct: float = Field(
        default_factory=lambda: cast(float, get_ndp("alpha_ct")),
        description="Coefficient for long-term effects on tensile strength (NDP)",
        gt=0,
        le=1.0,
    )

    aggregate_type: AggregateType = Field(
        default=AggregateType.QUARTZITE,
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
        return self.grade.f_ck

    @property
    def f_ck_cube(self) -> float:
        """Characteristic cube compressive strength at 28 days (§3.1.2), MPa."""
        return self.grade.f_ck_cube

    @property
    def f_cm(self) -> float:
        """Mean cylinder compressive strength (§Table 3.1): f_cm = f_ck + 8 (MPa)."""
        return self.grade.f_cm

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
        return 2.12 * log(1.0 + self.f_cm / 10.0)

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
    def f_ctd_accidental(self) -> float:
        """Accidental design tensile strength: f_ctd,acc = α_ct · f_ctk,0.05 / γ_c,acc (MPa)."""
        return self.alpha_ct * self.f_ctk_005 / self.gamma_c_accidental

    @property
    def E_cm(self) -> float:
        """
        Secant modulus of elasticity (§Table 3.1), MPa.

        E_cm = 22 · (f_cm / 10)^0.3 GPa, converted to MPa,
        adjusted for aggregate type per §3.1.3(2).
        """
        e_base_gpa = 22.0 * ((self.f_cm / 10.0) ** 0.3)
        factor = self.aggregate_type.e_cm_factor
        return to_mpa(e_base_gpa * factor, StressUnit.GPA)

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
    
    def find_mean_flexural_tensile_strength(self, section_height: float) -> float:
        '''Calculates the mean flexural tensile strength of concrete (§3.1.8(1)).

        Args:
            section_height: Geometrical height of section (mm)

        Returns:
            f_ctm,fl in MPa
        '''
        return find_mean_flexural_tensile_strength(self.f_ctm, section_height)

    def __str__(self) -> str:
        return f"{self.grade} (f_ck={self.f_ck} MPa, f_cd={self.f_cd:.1f} MPa)"
