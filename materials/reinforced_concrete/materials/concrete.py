"""
Concrete material properties according to Eurocode 2.

Implements characteristic strengths, design strengths, elastic modulus,
and stress-strain parameters for concrete grades C12/15 to C90/105.
"""

from typing import Literal, Optional, get_args
from pydantic import Field, field_validator, computed_field
from materials.core.base_material import BaseMaterial


# Concrete grades according to EC2 Table 3.1 (single source of truth)
ConcreteGrade = Literal[
    "C12/15", "C16/20", "C20/25", "C25/30", "C30/37", "C35/45", "C40/50",
    "C45/55", "C50/60", "C55/67", "C60/75", "C70/85", "C80/95", "C90/105"
]


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

    aggregate_type: Literal["quartzite", "limestone", "sandstone", "basalt"] = Field(
        default="quartzite",
        description="Aggregate type for elastic modulus adjustment (§3.1.3)",
    )

    density: Optional[float] = Field(
        default=2400.0,
        description="Concrete density in kg/m³",
        ge=2000,
        le=2600,
    )

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, v: str) -> str:
        """Validate concrete grade format."""
        valid_grades = get_args(ConcreteGrade)
        if v not in valid_grades:
            raise ValueError(
                f"Invalid concrete grade: {v}. "
                f"Must be one of {list(valid_grades)}"
            )
        return v

    @computed_field
    @property
    def f_ck(self) -> float:
        """
        Characteristic cylinder compressive strength at 28 days (§3.1.2).

        Returns:
            f_ck in MPa
        """
        return float(self.grade.split("/")[0].replace("C", ""))

    @computed_field
    @property
    def f_ck_cube(self) -> float:
        """
        Characteristic cube compressive strength at 28 days (§3.1.2).

        Returns:
            f_ck,cube in MPa
        """
        return float(self.grade.split("/")[1])

    @computed_field
    @property
    def f_cm(self) -> float:
        """
        Mean cylinder compressive strength (§Table 3.1).

        f_cm = f_ck + 8 MPa

        Returns:
            f_cm in MPa
        """
        return self.f_ck + 8.0

    @computed_field
    @property
    def f_cd(self) -> float:
        """
        Design compressive strength for ULS persistent/transient (§3.1.6).

        f_cd = α_cc · f_ck / γ_c

        Returns:
            f_cd in MPa
        """
        return self.alpha_cc * self.f_ck / self.gamma_c

    @computed_field
    @property
    def f_cd_accidental(self) -> float:
        """
        Design compressive strength for ULS accidental (§3.1.6).

        f_cd,acc = α_cc · f_ck / γ_c,acc

        Typically used for seismic design, impact, fire scenarios.

        Returns:
            f_cd,acc in MPa
        """
        return self.alpha_cc * self.f_ck / self.gamma_c_accidental

    @computed_field
    @property
    def f_ctm(self) -> float:
        """
        Mean tensile strength (§Table 3.1).

        f_ctm = 0.30 · f_ck^(2/3) for f_ck ≤ 50 MPa
        f_ctm = 2.12 · ln(1 + f_cm/10) for f_ck > 50 MPa

        Returns:
            f_ctm in MPa
        """
        if self.f_ck <= 50:
            return 0.30 * (self.f_ck ** (2/3))
        else:
            import math
            return 2.12 * math.log(1 + self.f_cm / 10)

    @computed_field
    @property
    def f_ctk_005(self) -> float:
        """
        5% fractile (lower) characteristic tensile strength (§Table 3.1).

        f_ctk,0.05 = 0.7 · f_ctm

        Returns:
            f_ctk,0.05 in MPa
        """
        return 0.7 * self.f_ctm

    @computed_field
    @property
    def f_ctk_095(self) -> float:
        """
        95% fractile (upper) characteristic tensile strength (§Table 3.1).

        f_ctk,0.95 = 1.3 · f_ctm

        Returns:
            f_ctk,0.95 in MPa
        """
        return 1.3 * self.f_ctm

    @computed_field
    @property
    def f_ctd(self) -> float:
        """
        Design tensile strength (§3.1.6).

        f_ctd = α_ct · f_ctk,0.05 / γ_c

        Returns:
            f_ctd in MPa
        """
        return self.alpha_ct * self.f_ctk_005 / self.gamma_c

    @computed_field
    @property
    def E_cm(self) -> float:
        """
        Secant modulus of elasticity (§Table 3.1).

        E_cm = 22 · (f_cm / 10)^0.3 GPa (converted to MPa)

        Adjusted for aggregate type per §3.1.3(2):
        - Basalt: E_cm × 1.2
        - Quartzite: E_cm × 1.0
        - Limestone: E_cm × 0.9
        - Sandstone: E_cm × 0.7

        Returns:
            E_cm in MPa
        """
        # Base modulus in GPa
        E_base_GPa = 22.0 * ((self.f_cm / 10) ** 0.3)

        # Aggregate adjustment factors
        aggregate_factors = {
            "basalt": 1.2,
            "quartzite": 1.0,
            "limestone": 0.9,
            "sandstone": 0.7,
        }

        factor = aggregate_factors[self.aggregate_type]

        # Convert to MPa
        return E_base_GPa * 1000.0 * factor

    def get_elastic_modulus(self) -> float:
        """
        Return elastic modulus (implements BaseMaterial abstract method).

        Returns:
            E_cm in MPa
        """
        return self.E_cm

    @computed_field
    @property
    def epsilon_c1(self) -> float:
        """
        Strain at peak stress for parabola-rectangle diagram (§Table 3.1).

        ε_c1 = 0.7 · f_cm^0.31 ≤ 2.8‰ for f_ck ≤ 50 MPa
        ε_c1 = 2.8 + 27 · [(98 - f_cm) / 100]^4 for f_ck > 50 MPa

        Returns:
            ε_c1 (dimensionless, e.g., 0.0022 for 2.2‰)
        """
        # TODO: CHECK THIS FORMULA FOR F_CK > 50 (NEW EC2?)
        if self.f_ck <= 50:
            return min(0.7 * (self.f_cm ** 0.31) / 1000, 0.0028)
        else:
            return (2.8 + 27.0 * (((98 - self.f_cm) / 100) ** 4)) / 1000

    @computed_field
    @property
    def epsilon_cu1(self) -> float:
        """
        Ultimate strain for parabola-rectangle diagram (§Table 3.1).

        ε_cu1 = 3.5‰ for f_ck ≤ 50 MPa
        ε_cu1 = 2.8 + 27 · [(98 - f_cm) / 100]^4 for f_ck > 50 MPa

        Returns:
            ε_cu1 (dimensionless)
        """
        if self.f_ck <= 50:
            return 0.0035
        else:
            return (2.8 + 27.0 * (((98 - self.f_cm) / 100) ** 4)) / 1000

    @computed_field
    @property
    def epsilon_c2(self) -> float:
        """
        Strain at reaching f_ck for parabola-rectangle (§Table 3.1).

        ε_c2 = 2.0‰ for f_ck ≤ 50 MPa
        ε_c2 = 2.0 + 0.085 · (f_ck - 50)^0.53 for f_ck > 50 MPa

        Returns:
            ε_c2 (dimensionless)
        """
        if self.f_ck <= 50:
            return 0.0020
        else:
            return (2.0 + 0.085 * ((self.f_ck - 50) ** 0.53)) / 1000

    @computed_field
    @property
    def epsilon_cu2(self) -> float:
        """
        Ultimate strain for parabola-rectangle (§Table 3.1).

        ε_cu2 = 3.5‰ for f_ck ≤ 50 MPa
        ε_cu2 = 2.6 + 35 · [(90 - f_ck) / 100]^4 for f_ck > 50 MPa

        Returns:
            ε_cu2 (dimensionless)
        """
        if self.f_ck <= 50:
            return 0.0035
        else:
            return (2.6 + 35.0 * (((90 - self.f_ck) / 100) ** 4)) / 1000

    @computed_field
    @property
    def n(self) -> float:
        """
        Exponent for parabola-rectangle diagram (§Table 3.1).

        n = 2.0 for f_ck ≤ 50 MPa
        n = 1.4 + 23.4 · [(90 - f_ck) / 100]^4 for f_ck > 50 MPa

        Returns:
            n (dimensionless)
        """
        if self.f_ck <= 50:
            return 2.0
        else:
            return 1.4 + 23.4 * (((90 - self.f_ck) / 100) ** 4)

    @computed_field
    @property
    def epsilon_c3(self) -> float:
        """
        Strain at reaching f_ck for bilinear diagram (§Table 3.1).

        ε_c3 = 1.75‰ for f_ck ≤ 50 MPa
        ε_c3 = 1.75 + 0.55 · [(f_ck - 50) / 40] for f_ck > 50 MPa

        Returns:
            ε_c3 (dimensionless)
        """
        if self.f_ck <= 50:
            return 0.00175
        else:
            return (1.75 + 0.55 * ((self.f_ck - 50) / 40)) / 1000

    @computed_field
    @property
    def epsilon_cu3(self) -> float:
        """
        Ultimate strain for bilinear diagram (§Table 3.1).

        ε_cu3 = 3.5‰ for f_ck ≤ 50 MPa
        ε_cu3 = 2.6 + 35 · [(90 - f_ck) / 100]^4 for f_ck > 50 MPa

        Returns:
            ε_cu3 (dimensionless)
        """
        if self.f_ck <= 50:
            return 0.0035
        else:
            return (2.6 + 35.0 * (((90 - self.f_ck) / 100) ** 4)) / 1000

    def __str__(self) -> str:
        """User-friendly representation."""
        return f"{self.grade} (f_ck={self.f_ck} MPa, f_cd={self.f_cd:.1f} MPa)"
