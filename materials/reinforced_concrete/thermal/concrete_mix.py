"""Concrete mix model for thermal analysis."""

from pydantic import BaseModel, Field, field_validator

from .binder import Binder


class ConcreteMix(BaseModel):
    """Defines concrete mix composition and thermal properties.

    Used for early-age thermal analysis including heat generation and temperature
    prediction. Mix properties affect adiabatic temperature rise and in-situ behavior.

    Attributes:
        cement_content: Total cement content including substitutes (kg/m³)
        concrete_placing_temp: Temperature of fresh concrete at placement (°C)
        binder: Binder composition (cement with optional ggbs/pfa substitutes)
        concrete_mass_density: Mass density of hardened concrete (kg/m³)
        concrete_thermal_conductivity: Thermal conductivity (W/m°C)
        specific_heat: Specific heat capacity (kJ/kg°C)
        mix_multiplier: Linear scaling factor for heat generation (dimensionless)

    Examples:
        >>> from materials.reinforced_concrete.thermal import Binder, ConcreteMix
        >>>
        >>> # Pure cement mix
        >>> mix1 = ConcreteMix(
        ...     cement_content=350,
        ...     concrete_placing_temp=20,
        ...     binder=Binder()
        ... )
        >>>
        >>> # GGBS blend mix
        >>> mix2 = ConcreteMix(
        ...     cement_content=400,
        ...     concrete_placing_temp=15,
        ...     binder=Binder(substitute_type="ggbs", substitute_percent=30)
        ... )
    """

    cement_content: float = Field(
        ...,
        gt=50.0,
        le=1000.0,
        description="Total cement content including substitutes (kg/m³)",
    )

    concrete_placing_temp: float = Field(
        ...,
        ge=5.0,
        le=50.0,
        description="Temperature of fresh concrete at placement (°C)",
    )

    binder: Binder = Field(
        ...,
        description="Binder composition with optional substitutes",
    )

    concrete_mass_density: float = Field(
        default=2400.0,
        gt=0.0,
        description="Mass density of hardened concrete (kg/m³)",
    )

    concrete_thermal_conductivity: float = Field(
        default=1.8,
        gt=0.0,
        description="Thermal conductivity (W/m°C)",
    )

    specific_heat: float = Field(
        default=1.0,
        gt=0.0,
        description="Specific heat capacity (kJ/kg°C)",
    )

    mix_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        description="Linear scaling factor for heat generation",
    )

    @classmethod
    def is_valid_cement_content(cls, cement_content: float) -> bool:
        """Check if cement content is within valid range (50, 1000] kg/m³."""
        return 50.0 < cement_content <= 1000.0

    @classmethod
    def is_valid_concrete_placing_temp(cls, concrete_placing_temp: float) -> bool:
        """Check if placing temperature is within valid range [5, 50] °C."""
        return 5.0 <= concrete_placing_temp <= 50.0

    def __repr__(self) -> str:
        return (
            f"ConcreteMix(cement={self.cement_content:.0f} kg/m³, "
            f"placing_temp={self.concrete_placing_temp:.0f}°C, "
            f"{self.binder})"
        )
