"""Adiabatic temperature rise model for early-age concrete.

Based on CIRIA C766 methods for predicting heat generation and temperature
development in early-age concrete under adiabatic conditions.
"""

from typing import Dict, List, Callable, Optional
from math import exp, sqrt
from pydantic import BaseModel, Field, field_validator, computed_field
import numpy as np

from .concrete_mix import ConcreteMix


class AdiabaticTemperature(BaseModel):
    """
    Adiabatic temperature rise prediction for early-age concrete.

    Models heat generation and temperature development in concrete cured under
    adiabatic conditions (no heat loss). Based on CIRIA C766 methodology.

    Attributes:
        mix: Concrete mix properties
        time_elapsed: Time since placing in hours
        rastrup_coefficient: Coefficient for temperature-time adjustment (default 12)
        test_mix_temp: Reference temperature for test mix calibration (°C, default 20)
        is_adjusted_for_placing_temp: Whether to adjust for placing temperature

    Computed properties:
        - Coefficients b, c, d for heat generation model
        - Activation time t2
        - GGBS calibration factor
        - Ultimate heat generation (q_41, q_ult)
        - Ultimate temperature rise (t_ult)

    References:
        - CIRIA C766: Early-age thermal crack control in concrete
        - Sections A2.2.1 to A2.2.3

    Examples:
        >>> from materials.reinforced_concrete.thermal import Binder, ConcreteMix, AdiabaticTemperature
        >>>
        >>> # Create mix
        >>> binder = Binder()
        >>> mix = ConcreteMix(cement_content=350, concrete_placing_temp=20, binder=binder)
        >>>
        >>> # Create adiabatic model at 24 hours
        >>> adiabatic = AdiabaticTemperature(mix=mix, time_elapsed=24.0)
        >>>
        >>> # Get temperature rise
        >>> temp = adiabatic.find_modelled_temperature_over_time()
        >>> print(f"Temperature at 24h: {temp:.1f}°C")
    """

    mix: ConcreteMix = Field(
        ...,
        description="Concrete mix properties",
    )

    time_elapsed: float = Field(
        ...,
        ge=0.0,
        description="Time since concrete placing in hours",
    )

    rastrup_coefficient: float = Field(
        default=12.0,
        gt=0.0,
        description="Rastrup coefficient for temperature-time adjustment",
    )

    test_mix_temp: float = Field(
        default=20.0,
        gt=0.0,
        description="Reference test mix temperature in °C",
    )

    is_adjusted_for_placing_temp: bool = Field(
        default=True,
        description="Whether to adjust calculations for placing temperature",
    )

    # Constants
    VALID_MIN_TIME: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def coefficient_b(self) -> float:
        """Coefficient b for heat generation model."""
        coefficient_b = 0.011724

        if self.is_adjusted_for_placing_temp:
            adjuster = (0.1387 * exp(0.0999 * self.mix.concrete_placing_temp) /
                       (0.1387 * exp(0.0999 * self.test_mix_temp)))
        else:
            adjuster = 1.0

        return coefficient_b * adjuster

    @computed_field
    @property
    def coefficient_c(self) -> float:
        """Coefficient c for heat generation model."""
        binder = self.mix.binder
        cem_1_factor = 1.6

        # Binder modification factor
        if binder.substitute_type == "pfa":
            binder_mod_factor = -0.001 * binder.substitute_percent
        elif binder.substitute_type == "ggbs":
            percent = binder.substitute_percent
            binder_mod_factor = -0.0072 * percent - 0.00003 * percent**2
        else:
            binder_mod_factor = 0.0

        coefficient_c = cem_1_factor + binder_mod_factor

        # Temperature adjustment (additive, not multiplicative)
        if self.is_adjusted_for_placing_temp:
            adjuster = (self.mix.concrete_placing_temp - self.test_mix_temp) / 2000
            coefficient_c += adjuster

        return coefficient_c

    @computed_field
    @property
    def coefficient_d(self) -> float:
        """Coefficient d for heat generation model."""
        binder = self.mix.binder
        cem_1_factor = 6.2

        # Binder modification factor
        if binder.substitute_type == "pfa":
            binder_mod_factor = 0.2131 * binder.substitute_percent
        elif binder.substitute_type == "ggbs":
            percent = binder.substitute_percent
            binder_mod_factor = 0.0848 * percent - 0.0004 * percent**2
        else:
            binder_mod_factor = 0.0

        coefficient_d = cem_1_factor + binder_mod_factor

        # Temperature adjustment (multiplicative)
        if self.is_adjusted_for_placing_temp:
            placing_temp = self.mix.concrete_placing_temp
            test_temp = self.test_mix_temp
            adjuster = ((0.0022 * placing_temp**2 - 0.1503 * placing_temp + 3.1483) /
                       (0.0022 * test_temp**2 - 0.1503 * test_temp + 3.1483))
        else:
            adjuster = 1.0

        return coefficient_d * adjuster

    @computed_field
    @property
    def activation_time_t2(self) -> float:
        """Activation time t2 in hours (start of second heat generation phase)."""
        binder = self.mix.binder
        cem_1_factor = 3.5

        # Binder modification factor
        if binder.substitute_type == "pfa":
            binder_mod_factor = 0.0236
        elif binder.substitute_type == "ggbs":
            binder_mod_factor = 0.0125
        else:
            binder_mod_factor = 0.0

        activation_time = cem_1_factor + binder_mod_factor * binder.substitute_percent

        # Adjust for placing temperature using Rastrup function
        if self.is_adjusted_for_placing_temp:
            activation_time = self.find_elapsed_time_adjusted_by_rastrup_function(activation_time)

        return activation_time

    @computed_field
    @property
    def ggbs_calibration_factor(self) -> float:
        """Calibration factor for GGBS mixes (1.0 for other mixes)."""
        binder = self.mix.binder

        if binder.substitute_type == "ggbs":
            return 1.0 - (0.15 / 75) * binder.substitute_percent
        else:
            return 1.0

    @computed_field
    @property
    def ultimate_heat_generation_q_41(self) -> float:
        """
        Ultimate heat generation q_41 in kJ/kg.

        Reference: CIRIA C766 A2.2.2
        """
        binder = self.mix.binder
        cem_1_factor1 = 352.0
        cem_1_factor2 = 338.4

        if binder.substitute_type == "pfa":
            percent = binder.substitute_percent
            q_41 = cem_1_factor2 - 2.99 * percent
            adjuster = (1.0 + (cem_1_factor1 - cem_1_factor2) / cem_1_factor2 -
                       0.027 * percent / 100)

        elif binder.substitute_type == "ggbs":
            percent = binder.substitute_percent
            q_41 = cem_1_factor2 - 60 * (percent / (100 - percent))**0.6
            adjuster = (1.0 + (cem_1_factor1 - cem_1_factor2) / cem_1_factor2 *
                       (100 - percent) / 100)

        else:  # Pure cement
            q_41 = cem_1_factor1
            adjuster = 1.0

        return q_41 * adjuster

    @computed_field
    @property
    def ultimate_heat_generation_q_ult(self) -> float:
        """
        Ultimate heat generation q_ult in kJ/kg.

        Reference: CIRIA C766 A2.2.3
        """
        binder = self.mix.binder
        q_41 = self.ultimate_heat_generation_q_41
        cem_1_factor = 0.925

        if binder.substitute_type == "pfa":
            percent = binder.substitute_percent
            binder_mod_factor = (cem_1_factor - 0.0034 * percent +
                                0.00002 * percent**2)

        elif binder.substitute_type == "ggbs":
            percent = binder.substitute_percent
            binder_mod_factor = (cem_1_factor - 0.0047 * percent +
                                0.00003 * percent**2)

        else:  # Pure cement
            binder_mod_factor = cem_1_factor

        return q_41 / binder_mod_factor

    @computed_field
    @property
    def ultimate_temperature_t_ult(self) -> float:
        """Ultimate temperature rise in °C (adiabatic conditions)."""
        q_ult = self.ultimate_heat_generation_q_ult
        t_ult = ((q_ult * self.mix.cement_content) /
                (self.mix.specific_heat * self.mix.concrete_mass_density))

        if self.is_adjusted_for_placing_temp:
            adjuster = 0.2 * (self.mix.concrete_placing_temp - self.test_mix_temp)
            t_ult -= adjuster

        return t_ult

    def find_elapsed_time_adjusted_by_rastrup_function(self, time_elapsed: float) -> float:
        """
        Adjust elapsed time for temperature differential using Rastrup function.

        For a rastrup_coefficient * 1 degree increase in placing temperature,
        the time to achieve the same heat generation is halved.

        Args:
            time_elapsed: Time in hours to adjust

        Returns:
            Adjusted time in hours
        """
        rastrup_time_adjuster = 2 ** ((self.test_mix_temp - self.mix.concrete_placing_temp) /
                                     self.rastrup_coefficient)
        return time_elapsed * rastrup_time_adjuster

    def find_total_heat_generated_q_over_time(self) -> float:
        """
        Calculate total heat generated at current time_elapsed.

        Units: kJ/kg

        Reference: CIRIA C766 A2.2.1
        """
        time_elapsed = self.time_elapsed
        activation_time_t2 = self.activation_time_t2

        # Second phase adjuster (activation time delay)
        if time_elapsed <= activation_time_t2:
            q_2_time_delay_adjuster = 0.0
        else:
            q_2_time_delay_adjuster = ((time_elapsed - activation_time_t2) /
                                      (time_elapsed - activation_time_t2 + self.coefficient_d))

        # First phase adjuster
        q_1_adjuster = 1.0 - exp(-self.coefficient_b * time_elapsed**self.coefficient_c)

        # Total heat generated
        total_heat = (self.ggbs_calibration_factor *
                     self.mix.mix_multiplier *
                     0.5 *
                     self.ultimate_heat_generation_q_ult *
                     (q_1_adjuster + q_2_time_delay_adjuster))

        return total_heat

    def find_modelled_temperature_over_time(self, is_temp_rise_only: bool = False) -> float:
        """
        Calculate modelled temperature at current time_elapsed.

        Args:
            is_temp_rise_only: If True, returns only temperature rise (excludes placing temp)

        Returns:
            Temperature in °C
        """
        time_elapsed = self.time_elapsed
        activation_time_t2 = self.activation_time_t2

        # Second phase adjuster
        if time_elapsed <= activation_time_t2:
            t_2_time_delay_adjuster = 0.0
        else:
            t_2_time_delay_adjuster = ((time_elapsed - activation_time_t2) /
                                      (time_elapsed - activation_time_t2 + self.coefficient_d))

        # First phase adjuster
        t_1_adjuster = 1.0 - exp(-self.coefficient_b * time_elapsed**self.coefficient_c)

        # Temperature rise
        temperature_multiplier = (self.ggbs_calibration_factor *
                                 self.mix.mix_multiplier *
                                 0.5 *
                                 self.ultimate_temperature_t_ult)
        adiabatic_temp_rise = temperature_multiplier * (t_1_adjuster + t_2_time_delay_adjuster)

        # Add placing temperature if requested
        if is_temp_rise_only:
            return adiabatic_temp_rise
        else:
            return adiabatic_temp_rise + self.mix.concrete_placing_temp

    def make_time_temps_dict(self,
                            number_of_time_intervals: int = 100,
                            is_temp_rise_only: bool = False
                            ) -> Dict[str, List[float]]:
        """
        Generate time-temperature data over the analysis period.

        Args:
            number_of_time_intervals: Number of intervals to subdivide time
            is_temp_rise_only: If True, returns only temperature rise

        Returns:
            Dictionary with 'time' and 'adiabatic_temps' lists
        """
        original_time = self.time_elapsed
        time_interval = original_time / number_of_time_intervals
        end_time = original_time + time_interval

        time_temps_dict = {"time": [], "adiabatic_temps": []}

        for time in np.arange(0, end_time, time_interval):
            # Create new instance with updated time (Pydantic immutability workaround)
            temp_instance = self.model_copy(update={"time_elapsed": float(time)})
            temp = temp_instance.find_modelled_temperature_over_time(is_temp_rise_only)

            time_temps_dict["time"].append(float(time))
            time_temps_dict["adiabatic_temps"].append(temp)

        return time_temps_dict

    def make_time_heat_dict(self, number_of_time_intervals: int = 100) -> Dict[str, List[float]]:
        """
        Generate time-heat generation data over the analysis period.

        Args:
            number_of_time_intervals: Number of intervals to subdivide time

        Returns:
            Dictionary with 'time' and 'heat' lists
        """
        original_time = self.time_elapsed
        time_interval = original_time / number_of_time_intervals
        end_time = original_time + time_interval

        time_heat_dict = {"time": [], "heat": []}

        for time in np.arange(0, end_time, time_interval):
            # Create new instance with updated time
            temp_instance = self.model_copy(update={"time_elapsed": float(time)})
            heat = temp_instance.find_total_heat_generated_q_over_time()

            time_heat_dict["time"].append(float(time))
            time_heat_dict["heat"].append(heat)

        return time_heat_dict

    @staticmethod
    def find_sadgrove_maturity_coefficient(avg_concrete_temp_during_time_interval: float) -> float:
        """
        Weaver and Sadgrove maturity coefficient.

        Args:
            avg_concrete_temp_during_time_interval: Average concrete temperature in °C

        Returns:
            Maturity coefficient (dimensionless)
        """
        concrete_temp = avg_concrete_temp_during_time_interval
        return ((concrete_temp + 16) / 36) ** 2

    def find_arrhenius_maturity_coefficient(self,
                                           avg_concrete_temp_during_time_interval: float,
                                           activation_energy: float) -> float:
        """
        Freiesleben Hansen and Pedersen (Arrhenius) maturity coefficient.

        Args:
            avg_concrete_temp_during_time_interval: Average concrete temperature in °C
            activation_energy: Activation energy in J/mol

        Returns:
            Maturity coefficient (dimensionless)
        """
        mod_absolute_zero = 273.15
        universal_gas_constant = 8.31446261815324  # J/mol/K

        concrete_temp_kelvin = avg_concrete_temp_during_time_interval + mod_absolute_zero
        test_mix_temp_kelvin = self.test_mix_temp + mod_absolute_zero

        return exp((-activation_energy / universal_gas_constant) *
                  ((1 / concrete_temp_kelvin) - (1 / test_mix_temp_kelvin)))

    def find_saul_maturity_coefficient(self,
                                      avg_concrete_temp_during_time_interval: float
                                      ) -> float:
        """
        Nurse-Saul maturity coefficient (linear relationship).

        Args:
            avg_concrete_temp_during_time_interval: Average concrete temperature in °C

        Returns:
            Maturity coefficient (dimensionless)
        """
        datum_temperature = -11.0  # Temperature at which no strength development occurs
        return ((self.test_mix_temp - datum_temperature) /
               (avg_concrete_temp_during_time_interval - datum_temperature))

    def find_sadgrove_maturity(self, number_of_time_intervals: int = 50) -> float:
        """Calculate Sadgrove maturity over time_elapsed period."""
        return self._calculate_maturity(
            self.find_sadgrove_maturity_coefficient,
            number_of_time_intervals
        )

    def find_arrhenius_maturity(self,
                               activation_energy: float,
                               number_of_time_intervals: int = 50
                               ) -> float:
        """Calculate Arrhenius maturity over time_elapsed period."""
        return self._calculate_maturity(
            lambda avg_temp: self.find_arrhenius_maturity_coefficient(avg_temp, activation_energy),
            number_of_time_intervals
        )

    def find_saul_maturity(self, number_of_time_intervals: int = 50) -> float:
        """Calculate Saul maturity over time_elapsed period."""
        return self._calculate_maturity(
            self.find_saul_maturity_coefficient,
            number_of_time_intervals
        )

    def _calculate_maturity(self,
                           maturity_coefficient_function: Callable[[float], float],
                           number_of_time_intervals: int
                           ) -> float:
        """
        Generic maturity calculation.

        Args:
            maturity_coefficient_function: Function to calculate maturity coefficient
            number_of_time_intervals: Number of time intervals

        Returns:
            Total maturity in hours
        """
        time_elapsed = self.time_elapsed
        time_interval = time_elapsed / number_of_time_intervals

        time_temps_dict = self.make_time_temps_dict(number_of_time_intervals)
        temps = time_temps_dict["adiabatic_temps"]

        # Calculate average temperatures for each interval
        avg_temps = [(temps[i] + temps[i - 1]) / 2
                    for i in range(1, number_of_time_intervals + 1)]

        # Sum maturity increments
        total_maturity = 0.0
        for avg_temp in avg_temps:
            maturity_coefficient = maturity_coefficient_function(avg_temp)
            maturity_increment = maturity_coefficient * time_interval
            total_maturity += maturity_increment

        return total_maturity

    @staticmethod
    def find_strength_maturity_relationship(ultimate_compressive_strength: float,
                                           characteristic_time_constant: float,
                                           shape_parameter: float,
                                           test_age: float) -> float:
        """
        Three Parameter Equation (Freiesleben Hansen and Pedersen).

        Predicts compressive strength from maturity.

        Args:
            ultimate_compressive_strength: Ultimate f_c in MPa
            characteristic_time_constant: Time constant in days
            shape_parameter: Shape parameter (dimensionless)
            test_age: Test age in days

        Returns:
            Compressive strength in MPa
        """
        return (ultimate_compressive_strength *
               exp(-characteristic_time_constant / test_age) ** shape_parameter)

    def __str__(self) -> str:
        """User-friendly representation."""
        return (f"AdiabaticTemperature(time={self.time_elapsed:.1f}h, "
               f"temp={self.find_modelled_temperature_over_time():.1f}°C)")
