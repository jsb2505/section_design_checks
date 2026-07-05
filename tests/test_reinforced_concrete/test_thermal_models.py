"""
Tests for thermal analysis models (binder, mix, adiabatic temperature).
"""

from __future__ import annotations

from math import exp

import pytest
from pydantic import ValidationError

from materials.reinforced_concrete.thermal import (
    AdiabaticTemperature,
    Binder,
    BinderSubstituteType,
    ConcreteMix,
)


def _make_mix(
    *,
    cement_content: float = 350.0,
    placing_temp: float = 20.0,
    substitute_type: BinderSubstituteType | None = None,
    substitute_percent: float = 0.0,
) -> ConcreteMix:
    return ConcreteMix(
        cement_content=cement_content,
        concrete_placing_temp=placing_temp,
        binder=Binder(
            substitute_type=substitute_type,
            substitute_percent=substitute_percent,
        ),
    )


class TestBinder:
    def test_default_is_pure_cement(self):
        binder = Binder()
        assert binder.substitute_type is None
        assert binder.substitute_percent == pytest.approx(0.0, abs=1e-12)
        assert binder.cement_percent == pytest.approx(100.0, abs=1e-12)
        assert binder.is_pure_cement is True
        assert repr(binder) == "Binder(100% cement)"

    def test_percent_is_normalized_to_zero_when_no_substitute_type(self):
        binder = Binder(substitute_percent=35.0)
        assert binder.substitute_type is None
        assert binder.substitute_percent == pytest.approx(0.0, abs=1e-12)

    def test_type_requires_positive_percent(self):
        with pytest.raises(ValueError, match="must be > 0"):
            Binder(substitute_type=BinderSubstituteType.GGBS, substitute_percent=0.0)

        # Negative values are rejected by field validation before model validator.
        with pytest.raises(ValidationError):
            Binder(substitute_type=BinderSubstituteType.GGBS, substitute_percent=-1.0)

    @pytest.mark.parametrize(
        ("substitute_type", "percent", "limit"),
        [
            (BinderSubstituteType.GGBS, 91.0, 90),
            (BinderSubstituteType.PFA, 71.0, 70),
        ],
    )
    def test_type_specific_upper_limits(self, substitute_type, percent, limit):
        with pytest.raises(ValueError, match=f"<= {limit}"):
            Binder(substitute_type=substitute_type, substitute_percent=percent)

    def test_repr_for_substituted_binder(self):
        binder = Binder(
            substitute_type=BinderSubstituteType.GGBS,
            substitute_percent=30.0,
        )
        text = repr(binder)
        assert "70.0% cement" in text
        assert "30.0%" in text
        assert "ggbs" in text


class TestConcreteMix:
    def test_repr_contains_core_mix_info(self):
        mix = _make_mix(placing_temp=18.0)
        text = repr(mix)
        assert "cement=350" in text
        assert "placing_temp=18" in text
        assert "substitute_type=None" in text

    def test_field_validation_and_assignment_validation(self):
        with pytest.raises(ValidationError):
            ConcreteMix(
                cement_content=40.0,
                concrete_placing_temp=20.0,
                binder=Binder(),
            )

        mix = _make_mix()
        with pytest.raises(ValidationError):
            mix.concrete_placing_temp = 60.0


class TestAdiabaticTemperature:
    def test_base_coefficients_without_temperature_adjustment(self):
        model = AdiabaticTemperature(
            mix=_make_mix(),
            time_elapsed=24.0,
            is_adjusted_for_placing_temp=False,
        )
        assert model.coefficient_b == pytest.approx(0.011724, rel=1e-12)
        assert model.coefficient_c == pytest.approx(1.6, rel=1e-12)
        assert model.coefficient_d == pytest.approx(6.2, rel=1e-12)
        assert model.activation_time_t2 == pytest.approx(3.5, rel=1e-12)

    def test_coefficients_with_temperature_adjustment(self):
        model = AdiabaticTemperature(
            mix=_make_mix(placing_temp=30.0),
            time_elapsed=24.0,
            test_mix_temp=20.0,
            rastrup_coefficient=12.0,
            is_adjusted_for_placing_temp=True,
        )

        expected_b = 0.011724 * exp(0.0999 * (30.0 - 20.0))
        expected_c = 1.6 + (30.0 - 20.0) / 2000.0
        expected_d_adjuster = (
            (0.0022 * 30.0**2 - 0.1503 * 30.0 + 3.1483)
            / (0.0022 * 20.0**2 - 0.1503 * 20.0 + 3.1483)
        )
        expected_d = 6.2 * expected_d_adjuster
        expected_t2 = 3.5 * (2 ** ((20.0 - 30.0) / 12.0))

        assert model.coefficient_b == pytest.approx(expected_b, rel=1e-12)
        assert model.coefficient_c == pytest.approx(expected_c, rel=1e-12)
        assert model.coefficient_d == pytest.approx(expected_d, rel=1e-12)
        assert model.activation_time_t2 == pytest.approx(expected_t2, rel=1e-12)

    def test_ggbs_specific_factors_and_ultimate_heat_values(self):
        pure = AdiabaticTemperature(mix=_make_mix(), time_elapsed=24.0)
        ggbs = AdiabaticTemperature(
            mix=_make_mix(substitute_type=BinderSubstituteType.GGBS, substitute_percent=50.0),
            time_elapsed=24.0,
        )
        pfa = AdiabaticTemperature(
            mix=_make_mix(substitute_type=BinderSubstituteType.PFA, substitute_percent=30.0),
            time_elapsed=24.0,
        )

        assert pure.ggbs_calibration_factor == pytest.approx(1.0, abs=1e-12)
        assert ggbs.ggbs_calibration_factor < 1.0
        assert pfa.ggbs_calibration_factor == pytest.approx(1.0, abs=1e-12)

        assert pure.ultimate_heat_generation_q_41 == pytest.approx(352.0, rel=1e-12)
        assert pure.ultimate_heat_generation_q_ult == pytest.approx(352.0 / 0.925, rel=1e-12)
        assert ggbs.ultimate_heat_generation_q_ult > 0.0
        assert pfa.ultimate_heat_generation_q_ult > 0.0

    def test_pfa_and_ggbs_coefficient_branches_without_temp_adjustment(self):
        pfa = AdiabaticTemperature(
            mix=_make_mix(substitute_type=BinderSubstituteType.PFA, substitute_percent=30.0),
            time_elapsed=24.0,
            is_adjusted_for_placing_temp=False,
        )
        ggbs = AdiabaticTemperature(
            mix=_make_mix(substitute_type=BinderSubstituteType.GGBS, substitute_percent=50.0),
            time_elapsed=24.0,
            is_adjusted_for_placing_temp=False,
        )

        assert pfa.coefficient_c == pytest.approx(1.6 - 0.001 * 30.0, rel=1e-12)
        assert pfa.coefficient_d == pytest.approx(6.2 + 0.2131 * 30.0, rel=1e-12)
        assert pfa.activation_time_t2 == pytest.approx(3.5 + 0.0236 * 30.0, rel=1e-12)

        assert ggbs.coefficient_c == pytest.approx(
            1.6 - 0.0072 * 50.0 - 0.00003 * 50.0**2, rel=1e-12
        )
        assert ggbs.coefficient_d == pytest.approx(
            6.2 + 0.0848 * 50.0 - 0.0004 * 50.0**2, rel=1e-12
        )
        assert ggbs.activation_time_t2 == pytest.approx(3.5 + 0.0125 * 50.0, rel=1e-12)

    def test_ultimate_heat_generation_q_ult_unknown_substitute_type_falls_back(self):
        model = AdiabaticTemperature(mix=_make_mix(), time_elapsed=24.0)
        object.__setattr__(model.mix.binder, "substitute_type", "custom-substitute")

        assert model.ultimate_heat_generation_q_ult == pytest.approx(352.0 / 0.925, rel=1e-12)

    def test_ultimate_temperature_adjustment_toggle(self):
        mix = _make_mix(placing_temp=25.0)
        adjusted = AdiabaticTemperature(
            mix=mix,
            time_elapsed=24.0,
            test_mix_temp=20.0,
            is_adjusted_for_placing_temp=True,
        )
        unadjusted = AdiabaticTemperature(
            mix=mix,
            time_elapsed=24.0,
            test_mix_temp=20.0,
            is_adjusted_for_placing_temp=False,
        )

        expected_delta = 0.2 * (mix.concrete_placing_temp - 20.0)
        assert unadjusted.ultimate_temperature_t_ult - adjusted.ultimate_temperature_t_ult == pytest.approx(
            expected_delta, rel=1e-12
        )

    def test_elapsed_time_adjusted_by_rastrup_function(self):
        model = AdiabaticTemperature(
            mix=_make_mix(placing_temp=30.0),
            time_elapsed=24.0,
            rastrup_coefficient=12.0,
            test_mix_temp=20.0,
        )
        adjusted = model.elapsed_time_adjusted_by_rastrup_function(10.0)
        expected = 10.0 * (2 ** ((20.0 - 30.0) / 12.0))
        assert adjusted == pytest.approx(expected, rel=1e-12)

    def test_heat_and_temperature_queries(self):
        model = AdiabaticTemperature(mix=_make_mix(), time_elapsed=24.0)
        assert model.find_total_heat_generated_q_at_time(0.0) == pytest.approx(0.0, abs=1e-12)
        assert model.find_total_heat_generated_q_at_time(24.0) > 0.0

        temp_rise = model.find_modelled_temperature_at_time(24.0, is_temp_rise_only=True)
        abs_temp = model.find_modelled_temperature_at_time(24.0, is_temp_rise_only=False)
        assert abs_temp - temp_rise == pytest.approx(model.mix.concrete_placing_temp, rel=1e-12)

        with pytest.raises(ValueError, match="time_elapsed must be >= 0"):
            model.find_total_heat_generated_q_at_time(-1.0)
        with pytest.raises(ValueError, match="time_elapsed must be >= 0"):
            model.find_modelled_temperature_at_time(-1.0)

        # Wrappers around self.time_elapsed
        assert model.get_total_heat_generated_q_over_time() == pytest.approx(
            model.find_total_heat_generated_q_at_time(model.time_elapsed),
            rel=1e-12,
        )
        assert model.get_modelled_temperature_over_time() == pytest.approx(
            model.find_modelled_temperature_at_time(model.time_elapsed),
            rel=1e-12,
        )

    def test_time_series_dict_helpers(self):
        model = AdiabaticTemperature(mix=_make_mix(), time_elapsed=12.0)

        temp_data = model.make_time_temps_dict(number_of_time_intervals=4, is_temp_rise_only=True)
        assert temp_data["time"][0] == pytest.approx(0.0, abs=1e-12)
        assert temp_data["time"][-1] == pytest.approx(12.0, abs=1e-12)
        assert len(temp_data["time"]) == 5
        assert len(temp_data["adiabatic_temps"]) == 5

        heat_data = model.make_time_heat_dict(number_of_time_intervals=4)
        assert heat_data["time"][0] == pytest.approx(0.0, abs=1e-12)
        assert heat_data["time"][-1] == pytest.approx(12.0, abs=1e-12)
        assert len(heat_data["heat"]) == 5

        with pytest.raises(ValueError, match="must be > 0"):
            model.make_time_temps_dict(number_of_time_intervals=0)
        with pytest.raises(ValueError, match="must be > 0"):
            model.make_time_heat_dict(number_of_time_intervals=0)

    def test_maturity_coefficients_and_validation(self):
        model = AdiabaticTemperature(mix=_make_mix(), time_elapsed=24.0, test_mix_temp=20.0)

        assert model._is_valid_concrete_temp(5.0) is True
        assert model._is_valid_concrete_temp(4.9) is False

        assert model.sadgrove_maturity_coefficient(20.0) > 0.0
        assert model.arrhenius_maturity_coefficient(20.0, activation_energy=40000.0) == pytest.approx(
            1.0, rel=1e-12
        )
        assert model.saul_maturity_coefficient(20.0) == pytest.approx(1.0, rel=1e-12)

        with pytest.raises(ValueError, match="Invalid concrete temperature"):
            model.sadgrove_maturity_coefficient(0.0)
        with pytest.raises(ValueError, match="Invalid concrete temperature"):
            model.arrhenius_maturity_coefficient(0.0, activation_energy=40000.0)
        with pytest.raises(ValueError, match="Invalid concrete temperature"):
            model.saul_maturity_coefficient(0.0)

    def test_calculate_maturity_and_wrappers(self, monkeypatch):
        model = AdiabaticTemperature(mix=_make_mix(), time_elapsed=4.0)

        # Deterministic 2-interval profile: [10, 20, 30] => avg temps [15, 25]
        monkeypatch.setattr(
            AdiabaticTemperature,
            "make_time_temps_dict",
            lambda self, number_of_time_intervals, is_temp_rise_only=False: {
                "time": [0.0, 2.0, 4.0],
                "adiabatic_temps": [10.0, 20.0, 30.0],
            },
        )

        maturity = model._calculate_maturity(
            maturity_coefficient_function=lambda avg_temp: avg_temp / 10.0,
            number_of_time_intervals=2,
        )
        # interval = 2h, coeffs = 1.5 and 2.5 => total = (1.5+2.5)*2 = 8
        assert maturity == pytest.approx(8.0, rel=1e-12)

        monkeypatch.setattr(AdiabaticTemperature, "_calculate_maturity", lambda self, fn, n: 123.0)
        assert model.sadgrove_maturity(number_of_time_intervals=5) == pytest.approx(123.0, rel=1e-12)
        assert model.arrhenius_maturity(activation_energy=40000.0, number_of_time_intervals=5) == pytest.approx(
            123.0, rel=1e-12
        )
        assert model.saul_maturity(number_of_time_intervals=5) == pytest.approx(123.0, rel=1e-12)

    def test_strength_maturity_relationship_and_str(self):
        f = AdiabaticTemperature.strength_maturity_relationship(
            ultimate_compressive_strength=60.0,
            characteristic_time_constant=2.0,
            shape_parameter=0.5,
            test_age=7.0,
        )
        expected = 60.0 * exp(-((2.0 / 7.0) ** 0.5))
        assert f == pytest.approx(expected, rel=1e-12)

        model = AdiabaticTemperature(mix=_make_mix(), time_elapsed=6.0)
        text = str(model)
        assert "AdiabaticTemperature(time=6.0h" in text
        assert "temp=" in text
