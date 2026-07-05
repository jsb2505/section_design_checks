"""
Tests for custom user-defined constitutive models.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from materials.reinforced_concrete.constitutive.custom_constitutive import (
    CustomConcreteModel,
    CustomSteelModel,
)


class TestCustomConcreteModel:
    def test_basic_interface_methods(self):
        model = CustomConcreteModel(
            stress_fn=lambda eps: 1000.0 * eps,
            ultimate_strain=0.0035,
            yield_stress=30.0,
        )

        assert model.get_stress(0.002) == pytest.approx(2.0, rel=1e-12)
        assert model.get_ultimate_strain() == pytest.approx(0.0035, rel=1e-12)
        assert model.get_yield_stress() == pytest.approx(30.0, rel=1e-12)

    def test_get_stress_array_uses_fallback_vectorization(self):
        model = CustomConcreteModel(
            stress_fn=lambda eps: 1000.0 * eps,
            ultimate_strain=0.0035,
            yield_stress=30.0,
        )
        strains = np.array([-0.001, 0.0, 0.002], dtype=float)

        out = model.get_stress_array(strains)
        np.testing.assert_allclose(out, np.array([-1.0, 0.0, 2.0], dtype=float), rtol=1e-12, atol=0.0)

    def test_get_stress_array_uses_custom_vectorized_callable(self):
        model = CustomConcreteModel(
            stress_fn=lambda eps: 9999.0,  # should not be used
            stress_array_fn=lambda arr: arr * 42.0,
            ultimate_strain=0.0035,
            yield_stress=30.0,
        )
        strains = np.array([0.0, 0.001], dtype=float)

        out = model.get_stress_array(strains)
        np.testing.assert_allclose(out, np.array([0.0, 0.042], dtype=float), rtol=1e-12, atol=0.0)

    def test_get_tangent_modulus_fallback_and_override(self):
        model_default = CustomConcreteModel(
            stress_fn=lambda eps: 100.0 * (eps ** 2),
            ultimate_strain=0.0035,
            yield_stress=30.0,
        )
        # d/deps [100 eps^2] at eps=0.01 is 2.0
        assert model_default.get_tangent_modulus(0.01) == pytest.approx(2.0, rel=1e-5)

        model_override = CustomConcreteModel(
            stress_fn=lambda eps: 0.0,
            tangent_modulus_fn=lambda eps: 1234.0 + eps,
            ultimate_strain=0.0035,
            yield_stress=30.0,
        )
        assert model_override.get_tangent_modulus(0.01) == pytest.approx(1234.01, rel=1e-12)

    def test_validation_rejects_non_positive_limits(self):
        with pytest.raises(ValidationError):
            CustomConcreteModel(
                stress_fn=lambda eps: eps,
                ultimate_strain=0.0,
                yield_stress=30.0,
            )

        with pytest.raises(ValidationError):
            CustomConcreteModel(
                stress_fn=lambda eps: eps,
                ultimate_strain=0.0035,
                yield_stress=0.0,
            )


class TestCustomSteelModel:
    def test_default_and_explicit_epsilon_y(self):
        model_default = CustomSteelModel(
            stress_fn=lambda eps: 200_000.0 * eps,
            ultimate_strain=0.05,
            yield_stress=500.0,
        )
        assert model_default.epsilon_y == pytest.approx(0.0025, rel=1e-12)

        model_explicit = CustomSteelModel(
            stress_fn=lambda eps: 200_000.0 * eps,
            ultimate_strain=0.05,
            yield_stress=500.0,
            epsilon_y=0.003,
        )
        assert model_explicit.epsilon_y == pytest.approx(0.003, rel=1e-12)

    def test_basic_interface_and_array_methods(self):
        model = CustomSteelModel(
            stress_fn=lambda eps: min(abs(eps) * 200_000.0, 500.0) * np.sign(eps),
            ultimate_strain=0.05,
            yield_stress=500.0,
        )

        assert model.get_stress(0.001) == pytest.approx(200.0, rel=1e-12)
        assert model.get_ultimate_strain() == pytest.approx(0.05, rel=1e-12)
        assert model.get_yield_stress() == pytest.approx(500.0, rel=1e-12)

        strains = np.array([-0.001, 0.0, 0.004], dtype=float)
        out = model.get_stress_array(strains)
        np.testing.assert_allclose(out, np.array([-200.0, 0.0, 500.0], dtype=float), rtol=1e-12, atol=0.0)

    def test_custom_array_and_tangent_modulus_overrides(self):
        model = CustomSteelModel(
            stress_fn=lambda eps: 0.0,
            stress_array_fn=lambda arr: arr + 7.0,
            tangent_modulus_fn=lambda eps: 321.0 - eps,
            ultimate_strain=0.05,
            yield_stress=500.0,
        )
        strains = np.array([0.0, 1.0], dtype=float)
        out = model.get_stress_array(strains)
        np.testing.assert_allclose(out, np.array([7.0, 8.0], dtype=float), rtol=1e-12, atol=0.0)
        assert model.get_tangent_modulus(0.25) == pytest.approx(320.75, rel=1e-12)

    def test_tangent_modulus_fallback_without_override(self):
        model = CustomSteelModel(
            stress_fn=lambda eps: 100.0 * (eps ** 2),
            ultimate_strain=0.05,
            yield_stress=500.0,
        )
        assert model.get_tangent_modulus(0.01) == pytest.approx(2.0, rel=1e-5)

    def test_validation_rejects_non_positive_limits(self):
        with pytest.raises(ValidationError):
            CustomSteelModel(
                stress_fn=lambda eps: eps,
                ultimate_strain=-0.1,
                yield_stress=500.0,
            )

        with pytest.raises(ValidationError):
            CustomSteelModel(
                stress_fn=lambda eps: eps,
                ultimate_strain=0.05,
                yield_stress=-1.0,
            )
