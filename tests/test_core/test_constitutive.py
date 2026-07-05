"""Tests for :mod:`materials.core.constitutive` base abstractions."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from materials.core.constitutive import BaseConstitutiveModel, StressStrainRelationship


class LinearQuadraticModel(BaseConstitutiveModel):
    """Simple constitutive model used to exercise default base-class behavior."""

    E: float = 200_000.0
    alpha: float = 1_000.0
    eps_u: float = 0.02
    f_y: float = 500.0

    def get_stress(self, strain: float) -> float:
        return self.E * strain + self.alpha * strain**2

    def get_ultimate_strain(self) -> float:
        return self.eps_u

    def get_yield_stress(self) -> float:
        return self.f_y


def test_base_constitutive_model_is_abstract():
    """`BaseConstitutiveModel` must not be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        BaseConstitutiveModel(name="invalid")


def test_default_get_stress_array_vectorizes_scalar_method():
    """Default array method should apply scalar `get_stress` element-wise."""
    model = LinearQuadraticModel(name="lin-quad")
    strains = np.array([-0.002, 0.0, 0.003], dtype=float)
    expected = np.array([model.get_stress(eps) for eps in strains], dtype=float)
    actual = model.get_stress_array(strains)
    assert np.allclose(actual, expected), (
        f"Vectorized stress mismatch. expected={expected.tolist()}, actual={actual.tolist()}"
    )


def test_default_get_tangent_modulus_uses_numerical_derivative():
    """Default tangent modulus should approximate dσ/dε for the model equation."""
    model = LinearQuadraticModel(name="lin-quad", E=210_000.0, alpha=2_000.0)
    strain = 0.0012
    # d/dε[E*ε + α*ε²] = E + 2αε
    expected = model.E + 2.0 * model.alpha * strain
    actual = model.get_tangent_modulus(strain)
    assert actual == pytest.approx(expected, rel=1e-5), (
        f"Tangent modulus mismatch at strain={strain}: expected={expected}, got={actual}"
    )


def test_default_get_tangent_modulus_array_vectorizes():
    """Default tangent array method should vectorize the scalar tangent method."""
    model = LinearQuadraticModel(name="lin-quad")
    strains = np.array([-0.001, 0.0, 0.001], dtype=float)
    expected = np.array([model.get_tangent_modulus(eps) for eps in strains], dtype=float)
    actual = model.get_tangent_modulus_array(strains)
    assert np.allclose(actual, expected), (
        f"Vectorized tangent mismatch. expected={expected.tolist()}, actual={actual.tolist()}"
    )


def test_constitutive_repr_contains_model_name():
    """`__repr__` should include both class name and user-facing model name."""
    model = LinearQuadraticModel(name="serviceability-model")
    rep = repr(model)
    assert "LinearQuadraticModel" in rep, f"repr missing class name: {rep}"
    assert "serviceability-model" in rep, f"repr missing model name: {rep}"


def test_runtime_protocol_check_for_stress_strain_relationship():
    """Concrete models implementing required methods should satisfy the protocol."""
    model = LinearQuadraticModel(name="protocol-check")
    assert isinstance(model, StressStrainRelationship), (
        "Model implementing get_stress/get_stress_array should satisfy StressStrainRelationship."
    )


def test_extra_fields_forbidden_by_base_model_config():
    """Unexpected model fields should be rejected by pydantic validation."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LinearQuadraticModel(name="bad", unknown_field=123)  # type: ignore[call-arg]
