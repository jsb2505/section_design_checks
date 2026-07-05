"""Tests for :mod:`materials.utils.helpers` utility functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from materials.utils.helpers import as_float, cot


def test_cot_regular_angle_matches_trig_identity():
    """cot(π/4) should be exactly 1 within floating-point tolerance."""
    value = cot(math.pi / 4)
    assert value == pytest.approx(1.0, rel=1e-12), f"Unexpected cot(pi/4): {value}"


def test_cot_near_right_angle_returns_zero_when_cos_is_tiny():
    """When |cos(theta)| is below tolerance, helper should return 0.0."""
    value = cot(math.pi / 2, tol=1e-12)
    assert value == pytest.approx(0.0, abs=1e-12), f"Expected ~0 near pi/2, got {value}"


def test_cot_raises_on_singular_default_mode():
    """Default singular handling should raise for sin(theta)≈0."""
    with pytest.raises(ZeroDivisionError, match="cot undefined"):
        cot(0.0)


@pytest.mark.parametrize(
    ("theta", "expected_sign"),
    [
        (0.0, 1.0),        # cos(0) > 0 => +inf
        (math.pi, -1.0),   # cos(pi) < 0 => -inf
    ],
)
def test_cot_singular_inf_mode_returns_signed_infinity(theta: float, expected_sign: float):
    """`singular='inf'` should preserve the expected cotangent sign."""
    value = cot(theta, singular="inf")
    assert math.isinf(value), f"Expected infinity for theta={theta}, got {value}"
    assert math.copysign(1.0, value) == expected_sign, (
        f"Unexpected infinity sign for theta={theta}: got {value}"
    )


def test_as_float_real_values_and_numpy_scalars():
    """`as_float` should convert native and numpy real scalars to Python float."""
    assert as_float(2) == pytest.approx(2.0, rel=1e-12)
    assert as_float(np.float64(3.5)) == pytest.approx(3.5, rel=1e-12)


def test_as_float_complex_extracts_real_component():
    """For complex-step values, helper should drop the imaginary perturbation."""
    value = as_float(7.25 + 1e-30j)
    assert value == pytest.approx(7.25, rel=1e-12), f"Expected real part only, got {value}"


def test_as_float_raises_for_non_numeric_input():
    """Non-convertible values should fail loudly with a useful exception."""
    with pytest.raises((TypeError, ValueError), match="could not convert|string"):
        as_float("not-a-number")
