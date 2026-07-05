"""
Tests for crack-width modular ratio handling (alpha_e = E_s / E_cm).
"""

import pytest

from materials.reinforced_concrete.code_checks.ec2_2004 import flexure_utils
from materials.reinforced_concrete.code_checks.ec2_2004.cracking_check import CrackingCheck


def test_calculate_modular_ratio_pure_helper():
    alpha_e = flexure_utils.calculate_modular_ratio(E_s=200000.0, E_cm=30000.0)
    assert alpha_e == pytest.approx(200000.0 / 30000.0, rel=1e-12)


def test_calculate_modular_ratio_rejects_nonpositive_E_cm():
    with pytest.raises(ValueError, match="E_cm must be > 0"):
        flexure_utils.calculate_modular_ratio(E_s=200000.0, E_cm=0.0)


def test_find_strain_difference_uses_passed_tension_zone_E_s(rectangular_beam_with_rebars, concrete_c30):
    check = CrackingCheck(section=rectangular_beam_with_rebars, concrete=concrete_c30)

    sigma_s = 200.0
    rho_p_eff = 0.01

    E_s_low = 100000.0
    E_s_high = 220000.0

    eps_low = check.find_strain_difference(sigma_s=sigma_s, rho_p_eff=rho_p_eff, E_s=E_s_low)
    eps_high = check.find_strain_difference(sigma_s=sigma_s, rho_p_eff=rho_p_eff, E_s=E_s_high)

    alpha_e_low = flexure_utils.calculate_modular_ratio(E_s=E_s_low, E_cm=concrete_c30.get_elastic_modulus())
    expected_low = max(
        (sigma_s - check.k_t * concrete_c30.f_ctm * (1.0 + alpha_e_low * rho_p_eff) / rho_p_eff) / E_s_low,
        0.6 * sigma_s / E_s_low,
    )

    assert eps_low == pytest.approx(expected_low, rel=1e-12)
    assert eps_low > eps_high
