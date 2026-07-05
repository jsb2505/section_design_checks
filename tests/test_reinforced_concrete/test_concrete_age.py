"""
Tests for age-dependent concrete material properties.
"""

from __future__ import annotations

from math import exp, sqrt

import pytest
from pydantic import ValidationError

import materials.reinforced_concrete.materials.concrete_age as concrete_age_mod
from materials.reinforced_concrete.materials import ConcreteMaterial
from materials.reinforced_concrete.materials.concrete_age import CementClass, ConcreteAge


class TestCementClass:
    def test_s_coefficient_values(self):
        assert CementClass.R.s_coefficient == pytest.approx(0.20, rel=1e-12)
        assert CementClass.N.s_coefficient == pytest.approx(0.25, rel=1e-12)
        assert CementClass.S.s_coefficient == pytest.approx(0.38, rel=1e-12)


class TestConcreteAge:
    def test_validation_requires_age_greater_than_3_days(self):
        concrete = ConcreteMaterial(grade="C30/37")
        with pytest.raises(ValidationError):
            ConcreteAge(concrete=concrete, age=3.0, cement_class=CementClass.N)

    def test_properties_for_age_less_than_28_days(self):
        concrete = ConcreteMaterial(grade="C30/37")
        age = ConcreteAge(concrete=concrete, age=7.0, cement_class=CementClass.N)

        beta = exp(CementClass.N.s_coefficient * (1.0 - sqrt(28.0 / 7.0)))
        assert age.beta_cc_t == pytest.approx(beta, rel=1e-12)
        assert age.f_cm_t == pytest.approx(beta * concrete.f_cm, rel=1e-12)
        assert age.f_ck_t == pytest.approx(age.f_cm_t - 8.0, rel=1e-12)
        assert age.f_ctm_t == pytest.approx(concrete.f_ctm * beta, rel=1e-12)
        assert age.f_ctd_t == pytest.approx(concrete.alpha_ct * 0.7 * age.f_ctm_t / concrete.gamma_c, rel=1e-12)
        assert age.E_cm_t == pytest.approx(concrete.E_cm * ((age.f_cm_t / concrete.f_cm) ** 0.3), rel=1e-12)

    def test_properties_for_age_greater_than_or_equal_28_days(self):
        concrete = ConcreteMaterial(grade="C30/37")
        age = ConcreteAge(concrete=concrete, age=56.0, cement_class=CementClass.R)

        beta = exp(CementClass.R.s_coefficient * (1.0 - sqrt(28.0 / 56.0)))
        expected_f_ctm_t = concrete.f_ctm * (beta ** (2.0 / 3.0))
        assert age.beta_cc_t == pytest.approx(beta, rel=1e-12)
        assert age.f_ctm_t == pytest.approx(expected_f_ctm_t, rel=1e-12)

    def test_find_mean_flexural_tensile_strength_delegates_to_helper(self, monkeypatch):
        concrete = ConcreteMaterial(grade="C30/37")
        age = ConcreteAge(concrete=concrete, age=14.0, cement_class=CementClass.N)
        captured = {}

        def _fake_helper(f_ctm_t, section_height):
            captured["f_ctm_t"] = f_ctm_t
            captured["section_height"] = section_height
            return 9.87

        monkeypatch.setattr(concrete_age_mod, "find_mean_flexural_tensile_strength", _fake_helper)

        out = age.find_mean_flexural_tensile_strength(section_height=450.0)
        assert out == pytest.approx(9.87, rel=1e-12)
        assert captured["f_ctm_t"] == pytest.approx(age.f_ctm_t, rel=1e-12)
        assert captured["section_height"] == pytest.approx(450.0, rel=1e-12)

    def test_assignment_validation_and_string_representations(self):
        concrete = ConcreteMaterial(grade="C30/37")
        age = ConcreteAge(concrete=concrete, age=10.0, cement_class=CementClass.S)

        with pytest.raises(ValidationError):
            age.age = 2.0

        s = str(age)
        r = repr(age)
        assert concrete.grade in s
        assert "cement class" in s
        assert "ConcreteAge" in r
        assert "age=10.0 days" in r
