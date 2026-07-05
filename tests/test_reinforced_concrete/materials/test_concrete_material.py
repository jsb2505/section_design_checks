"""
Tests for reinforced_concrete.materials.concrete module.
"""

import pytest
from pydantic import ValidationError
from materials.reinforced_concrete.materials import ConcreteMaterial
from materials.reinforced_concrete.materials.concrete import find_mean_flexural_tensile_strength
import materials.reinforced_concrete.materials.concrete as concrete_mod


class TestConcreteMaterial:
    """Tests for ConcreteMaterial class."""

    def test_create_c30_concrete(self, concrete_c30):
        """Test creating C30/37 concrete."""
        assert concrete_c30.grade == "C30/37"
        assert concrete_c30.f_ck == 30.0
        assert concrete_c30.f_ck_cube == 37.0

    def test_invalid_grade(self):
        """Test that invalid grades are rejected."""
        with pytest.raises(ValidationError):
            ConcreteMaterial(grade="C100/120")

    def test_characteristic_strengths(self, concrete_c30):
        """Test characteristic strength calculations."""
        assert concrete_c30.f_ck == 30.0
        assert concrete_c30.f_ck_cube == 37.0
        assert concrete_c30.f_cm == 38.0  # f_ck + 8

    def test_design_strength(self, concrete_c30):
        """Test design strength calculation."""
        # f_cd = alpha_cc * f_ck / gamma_c = 1.0 * 30 / 1.5 = 20.0
        # EU default alpha_cc = 1.0 (EC2 §3.1.6(1)P)
        assert concrete_c30.f_cd == pytest.approx(20.0)

    def test_custom_gamma_c(self):
        """Test custom partial factor."""
        concrete = ConcreteMaterial(grade="C30/37", gamma_c=1.2)
        # f_cd = 1.0 * 30 / 1.2 = 25.0 (EU default alpha_cc=1.0)
        assert concrete.f_cd == pytest.approx(25.0)

    def test_custom_alpha_cc(self):
        """Test custom alpha_cc."""
        concrete = ConcreteMaterial(grade="C30/37", alpha_cc=1.0)
        assert concrete.f_cd == pytest.approx(20.0)  # 1.0 * 30 / 1.5

    def test_tensile_strength_normal_concrete(self, concrete_c30):
        """Test tensile strength for f_ck ≤ 50."""
        # f_ctm = 0.30 * f_ck^(2/3) = 0.30 * 30^(2/3)
        expected = 0.30 * (30 ** (2/3))
        assert concrete_c30.f_ctm == pytest.approx(expected, rel=1e-6)

    def test_tensile_strength_high_strength_concrete(self):
        """Test tensile strength for f_ck > 50."""
        import math
        # C50/60 is at boundary (uses first formula)
        # Use C60/75 for testing second formula
        concrete_c60 = ConcreteMaterial(grade="C60/75")
        # f_ctm = 2.12 * ln(1 + f_cm/10)
        expected = 2.12 * math.log(1 + concrete_c60.f_cm / 10)
        assert concrete_c60.f_ctm == pytest.approx(expected, rel=1e-6)

    def test_tensile_strength_fractiles(self, concrete_c30):
        """Test 5% and 95% fractiles of tensile strength."""
        # f_ctk,0.05 = 0.7 * f_ctm
        assert concrete_c30.f_ctk_005 == pytest.approx(0.7 * concrete_c30.f_ctm)
        # f_ctk,0.95 = 1.3 * f_ctm
        assert concrete_c30.f_ctk_095 == pytest.approx(1.3 * concrete_c30.f_ctm)

    def test_design_tensile_strength(self, concrete_c30):
        """Test design tensile strength."""
        # f_ctd = alpha_ct * f_ctk,0.05 / gamma_c
        expected = 1.0 * concrete_c30.f_ctk_005 / 1.5
        assert concrete_c30.f_ctd == pytest.approx(expected)

    def test_elastic_modulus(self, concrete_c30):
        """Test elastic modulus calculation."""
        # E_cm = 22 * (f_cm / 10)^0.3 GPa
        import math
        E_base_GPa = 22.0 * ((38.0 / 10) ** 0.3)
        expected_MPa = E_base_GPa * 1000.0 * 1.0  # quartzite factor
        assert concrete_c30.E_cm == pytest.approx(expected_MPa, rel=1e-3)

    def test_elastic_modulus_aggregate_types(self):
        """Test elastic modulus with different aggregate types."""
        base = ConcreteMaterial(grade="C30/37", aggregate_type="quartzite")
        limestone = ConcreteMaterial(grade="C30/37", aggregate_type="limestone")
        basalt = ConcreteMaterial(grade="C30/37", aggregate_type="basalt")
        sandstone = ConcreteMaterial(grade="C30/37", aggregate_type="sandstone")

        # Check factors
        assert limestone.E_cm == pytest.approx(base.E_cm * 0.9, rel=1e-6)
        assert basalt.E_cm == pytest.approx(base.E_cm * 1.2, rel=1e-6)
        assert sandstone.E_cm == pytest.approx(base.E_cm * 0.7, rel=1e-6)

    def test_get_elastic_modulus_method(self, concrete_c30):
        """Test get_elastic_modulus() method."""
        assert concrete_c30.get_elastic_modulus() == concrete_c30.E_cm

    def test_strain_parameters_normal_strength(self, concrete_c30):
        """Test strain parameters for f_ck ≤ 50."""
        # All should be the standard values for normal strength
        assert concrete_c30.epsilon_c2 == 0.002
        assert concrete_c30.epsilon_cu2 == 0.0035
        assert concrete_c30.epsilon_c3 == 0.00175
        assert concrete_c30.epsilon_cu3 == 0.0035
        assert concrete_c30.n == 2.0

    def test_strain_parameters_high_strength(self):
        """Test strain parameters for f_ck > 50."""
        # C50 is exactly at boundary, so use > 50 check
        # For C50: epsilon_c2 = 0.002 (not greater)
        # For higher grades, these should be calculated values
        concrete_c60 = ConcreteMaterial(grade="C60/75")
        assert concrete_c60.epsilon_c2 > 0.002
        assert concrete_c60.epsilon_cu2 < 0.0035
        assert concrete_c60.epsilon_c3 > 0.00175
        assert concrete_c60.n < 2.0

    def test_all_concrete_grades(self):
        """Test all valid concrete grades can be created."""
        grades = [
            "C12/15", "C16/20", "C20/25", "C25/30", "C30/37", "C35/45", "C40/50",
            "C45/55", "C50/60", "C55/67", "C60/75", "C70/85", "C80/95", "C90/105"
        ]
        for grade in grades:
            concrete = ConcreteMaterial(grade=grade)
            assert concrete.grade == grade
            assert concrete.f_ck > 0
            assert concrete.f_cd > 0

    def test_density_default(self, concrete_c30):
        """Test default density."""
        assert concrete_c30.density == 2400.0

    def test_custom_density(self):
        """Test custom density."""
        concrete = ConcreteMaterial(grade="C30/37", density=2500.0)
        assert concrete.density == 2500.0

    def test_density_validation(self):
        """Test density validation - must be non-negative."""
        # Negative density should fail
        with pytest.raises(ValidationError):
            ConcreteMaterial(grade="C30/37", density=-100.0)

        # Positive values are now accepted (no upper bound)
        concrete1 = ConcreteMaterial(grade="C30/37", density=1000.0)
        assert concrete1.density == 1000.0
        concrete2 = ConcreteMaterial(grade="C30/37", density=3000.0)
        assert concrete2.density == 3000.0

    def test_str_representation(self, concrete_c30):
        """Test __str__ method."""
        s = str(concrete_c30)
        assert "C30/37" in s
        assert "30.0" in s  # f_ck
        assert "20.0" in s  # f_cd (EU default alpha_cc=1.0)

    def test_readonly_properties(self, concrete_c30):
        """Test that computed fields are read-only."""
        with pytest.raises((AttributeError, ValidationError, TypeError)):
            concrete_c30.f_ck = 40.0

    def test_validate_assignment(self, concrete_c30):
        """Test that changes trigger validation."""
        # Valid change
        concrete_c30.gamma_c = 1.2
        assert concrete_c30.gamma_c == 1.2

        # Invalid change
        with pytest.raises(ValidationError):
            concrete_c30.gamma_c = -1.0

    def test_json_serialization(self, concrete_c30):
        """Test JSON serialization."""
        json_data = concrete_c30.model_dump(mode="json")
        assert json_data["grade"] == "C30/37"
        assert json_data["gamma_c"] == 1.5
        # f_ck is a property derived from grade, not included in model_dump()
        assert concrete_c30.f_ck == 30.0

    def test_json_deserialization(self):
        """Test creating from JSON."""
        json_data = {
            "grade": "C30/37",
            "gamma_c": 1.5,
            "alpha_cc": 1.0,
        }
        concrete = ConcreteMaterial(**json_data)
        assert concrete.grade == "C30/37"
        assert concrete.f_ck == 30.0

    def test_50mpa_boundary_low(self):
        """Test that C50/60 (f_ck=50) uses the ≤50 branch."""
        c50 = ConcreteMaterial(grade="C50/60")
        # ≤50 branch: fixed values
        assert c50.f_ctm == pytest.approx(0.30 * (50 ** (2.0 / 3.0)), rel=1e-6)
        assert c50.epsilon_c2 == 0.002
        assert c50.epsilon_cu2 == 0.0035
        assert c50.epsilon_c3 == 0.00175
        assert c50.epsilon_cu3 == 0.0035
        assert c50.n == 2.0

    def test_50mpa_boundary_high(self):
        """Test that C55/67 (f_ck=55) uses the >50 branch."""
        import math
        c55 = ConcreteMaterial(grade="C55/67")
        # >50 branch: calculated values
        assert c55.f_ctm == pytest.approx(2.12 * math.log(1 + c55.f_cm / 10), rel=1e-6)
        assert c55.epsilon_c2 > 0.002
        assert c55.epsilon_cu2 < 0.0035
        assert c55.epsilon_c3 > 0.00175
        assert c55.n < 2.0

    def test_accidental_tensile_strength_and_high_strength_ultimate_branches(self):
        """Test accidental tensile strength and high strength ultimate branches."""
        c60 = ConcreteMaterial(grade="C60/75")
        expected_fctd_acc = c60.alpha_ct * c60.f_ctk_005 / c60.gamma_c_accidental
        assert c60.f_ctd_accidental == pytest.approx(expected_fctd_acc, rel=1e-12)
        assert c60.epsilon_cu1 < 0.0035
        assert c60.epsilon_cu3 < 0.0035

    def test_accidental_and_shear_design_strength_properties(self):
        """Test accidental and shear design strength properties."""
        c30 = ConcreteMaterial(grade="C30/37")
        assert c30.f_cd_accidental == pytest.approx(c30.alpha_cc * c30.f_ck / c30.gamma_c_accidental, rel=1e-12)
        assert c30.f_cd_shear == pytest.approx(c30.alpha_cc_shear * c30.f_ck / c30.gamma_c, rel=1e-12)
        assert c30.f_cd_shear_accidental == pytest.approx(
            c30.alpha_cc_shear * c30.f_ck / c30.gamma_c_accidental,
            rel=1e-12,
        )

    def test_epsilon_c1_and_epsilon_cu1_branches(self):
        """Test epsilon c1 and epsilon cu1 branches."""
        c30 = ConcreteMaterial(grade="C30/37")
        c90 = ConcreteMaterial(grade="C90/105")
        assert c30.epsilon_cu1 == pytest.approx(0.0035, rel=1e-12)
        assert c30.epsilon_c1 < 0.0028
        # High-grade branch should still be capped at 2.8‰
        assert c90.epsilon_c1 <= 0.0028 + 1e-12

    def test_mean_flexural_tensile_strength_function_and_method(self, concrete_c30):
        # h = 100 mm -> 0.1 m, factor = max(1.6 - 0.1, 1.0) = 1.5
        """Test mean flexural tensile strength function and method."""
        direct = find_mean_flexural_tensile_strength(concrete_c30.f_ctm, 100.0)
        assert direct == pytest.approx(concrete_c30.f_ctm * 1.5, rel=1e-12)
        assert concrete_c30.find_mean_flexural_tensile_strength(100.0) == pytest.approx(direct, rel=1e-12)

        # Large section height should floor factor at 1.0
        large = find_mean_flexural_tensile_strength(concrete_c30.f_ctm, 3000.0)
        assert large == pytest.approx(concrete_c30.f_ctm, rel=1e-12)

    def test_ndp_minimum_cylinder_strength_limit(self, monkeypatch: pytest.MonkeyPatch):
        """Test ndp minimum cylinder strength limit."""
        def _fake_get_ndp(key: str):
            values = {
                "f_ck_min": 20.0,
                "f_ck_cube_min": None,
                "f_ck_max": None,
                "f_ck_cube_max": None,
            }
            return values.get(key)

        monkeypatch.setattr(concrete_mod, "get_ndp", _fake_get_ndp)
        with pytest.raises(ValueError, match="below the minimum cylinder strength"):
            ConcreteMaterial(grade="C12/15")

    def test_ndp_minimum_cube_strength_limit(self, monkeypatch: pytest.MonkeyPatch):
        """Test ndp minimum cube strength limit."""
        def _fake_get_ndp(key: str):
            values = {
                "f_ck_min": None,
                "f_ck_cube_min": 20.0,
                "f_ck_max": None,
                "f_ck_cube_max": None,
            }
            return values.get(key)

        monkeypatch.setattr(concrete_mod, "get_ndp", _fake_get_ndp)
        with pytest.raises(ValueError, match="below the minimum cube strength"):
            ConcreteMaterial(grade="C12/15")

    def test_ndp_maximum_cylinder_strength_limit(self, monkeypatch: pytest.MonkeyPatch):
        """Test ndp maximum cylinder strength limit."""
        def _fake_get_ndp(key: str):
            values = {
                "f_ck_min": None,
                "f_ck_cube_min": None,
                "f_ck_max": 25.0,
                "f_ck_cube_max": None,
            }
            return values.get(key)

        monkeypatch.setattr(concrete_mod, "get_ndp", _fake_get_ndp)
        with pytest.raises(ValueError, match="exceeds the maximum cylinder strength"):
            ConcreteMaterial(grade="C30/37")

    def test_ndp_maximum_cube_strength_limit(self, monkeypatch: pytest.MonkeyPatch):
        """Test ndp maximum cube strength limit."""
        def _fake_get_ndp(key: str):
            values = {
                "f_ck_min": None,
                "f_ck_cube_min": None,
                "f_ck_max": None,
                "f_ck_cube_max": 30.0,
            }
            return values.get(key)

        monkeypatch.setattr(concrete_mod, "get_ndp", _fake_get_ndp)
        with pytest.raises(ValueError, match="exceeds the maximum cube strength"):
            ConcreteMaterial(grade="C30/37")

    def test_overrides_skip_ndp_validation_and_drive_properties(self, monkeypatch: pytest.MonkeyPatch):
        """Test overrides skip ndp validation and drive properties."""
        original_get_ndp = concrete_mod.get_ndp

        def _fake_get_ndp(key: str):
            if key == "f_ck_min":
                # Would reject C12/15 unless override short-circuits validator.
                return 999.0
            return original_get_ndp(key)

        monkeypatch.setattr(concrete_mod, "get_ndp", _fake_get_ndp)
        concrete = ConcreteMaterial(
            grade="C12/15",
            f_ck_override=30.0,
            f_cm_override=38.0,
            E_cm_override=32000.0,
        )
        assert concrete.f_ck == pytest.approx(30.0, rel=1e-12)
        assert concrete.f_cm == pytest.approx(38.0, rel=1e-12)
        assert concrete.E_cm == pytest.approx(32000.0, rel=1e-12)
