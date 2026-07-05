"""
Tests for reinforced_concrete.materials.reinforcing_steel module.
"""

import pytest
from pydantic import ValidationError

from materials.reinforced_concrete.materials import ReinforcingSteel


class TestReinforcingSteel:
    """Tests for ReinforcingSteel class."""

    def test_create_b500b_steel(self, steel_b500b):
        """Test creating B500B steel."""
        assert steel_b500b.grade == "B500B"
        assert steel_b500b.f_yk == 500.0

    def test_invalid_grade(self):
        """Test that invalid grades are rejected."""
        with pytest.raises(ValidationError):
            ReinforcingSteel(grade="B600C")

    def test_all_grades(self):
        """Test all valid grades can be created."""
        for grade in ["B500A", "B500B", "B500C"]:
            steel = ReinforcingSteel(grade=grade)
            assert steel.grade == grade
            assert steel.f_yk == 500.0

    def test_characteristic_strength(self, steel_b500b):
        """Test characteristic yield strength."""
        assert steel_b500b.f_yk == 500.0

    def test_design_strength(self, steel_b500b):
        """Test design yield strength."""
        # f_yd = f_yk / gamma_s = 500 / 1.15
        expected = 500.0 / 1.15
        assert steel_b500b.f_yd == pytest.approx(expected, rel=1e-6)

    def test_custom_gamma_s(self):
        """Test custom partial factor."""
        steel = ReinforcingSteel(grade="B500B", gamma_s=1.0)
        assert steel.f_yd == 500.0

    def test_accidental_design_strength(self, steel_b500b):
        """Test design strength for accidental combinations."""
        # gamma_s_accidental defaults to 1.0
        assert steel_b500b.f_yd_accidental == 500.0

    def test_elastic_modulus(self, steel_b500b):
        """Test elastic modulus."""
        assert steel_b500b.E_s == 200_000.0

    def test_get_elastic_modulus_method(self, steel_b500b):
        """Test get_elastic_modulus() method."""
        assert steel_b500b.get_elastic_modulus() == 200_000.0

    def test_tensile_strength_ratios(self):
        """Test f_t/f_yk ratios for different grades."""
        b500a = ReinforcingSteel(grade="B500A")
        b500b = ReinforcingSteel(grade="B500B")
        b500c = ReinforcingSteel(grade="B500C")

        assert b500a.f_t == pytest.approx(500.0 * 1.05)
        assert b500b.f_t == pytest.approx(500.0 * 1.08)
        assert b500c.f_t == pytest.approx(500.0 * 1.15)
        assert b500b.f_td == pytest.approx(b500b.f_t / b500b.gamma_s, rel=1e-12)
        assert b500b.f_td_accidental == pytest.approx(
            b500b.f_t / b500b.gamma_s_accidental,
            rel=1e-12,
        )

    def test_characteristic_yield_strain(self, steel_b500b):
        """Test characteristic yield strain."""
        # epsilon_yk = f_yk / E_s
        expected = 500.0 / 200_000.0
        assert steel_b500b.epsilon_yk == pytest.approx(expected)

    def test_design_yield_strain(self, steel_b500b):
        """Test design yield strain."""
        # epsilon_yd = f_yd / E_s
        expected = steel_b500b.f_yd / 200_000.0
        assert steel_b500b.epsilon_yd == pytest.approx(expected)

    def test_ultimate_strains(self):
        """Test ultimate strains for different grades."""
        b500a = ReinforcingSteel(grade="B500A")
        b500b = ReinforcingSteel(grade="B500B")
        b500c = ReinforcingSteel(grade="B500C")

        assert b500a.epsilon_uk == 0.025
        assert b500b.epsilon_uk == 0.050
        assert b500c.epsilon_uk == 0.075

    def test_design_ultimate_strain(self, steel_b500b):
        """Test design ultimate strain."""
        # epsilon_ud = 0.9 * epsilon_uk
        expected = 0.9 * 0.050
        assert steel_b500b.epsilon_ud == pytest.approx(expected)

    def test_k_ratio(self):
        """Test k = f_t/f_yk ratio."""
        b500a = ReinforcingSteel(grade="B500A")
        b500b = ReinforcingSteel(grade="B500B")
        b500c = ReinforcingSteel(grade="B500C")

        assert b500a.k_ratio == pytest.approx(1.05)
        assert b500b.k_ratio == pytest.approx(1.08)
        assert b500c.k_ratio == pytest.approx(1.15)

    def test_ductility_class(self):
        """Test ductility class mapping."""
        b500a = ReinforcingSteel(grade="B500A")
        b500b = ReinforcingSteel(grade="B500B")
        b500c = ReinforcingSteel(grade="B500C")

        assert b500a.ductility_class == "A"
        assert b500b.ductility_class == "B"
        assert b500c.ductility_class == "C"

    def test_density_default(self, steel_b500b):
        """Test default density."""
        assert steel_b500b.density == 7850.0

    def test_custom_density(self):
        """Test custom density."""
        steel = ReinforcingSteel(grade="B500B", density=7800.0)
        assert steel.density == 7800.0

    def test_density_validation(self):
        """Test density validation - must be non-negative."""
        # Negative density should fail
        with pytest.raises(ValidationError):
            ReinforcingSteel(grade="B500B", density=-100.0)

        # Positive values are now accepted (no upper bound)
        steel1 = ReinforcingSteel(grade="B500B", density=5000.0)
        assert steel1.density == 5000.0
        steel2 = ReinforcingSteel(grade="B500B", density=9000.0)
        assert steel2.density == 9000.0

    def test_str_representation(self):
        """Test __str__ method."""
        steel = ReinforcingSteel(grade="B500B")
        s = str(steel)
        assert "B500B" in s
        assert "500" in s  # f_yk
        assert "Class B" in s

    def test_json_serialization(self, steel_b500b):
        """Test JSON serialization."""
        json_data = steel_b500b.model_dump()
        assert json_data["grade"] == "B500B"
        assert json_data["gamma_s"] == 1.15
        # f_yk is now a property, not included in model_dump()
        assert steel_b500b.f_yk == 500.0

    def test_validate_assignment(self, steel_b500b):
        """Test that changes trigger validation."""
        # Valid change
        steel_b500b.gamma_s = 1.0
        assert steel_b500b.gamma_s == 1.0

        # Invalid change
        with pytest.raises(ValidationError):
            steel_b500b.gamma_s = -1.0

    def test_classmethod_strength_helpers(self):
        """Test classmethod strength helpers."""
        assert ReinforcingSteel.f_yk_for() == pytest.approx(500.0, rel=1e-12)
        assert ReinforcingSteel.f_yk_for("B500C") == pytest.approx(500.0, rel=1e-12)
        assert ReinforcingSteel.f_yd_for(grade="B500A", gamma_s=1.25) == pytest.approx(400.0, rel=1e-12)
        assert ReinforcingSteel.f_yd_accidental_for(grade="B500B", gamma_s_accidental=1.0) == pytest.approx(
            500.0,
            rel=1e-12,
        )
