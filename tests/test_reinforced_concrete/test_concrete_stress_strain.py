"""
Tests for reinforced_concrete.constitutive.concrete_stress_strain module.
"""

import pytest
import numpy as np
from materials.reinforced_concrete.constitutive import (
    ConcreteStressStrainSchematic,
    ConcreteStressStrainParabolaRectangle,
    ConcreteStressStrainBilinear,
    create_concrete_stress_strain,
)


class TestConcreteStressStrainSchematic:
    """Tests for ConcreteStressStrainSchematic class."""

    @pytest.fixture
    def model_c30(self, concrete_c30):
        """Schematic model for C30/37."""
        return ConcreteStressStrainSchematic(concrete=concrete_c30)

    def test_create_model(self, model_c30):
        """Test creating schematic model."""
        assert model_c30.name == "EC2 Schematic"
        assert model_c30.concrete.grade == "C30/37"

    def test_k_parameter(self, model_c30, concrete_c30):
        """Test k parameter calculation."""
        # k = 1.05 · E_cm · |ε_c1| / f_cm
        expected = 1.05 * concrete_c30.E_cm * abs(concrete_c30.epsilon_c1) / concrete_c30.f_cm
        assert model_c30.k == pytest.approx(expected, rel=1e-6)

    def test_stress_at_zero_strain(self, model_c30):
        """Test that stress is zero at zero strain."""
        assert model_c30.get_stress(0.0) == 0.0

    def test_stress_at_peak_strain(self, model_c30, concrete_c30):
        """Test stress at peak strain ε_c1."""
        stress = model_c30.get_stress(concrete_c30.epsilon_c1)
        # Should be close to f_cm
        assert stress == pytest.approx(concrete_c30.f_cm, rel=0.05)

    def test_stress_at_ultimate_strain(self, model_c30, concrete_c30):
        """Test stress at ultimate strain."""
        stress = model_c30.get_stress(concrete_c30.epsilon_cu1)
        # Should still be positive but lower than peak
        assert 0 < stress < concrete_c30.f_cm

    def test_stress_beyond_ultimate(self, model_c30, concrete_c30):
        """Test that stress is zero beyond ultimate strain."""
        stress = model_c30.get_stress(concrete_c30.epsilon_cu1 + 0.001)
        assert stress == 0.0

    def test_no_tension(self, model_c30):
        """Test that concrete has no tension capacity."""
        stress = model_c30.get_stress(-0.001)
        assert stress == 0.0

    def test_stress_array_vectorized(self, model_c30, concrete_c30):
        """Test vectorized stress calculation."""
        strains = np.array([0.0, 0.001, 0.002, 0.0035, 0.005])
        stresses = model_c30.get_stress_array(strains)

        assert isinstance(stresses, np.ndarray)
        assert len(stresses) == len(strains)
        assert stresses[0] == 0.0  # Zero strain
        assert stresses[-1] == 0.0  # Beyond ultimate

    def test_get_ultimate_strain(self, model_c30, concrete_c30):
        """Test get_ultimate_strain method."""
        assert model_c30.get_ultimate_strain() == concrete_c30.epsilon_cu1

    def test_get_yield_stress(self, model_c30, concrete_c30):
        """Test get_yield_stress method."""
        assert model_c30.get_yield_stress() == concrete_c30.f_cm


class TestConcreteStressStrainParabolaRectangle:
    """Tests for ConcreteStressStrainParabolaRectangle class."""

    @pytest.fixture
    def model_c30_design(self, concrete_c30):
        """Parabola-rectangle model with design strength."""
        return ConcreteStressStrainParabolaRectangle(
            concrete=concrete_c30,
            use_characteristic=False,
        )

    @pytest.fixture
    def model_c30_characteristic(self, concrete_c30):
        """Parabola-rectangle model with characteristic strength."""
        return ConcreteStressStrainParabolaRectangle(
            concrete=concrete_c30,
            use_characteristic=True,
        )

    def test_create_model(self, model_c30_design):
        """Test creating parabola-rectangle model."""
        assert model_c30_design.name == "EC2 Parabola-Rectangle"

    def test_design_vs_characteristic_strength(self, model_c30_design, model_c30_characteristic):
        """Test that model uses correct strength."""
        assert model_c30_design.f_c == model_c30_design.concrete.f_cd
        assert model_c30_characteristic.f_c == model_c30_characteristic.concrete.f_ck

    def test_stress_at_zero(self, model_c30_design):
        """Test stress at zero strain."""
        assert model_c30_design.get_stress(0.0) == 0.0

    def test_parabolic_region(self, model_c30_design, concrete_c30):
        """Test stress in parabolic region (0 < ε ≤ ε_c2)."""
        strain = concrete_c30.epsilon_c2 / 2  # Mid-point
        stress = model_c30_design.get_stress(strain)

        # Should be positive and less than f_cd
        assert 0 < stress < concrete_c30.f_cd

    def test_stress_at_ec2(self, model_c30_design, concrete_c30):
        """Test stress at ε_c2 (transition to rectangle)."""
        stress = model_c30_design.get_stress(concrete_c30.epsilon_c2)
        # Should equal f_cd
        assert stress == pytest.approx(concrete_c30.f_cd, rel=1e-6)

    def test_rectangular_region(self, model_c30_design, concrete_c30):
        """Test stress in rectangular region (ε_c2 < ε ≤ ε_cu2)."""
        strain = (concrete_c30.epsilon_c2 + concrete_c30.epsilon_cu2) / 2
        stress = model_c30_design.get_stress(strain)

        # Should equal f_cd (constant)
        assert stress == pytest.approx(concrete_c30.f_cd, rel=1e-6)

    def test_stress_at_ultimate(self, model_c30_design, concrete_c30):
        """Test stress at ultimate strain."""
        stress = model_c30_design.get_stress(concrete_c30.epsilon_cu2)
        assert stress == pytest.approx(concrete_c30.f_cd, rel=1e-6)

    def test_beyond_ultimate(self, model_c30_design, concrete_c30):
        """Test stress beyond ultimate strain."""
        stress = model_c30_design.get_stress(concrete_c30.epsilon_cu2 + 0.001)
        assert stress == 0.0

    def test_no_tension(self, model_c30_design):
        """Test no tension."""
        stress = model_c30_design.get_stress(-0.001)
        assert stress == 0.0

    def test_stress_array(self, model_c30_design, concrete_c30):
        """Test vectorized calculation."""
        strains = np.linspace(0, concrete_c30.epsilon_cu2 * 1.5, 100)
        stresses = model_c30_design.get_stress_array(strains)

        assert isinstance(stresses, np.ndarray)
        assert len(stresses) == len(strains)

        # Check specific points
        assert stresses[0] == 0.0
        assert stresses[-1] == 0.0  # Beyond ultimate

    def test_get_ultimate_strain(self, model_c30_design, concrete_c30):
        """Test get_ultimate_strain."""
        assert model_c30_design.get_ultimate_strain() == concrete_c30.epsilon_cu2

    def test_get_yield_stress(self, model_c30_design, concrete_c30):
        """Test get_yield_stress."""
        assert model_c30_design.get_yield_stress() == concrete_c30.f_cd


class TestConcreteStressStrainBilinear:
    """Tests for ConcreteStressStrainBilinear class."""

    @pytest.fixture
    def model_c30(self, concrete_c30):
        """Bilinear model for C30/37."""
        return ConcreteStressStrainBilinear(concrete=concrete_c30)

    def test_create_model(self, model_c30):
        """Test creating bilinear model."""
        assert model_c30.name == "EC2 Bilinear"

    def test_stress_at_zero(self, model_c30):
        """Test stress at zero strain."""
        assert model_c30.get_stress(0.0) == 0.0

    def test_linear_region(self, model_c30, concrete_c30):
        """Test stress in linear region (0 < ε ≤ ε_c3)."""
        strain = concrete_c30.epsilon_c3 / 2
        stress = model_c30.get_stress(strain)

        # Should be linear: stress = f_cd * ε / ε_c3
        expected = concrete_c30.f_cd * strain / concrete_c30.epsilon_c3
        assert stress == pytest.approx(expected, rel=1e-6)

    def test_stress_at_ec3(self, model_c30, concrete_c30):
        """Test stress at ε_c3."""
        stress = model_c30.get_stress(concrete_c30.epsilon_c3)
        assert stress == pytest.approx(concrete_c30.f_cd, rel=1e-6)

    def test_constant_region(self, model_c30, concrete_c30):
        """Test stress in constant region (ε_c3 < ε ≤ ε_cu3)."""
        strain = (concrete_c30.epsilon_c3 + concrete_c30.epsilon_cu3) / 2
        stress = model_c30.get_stress(strain)
        assert stress == pytest.approx(concrete_c30.f_cd, rel=1e-6)

    def test_beyond_ultimate(self, model_c30, concrete_c30):
        """Test stress beyond ultimate."""
        stress = model_c30.get_stress(concrete_c30.epsilon_cu3 + 0.001)
        assert stress == 0.0

    def test_get_ultimate_strain(self, model_c30, concrete_c30):
        """Test get_ultimate_strain."""
        assert model_c30.get_ultimate_strain() == concrete_c30.epsilon_cu3


class TestCreateConcreteStressStrain:
    """Tests for factory function."""

    def test_create_schematic(self, concrete_c30):
        """Test creating schematic model."""
        model = create_concrete_stress_strain(concrete_c30, "schematic")
        assert isinstance(model, ConcreteStressStrainSchematic)

    def test_create_parabola_rectangle(self, concrete_c30):
        """Test creating parabola-rectangle model."""
        model = create_concrete_stress_strain(concrete_c30, "parabola-rectangle")
        assert isinstance(model, ConcreteStressStrainParabolaRectangle)

    def test_create_bilinear(self, concrete_c30):
        """Test creating bilinear model."""
        model = create_concrete_stress_strain(concrete_c30, "bilinear")
        assert isinstance(model, ConcreteStressStrainBilinear)

    def test_invalid_model_type(self, concrete_c30):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_concrete_stress_strain(concrete_c30, "invalid")

    def test_use_characteristic_flag(self, concrete_c30):
        """Test use_characteristic flag."""
        model = create_concrete_stress_strain(
            concrete_c30,
            "parabola-rectangle",
            use_characteristic=True
        )
        assert model.f_c == concrete_c30.f_ck
