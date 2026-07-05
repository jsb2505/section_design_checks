"""
Tests for reinforced_concrete.constitutive.concrete_stress_strain module.
"""

import numpy as np
import pytest

import section_design_checks.reinforced_concrete.constitutive.concrete_stress_strain as css_mod
from section_design_checks.reinforced_concrete.constitutive import (
    ConcreteModelType,
    ConcreteStressStrainBilinear,
    ConcreteStressStrainLinearElastic,
    ConcreteStressStrainParabolaRectangle,
    ConcreteStressStrainSchematic,
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

    def test_accidental_strength(self, concrete_c30):
        """Test that model uses f_cd_accidental when use_accidental=True."""
        model_accidental = ConcreteStressStrainParabolaRectangle(
            concrete=concrete_c30,
            use_accidental=True,
        )
        assert model_accidental.f_c == concrete_c30.f_cd_accidental
        # f_cd_accidental should be higher than f_cd (due to reduced partial factor)
        assert concrete_c30.f_cd_accidental > concrete_c30.f_cd

    def test_cannot_use_both_characteristic_and_accidental(self, concrete_c30):
        """Test that using both flags raises an error."""
        with pytest.raises(ValueError, match="Cannot set both use_characteristic=True and use_accidental=True"):
            ConcreteStressStrainParabolaRectangle(
                concrete=concrete_c30,
                use_characteristic=True,
                use_accidental=True,
            )

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


class TestEC2ConfinedConcrete:
    """Tests for EC2 §3.1.9 confined concrete in parabola-rectangle model."""

    @pytest.fixture
    def concrete_c30(self):
        """C30/37 concrete material for confinement tests."""
        from section_design_checks.reinforced_concrete.materials.concrete import ConcreteMaterial
        return ConcreteMaterial(grade="C30/37")

    def test_unconfined_default(self, concrete_c30):
        """Test that model is unconfined by default."""
        model = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30)
        assert model.sigma_2 is None
        assert model.is_ec2_confined is False

    def test_sigma_2_zero_not_confined(self, concrete_c30):
        """Test that sigma_2=0 is treated as unconfined."""
        model = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, sigma_2=0.0)
        assert model.is_ec2_confined is False
        assert model.f_ck_c == concrete_c30.f_ck

    def test_confined_strength_low_confinement(self, concrete_c30):
        """Test confined strength for σ₂ ≤ 0.05·fck (Eq. 3.24)."""
        # σ₂ = 1.0 MPa, which is < 0.05 * 30 = 1.5 MPa
        sigma_2 = 1.0
        model = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, sigma_2=sigma_2)

        assert model.is_ec2_confined is True

        # fck,c = fck(1.000 + 5.0·σ₂/fck) = 30 * (1 + 5*1/30) = 30 * 1.167 = 35.0
        expected_f_ck_c = 30.0 * (1.0 + 5.0 * sigma_2 / 30.0)
        assert model.f_ck_c == pytest.approx(expected_f_ck_c, rel=1e-6)

    def test_confined_strength_high_confinement(self, concrete_c30):
        """Test confined strength for σ₂ > 0.05·fck (Eq. 3.25)."""
        # σ₂ = 3.0 MPa, which is > 0.05 * 30 = 1.5 MPa
        sigma_2 = 3.0
        model = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, sigma_2=sigma_2)

        # fck,c = fck(1.125 + 2.5·σ₂/fck) = 30 * (1.125 + 2.5*3/30) = 30 * 1.375 = 41.25
        expected_f_ck_c = 30.0 * (1.125 + 2.5 * sigma_2 / 30.0)
        assert model.f_ck_c == pytest.approx(expected_f_ck_c, rel=1e-6)

    def test_confined_strain_at_peak(self, concrete_c30):
        """Test confined strain at peak per EC2 Eq. 3.26."""
        sigma_2 = 2.0
        model = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, sigma_2=sigma_2)

        # εc2,c = εc2·(fck,c/fck)²
        f_ck_c = model.f_ck_c
        strength_ratio = f_ck_c / concrete_c30.f_ck
        expected_eps_c2_c = concrete_c30.epsilon_c2 * (strength_ratio ** 2)
        assert model.epsilon_c2_c == pytest.approx(expected_eps_c2_c, rel=1e-6)

    def test_confined_ultimate_strain(self, concrete_c30):
        """Test confined ultimate strain per EC2 Eq. 3.27."""
        sigma_2 = 2.0
        model = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, sigma_2=sigma_2)

        # εcu2,c = εcu2 + 0.2·σ₂/fck
        expected_eps_cu2_c = concrete_c30.epsilon_cu2 + 0.2 * sigma_2 / concrete_c30.f_ck
        assert model.epsilon_cu2_c == pytest.approx(expected_eps_cu2_c, rel=1e-6)

    def test_confined_design_strength(self, concrete_c30):
        """Test that confined design strength uses design reduction factor."""
        sigma_2 = 2.0
        model = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, sigma_2=sigma_2)

        # f_c = f_ck_c * alpha_cc / gamma_c
        design_factor = concrete_c30.alpha_cc / concrete_c30.gamma_c
        expected_f_c = model.f_ck_c * design_factor
        assert model.f_c == pytest.approx(expected_f_c, rel=1e-6)

    def test_confined_stress_higher_than_unconfined(self, concrete_c30):
        """Test that confined concrete gives higher peak stress (f_c)."""
        model_unconfined = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30)
        model_confined = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, sigma_2=2.0)

        # Confined concrete should have higher peak stress
        assert model_confined.f_c > model_unconfined.f_c

        # At their respective peak strains, both should reach their peak stress
        stress_unconfined = model_unconfined.get_stress(model_unconfined.epsilon_c2_eff)
        stress_confined = model_confined.get_stress(model_confined.epsilon_c2_eff)

        assert stress_confined == pytest.approx(model_confined.f_c, rel=1e-6)
        assert stress_unconfined == pytest.approx(model_unconfined.f_c, rel=1e-6)
        assert stress_confined > stress_unconfined

    def test_confined_ultimate_strain_higher(self, concrete_c30):
        """Test that confined concrete has higher ultimate strain."""
        model_unconfined = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30)
        model_confined = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, sigma_2=2.0)

        assert model_confined.get_ultimate_strain() > model_unconfined.get_ultimate_strain()

    def test_effective_strain_properties(self, concrete_c30):
        """Test that effective strain properties work correctly."""
        sigma_2 = 2.0
        model = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, sigma_2=sigma_2)

        # Effective properties should equal confined values
        assert model.epsilon_c2_eff == model.epsilon_c2_c
        assert model.epsilon_cu2_eff == model.epsilon_cu2_c

        # Unconfined model should use base values
        model_unconf = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30)
        assert model_unconf.epsilon_c2_eff == concrete_c30.epsilon_c2
        assert model_unconf.epsilon_cu2_eff == concrete_c30.epsilon_cu2

    def test_stress_array_confined(self, concrete_c30):
        """Test vectorized stress calculation with confinement."""
        model = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, sigma_2=2.0)

        strains = np.linspace(0, model.epsilon_cu2_eff * 1.2, 50)
        stresses = model.get_stress_array(strains)

        # Basic sanity checks
        assert stresses[0] == 0.0  # Zero strain
        assert stresses[-1] == 0.0  # Beyond ultimate

        # Max stress should be at or near f_c
        assert np.max(stresses) == pytest.approx(model.f_c, rel=0.01)

    def test_confined_with_use_characteristic(self, concrete_c30):
        """Test that use_characteristic=True returns f_ck_c without reduction."""
        sigma_2 = 2.0
        model = ConcreteStressStrainParabolaRectangle(
            concrete=concrete_c30, sigma_2=sigma_2, use_characteristic=True
        )

        # f_c should equal f_ck_c (no alpha_cc/gamma_c reduction)
        assert model.f_c == pytest.approx(model.f_ck_c, rel=1e-6)
        assert model.f_c > concrete_c30.f_ck  # Confined > unconfined

    def test_confined_with_use_accidental(self, concrete_c30):
        """Test that use_accidental=True uses gamma_c_accidental."""
        sigma_2 = 2.0
        model = ConcreteStressStrainParabolaRectangle(
            concrete=concrete_c30, sigma_2=sigma_2, use_accidental=True
        )

        # f_c = f_ck_c * alpha_cc / gamma_c_accidental
        accidental_factor = concrete_c30.alpha_cc / concrete_c30.gamma_c_accidental
        expected_f_c = model.f_ck_c * accidental_factor
        assert model.f_c == pytest.approx(expected_f_c, rel=1e-6)

        # Accidental should be higher than normal design (lower gamma_c)
        model_design = ConcreteStressStrainParabolaRectangle(
            concrete=concrete_c30, sigma_2=sigma_2
        )
        assert model.f_c > model_design.f_c

    def test_confined_strength_ordering(self, concrete_c30):
        """Test that characteristic > accidental > design for confined concrete."""
        sigma_2 = 2.0

        model_char = ConcreteStressStrainParabolaRectangle(
            concrete=concrete_c30, sigma_2=sigma_2, use_characteristic=True
        )
        model_acc = ConcreteStressStrainParabolaRectangle(
            concrete=concrete_c30, sigma_2=sigma_2, use_accidental=True
        )
        model_des = ConcreteStressStrainParabolaRectangle(
            concrete=concrete_c30, sigma_2=sigma_2
        )

        # Characteristic > Accidental > Design
        assert model_char.f_c > model_acc.f_c > model_des.f_c


class TestConcreteStressStrainBilinear:
    """Tests for ConcreteStressStrainBilinear class."""

    @pytest.fixture
    def model_c30(self, concrete_c30):
        """Bilinear model for C30/37."""
        return ConcreteStressStrainBilinear(concrete=concrete_c30)

    def test_create_model(self, model_c30):
        """Test creating bilinear model."""
        assert model_c30.name == "EC2 Bilinear"

    def test_accidental_strength(self, concrete_c30):
        """Test that model uses f_cd_accidental when use_accidental=True."""
        model_accidental = ConcreteStressStrainBilinear(
            concrete=concrete_c30,
            use_accidental=True,
        )
        assert model_accidental.f_c == concrete_c30.f_cd_accidental
        # f_cd_accidental should be higher than f_cd (due to reduced partial factor)
        assert concrete_c30.f_cd_accidental > concrete_c30.f_cd

    def test_cannot_use_both_characteristic_and_accidental(self, concrete_c30):
        """Test that using both flags raises an error."""
        with pytest.raises(ValueError, match="Cannot set both use_characteristic=True and use_accidental=True"):
            ConcreteStressStrainBilinear(
                concrete=concrete_c30,
                use_characteristic=True,
                use_accidental=True,
            )

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

    def test_use_accidental_flag(self, concrete_c30):
        """Test use_accidental flag."""
        model = create_concrete_stress_strain(
            concrete_c30,
            "parabola-rectangle",
            use_accidental=True
        )
        assert model.f_c == concrete_c30.f_cd_accidental


class TestConcreteStressStrainAdditionalBranches:
    """Tests for TestConcreteStressStrainAdditionalBranches."""
    def test_apply_ultimate_tolerance_clip_near_band(self):
        """Test apply ultimate tolerance clip near band."""
        strains = np.array([0.0035005], dtype=float)
        clipped, killed = css_mod._apply_ultimate_tolerance_clip(
            strains=strains, epsilon_cu=0.0035, tol=0.001
        )
        assert not bool(killed[0])
        assert clipped[0] == pytest.approx(0.0035, rel=1e-12)

    def test_apply_ultimate_tolerance_clip_complex_and_zero_tol(self):
        """Test apply ultimate tolerance clip complex and zero tol."""
        complex_strains = np.array([0.001 + 1e-30j, 0.004 + 2e-30j], dtype=np.complex128)
        clipped_c, killed_c = css_mod._apply_ultimate_tolerance_clip(
            strains=complex_strains, epsilon_cu=0.0035, tol=1e-6
        )
        assert np.array_equal(clipped_c, complex_strains)
        assert not np.any(killed_c)

        real_strains = np.array([0.002, 0.004], dtype=float)
        clipped_r, killed_r = css_mod._apply_ultimate_tolerance_clip(
            strains=real_strains, epsilon_cu=0.0035, tol=0.0
        )
        assert np.array_equal(clipped_r, real_strains)
        assert np.array_equal(killed_r, np.array([False, True]))

    def test_schematic_validation_and_scalar_guards(self, concrete_c30, monkeypatch):
        """Test schematic validation and scalar guards."""
        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "f_cm", property(lambda self: 0.0))
            with pytest.raises(ValueError, match="f_cm must be > 0"):
                ConcreteStressStrainSchematic(concrete=concrete_c30)

        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "epsilon_c1", property(lambda self: 0.0))
            with pytest.raises(ValueError, match="epsilon_c1 must be non-zero"):
                ConcreteStressStrainSchematic(concrete=concrete_c30)

        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "epsilon_cu1", property(lambda self: 0.0))
            with pytest.raises(ValueError, match="epsilon_cu1 must be > 0"):
                ConcreteStressStrainSchematic(concrete=concrete_c30)

        model = ConcreteStressStrainSchematic(concrete=concrete_c30)
        # Clip to epsilon_cu when in tolerance band.
        s_clip = model.get_stress(concrete_c30.epsilon_cu1 + 0.5 * model.ultimate_strain_tol)
        s_ref = model.get_stress(concrete_c30.epsilon_cu1)
        assert s_clip == pytest.approx(s_ref, rel=1e-12)

        # Denominator singularity guard.
        with monkeypatch.context() as m:
            m.setattr(type(model), "k", property(lambda self: 1.0))
            assert model.get_stress(abs(concrete_c30.epsilon_c1)) == pytest.approx(0.0, rel=1e-12)

    def test_schematic_array_all_killed(self, concrete_c30):
        """Test schematic array all killed."""
        model = ConcreteStressStrainSchematic(concrete=concrete_c30, ultimate_strain_tol=0.0)
        strains = np.array([concrete_c30.epsilon_cu1 + 1e-3, concrete_c30.epsilon_cu1 + 2e-3])
        stresses = model.get_stress_array(strains)
        assert np.allclose(stresses, 0.0)
        assert np.allclose(model.get_stress_array(np.array([-1e-3, 0.0])), 0.0)

    def test_parabola_validation_tangent_and_complex_array(self, concrete_c30, monkeypatch):
        """Test parabola validation tangent and complex array."""
        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "epsilon_c2", property(lambda self: 0.0))
            with pytest.raises(ValueError, match="epsilon_c2 must be > 0"):
                ConcreteStressStrainParabolaRectangle(concrete=concrete_c30)

        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "epsilon_cu2", property(lambda self: 0.0))
            with pytest.raises(ValueError, match="epsilon_cu2 must be > 0"):
                ConcreteStressStrainParabolaRectangle(concrete=concrete_c30)

        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "epsilon_cu2", property(lambda self: 0.001))
            m.setattr(type(concrete_c30), "epsilon_c2", property(lambda self: 0.002))
            with pytest.raises(ValueError, match="epsilon_cu2 must be >= epsilon_c2"):
                ConcreteStressStrainParabolaRectangle(concrete=concrete_c30)

        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "f_cd", property(lambda self: -1.0))
            with pytest.raises(ValueError, match="strength f_c must be > 0"):
                ConcreteStressStrainParabolaRectangle(concrete=concrete_c30)

        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "n", property(lambda self: 0.0))
            with pytest.raises(ValueError, match="exponent n must be > 0"):
                ConcreteStressStrainParabolaRectangle(concrete=concrete_c30)

        model = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30)
        # Unconfined property branches.
        assert model.epsilon_c2_c == pytest.approx(concrete_c30.epsilon_c2, rel=1e-12)
        assert model.epsilon_cu2_c == pytest.approx(concrete_c30.epsilon_cu2, rel=1e-12)
        # Clip in scalar path (tolerance band).
        s_clip = model.get_stress(model.epsilon_cu2_eff + 0.5 * model.ultimate_strain_tol)
        assert s_clip == pytest.approx(model.f_c, rel=1e-12)

        # Complex array branch reconstruction.
        strains = np.array([0.0 + 0j, model.epsilon_c2_eff * 0.5 + 1e-30j], dtype=np.complex128)
        stresses = model.get_stress_array(strains)
        assert np.iscomplexobj(stresses)
        assert np.allclose(model.get_stress_array(np.array([-1e-3, 0.0])), 0.0)
        model_zero_tol = ConcreteStressStrainParabolaRectangle(concrete=concrete_c30, ultimate_strain_tol=0.0)
        killed = model_zero_tol.get_stress_array(
            np.array([model_zero_tol.epsilon_cu2_eff + 1e-3, model_zero_tol.epsilon_cu2_eff + 2e-3])
        )
        assert np.allclose(killed, 0.0)

        # Scalar tangent branches.
        assert model.get_tangent_modulus(-1e-4) == pytest.approx(0.0, rel=1e-12)
        assert model.get_tangent_modulus(model.epsilon_cu2_eff + 1e-3) == pytest.approx(0.0, rel=1e-12)
        assert model.get_tangent_modulus(model.epsilon_c2_eff + 1e-6) == pytest.approx(0.0, rel=1e-12)
        mid = 0.5 * model.epsilon_c2_eff
        e_t_mid = model.get_tangent_modulus(mid)
        assert e_t_mid > 0.0

        # Array tangent branches.
        et = model.get_tangent_modulus_array(np.array([-1e-4, mid, model.epsilon_c2_eff + 1e-6]))
        assert et[0] == pytest.approx(0.0, rel=1e-12)
        assert et[1] > 0.0
        assert et[2] == pytest.approx(0.0, rel=1e-12)

    def test_bilinear_validation_and_additional_paths(self, concrete_c30, monkeypatch):
        """Test bilinear validation and additional paths."""
        model_char = ConcreteStressStrainBilinear(concrete=concrete_c30, use_characteristic=True)
        assert model_char.f_c == pytest.approx(concrete_c30.f_ck, rel=1e-12)

        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "epsilon_c3", property(lambda self: 0.0))
            with pytest.raises(ValueError, match="epsilon_c3 must be > 0"):
                ConcreteStressStrainBilinear(concrete=concrete_c30)

        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "epsilon_cu3", property(lambda self: 0.0))
            with pytest.raises(ValueError, match="epsilon_cu3 must be > 0"):
                ConcreteStressStrainBilinear(concrete=concrete_c30)

        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "epsilon_cu3", property(lambda self: 0.001))
            m.setattr(type(concrete_c30), "epsilon_c3", property(lambda self: 0.002))
            with pytest.raises(ValueError, match="epsilon_cu3 must be >= epsilon_c3"):
                ConcreteStressStrainBilinear(concrete=concrete_c30)

        with monkeypatch.context() as m:
            m.setattr(type(concrete_c30), "f_cd", property(lambda self: -1.0))
            with pytest.raises(ValueError, match="strength f_c must be > 0"):
                ConcreteStressStrainBilinear(concrete=concrete_c30)

        model = ConcreteStressStrainBilinear(concrete=concrete_c30)
        s_clip = model.get_stress(concrete_c30.epsilon_cu3 + 0.5 * model.ultimate_strain_tol)
        assert s_clip == pytest.approx(model.f_c, rel=1e-12)

        # All valid compression strains killed by ultimate limit.
        model_zero_tol = ConcreteStressStrainBilinear(concrete=concrete_c30, ultimate_strain_tol=0.0)
        strains = np.array([concrete_c30.epsilon_cu3 + 1e-3, concrete_c30.epsilon_cu3 + 2e-3])
        stresses = model_zero_tol.get_stress_array(strains)
        assert np.allclose(stresses, 0.0)
        assert np.allclose(model.get_stress_array(np.array([-1e-3, 0.0])), 0.0)
        arr = np.array([0.0, concrete_c30.epsilon_c3 * 0.5, 0.5 * (concrete_c30.epsilon_c3 + concrete_c30.epsilon_cu3)])
        out = model.get_stress_array(arr)
        assert out[0] == pytest.approx(0.0, rel=1e-12)
        assert out[1] == pytest.approx(model.f_c * 0.5, rel=1e-12)
        assert out[2] == pytest.approx(model.f_c, rel=1e-12)

        assert model.get_yield_stress() == pytest.approx(model.f_c, rel=1e-12)

    def test_linear_elastic_model_branches_and_factory(self, concrete_c30):
        """Test linear elastic model branches and factory."""
        m_no_tension = ConcreteStressStrainLinearElastic(concrete=concrete_c30, include_tension=False)
        assert m_no_tension.E_mod == pytest.approx(concrete_c30.E_cm, rel=1e-12)
        assert m_no_tension.cracking_strain < 0.0
        assert m_no_tension.get_stress(1e-4) == pytest.approx(m_no_tension.E_mod * 1e-4, rel=1e-12)
        assert m_no_tension.get_stress(0.0) == pytest.approx(0.0, rel=1e-12)
        assert m_no_tension.get_stress(-1e-4) == pytest.approx(0.0, rel=1e-12)

        m_tension = ConcreteStressStrainLinearElastic(
            concrete=concrete_c30,
            elastic_modulus=20_000.0,
            include_tension=True,
        )
        assert m_tension.E_mod == pytest.approx(20_000.0, rel=1e-12)
        # In-tension below crack limit -> linear.
        in_tension = 0.5 * m_tension.cracking_strain
        assert m_tension.get_stress(in_tension) == pytest.approx(m_tension.E_mod * in_tension, rel=1e-12)
        # Beyond cracking -> zero.
        assert m_tension.get_stress(1.1 * m_tension.cracking_strain) == pytest.approx(0.0, rel=1e-12)

        arr = np.array([1e-4, -1e-4, 1.2 * m_tension.cracking_strain, 0.0])
        stresses = m_tension.get_stress_array(arr)
        assert stresses[0] > 0.0
        assert stresses[1] < 0.0
        assert stresses[2] == pytest.approx(0.0, rel=1e-12)
        assert stresses[3] == pytest.approx(0.0, rel=1e-12)

        assert m_tension.get_tangent_modulus(1e-4) == pytest.approx(m_tension.E_mod, rel=1e-12)
        assert m_tension.get_tangent_modulus(in_tension) == pytest.approx(m_tension.E_mod, rel=1e-12)
        assert m_tension.get_tangent_modulus(1.1 * m_tension.cracking_strain) == pytest.approx(0.0, rel=1e-12)

        et = m_tension.get_tangent_modulus_array(arr)
        assert et[0] == pytest.approx(m_tension.E_mod, rel=1e-12)
        assert et[1] == pytest.approx(m_tension.E_mod, rel=1e-12)
        assert et[2] == pytest.approx(0.0, rel=1e-12)
        assert m_tension.get_yield_stress() == pytest.approx(concrete_c30.f_ck, rel=1e-12)
        assert m_tension.get_ultimate_strain() == pytest.approx(0.01, rel=1e-12)

        m_factory = create_concrete_stress_strain(
            concrete=concrete_c30,
            model_type=ConcreteModelType.LINEAR_ELASTIC,
            elastic_modulus=15_000.0,
            include_tension=True,
        )
        assert isinstance(m_factory, ConcreteStressStrainLinearElastic)
        assert m_factory.E_mod == pytest.approx(15_000.0, rel=1e-12)
