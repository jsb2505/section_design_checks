"""
Tests for reinforced_concrete.constitutive.steel_stress_strain module.
"""

import numpy as np
import pytest

from materials.reinforced_concrete.constitutive import (
    SteelStressStrainEC2,
    create_steel_stress_strain,
)
from materials.reinforced_concrete.materials import ReinforcingSteel


class TestSteelStressStrainEC2:
    """Tests for SteelStressStrainEC2 class."""

    @pytest.fixture
    def model_inclined(self, steel_b500b):
        """Inclined branch model (with strain hardening)."""
        return SteelStressStrainEC2(steel=steel_b500b, branch_type="inclined")

    @pytest.fixture
    def model_horizontal(self, steel_b500b):
        """Horizontal branch model (perfectly plastic)."""
        return SteelStressStrainEC2(steel=steel_b500b, branch_type="horizontal")

    def test_create_model_inclined(self, model_inclined):
        """Test creating inclined branch model."""
        assert model_inclined.name == "EC2 Steel"
        assert model_inclined.branch_type == "inclined"

    def test_create_model_horizontal(self, model_horizontal):
        """Test creating horizontal branch model."""
        assert model_horizontal.branch_type == "horizontal"

    def test_design_vs_characteristic(self, steel_b500b):
        """Test design vs characteristic strength."""
        design_model = SteelStressStrainEC2(steel=steel_b500b, use_characteristic=False)
        char_model = SteelStressStrainEC2(steel=steel_b500b, use_characteristic=True)

        assert design_model.f_y == steel_b500b.f_yd
        assert char_model.f_y == steel_b500b.f_yk

    def test_accidental_strength(self, steel_b500b):
        """Test that model uses f_yd_accidental when use_accidental=True."""
        model_accidental = SteelStressStrainEC2(steel=steel_b500b, use_accidental=True)
        assert model_accidental.f_y == steel_b500b.f_yd_accidental
        # f_yd_accidental should be higher than f_yd (due to reduced partial factor)
        assert steel_b500b.f_yd_accidental > steel_b500b.f_yd

    def test_characteristic_tensile_strength_selection(self, steel_b500b):
        """Characteristic mode should use f_t (not design-reduced f_td)."""
        model_char = SteelStressStrainEC2(steel=steel_b500b, use_characteristic=True)
        assert model_char.f_t == pytest.approx(steel_b500b.f_t, rel=1e-12)

    def test_accidental_tensile_strength_selection(self, steel_b500b):
        """Accidental mode should use f_td_accidental for inclined branch top stress."""
        model_acc = SteelStressStrainEC2(steel=steel_b500b, use_accidental=True)
        assert model_acc.f_t == pytest.approx(steel_b500b.f_td_accidental, rel=1e-12)

    def test_cannot_use_both_characteristic_and_accidental(self, steel_b500b):
        """Test that using both flags raises an error."""
        with pytest.raises(ValueError, match="Cannot set both use_characteristic=True and use_accidental=True"):
            SteelStressStrainEC2(
                steel=steel_b500b,
                use_characteristic=True,
                use_accidental=True,
            )

    def test_inclined_branch_requires_epsilon_ud_greater_than_epsilon_y(self):
        """Validator should reject inclined model when epsilon_ud <= epsilon_y."""
        pathological = ReinforcingSteel(grade="B500B", gamma_s=0.01)
        with pytest.raises(ValueError, match="epsilon_ud .* must be > epsilon_y"):
            SteelStressStrainEC2(steel=pathological, branch_type="inclined")

    def test_yield_strain(self, model_inclined, steel_b500b):
        """Test yield strain calculation."""
        expected = steel_b500b.f_yd / steel_b500b.E_s
        assert model_inclined.epsilon_y == pytest.approx(expected, rel=1e-6)

    def test_stress_at_zero(self, model_inclined):
        """Test stress at zero strain."""
        assert model_inclined.get_stress(0.0) == 0.0

    def test_elastic_region_tension(self, model_inclined, steel_b500b):
        """Test elastic region in tension."""
        strain = model_inclined.epsilon_y / 2
        stress = model_inclined.get_stress(strain)

        # σ = E · ε
        expected = steel_b500b.E_s * strain
        assert stress == pytest.approx(expected, rel=1e-6)

    def test_stress_at_yield(self, model_inclined):
        """Test stress at yield."""
        stress = model_inclined.get_stress(model_inclined.epsilon_y)
        assert stress == pytest.approx(model_inclined.f_y, rel=1e-3)

    def test_plastic_region_inclined(self, model_inclined, steel_b500b):
        """Test plastic region with strain hardening."""
        # Mid-point between yield and ultimate
        strain = (model_inclined.epsilon_y + steel_b500b.epsilon_ud) / 2
        stress = model_inclined.get_stress(strain)

        # Should be between f_y and f_t (both using design values)
        assert model_inclined.f_y < stress < model_inclined.f_t

    def test_plastic_region_horizontal(self, model_horizontal):
        """Test plastic region for perfectly plastic."""
        strain = model_horizontal.epsilon_y * 2  # Beyond yield
        stress = model_horizontal.get_stress(strain)

        # Should equal f_y (constant)
        assert stress == pytest.approx(model_horizontal.f_y, rel=1e-6)

    def test_stress_at_ultimate_inclined(self, model_inclined, steel_b500b):
        """Test stress at ultimate strain for inclined branch."""
        stress = model_inclined.get_stress(steel_b500b.epsilon_ud)
        # Should reach f_t (design tensile strength f_td)
        assert stress == pytest.approx(model_inclined.f_t, rel=1e-3)

    def test_beyond_ultimate(self, model_inclined, steel_b500b):
        """Test stress beyond ultimate strain - maintains ultimate stress."""
        stress = model_inclined.get_stress(steel_b500b.epsilon_ud + 0.01)
        # New behavior: maintains ultimate stress f_t beyond ultimate strain
        assert stress == pytest.approx(model_inclined.f_t, rel=1e-3)

    def test_compression_elastic(self, model_inclined, steel_b500b):
        """Test elastic compression."""
        strain = -model_inclined.epsilon_y / 2
        stress = model_inclined.get_stress(strain)

        expected = steel_b500b.E_s * strain  # Negative
        assert stress == pytest.approx(expected, rel=1e-6)

    def test_compression_plastic(self, model_inclined):
        """Test plastic compression."""
        strain = -model_inclined.epsilon_y * 2  # Beyond yield in compression
        stress = model_inclined.get_stress(strain)

        # Should be negative. For inclined model, compression also has strain hardening
        # so magnitude will be slightly higher than f_y
        assert stress < 0
        assert abs(stress) >= model_inclined.f_y  # At least f_y, possibly higher due to hardening

    def test_stress_array(self, model_inclined, steel_b500b):
        """Test vectorized calculation."""
        strains = np.linspace(-0.01, 0.05, 100)
        stresses = model_inclined.get_stress_array(strains)

        assert isinstance(stresses, np.ndarray)
        assert len(stresses) == len(strains)

        # Check elastic region
        elastic_mask = np.abs(strains) <= model_inclined.epsilon_y
        expected_elastic = steel_b500b.E_s * strains[elastic_mask]
        np.testing.assert_allclose(
            stresses[elastic_mask],
            expected_elastic,
            rtol=1e-5
        )

    def test_stress_array_all_elastic_returns_early(self, model_inclined, steel_b500b):
        """Cover early return when no plastic strains are present."""
        strains = np.array([-0.5, 0.0, 0.5]) * model_inclined.epsilon_y
        stresses = model_inclined.get_stress_array(strains)
        np.testing.assert_allclose(stresses, steel_b500b.E_s * strains, rtol=1e-12)

    def test_stress_array_horizontal_plastic_branch(self, model_horizontal):
        """Cover horizontal-branch vectorized plastic path."""
        strains = np.array([0.0, model_horizontal.epsilon_y * 1.5, -model_horizontal.epsilon_y * 2.0])
        stresses = model_horizontal.get_stress_array(strains)
        assert stresses[0] == pytest.approx(0.0, rel=1e-12)
        assert stresses[1] == pytest.approx(model_horizontal.f_y, rel=1e-12)
        assert stresses[2] == pytest.approx(-model_horizontal.f_y, rel=1e-12)

    def test_stress_array_complex_step_branch(self, model_inclined):
        """Complex input should use the complex-step plastic branch (no clipping)."""
        eps_y = model_inclined.epsilon_y
        strains = np.array([eps_y * 0.5 + 1e-30j, eps_y * 1.5 + 1e-30j], dtype=np.complex128)
        stresses = model_inclined.get_stress_array(strains)
        assert np.iscomplexobj(stresses)
        # Plastic entry should remain above yield for inclined branch.
        assert np.real(stresses[1]) > model_inclined.f_y

    def test_get_stress_tension_only(self, model_inclined):
        """Test get_stress_tension_only method."""
        # Tension
        stress_tension = model_inclined.get_stress_tension_only(0.01)
        assert stress_tension > 0

        # Compression (should return 0)
        stress_compression = model_inclined.get_stress_tension_only(-0.01)
        assert stress_compression == 0.0

    def test_get_stress_compression_only(self, model_inclined):
        """Test get_stress_compression_only method."""
        # Compression
        stress_compression = model_inclined.get_stress_compression_only(-0.01)
        assert stress_compression < 0

        # Tension (should return 0)
        stress_tension = model_inclined.get_stress_compression_only(0.01)
        assert stress_tension == 0.0

    def test_get_ultimate_strain(self, model_inclined, steel_b500b):
        """Test get_ultimate_strain method."""
        assert model_inclined.get_ultimate_strain() == steel_b500b.epsilon_ud

    def test_get_yield_stress(self, model_inclined):
        """Test get_yield_stress method."""
        assert model_inclined.get_yield_stress() == model_inclined.f_y

    def test_get_tangent_modulus_scalar_branches(self, model_inclined, model_horizontal, steel_b500b):
        """Cover elastic, hardening, and clipped scalar tangent branches."""
        # Elastic branch
        assert model_inclined.get_tangent_modulus(model_inclined.epsilon_y * 0.5) == pytest.approx(
            steel_b500b.E_s, rel=1e-12
        )
        # Horizontal post-yield branch
        assert model_horizontal.get_tangent_modulus(model_horizontal.epsilon_y * 2.0) == pytest.approx(0.0, rel=1e-12)
        # Inclined hardening branch
        eps_mid = 0.5 * (model_inclined.epsilon_y + steel_b500b.epsilon_ud)
        e_hard = (model_inclined.f_t - model_inclined.f_y) / (steel_b500b.epsilon_ud - model_inclined.epsilon_y)
        assert model_inclined.get_tangent_modulus(eps_mid) == pytest.approx(e_hard, rel=1e-12)
        # Beyond ultimate (clipped flat)
        assert model_inclined.get_tangent_modulus(steel_b500b.epsilon_ud + 0.01) == pytest.approx(0.0, rel=1e-12)

    def test_get_tangent_modulus_array_branches(self, model_inclined, model_horizontal, steel_b500b):
        """Cover vectorized elastic-only, horizontal, and inclined hardening branches."""
        # Elastic-only input (exits early with no plastic values)
        elastic = np.array([-0.5, 0.0, 0.5]) * model_inclined.epsilon_y
        et_elastic = model_inclined.get_tangent_modulus_array(elastic)
        np.testing.assert_allclose(et_elastic, steel_b500b.E_s, rtol=1e-12)

        # Horizontal branch: plastic values should remain zero tangent
        horiz_vals = np.array([0.0, model_horizontal.epsilon_y * 1.5, -model_horizontal.epsilon_y * 2.0])
        et_horizontal = model_horizontal.get_tangent_modulus_array(horiz_vals)
        assert et_horizontal[0] == pytest.approx(steel_b500b.E_s, rel=1e-12)
        assert et_horizontal[1] == pytest.approx(0.0, rel=1e-12)
        assert et_horizontal[2] == pytest.approx(0.0, rel=1e-12)

        # Inclined branch: hardening in plastic-but-below-ultimate, zero beyond ultimate
        eps_mid = 0.5 * (model_inclined.epsilon_y + steel_b500b.epsilon_ud)
        vals = np.array([0.0, eps_mid, steel_b500b.epsilon_ud + 0.01])
        et_inclined = model_inclined.get_tangent_modulus_array(vals)
        e_hard = (model_inclined.f_t - model_inclined.f_y) / (steel_b500b.epsilon_ud - model_inclined.epsilon_y)
        assert et_inclined[0] == pytest.approx(steel_b500b.E_s, rel=1e-12)
        assert et_inclined[1] == pytest.approx(e_hard, rel=1e-12)
        assert et_inclined[2] == pytest.approx(0.0, rel=1e-12)


class TestCreateSteelStressStrain:
    """Tests for factory function."""

    def test_create_inclined(self, steel_b500b):
        """Test creating inclined branch model."""
        model = create_steel_stress_strain(steel_b500b, "inclined")
        assert isinstance(model, SteelStressStrainEC2)
        assert model.branch_type == "inclined"

    def test_create_horizontal(self, steel_b500b):
        """Test creating horizontal branch model."""
        model = create_steel_stress_strain(steel_b500b, "horizontal")
        assert isinstance(model, SteelStressStrainEC2)
        assert model.branch_type == "horizontal"

    def test_default_branch_type(self, steel_b500b):
        """Test default branch type."""
        model = create_steel_stress_strain(steel_b500b)
        assert model.branch_type == "inclined"

    def test_use_characteristic_flag(self, steel_b500b):
        """Test use_characteristic flag."""
        model = create_steel_stress_strain(steel_b500b, use_characteristic=True)
        assert model.f_y == steel_b500b.f_yk

    def test_use_accidental_flag(self, steel_b500b):
        """Test use_accidental flag."""
        model = create_steel_stress_strain(steel_b500b, use_accidental=True)
        assert model.f_y == steel_b500b.f_yd_accidental

    def test_comparison_inclined_vs_horizontal(self, steel_b500b):
        """Test that inclined has higher stress at large strains."""
        inclined = create_steel_stress_strain(steel_b500b, "inclined")
        horizontal = create_steel_stress_strain(steel_b500b, "horizontal")

        # At large strain (near ultimate)
        strain = steel_b500b.epsilon_ud * 0.9
        stress_inclined = inclined.get_stress(strain)
        stress_horizontal = horizontal.get_stress(strain)

        # Inclined should have strain hardening
        assert stress_inclined > stress_horizontal
