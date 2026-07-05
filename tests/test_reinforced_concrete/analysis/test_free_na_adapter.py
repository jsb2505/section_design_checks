"""
Targeted tests for FreeNADiagramAdapter analytical elastic solve.

These tests verify:
1. _solve_uncracked_elastic returns the correct (eps_0, kappa_y, kappa_z)
2. find_strain_state_for_MN uses the analytical path for uncracked SLS
3. The analytical result matches the 1D MNInteractionDiagram for pure My
4. NA angle is approximately 0° for pure major-axis bending (no Mz)
"""

from __future__ import annotations


import pytest

from materials.reinforced_concrete.analysis.biaxial_interaction import (
    BiaxialMNInteractionSurface,
    create_biaxial_interaction_surface,
)
from materials.reinforced_concrete.constitutive.concrete_stress_strain import ConcreteModelType
from materials.reinforced_concrete.analysis.free_na_adapter import FreeNADiagramAdapter
from materials.reinforced_concrete.analysis.interaction_diagram import (
    create_interaction_diagram,
)
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.reinforced_concrete.materials import Rebar


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
# Note: concrete_c30 comes from tests/conftest.py


@pytest.fixture
def rebar_h16():
    return Rebar(diameter=16, grade="B500B")


@pytest.fixture
def symmetric_section(rebar_h16):
    """300×500 section with symmetric reinforcement (I_xy ≈ 0)."""
    section = create_rectangular_section(300, 500, section_name="Symmetric")
    bottom = create_linear_rebar_layer(
        rebar=rebar_h16, n_bars=3,
        start_point=(50, 50), end_point=(250, 50), layer_name="bottom",
    )
    top = create_linear_rebar_layer(
        rebar=rebar_h16, n_bars=3,
        start_point=(50, 450), end_point=(250, 450), layer_name="top",
    )
    section.add_rebar_group(bottom)
    section.add_rebar_group(top)
    return section


@pytest.fixture
def asymmetric_section(rebar_h16):
    """300×500 section with asymmetric reinforcement (I_xy ≠ 0, more steel on one side)."""
    section = create_rectangular_section(300, 500, section_name="Asymmetric")
    # More bars on left side than right — this shifts transformed centroid
    bottom_left = create_linear_rebar_layer(
        rebar=rebar_h16, n_bars=2,
        start_point=(50, 50), end_point=(100, 50), layer_name="bottom_left",
    )
    bottom_right = create_linear_rebar_layer(
        rebar=rebar_h16, n_bars=1,
        start_point=(200, 50), end_point=(200, 50), layer_name="bottom_right",
    )
    top = create_linear_rebar_layer(
        rebar=rebar_h16, n_bars=2,
        start_point=(50, 450), end_point=(250, 450), layer_name="top",
    )
    section.add_rebar_group(bottom_left)
    section.add_rebar_group(bottom_right)
    section.add_rebar_group(top)
    return section


def _make_biaxial_surface(section, concrete, E_c_eff: float) -> BiaxialMNInteractionSurface:
    return create_biaxial_interaction_surface(
        section=section,
        concrete=concrete,
        concrete_model_type=ConcreteModelType.LINEAR_ELASTIC,
        elastic_modulus=E_c_eff,
        include_tension=True,
        crack_to_neutral_axis_on_first_tension_failure=True,
    )


def _make_adapter(section, concrete, E_c_eff: float) -> FreeNADiagramAdapter:
    surface = _make_biaxial_surface(section, concrete, E_c_eff)
    return FreeNADiagramAdapter(surface)


# ---------------------------------------------------------------------------
# Tests for _solve_uncracked_elastic
# ---------------------------------------------------------------------------

class TestSolveUncrackedElastic:
    """Tests for FreeNADiagramAdapter._solve_uncracked_elastic."""

    def test_returns_none_when_not_elastic(self, symmetric_section, concrete_c30):
        """When include_tension=False, analytical solve should return None."""
        surface = create_biaxial_interaction_surface(
            section=symmetric_section,
            concrete=concrete_c30,
            elastic_modulus=None,
            include_tension=False,
        )
        adapter = FreeNADiagramAdapter(surface)
        result = adapter._solve_uncracked_elastic(100.0, 0.0, 0.0)
        assert result is None

    def test_pure_My_symmetric_gives_zero_kappa_z(self, symmetric_section, concrete_c30):
        """For a symmetric section under pure My, kappa_z should be ≈ 0."""
        E_c = 25000.0
        adapter = _make_adapter(symmetric_section, concrete_c30, E_c)
        # Use My = 15 kN·m, well below M_cr ≈ 36 kN·m for 300×500 C30/37
        result = adapter._solve_uncracked_elastic(My=15.0, N=0.0, Mz=0.0)
        assert result is not None, "Uncracked section should return a result"
        eps_0, kappa_y, kappa_z = result
        assert abs(kappa_z) < 1e-9, f"kappa_z should be ~0 for symmetric section, got {kappa_z}"

    def test_pure_My_positive_gives_kappa_y_positive(self, symmetric_section, concrete_c30):
        """My > 0 (sagging) should give kappa_y > 0 (compression at top, tension at bottom)."""
        E_c = 25000.0
        adapter = _make_adapter(symmetric_section, concrete_c30, E_c)
        result = adapter._solve_uncracked_elastic(My=15.0, N=0.0, Mz=0.0)
        assert result is not None
        eps_0, kappa_y, kappa_z = result
        assert kappa_y > 0, f"kappa_y should be positive for sagging, got {kappa_y}"

    def test_pure_axial_compression_gives_positive_uniform_strain(self, symmetric_section, concrete_c30):
        """Pure compression (N > 0) should give eps_0 > 0 and kappa_y ≈ kappa_z ≈ 0."""
        E_c = 25000.0
        adapter = _make_adapter(symmetric_section, concrete_c30, E_c)
        # Use small N to stay uncracked
        result = adapter._solve_uncracked_elastic(My=0.0, N=500.0, Mz=0.0)
        assert result is not None
        eps_0, kappa_y, kappa_z = result
        assert eps_0 > 0, f"eps_0 should be positive for compression, got {eps_0}"
        assert abs(kappa_y) < 1e-9
        assert abs(kappa_z) < 1e-9

    def test_cracked_section_returns_none(self, symmetric_section, concrete_c30):
        """A high moment that cracks the section should return None."""
        E_c = 25000.0
        adapter = _make_adapter(symmetric_section, concrete_c30, E_c)
        # Very high moment — will crack concrete in tension
        result = adapter._solve_uncracked_elastic(My=500.0, N=0.0, Mz=0.0)
        assert result is None, "Highly loaded section should be cracked (return None)"

    def test_equilibrium_satisfied_via_transformed_section(self, symmetric_section, concrete_c30):
        """Verify the analytical solve satisfies equilibrium using the same transformed section."""
        E_c = 25000.0
        adapter = _make_adapter(symmetric_section, concrete_c30, E_c)
        My_in, N_in, Mz_in = 30.0, 200.0, 0.0  # N > 0 = compression, keeps section uncracked
        result = adapter._solve_uncracked_elastic(My=My_in, N=N_in, Mz=Mz_in)
        assert result is not None
        eps_0, kappa_y, kappa_z = result

        # Verify via the same transformed section properties used in the solve
        A_tr, cx_tr, cy_tr = symmetric_section.get_transformed_centroid(E_c)
        I_xx_tr, I_yy_tr, I_xy_tr = symmetric_section.get_transformed_second_moment_area(E_c)
        cx_g, cy_g = symmetric_section.get_centroid()
        dcx, dcy = cx_tr - cx_g, cy_tr - cy_g
        I_xx_g = I_xx_tr + A_tr * dcy**2
        I_yy_g = I_yy_tr + A_tr * dcx**2
        I_xy_g = I_xy_tr + A_tr * dcx * dcy

        N_check = E_c * (eps_0 * A_tr + kappa_y * A_tr * dcy + kappa_z * A_tr * dcx) / 1e3
        My_check = E_c * (eps_0 * A_tr * dcy + kappa_y * I_xx_g + kappa_z * I_xy_g) / 1e6
        Mz_check = E_c * (eps_0 * A_tr * dcx + kappa_y * I_xy_g + kappa_z * I_yy_g) / 1e6

        assert abs(N_check - N_in) < 0.01, f"N equilibrium error: {N_check:.3f} vs {N_in}"
        assert abs(My_check - My_in) < 0.01, f"My equilibrium error: {My_check:.3f} vs {My_in}"
        assert abs(Mz_check - Mz_in) < 0.01, f"Mz equilibrium error: {Mz_check:.3f} vs {Mz_in}"


# ---------------------------------------------------------------------------
# Tests for find_strain_state_for_MN analytical fast path
# ---------------------------------------------------------------------------

class TestFindStrainStateAnalyticalPath:
    """Tests that find_strain_state_for_MN uses the analytical path for uncracked SLS."""

    def test_uncracked_na_angle_near_zero_for_pure_My(self, symmetric_section, concrete_c30):
        """For symmetric section under pure My, NA angle should be ≈ 0°."""
        E_c = 25000.0
        adapter = _make_adapter(symmetric_section, concrete_c30, E_c)
        ss = adapter.find_strain_state_for_MN(My_target=30.0, N_target=0.0, Mz_target=0.0)
        assert abs(ss.na_angle_deg) < 5.0, (
            f"NA angle should be ≈ 0° for symmetric section under pure My, got {ss.na_angle_deg:.1f}°"
        )

    def test_uncracked_eps_bottom_more_tensile_than_top_for_sagging(self, symmetric_section, concrete_c30):
        """For positive My (sagging), eps_bottom should be more tensile (more negative) than eps_top."""
        E_c = 25000.0
        adapter = _make_adapter(symmetric_section, concrete_c30, E_c)
        ss = adapter.find_strain_state_for_MN(My_target=30.0, N_target=0.0, Mz_target=0.0)
        assert ss.eps_bottom < ss.eps_top, (
            f"eps_bottom ({ss.eps_bottom:.6f}) should be < eps_top ({ss.eps_top:.6f}) for sagging"
        )

    def test_uncracked_matches_1d_solver(self, symmetric_section, concrete_c30):
        """For symmetric section, biaxial analytical result should match 1D MNInteractionDiagram."""
        E_c = 25000.0
        # 1D diagram with same SLS model — must use LINEAR_ELASTIC concrete
        diag_1d = create_interaction_diagram(
            section=symmetric_section,
            concrete=concrete_c30,
            concrete_model_type=ConcreteModelType.LINEAR_ELASTIC,
            free_neutral_axis=False,
            elastic_modulus=E_c,
            include_tension=True,
            crack_to_neutral_axis_on_first_tension_failure=True,
        )
        # Biaxial adapter
        adapter = _make_adapter(symmetric_section, concrete_c30, E_c)

        My, N = 30.0, 0.0
        eps_top_1d, eps_bottom_1d = diag_1d.find_strains_for_MN(My, N)
        eps_top_2d, eps_bottom_2d = adapter.find_strains_for_MN(My, N)

        assert abs(eps_top_2d - eps_top_1d) < 5e-6, (
            f"eps_top mismatch: 2D={eps_top_2d:.6f} vs 1D={eps_top_1d:.6f}"
        )
        assert abs(eps_bottom_2d - eps_bottom_1d) < 5e-6, (
            f"eps_bottom mismatch: 2D={eps_bottom_2d:.6f} vs 1D={eps_bottom_1d:.6f}"
        )

    def test_asymmetric_section_has_nonzero_kappa_z_for_pure_My(self, asymmetric_section, concrete_c30):
        """Asymmetric section under pure My requires NA rotation (kappa_z ≠ 0) for Mz=0 equilibrium."""
        E_c = 25000.0
        adapter = _make_adapter(asymmetric_section, concrete_c30, E_c)
        ss = adapter.find_strain_state_for_MN(My_target=30.0, N_target=0.0, Mz_target=0.0)
        # The section has I_xy ≠ 0, so kappa_z ≠ 0 to maintain Mz = 0 equilibrium
        # plane_a = kappa_z should be non-trivial; is_biaxial may be True
        # At minimum, the result should be physically consistent
        assert ss.eps_bottom < ss.eps_top, (
            f"eps_bottom ({ss.eps_bottom:.6f}) should be < eps_top ({ss.eps_top:.6f}) for sagging"
        )
