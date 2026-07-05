"""
Extreme case tests for analytical Jacobian implementation.

Tests cover:
1. Very high strain values (near/beyond material limits)
2. Mixed extreme strain states (one end extreme, other end normal)
3. Jacobian singularity detection
4. Numerical stability with extreme stiffness ratios
5. Cracked section behavior (tension side)
"""

import numpy as np
import pytest

from section_design_checks.core.geometry import Point2D
from section_design_checks.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from section_design_checks.reinforced_concrete.geometry import RebarGroup, create_rectangular_section
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar


@pytest.fixture
def test_diagram():
    """Create a test M-N diagram for verification."""
    # Use default hook_ref=1 (positive quadrant: 0 to 300 in x, 0 to 500 in y)
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    # Bottom bars (50mm cover from edges)
    bottom_positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    # Top bars (50mm cover from edges)
    top_positions = [Point2D(x=50, y=450), Point2D(x=250, y=450)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    concrete = ConcreteMaterial(grade="C30/37")

    return MNInteractionDiagram(
        section=section,
        concrete=concrete,
        use_characteristic=False,
        use_accidental=False,
    )


class TestAnalyticalJacobianExtremeStrains:
    """Test Jacobian computation with extreme strain values."""

    def test_both_ends_at_ultimate_compression(self, test_diagram):
        """Test Jacobian when both ends are at ultimate compression strain."""
        eps_cu = test_diagram.concrete_model.get_ultimate_strain()

        # Both ends at ultimate (pure compression at limit)
        J = test_diagram._compute_analytical_jacobian(eps_cu, eps_cu)

        # Jacobian should be well-defined
        assert J.shape == (2, 2)
        assert not np.any(np.isnan(J)), "Jacobian contains NaN"
        assert not np.any(np.isinf(J)), "Jacobian contains Inf"

        # At ultimate strain with horizontal branch, stiffness should be very low
        # (concrete E_t = 0 in rectangular region, only steel contributes)
        assert np.all(np.abs(J) < 1e10), "Jacobian values are unreasonably large"

    def test_one_end_extreme_compression(self, test_diagram):
        """Test Jacobian with one end at extreme compression, other moderate."""
        eps_cu = test_diagram.concrete_model.get_ultimate_strain()

        # Top at ultimate, bottom at moderate compression
        eps_top = eps_cu
        eps_bottom = 0.001

        J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        assert J.shape == (2, 2)
        assert not np.any(np.isnan(J)), "Jacobian contains NaN"
        assert not np.any(np.isinf(J)), "Jacobian contains Inf"

    def test_one_end_extreme_tension(self, test_diagram):
        """Test Jacobian with one end in extreme tension."""
        eps_y = test_diagram.steel_models[0].epsilon_y

        # Top compressed, bottom in high tension
        eps_top = 0.002
        eps_bottom = -eps_y * 5.0  # Well into plastic range

        J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        assert J.shape == (2, 2)
        assert not np.any(np.isnan(J)), "Jacobian contains NaN"
        assert not np.any(np.isinf(J)), "Jacobian contains Inf"

    def test_both_ends_in_tension(self, test_diagram):
        """Test Jacobian when both ends are in tension (unusual case)."""
        eps_y = test_diagram.steel_models[0].epsilon_y

        # Both ends in tension (concrete contribution minimal)
        eps_top = -eps_y * 0.5
        eps_bottom = -eps_y * 2.0

        J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        assert J.shape == (2, 2)
        assert not np.any(np.isnan(J)), "Jacobian contains NaN"
        assert not np.any(np.isinf(J)), "Jacobian contains Inf"

        # In tension-dominated case, stiffness should be relatively low
        # (only steel contributes significantly)

    def test_very_small_strain_difference(self, test_diagram):
        """Test Jacobian when strain difference is very small (near uniform)."""
        # Nearly uniform compression (small curvature)
        eps_top = 0.001000
        eps_bottom = 0.001001  # Very small difference

        J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        assert J.shape == (2, 2)
        assert not np.any(np.isnan(J)), "Jacobian contains NaN"
        assert not np.any(np.isinf(J)), "Jacobian contains Inf"


class TestAnalyticalJacobianNumericalStability:
    """Test numerical stability of Jacobian computation."""

    def test_jacobian_condition_number(self, test_diagram):
        """Test that Jacobian condition number is reasonable."""
        # Test at various strain states
        test_cases = [
            (0.001, -0.002),   # Typical bending
            (0.0015, 0.0015),  # Pure compression
            (0.002, -0.003),   # Large bending
        ]

        for eps_top, eps_bottom in test_cases:
            J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

            # Compute condition number
            cond = np.linalg.cond(J)

            # Condition number should not be excessively large
            # (< 1e10 indicates reasonable conditioning)
            assert cond < 1e10, \
                f"Jacobian poorly conditioned at eps=({eps_top}, {eps_bottom}): cond={cond}"

    def test_jacobian_determinant_nonzero(self, test_diagram):
        """Test that Jacobian determinant is non-zero (invertible)."""
        # Test at various strain states
        test_cases = [
            (0.001, -0.002),   # Typical bending
            (0.0015, 0.0015),  # Pure compression
            (0.002, -0.003),   # Large bending
            (0.0005, -0.0005), # Small strains
        ]

        for eps_top, eps_bottom in test_cases:
            J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

            det = np.linalg.det(J)

            # Determinant should be non-zero (Jacobian is invertible)
            assert abs(det) > 1e-10, \
                f"Jacobian nearly singular at eps=({eps_top}, {eps_bottom}): det={det}"

    def test_jacobian_symmetry_properties(self, test_diagram):
        """Test symmetry properties of Jacobian for symmetric loading."""
        # For pure compression (symmetric), certain symmetry should hold
        eps_uniform = 0.0015

        J = test_diagram._compute_analytical_jacobian(eps_uniform, eps_uniform)

        # For uniform strain (no curvature), both columns should be similar
        # because changing either eps_top or eps_bottom has similar effect
        J_col0 = J[:, 0]
        J_col1 = J[:, 1]

        # Columns should be close (not identical due to fiber positions)
        # but ratio should be near 1.0
        ratio = np.linalg.norm(J_col0) / np.linalg.norm(J_col1)
        assert 0.5 < ratio < 2.0, \
            f"Jacobian columns unexpectedly different for uniform strain: ratio={ratio}"


class TestAnalyticalJacobianVsNumerical:
    """Compare analytical Jacobian to numerical in extreme cases."""

    def test_extreme_compression_matches_numerical(self, test_diagram):
        """Verify analytical matches numerical for extreme compression."""
        eps_cu = test_diagram.concrete_model.get_ultimate_strain()

        eps_top = eps_cu * 0.95
        eps_bottom = eps_cu * 0.9

        # Analytical
        J_analytical = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        # Numerical (2-point finite difference)
        h = 1e-8
        def f(eps_pair):
            point = test_diagram.calculate_point_from_end_strains(eps_pair[0], eps_pair[1])
            return np.array([point.N, point.M])

        J_numerical = np.zeros((2, 2))
        for i in range(2):
            eps_plus = np.array([eps_top, eps_bottom])
            eps_plus[i] += h
            f_plus = f(eps_plus)

            eps_minus = np.array([eps_top, eps_bottom])
            eps_minus[i] -= h
            f_minus = f(eps_minus)

            J_numerical[:, i] = (f_plus - f_minus) / (2 * h)

        # Compare
        max_element = np.max(np.abs(J_analytical))
        atol = max(1.0, max_element * 0.02)  # 2% tolerance

        np.testing.assert_allclose(
            J_analytical, J_numerical, rtol=0.02, atol=atol,
            err_msg="Jacobian mismatch at extreme compression"
        )

    def test_extreme_tension_matches_numerical(self, test_diagram):
        """Verify analytical matches numerical for extreme tension."""
        eps_y = test_diagram.steel_models[0].epsilon_y

        eps_top = 0.001
        eps_bottom = -eps_y * 4.0  # High tension

        # Analytical
        J_analytical = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        # Numerical
        h = 1e-8
        def f(eps_pair):
            point = test_diagram.calculate_point_from_end_strains(eps_pair[0], eps_pair[1])
            return np.array([point.N, point.M])

        J_numerical = np.zeros((2, 2))
        for i in range(2):
            eps_plus = np.array([eps_top, eps_bottom])
            eps_plus[i] += h
            f_plus = f(eps_plus)

            eps_minus = np.array([eps_top, eps_bottom])
            eps_minus[i] -= h
            f_minus = f(eps_minus)

            J_numerical[:, i] = (f_plus - f_minus) / (2 * h)

        # Compare
        max_element = np.max(np.abs(J_analytical))
        atol = max(1.0, max_element * 0.02)

        np.testing.assert_allclose(
            J_analytical, J_numerical, rtol=0.02, atol=atol,
            err_msg="Jacobian mismatch at extreme tension"
        )


class TestAnalyticalJacobianCrackedSection:
    """Test Jacobian behavior when section is cracked (tension zone)."""

    def test_jacobian_with_cracked_tension_zone(self, test_diagram):
        """Test Jacobian when bottom half is cracked (in tension)."""
        # Large bending with tension at bottom
        eps_top = 0.002  # Compression
        eps_bottom = -0.005  # Tension (cracked concrete)

        J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        assert J.shape == (2, 2)
        assert not np.any(np.isnan(J)), "Jacobian contains NaN"
        assert not np.any(np.isinf(J)), "Jacobian contains Inf"

        # Jacobian should still be well-conditioned (steel provides stiffness)
        cond = np.linalg.cond(J)
        assert cond < 1e8, f"Jacobian poorly conditioned with cracked section: cond={cond}"

    def test_jacobian_transitions_through_cracking(self, test_diagram):
        """Test Jacobian as section transitions from uncracked to cracked."""
        eps_top = 0.001

        # Test at different bottom strains (compression → tension)
        eps_bottom_values = [0.001, 0.0, -0.001, -0.003, -0.005]

        previous_J = None
        for eps_bottom in eps_bottom_values:
            J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

            # Should be well-defined at all stages
            assert not np.any(np.isnan(J))
            assert not np.any(np.isinf(J))

            # Jacobian should change smoothly (no sudden jumps)
            if previous_J is not None:
                # Relative change should not be extreme
                rel_change = np.linalg.norm(J - previous_J) / np.linalg.norm(previous_J)
                assert rel_change < 2.0, \
                    f"Jacobian changed too abruptly at eps_bottom={eps_bottom}"

            previous_J = J.copy()


class TestAnalyticalJacobianSpecialCases:
    """Test special edge cases for Jacobian computation."""

    def test_jacobian_at_zero_strain(self, test_diagram):
        """Test Jacobian at zero strain (unloaded section)."""
        eps_top = 0.0
        eps_bottom = 0.0

        J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        assert J.shape == (2, 2)
        assert not np.any(np.isnan(J))
        assert not np.any(np.isinf(J))

        # At zero strain, should have elastic stiffness
        # Jacobian should be non-zero (section has stiffness)
        assert np.linalg.norm(J) > 0

    def test_jacobian_with_reversed_strains(self, test_diagram):
        """Test Jacobian when top strain < bottom strain (reversed gradient)."""
        # Unusual case: bottom more compressed than top
        eps_top = 0.0005
        eps_bottom = 0.002

        J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        assert J.shape == (2, 2)
        assert not np.any(np.isnan(J))
        assert not np.any(np.isinf(J))

    def test_jacobian_consistency_across_strain_range(self, test_diagram):
        """Test that Jacobian remains consistent across full strain range."""
        eps_cu = test_diagram.concrete_model.get_ultimate_strain()
        eps_y = test_diagram.steel_models[0].epsilon_y

        # Test many points across the strain space
        n_tests = 20
        all_valid = True

        for i in range(n_tests):
            # Random strains within reasonable bounds
            eps_top = np.random.uniform(-eps_y, eps_cu)
            eps_bottom = np.random.uniform(-eps_y * 2, eps_cu)

            try:
                J = test_diagram._compute_analytical_jacobian(eps_top, eps_bottom)

                # Check validity
                if np.any(np.isnan(J)) or np.any(np.isinf(J)):
                    all_valid = False
                    break

                # Check invertibility
                det = np.linalg.det(J)
                if abs(det) < 1e-12:
                    all_valid = False
                    break

            except Exception:
                all_valid = False
                break

        assert all_valid, "Jacobian failed validity checks for some strain combinations"
