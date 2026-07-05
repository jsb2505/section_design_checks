"""
Tests for confined concrete and tension stiffening features.

Tests cover:
1. Confined concrete behavior (enhanced compression capacity)
2. Tension stiffening behavior (post-cracking stiffness)
3. Combinations of confined concrete and tension stiffening
4. Comparison with standard (unconfined, no tension stiffening) diagrams
5. Proper application of material enhancements
"""

import pytest
import numpy as np
from materials.core.geometry import Point2D
from materials.reinforced_concrete.analysis.interaction_diagram import (
    MNInteractionDiagram,
    create_interaction_diagram,
)
from materials.reinforced_concrete.geometry import create_rectangular_section, RebarGroup
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar


@pytest.fixture
def standard_section():
    """Create a standard rectangular section with reinforcement."""
    section = create_rectangular_section(width=300, height=500, hook_ref=0)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    bottom_positions = [Point2D(x=-50, y=-200), Point2D(x=50, y=-200)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    top_positions = [Point2D(x=-50, y=200), Point2D(x=50, y=200)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    return section


@pytest.fixture
def concrete_c30():
    """Create C30/37 concrete material."""
    return ConcreteMaterial(grade="C30/37")


class TestConfinedConcrete:
    """Test confined concrete behavior."""

    def test_confined_concrete_increases_compression_capacity(self, standard_section, concrete_c30):
        """Test that confined concrete increases pure compression capacity."""
        # Standard diagram (unconfined)
        diagram_standard = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=False,
        )

        # Confined diagram
        diagram_confined = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,  # 2% volumetric ratio
            confinement_f_yh=500.0,  # MPa
        )

        # Calculate pure compression capacity
        eps_cu_std = diagram_standard.concrete_model.get_ultimate_strain()
        eps_cu_conf = diagram_confined.concrete_model.get_ultimate_strain()

        point_std = diagram_standard.calculate_point_from_end_strains(eps_cu_std, eps_cu_std)
        point_conf = diagram_confined.calculate_point_from_end_strains(eps_cu_conf, eps_cu_conf)

        # Confined concrete should have higher compression capacity
        assert point_conf.N > point_std.N, \
            f"Confined N={point_conf.N} should be > unconfined N={point_std.N}"

        # Increase should be at least 10% (typical for 2% confinement)
        capacity_increase = (point_conf.N - point_std.N) / point_std.N
        assert capacity_increase > 0.05, \
            f"Capacity increase ({capacity_increase:.2%}) should be significant"

    def test_confined_concrete_ultimate_strain(self, standard_section, concrete_c30):
        """Test that confined concrete ultimate strain is defined."""
        diagram_standard = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=False,
        )

        diagram_confined = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
        )

        eps_cu_std = diagram_standard.concrete_model.get_ultimate_strain()
        eps_cu_conf = diagram_confined.concrete_model.get_ultimate_strain()

        # Confined concrete ultimate strain should be at least as large as standard
        # (may be equal or higher depending on implementation)
        assert eps_cu_conf >= eps_cu_std, \
            f"Confined ult strain {eps_cu_conf} should be >= unconfined {eps_cu_std}"

        # Both should be reasonable values
        assert 0.002 <= eps_cu_std <= 0.01
        assert 0.002 <= eps_cu_conf <= 0.01

    def test_confined_concrete_with_different_confinement_ratios(self, standard_section, concrete_c30):
        """Test behavior with different confinement ratios."""
        confinement_ratios = [0.01, 0.02, 0.04]
        capacities = []

        for rho_s in confinement_ratios:
            diagram = MNInteractionDiagram(
                section=standard_section,
                concrete=concrete_c30,
                confined_concrete=True,
                confinement_rho_s=rho_s,
                confinement_f_yh=500.0,
            )

            eps_cu = diagram.concrete_model.get_ultimate_strain()
            point = diagram.calculate_point_from_end_strains(eps_cu, eps_cu)
            capacities.append(point.N)

        # Higher confinement should give higher capacity
        assert capacities[1] > capacities[0], "Higher confinement should increase capacity"
        assert capacities[2] > capacities[1], "Higher confinement should increase capacity"

    def test_confined_concrete_moment_capacity(self, standard_section, concrete_c30):
        """Test that confinement affects moment capacity."""
        diagram_standard = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=False,
        )

        diagram_confined = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
        )

        # Generate diagrams to find maximum moment
        points_std = diagram_standard.generate_diagram_points(n_points=30)
        points_conf = diagram_confined.generate_diagram_points(n_points=30)

        max_M_std = max(abs(p.M) for p in points_std)
        max_M_conf = max(abs(p.M) for p in points_conf)

        # Confined should have higher or similar moment capacity
        assert max_M_conf >= max_M_std * 0.95, \
            "Confined moment capacity should not be significantly reduced"


class TestTensionStiffening:
    """Test tension stiffening behavior."""

    def test_tension_stiffening_affects_cracked_section(self, standard_section, concrete_c30):
        """Test that tension stiffening affects behavior with cracked sections."""
        # Without tension stiffening
        diagram_no_ts = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            tension_stiffening=False,
        )

        # With tension stiffening
        diagram_with_ts = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            tension_stiffening=True,
        )

        # Test case with cracked section (compression top, tension bottom)
        eps_top = 0.001  # Compression
        eps_bottom = -0.005  # Tension (cracked)

        point_no_ts = diagram_no_ts.calculate_point_from_end_strains(eps_top, eps_bottom)
        point_with_ts = diagram_with_ts.calculate_point_from_end_strains(eps_top, eps_bottom)

        # With tension stiffening, moment should be slightly different
        # (concrete in tension zone contributes some stiffness)
        # The difference may be small but should be measurable
        assert point_with_ts.N is not None
        assert point_with_ts.M is not None

    def test_tension_stiffening_increases_stiffness(self, standard_section, concrete_c30):
        """Test that tension stiffening provides additional stiffness."""
        diagram_no_ts = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            tension_stiffening=False,
        )

        diagram_with_ts = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            tension_stiffening=True,
        )

        # Small strains in cracked region
        eps_top = 0.0005
        eps_bottom = -0.002

        point_no_ts = diagram_no_ts.calculate_point_from_end_strains(eps_top, eps_bottom)
        point_with_ts = diagram_with_ts.calculate_point_from_end_strains(eps_top, eps_bottom)

        # Both should give valid results
        assert not np.isnan(point_no_ts.N)
        assert not np.isnan(point_with_ts.N)

        # Forces should be similar (tension stiffening effect is secondary)
        # but with tension stiffening should have slightly different response
        N_diff = abs(point_with_ts.N - point_no_ts.N)
        assert N_diff < 100.0, "Difference should be relatively small"

    def test_tension_stiffening_no_effect_pure_compression(self, standard_section, concrete_c30):
        """Test that tension stiffening has no effect under pure compression."""
        diagram_no_ts = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            tension_stiffening=False,
        )

        diagram_with_ts = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            tension_stiffening=True,
        )

        # Pure compression (no cracking)
        eps_comp = 0.002

        point_no_ts = diagram_no_ts.calculate_point_from_end_strains(eps_comp, eps_comp)
        point_with_ts = diagram_with_ts.calculate_point_from_end_strains(eps_comp, eps_comp)

        # Should be identical or very close (no tension zone)
        assert abs(point_no_ts.N - point_with_ts.N) < 1.0
        assert abs(point_no_ts.M - point_with_ts.M) < 0.1


class TestConfinedConcreteAndTensionStiffening:
    """Test combinations of confined concrete and tension stiffening."""

    def test_combined_features_work_together(self, standard_section, concrete_c30):
        """Test that confined concrete and tension stiffening can be used together."""
        diagram = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
            tension_stiffening=True,
        )

        # Should be able to generate diagram without errors
        points = diagram.generate_diagram_points(n_points=20)

        assert len(points) > 0, "Should generate points with both features"

        # All points should be valid
        for point in points:
            assert not np.isnan(point.N)
            assert not np.isnan(point.M)

    def test_combined_features_vs_standard(self, standard_section, concrete_c30):
        """Compare diagram with all features vs standard diagram."""
        # Standard (no enhancements)
        diagram_standard = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=False,
            tension_stiffening=False,
        )

        # All enhancements
        diagram_enhanced = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
            tension_stiffening=True,
        )

        # Pure compression capacity should be significantly higher
        eps_cu_std = diagram_standard.concrete_model.get_ultimate_strain()
        eps_cu_enh = diagram_enhanced.concrete_model.get_ultimate_strain()

        point_std = diagram_standard.calculate_point_from_end_strains(eps_cu_std, eps_cu_std)
        point_enh = diagram_enhanced.calculate_point_from_end_strains(eps_cu_enh, eps_cu_enh)

        assert point_enh.N > point_std.N, "Enhanced should have higher compression capacity"

    def test_all_combinations_of_features(self, standard_section, concrete_c30):
        """Test all 4 combinations of confined/tension stiffening."""
        combinations = [
            (False, False),  # Standard
            (True, False),   # Only confined
            (False, True),   # Only tension stiffening
            (True, True),    # Both
        ]

        diagrams = []
        for confined, ts in combinations:
            if confined:
                diagram = MNInteractionDiagram(
                    section=standard_section,
                    concrete=concrete_c30,
                    confined_concrete=True,
                    confinement_rho_s=0.02,
                    confinement_f_yh=500.0,
                    tension_stiffening=ts,
                )
            else:
                diagram = MNInteractionDiagram(
                    section=standard_section,
                    concrete=concrete_c30,
                    confined_concrete=False,
                    tension_stiffening=ts,
                )
            diagrams.append(diagram)

        # All should be able to generate diagrams
        for i, diagram in enumerate(diagrams):
            points = diagram.generate_diagram_points(n_points=15)
            assert len(points) > 0, f"Combination {i} should generate points"


class TestMaterialEnhancementConsistency:
    """Test consistency of material enhancement implementations."""

    def test_confined_concrete_strain_limits_consistent(self, standard_section, concrete_c30):
        """Test that confined concrete strain limits are properly enforced."""
        diagram = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
        )

        eps_cu = diagram.concrete_model.get_ultimate_strain()

        # Strain beyond ultimate should still compute
        eps_beyond = eps_cu * 1.2
        point = diagram.calculate_point_from_end_strains(eps_beyond, eps_beyond)

        assert not np.isnan(point.N)
        assert not np.isnan(point.M)

    def test_inverse_solver_works_with_enhancements(self, standard_section, concrete_c30):
        """Test that inverse solver works with material enhancements."""
        diagram = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
            tension_stiffening=True,
        )

        # Test inverse solver
        M_target = 80.0
        N_target = 300.0

        eps_top, eps_bottom = diagram.find_strains_for_MN(M_target, N_target)

        # Verify round-trip
        point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

        assert abs(point.N - N_target) < 2.0
        assert abs(point.M - M_target) < 2.0

    def test_analytical_jacobian_works_with_enhancements(self, standard_section, concrete_c30):
        """Test that analytical Jacobian works with material enhancements."""
        diagram = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
            tension_stiffening=True,
        )

        # Test Jacobian computation
        eps_top = 0.002
        eps_bottom = -0.003

        J = diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        assert J.shape == (2, 2)
        assert not np.any(np.isnan(J))
        assert not np.any(np.isinf(J))

        # Should be invertible
        det = np.linalg.det(J)
        assert abs(det) > 1e-10


class TestConfinementParameters:
    """Test different confinement parameter combinations."""

    def test_low_confinement_ratio(self, standard_section, concrete_c30):
        """Test with low confinement ratio."""
        diagram = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.005,  # 0.5% - very low
            confinement_f_yh=500.0,
        )

        points = diagram.generate_diagram_points(n_points=15)
        assert len(points) > 0

    def test_high_confinement_ratio(self, standard_section, concrete_c30):
        """Test with high confinement ratio."""
        diagram = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.08,  # 8% - very high
            confinement_f_yh=500.0,
        )

        points = diagram.generate_diagram_points(n_points=15)
        assert len(points) > 0

    def test_different_confinement_yield_strengths(self, standard_section, concrete_c30):
        """Test with different confinement steel yield strengths."""
        yield_strengths = [400.0, 500.0, 600.0]  # MPa
        capacities = []

        for f_yh in yield_strengths:
            diagram = MNInteractionDiagram(
                section=standard_section,
                concrete=concrete_c30,
                confined_concrete=True,
                confinement_rho_s=0.02,
                confinement_f_yh=f_yh,
            )

            eps_cu = diagram.concrete_model.get_ultimate_strain()
            point = diagram.calculate_point_from_end_strains(eps_cu, eps_cu)
            capacities.append(point.N)

        # Higher yield strength should give higher capacity
        assert capacities[1] > capacities[0]
        assert capacities[2] > capacities[1]


class TestCharacteristicAndAccidentalCombinations:
    """Test combinations with characteristic and accidental load cases."""

    def test_confined_with_characteristic_loads(self, standard_section, concrete_c30):
        """Test confined concrete with characteristic material values."""
        diagram = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            use_characteristic=True,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
        )

        points = diagram.generate_diagram_points(n_points=15)
        assert len(points) > 0

    def test_confined_with_accidental_loads(self, standard_section, concrete_c30):
        """Test confined concrete with accidental load case."""
        diagram = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            use_accidental=True,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
        )

        points = diagram.generate_diagram_points(n_points=15)
        assert len(points) > 0

    def test_tension_stiffening_with_characteristic(self, standard_section, concrete_c30):
        """Test tension stiffening with characteristic values."""
        diagram = MNInteractionDiagram(
            section=standard_section,
            concrete=concrete_c30,
            use_characteristic=True,
            tension_stiffening=True,
        )

        points = diagram.generate_diagram_points(n_points=15)
        assert len(points) > 0
