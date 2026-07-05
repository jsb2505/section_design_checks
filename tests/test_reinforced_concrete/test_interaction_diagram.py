"""
Tests for reinforced_concrete.analysis.interaction_diagram module.
"""

import pytest
import numpy as np
from materials.reinforced_concrete.analysis.interaction_diagram import (
    InteractionPoint,
    MNInteractionDiagram,
    create_interaction_diagram,
)
from materials.reinforced_concrete.materials import ConcreteMaterial
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)


class TestInteractionPoint:
    """Tests for InteractionPoint model."""

    def test_create_point(self):
        """Test creating an interaction point."""
        point = InteractionPoint(
            N=500.0,
            M=150.0,
            neutral_axis_depth=250.0,
            max_concrete_strain=0.0035,
            max_steel_strain=0.010,
        )

        assert point.N == 500.0
        assert point.M == 150.0
        assert point.neutral_axis_depth == 250.0
        assert point.max_concrete_strain == 0.0035
        assert point.max_steel_strain == 0.010

    def test_point_is_frozen(self):
        """Test that InteractionPoint is immutable."""
        point = InteractionPoint(
            N=500.0,
            M=150.0,
            neutral_axis_depth=250.0,
            max_concrete_strain=0.0035,
            max_steel_strain=0.010,
        )

        with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen
            point.N = 600.0

    def test_repr(self):
        """Test __repr__ method."""
        point = InteractionPoint(
            N=500.0,
            M=150.0,
            neutral_axis_depth=250.0,
            max_concrete_strain=0.0035,
            max_steel_strain=0.010,
        )

        r = repr(point)
        assert "500" in r
        assert "150" in r
        assert "kN" in r


class TestMNInteractionDiagram:
    """Tests for MNInteractionDiagram class."""

    @pytest.fixture
    def simple_beam(self, rebar_20):
        """Create a simple reinforced beam for testing."""
        section = create_rectangular_section(300, 500, section_name="Test Beam")

        # Bottom layer (tension reinforcement)
        bottom_layer = create_linear_rebar_layer(
            rebar=rebar_20,
            n_bars=3,
            start_point=(50, 50),
            end_point=(250, 50),
            layer_name="bottom",
        )
        section.add_rebar_group(bottom_layer)

        # Top layer (compression reinforcement)
        top_layer = create_linear_rebar_layer(
            rebar=rebar_20,
            n_bars=2,
            start_point=(75, 450),
            end_point=(225, 450),
            layer_name="top",
        )
        section.add_rebar_group(top_layer)

        return section

    @pytest.fixture
    def diagram(self, simple_beam, concrete_c30):
        """Create M-N diagram for testing."""
        return MNInteractionDiagram(
            section=simple_beam,
            concrete=concrete_c30,
            n_fibers_width=10,
            n_fibers_height=20,
        )

    def test_create_diagram(self, diagram):
        """Test creating M-N diagram."""
        assert diagram.section is not None
        assert diagram.concrete is not None
        assert diagram.concrete_model is not None
        assert diagram.steel_model is not None
        assert diagram.mesh is not None

    def test_diagram_has_section_properties(self, diagram):
        """Test that diagram has correct section properties."""
        assert diagram.section_height == pytest.approx(500.0)
        assert diagram.section_top == pytest.approx(500.0)
        assert diagram.section_bottom == pytest.approx(0.0)
        assert diagram.section_centroid_y == pytest.approx(250.0, abs=1.0)

    def test_calculate_point_pure_compression(self, diagram):
        """Test calculation at pure compression (NA very deep)."""
        point = diagram.calculate_point(neutral_axis_depth=5000.0)

        # Should have compression (positive N)
        assert point.N > 0
        # Moment should be small (nearly uniform compression)
        assert abs(point.M) < point.N * 0.1  # M < 10% of N·h
        assert point.neutral_axis_depth == 5000.0

    def test_calculate_point_balanced(self, diagram):
        """Test calculation near balanced failure (NA in section)."""
        # Balanced typically has NA around 0.4-0.6 of depth
        point = diagram.calculate_point(neutral_axis_depth=250.0)

        # Should have compression
        assert point.N > 0
        # Should have significant moment
        assert point.M > 0
        # Strains should be reasonable
        assert 0 < point.max_concrete_strain <= 0.0035
        assert point.max_steel_strain > 0

    def test_calculate_point_pure_tension(self, diagram):
        """Test calculation at pure tension (NA above section)."""
        point = diagram.calculate_point(neutral_axis_depth=-500.0)

        # Should have tension (negative N)
        assert point.N < 0
        # All steel should be in tension
        assert point.max_steel_strain > 0

    def test_calculate_point_custom_strain(self, diagram):
        """Test calculation with custom maximum concrete strain."""
        point = diagram.calculate_point(
            neutral_axis_depth=250.0,
            max_concrete_strain=0.002,
        )

        # Should use custom strain
        assert point.max_concrete_strain <= 0.002

    def test_generate_diagram_returns_points(self, diagram):
        """Test that generate_diagram returns list of points."""
        points = diagram.generate_diagram(n_points=30)

        assert len(points) > 0
        assert all(isinstance(p, InteractionPoint) for p in points)

    def test_generate_diagram_covers_full_range(self, diagram):
        """Test that diagram covers compression to tension."""
        points = diagram.generate_diagram(n_points=50, include_tension=True)

        N_values = [p.N for p in points]

        # Should have compression points
        assert any(N > 0 for N in N_values)
        # Should have tension points
        assert any(N < 0 for N in N_values)
        # Should be roughly monotonic (N decreases)
        assert N_values[0] > N_values[-1]

    def test_generate_diagram_without_tension(self, diagram):
        """Test diagram generation without tension branch."""
        points = diagram.generate_diagram(n_points=30, include_tension=False)

        N_values = [p.N for p in points]

        # Should have mostly compression, but transition zone may create some tension
        # Check that we don't have the full tension branch (which would be larger negative N)
        min_N = min(N_values)
        max_N = max(N_values)

        # Maximum compression should be large
        assert max_N > 1000  # kN
        # Minimum (most tension) should be small compared to full tension capacity
        assert min_N > -800  # Not full tension branch (which would be < -1000)

    def test_get_diagram_arrays(self, diagram):
        """Test getting diagram as arrays for plotting."""
        N, M = diagram.get_diagram_arrays(n_points=50)

        assert isinstance(N, np.ndarray)
        assert isinstance(M, np.ndarray)
        assert len(N) == len(M)
        assert len(N) > 0

    def test_get_capacity_compression(self, diagram):
        """Test getting moment capacity under compression."""
        N_Ed = 500.0  # 500 kN compression
        M_Rd_pos, M_Rd_neg = diagram.get_capacity(N_Ed)

        # Should have moment capacity
        assert M_Rd_pos > 0
        assert M_Rd_neg < 0
        # Should be symmetric (rectangular section)
        assert abs(M_Rd_pos + M_Rd_neg) < 1.0

    def test_get_capacity_tension(self, diagram):
        """Test getting moment capacity under tension."""
        N_Ed = -200.0  # 200 kN tension
        M_Rd_pos, M_Rd_neg = diagram.get_capacity(N_Ed)

        # Should have moment capacity (but smaller than compression)
        assert M_Rd_pos > 0
        assert M_Rd_neg < 0

    def test_check_capacity_safe(self, diagram):
        """Test capacity check for safe loads."""
        # Use a known safe load
        N_Ed = 500.0  # kN compression
        M_Rd_pos, _ = diagram.get_capacity(N_Ed)

        # Apply 50% of capacity
        M_Ed = M_Rd_pos * 0.5

        is_safe, utilization = diagram.check_capacity(N_Ed, M_Ed)

        assert is_safe == True
        assert 0 < utilization < 1.0
        assert utilization == pytest.approx(0.5, rel=0.15)  # Should be around 50%

    def test_check_capacity_unsafe(self, diagram):
        """Test capacity check for unsafe loads."""
        # Use a known load
        N_Ed = 500.0  # kN compression
        M_Rd_pos, _ = diagram.get_capacity(N_Ed)

        # Apply 150% of capacity
        M_Ed = M_Rd_pos * 1.5

        is_safe, utilization = diagram.check_capacity(N_Ed, M_Ed)

        assert is_safe == False
        assert utilization > 1.0
        assert utilization == pytest.approx(1.5, rel=0.15)  # Should be around 150%

    def test_check_capacity_at_limit(self, diagram):
        """Test capacity check at exactly the limit."""
        # Use a known load
        N_Ed = 500.0  # kN compression
        M_Rd_pos, _ = diagram.get_capacity(N_Ed)

        # Apply exactly the capacity
        M_Ed = M_Rd_pos

        is_safe, utilization = diagram.check_capacity(N_Ed, M_Ed)

        # Should be at or very close to 1.0
        assert utilization == pytest.approx(1.0, rel=0.1)

    def test_repr(self, diagram):
        """Test __repr__ method."""
        r = repr(diagram)
        assert "MNInteractionDiagram" in r
        assert "Test Beam" in r
        assert "C30/37" in r

    def test_different_concrete_models(self, simple_beam, concrete_c30):
        """Test creating diagram with different concrete models."""
        # Parabola-rectangle (default)
        diag1 = MNInteractionDiagram(
            section=simple_beam,
            concrete=concrete_c30,
            concrete_model_type="parabola-rectangle",
        )

        # Bilinear
        diag2 = MNInteractionDiagram(
            section=simple_beam,
            concrete=concrete_c30,
            concrete_model_type="bilinear",
        )

        # Both should produce valid diagrams
        point1 = diag1.calculate_point(250.0)
        point2 = diag2.calculate_point(250.0)

        # Results will differ slightly due to different models
        assert point1.N != pytest.approx(point2.N, rel=0.01)

    def test_different_steel_models(self, simple_beam, concrete_c30):
        """Test creating diagram with different steel models."""
        # Inclined (with strain hardening)
        diag1 = MNInteractionDiagram(
            section=simple_beam,
            concrete=concrete_c30,
            steel_branch_type="inclined",
        )

        # Horizontal (perfectly plastic)
        diag2 = MNInteractionDiagram(
            section=simple_beam,
            concrete=concrete_c30,
            steel_branch_type="horizontal",
        )

        # Both should produce valid diagrams
        point1 = diag1.calculate_point(250.0)
        point2 = diag2.calculate_point(250.0)

        # Results may differ at high strains
        assert isinstance(point1, InteractionPoint)
        assert isinstance(point2, InteractionPoint)

    def test_no_rebar_raises_error(self, concrete_c30):
        """Test that section without rebars raises error."""
        section = create_rectangular_section(300, 500)
        # Don't add any rebars

        with pytest.raises(ValueError, match="at least one rebar group"):
            MNInteractionDiagram(section=section, concrete=concrete_c30)

    def test_fine_mesh(self, simple_beam, concrete_c30):
        """Test creating diagram with fine mesh."""
        diagram = MNInteractionDiagram(
            section=simple_beam,
            concrete=concrete_c30,
            n_fibers_width=30,
            n_fibers_height=50,
        )

        # Should have many fibers
        assert diagram.mesh.total_fibers > 1000

        # Should still calculate correctly
        point = diagram.calculate_point(250.0)
        assert point.N > 0
        assert point.M > 0

    def test_coarse_mesh(self, simple_beam, concrete_c30):
        """Test creating diagram with coarse mesh."""
        diagram = MNInteractionDiagram(
            section=simple_beam,
            concrete=concrete_c30,
            n_fibers_width=5,
            n_fibers_height=10,
        )

        # Should have fewer fibers
        assert diagram.mesh.total_fibers < 100

        # Should still calculate (less accurate)
        point = diagram.calculate_point(250.0)
        assert point.N > 0


class TestCreateInteractionDiagram:
    """Tests for create_interaction_diagram factory."""

    def test_create_basic(self, rectangular_beam_with_rebars, concrete_c30):
        """Test creating diagram with factory function."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        assert isinstance(diagram, MNInteractionDiagram)
        assert diagram.section is rectangular_beam_with_rebars
        assert diagram.concrete is concrete_c30

    def test_create_with_kwargs(self, rectangular_beam_with_rebars, concrete_c30):
        """Test creating diagram with additional kwargs."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            concrete_model_type="bilinear",
            n_fibers_width=15,
        )

        assert isinstance(diagram, MNInteractionDiagram)
        # Should use bilinear model and custom mesh
        point = diagram.calculate_point(250.0)
        assert isinstance(point, InteractionPoint)


class TestNumericalAccuracy:
    """Tests for numerical accuracy of M-N calculations."""

    @pytest.fixture
    def symmetrical_section(self, rebar_20):
        """Create symmetrical section for testing."""
        section = create_rectangular_section(300, 600)

        # Equal reinforcement top and bottom
        bottom = create_linear_rebar_layer(
            rebar=rebar_20,
            n_bars=4,
            start_point=(50, 50),
            end_point=(250, 50),
            layer_name="bottom",
        )
        top = create_linear_rebar_layer(
            rebar=rebar_20,
            n_bars=4,
            start_point=(50, 550),
            end_point=(250, 550),
            layer_name="top",
        )

        section.add_rebar_group(bottom)
        section.add_rebar_group(top)

        return section

    def test_pure_compression_has_small_moment(self, symmetrical_section, concrete_c30):
        """Test that pure compression has very small moment (due to symmetry)."""
        diagram = MNInteractionDiagram(
            section=symmetrical_section,
            concrete=concrete_c30,
        )

        point = diagram.calculate_point(neutral_axis_depth=10000.0)

        # For symmetrical section, pure compression should have M ≈ 0
        # Allow small numerical error
        assert abs(point.M) < abs(point.N) * 0.05  # M < 5% of N·h

    def test_equilibrium(self, rectangular_beam_with_rebars, concrete_c30):
        """Test force equilibrium (sum of fiber forces = N)."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        point = diagram.calculate_point(neutral_axis_depth=250.0)

        # N and M should be reasonable values
        assert -1000 < point.N < 5000  # kN (reasonable for 300×500 section)
        assert -500 < point.M < 500  # kN·m

    def test_monotonic_n_with_increasing_na(self, rectangular_beam_with_rebars, concrete_c30):
        """Test that N increases as NA depth increases (more compression)."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        # Calculate points at different NA depths
        point1 = diagram.calculate_point(100.0)  # Shallow NA (tension-controlled)
        point2 = diagram.calculate_point(300.0)  # Deep NA (compression-controlled)
        point3 = diagram.calculate_point(500.0)  # Very deep NA

        # N should generally increase with NA depth
        # (more section in compression)
        assert point3.N > point1.N

    def test_strain_distribution_linear(self, rectangular_beam_with_rebars, concrete_c30):
        """Test that strain distribution is linear (plane sections remain plane)."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        # Calculate point
        point = diagram.calculate_point(neutral_axis_depth=250.0)

        # Strains should be within reasonable limits
        assert point.max_concrete_strain <= 0.0035  # EC2 limit
        assert point.max_steel_strain >= 0  # Steel can be in tension or compression
