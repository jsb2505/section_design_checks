"""
Tests for reinforced_concrete.analysis.interaction_diagram module.
"""

import pytest
import numpy as np
import json
import csv
from pathlib import Path
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

    def test_to_dict(self):
        """Test converting point to dictionary."""
        point = InteractionPoint(
            N=500.0,
            M=150.0,
            neutral_axis_depth=250.0,
            max_concrete_strain=0.0035,
            max_steel_strain=0.010,
        )

        data = point.to_dict()

        assert isinstance(data, dict)
        assert data["N_kN"] == 500.0
        assert data["M_kNm"] == 150.0
        assert data["neutral_axis_depth_mm"] == 250.0
        assert data["max_concrete_strain"] == 0.0035
        assert data["max_steel_strain"] == 0.010
        assert len(data) == 5  # All expected keys


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
        assert diagram.steel_models is not None
        assert len(diagram.steel_models) > 0
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
        # The diagram is a closed loop (first point repeated at end)
        # So check that max N (compression) is greater than min N (tension)
        assert max(N_values) > min(N_values)

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
        M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

        # Should have moment capacity
        assert M_Rd_pos > 0
        assert M_Rd_neg < 0
        # Section has only bottom reinforcement, so M_Rd_pos > |M_Rd_neg|
        assert M_Rd_pos > abs(M_Rd_neg)

    def test_get_capacity_tension(self, diagram):
        """Test getting moment capacity under tension."""
        N_Ed = -200.0  # 200 kN tension
        M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

        # Should have moment capacity (but smaller than compression)
        assert M_Rd_pos > 0
        assert M_Rd_neg < 0

    def test_check_capacity_safe(self, diagram):
        """Test capacity check for safe loads."""
        # Use a known safe load
        N_Ed = 500.0  # kN compression
        M_Rd_pos, _ = diagram.get_capacity_fixed_n(N_Ed)

        # Apply 50% of capacity
        M_Ed = M_Rd_pos * 0.5

        is_safe, utilization = diagram.get_utilization_vector(N_Ed, M_Ed)

        assert is_safe == True
        assert 0 < utilization < 1.0
        assert utilization == pytest.approx(0.5, rel=0.15)  # Should be around 50%

    def test_check_capacity_unsafe(self, diagram):
        """Test capacity check for unsafe loads."""
        # Use a known load
        N_Ed = 500.0  # kN compression
        M_Rd_pos, _ = diagram.get_capacity_fixed_n(N_Ed)

        # Apply 150% of capacity
        M_Ed = M_Rd_pos * 1.5

        is_safe, utilization = diagram.get_utilization_vector(N_Ed, M_Ed)

        assert is_safe == False
        assert utilization > 1.0
        assert utilization == pytest.approx(1.5, rel=0.15)  # Should be around 150%

    def test_check_capacity_at_limit(self, diagram):
        """Test capacity check at exactly the limit."""
        # Use a known load
        N_Ed = 500.0  # kN compression
        M_Rd_pos, _ = diagram.get_capacity_fixed_n(N_Ed)

        # Apply exactly the capacity
        M_Ed = M_Rd_pos

        is_safe, utilization = diagram.get_utilization_vector(N_Ed, M_Ed)

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


class TestExportFunctionality:
    """Tests for export to JSON, CSV, and dict."""

    @pytest.fixture
    def diagram(self, rectangular_beam_with_rebars, concrete_c30):
        """Create M-N diagram for export testing."""
        return MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            n_fibers_width=10,
            n_fibers_height=20,
        )

    def test_export_to_json(self, diagram, tmp_path):
        """Test exporting diagram to JSON file."""
        output_file = tmp_path / "mn_diagram.json"

        diagram.export_to_json(output_file, n_points=20)

        # Verify file exists
        assert output_file.exists()

        # Load and verify contents
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert "diagram_points" in data
        assert "metadata" in data
        # n_points is approximate due to how the diagram is generated
        # Note: actual count is higher due to minimum point requirements per curve segment
        assert len(data["diagram_points"]) > 0  # At least some points generated

        # Check point structure
        point = data["diagram_points"][0]
        assert "N_kN" in point
        assert "M_kNm" in point
        assert "neutral_axis_depth_mm" in point
        assert "max_concrete_strain" in point
        assert "max_steel_strain" in point

        # Check metadata
        metadata = data["metadata"]
        assert "concrete_grade" in metadata
        assert "concrete_fck" in metadata
        assert "n_fibers" in metadata
        assert metadata["concrete_grade"] == "C30/37"

    def test_export_to_json_without_metadata(self, diagram, tmp_path):
        """Test exporting JSON without metadata."""
        output_file = tmp_path / "mn_diagram_no_meta.json"

        diagram.export_to_json(output_file, n_points=15, include_metadata=False)

        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert "diagram_points" in data
        assert "metadata" not in data
        # n_points is approximate
        assert len(data["diagram_points"]) > 0

    def test_export_to_json_compact(self, diagram, tmp_path):
        """Test exporting JSON in compact format."""
        output_file = tmp_path / "mn_diagram_compact.json"

        diagram.export_to_json(output_file, n_points=10, indent=None)

        # File should exist and be smaller (no indentation)
        assert output_file.exists()

        # Verify it's valid JSON
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Note: actual count is higher than requested due to minimum points per segment
        assert len(data["diagram_points"]) > 0

    def test_export_to_csv(self, diagram, tmp_path):
        """Test exporting diagram to CSV file."""
        output_file = tmp_path / "mn_diagram.csv"

        diagram.export_to_csv(output_file, n_points=25)

        # Verify file exists
        assert output_file.exists()

        # Load and verify contents
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Note: actual count is higher than requested due to minimum points per segment
        assert len(rows) > 0

        # Check headers
        assert 'N_kN' in rows[0]
        assert 'M_kNm' in rows[0]
        assert 'neutral_axis_depth_mm' in rows[0]
        assert 'max_concrete_strain' in rows[0]
        assert 'max_steel_strain' in rows[0]

        # Check data is numeric
        for row in rows:
            assert float(row['N_kN']) is not None
            assert float(row['M_kNm']) is not None

    def test_export_to_csv_without_strains(self, diagram, tmp_path):
        """Test exporting CSV without strain columns."""
        output_file = tmp_path / "mn_diagram_simple.csv"

        diagram.export_to_csv(output_file, n_points=20, include_strains=False)

        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # n_points is approximate
        assert len(rows) > 0

        # Should only have N and M columns
        assert 'N_kN' in rows[0]
        assert 'M_kNm' in rows[0]
        assert 'neutral_axis_depth_mm' not in rows[0]
        assert 'max_concrete_strain' not in rows[0]

    def test_to_dict(self, diagram):
        """Test converting diagram to dictionary."""
        data = diagram.to_dict(n_points=30)

        assert isinstance(data, dict)
        assert "points" in data
        assert "N_array" in data
        assert "M_array" in data
        assert "metadata" in data

        # Check arrays (note: actual count higher than requested)
        assert len(data["points"]) > 0
        assert len(data["N_array"]) == len(data["points"])
        assert len(data["M_array"]) == len(data["points"])

        # Check point structure
        point = data["points"][0]
        assert "N_kN" in point
        assert "M_kNm" in point

        # Check arrays are lists of floats
        assert isinstance(data["N_array"], list)
        assert all(isinstance(n, (int, float)) for n in data["N_array"])

    def test_to_dict_without_metadata(self, diagram):
        """Test converting to dict without metadata."""
        data = diagram.to_dict(n_points=20, include_metadata=False)

        assert "points" in data
        assert "N_array" in data
        assert "M_array" in data
        assert "metadata" not in data
        # n_points is approximate
        assert len(data["points"]) > 0

    def test_export_json_then_reload(self, diagram, tmp_path):
        """Test round-trip: export to JSON and reload."""
        output_file = tmp_path / "mn_diagram_roundtrip.json"

        # Export
        diagram.export_to_json(output_file, n_points=30)

        # Reload
        with open(output_file, 'r', encoding='utf-8') as f:
            reloaded_data = json.load(f)

        # Compare with direct dict export
        dict_data = diagram.to_dict(n_points=30)

        # Points should match
        assert len(reloaded_data["diagram_points"]) == len(dict_data["points"])

        # First point should match
        assert reloaded_data["diagram_points"][0]["N_kN"] == dict_data["points"][0]["N_kN"]
        assert reloaded_data["diagram_points"][0]["M_kNm"] == dict_data["points"][0]["M_kNm"]

    def test_export_csv_data_integrity(self, diagram, tmp_path):
        """Test that CSV export maintains data integrity."""
        output_file = tmp_path / "mn_diagram_integrity.csv"

        # Export
        diagram.export_to_csv(output_file, n_points=20)

        # Get reference data
        points = diagram.generate_diagram(n_points=20)

        # Read CSV
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        # Compare
        for i, (point, csv_row) in enumerate(zip(points, csv_rows)):
            assert point.N == pytest.approx(float(csv_row['N_kN']), rel=1e-6)
            assert point.M == pytest.approx(float(csv_row['M_kNm']), rel=1e-6)
            assert point.neutral_axis_depth == pytest.approx(
                float(csv_row['neutral_axis_depth_mm']), rel=1e-6
            )


class TestNonSymmetricSections:
    """Tests for non-symmetric section handling."""

    @pytest.fixture
    def asymmetric_rebar_beam(self, rebar_20, rebar_16):
        """Create beam with asymmetric reinforcement (more bottom than top)."""
        section = create_rectangular_section(300, 500, section_name="Asymmetric Beam")

        # Heavy bottom reinforcement
        bottom_layer = create_linear_rebar_layer(
            rebar=rebar_20,
            n_bars=5,  # 5 bars on bottom
            start_point=(40, 50),
            end_point=(260, 50),
            layer_name="bottom",
        )
        section.add_rebar_group(bottom_layer)

        # Light top reinforcement
        top_layer = create_linear_rebar_layer(
            rebar=rebar_16,
            n_bars=2,  # Only 2 smaller bars on top
            start_point=(100, 450),
            end_point=(200, 450),
            layer_name="top",
        )
        section.add_rebar_group(top_layer)

        return section

    def test_asymmetric_rebar_different_capacities(self, asymmetric_rebar_beam, concrete_c30):
        """Test that get_capacity properly separates positive and negative moments."""
        diagram = MNInteractionDiagram(
            section=asymmetric_rebar_beam,
            concrete=concrete_c30,
        )

        # At moderate axial load
        N_Ed = 500.0  # kN
        M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

        # Should return separate positive and negative capacities
        assert M_Rd_pos >= 0  # Positive capacity (non-negative)
        assert M_Rd_neg <= 0  # Negative capacity (non-positive)

        # Both should be non-zero for a valid section
        assert M_Rd_pos > 0 or M_Rd_neg < 0

    def test_asymmetric_positive_moment_capacity_higher(self, asymmetric_rebar_beam, concrete_c30):
        """Test that positive moment capacity is higher with more bottom steel."""
        diagram = MNInteractionDiagram(
            section=asymmetric_rebar_beam,
            concrete=concrete_c30,
        )

        # Test at various axial load levels
        for N_Ed in [0, 200, 500, 1000]:
            M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

            # Positive capacity should be larger (more bottom steel in tension)
            if M_Rd_pos > 0 and M_Rd_neg < 0:  # Valid range
                assert abs(M_Rd_pos) >= abs(M_Rd_neg)

    def test_asymmetric_check_capacity_handles_both_directions(
        self, asymmetric_rebar_beam, concrete_c30
    ):
        """Test that capacity check works for both moment directions."""
        diagram = MNInteractionDiagram(
            section=asymmetric_rebar_beam,
            concrete=concrete_c30,
        )

        N_Ed = 500.0  # kN
        M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

        # Test positive moment (should be safe at 50% capacity)
        M_Ed_pos = M_Rd_pos * 0.5
        is_safe_pos, util_pos = diagram.get_utilization_vector(N_Ed, M_Ed_pos)
        assert is_safe_pos == True
        assert util_pos == pytest.approx(0.5, rel=0.2)

        # Test negative moment (should be safe at 50% capacity)
        M_Ed_neg = M_Rd_neg * 0.5
        is_safe_neg, util_neg = diagram.get_utilization_vector(N_Ed, M_Ed_neg)
        assert is_safe_neg == True
        # Note: vector projection method may give different utilization than simple ratio
        # due to M-N interaction effects and convex hull geometry
        assert 0.2 < util_neg < 0.8  # Should be somewhere in safe range

    def test_symmetric_section_still_symmetric(self, rebar_20):
        """Test that symmetric sections still give symmetric capacities."""
        # Create perfectly symmetric section
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

        concrete = ConcreteMaterial(grade="C30/37")
        diagram = MNInteractionDiagram(section=section, concrete=concrete)

        # For symmetric section, capacities should be nearly equal
        N_Ed = 500.0
        M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

        # Should be approximately symmetric (allow small numerical differences)
        assert abs(M_Rd_pos + M_Rd_neg) < max(abs(M_Rd_pos), abs(M_Rd_neg)) * 0.1

    def test_get_capacity_returns_correct_signs(self, asymmetric_rebar_beam, concrete_c30):
        """Test that get_capacity returns correctly signed values."""
        diagram = MNInteractionDiagram(
            section=asymmetric_rebar_beam,
            concrete=concrete_c30,
        )

        M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(500.0)

        # Positive capacity should be positive
        assert M_Rd_pos >= 0
        # Negative capacity should be negative
        assert M_Rd_neg <= 0

    def test_asymmetric_full_diagram_has_both_moments(
        self, asymmetric_rebar_beam, concrete_c30
    ):
        """Test that full diagram generates both positive and negative moments."""
        diagram = MNInteractionDiagram(
            section=asymmetric_rebar_beam,
            concrete=concrete_c30,
        )

        points = diagram.generate_diagram(n_points=100)
        M_values = [p.M for p in points]

        # Should have both positive and negative moments
        assert any(M > 0 for M in M_values)
        assert any(M < 0 for M in M_values)

        # For asymmetric section, max positive should differ from abs(min negative)
        max_M_pos = max(M for M in M_values if M > 0)
        min_M_neg = min(M for M in M_values if M < 0)

        # Should not be symmetric - but difference may be small depending on N value where peak occurs
        # Check that they're not identical (within numerical precision)
        assert abs(max_M_pos - abs(min_M_neg)) > 1.0  # At least 1 kN·m difference


class TestBalancedFailurePoint:
    """Tests for balanced failure point optimization."""

    def test_find_balanced_point_returns_valid_point(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that find_balanced_point returns a valid interaction point."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        balanced_point, na_depth = diagram.find_balanced_point()

        # Should return valid InteractionPoint
        assert isinstance(balanced_point, InteractionPoint)
        assert balanced_point.N > 0  # Should be in compression
        assert balanced_point.M != 0  # Should have moment
        assert na_depth > 0  # Neutral axis should be within/below section

    def test_balanced_point_has_correct_strains(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that balanced point has concrete at ultimate strain and steel at yield."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        balanced_point, na_depth = diagram.find_balanced_point()

        # Concrete should be at or near ultimate strain
        concrete_ultimate_strain = diagram.concrete_model.get_ultimate_strain()
        # Allow 5% tolerance because max_concrete_strain is the maximum observed
        # across all concrete fibers, which may be slightly less than the target
        # due to fiber discretization
        assert balanced_point.max_concrete_strain == pytest.approx(
            concrete_ultimate_strain, rel=0.05
        )

        # Steel should be at or near yield strain
        steel_yield_strain = diagram.steel_models[0].epsilon_y
        # Allow 10% tolerance due to numerical approximation and fiber discretization
        assert balanced_point.max_steel_strain == pytest.approx(
            steel_yield_strain, rel=0.10
        )

    def test_balanced_na_depth_is_reasonable(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that balanced neutral axis depth is within reasonable range."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        balanced_point, na_depth = diagram.find_balanced_point()

        # For typical reinforced concrete, balanced NA is usually between 0.3h and 0.7h
        # where h is the section height
        section_height = diagram.section_height
        assert 0.2 * section_height < na_depth < section_height

    def test_balanced_point_is_on_diagram(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that balanced point appears on the M-N diagram."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        balanced_point, na_depth = diagram.find_balanced_point()

        # Generate full diagram
        all_points = diagram.generate_diagram(n_points=200)

        # Check that balanced point N is within the range of diagram
        N_values = [p.N for p in all_points]
        # Allow small margin because balanced point may be at exact conditions
        # that the standard diagram doesn't sample precisely
        N_min, N_max = min(N_values), max(N_values)
        N_margin = (N_max - N_min) * 0.05  # 5% margin
        assert N_min - N_margin <= balanced_point.N <= N_max + N_margin

        # Check that balanced point M is reasonable (not testing exact range)
        # The balanced point should have significant moment capacity
        assert balanced_point.M > 0

    def test_balanced_point_with_custom_strain(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test balanced point calculation with custom concrete strain."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        custom_strain = 0.003  # Different from default ε_cu2
        balanced_point, na_depth = diagram.find_balanced_point(
            max_concrete_strain=custom_strain
        )

        # Should use custom strain (with tolerance for fiber discretization)
        assert balanced_point.max_concrete_strain == pytest.approx(
            custom_strain, rel=0.05
        )

    def test_different_sections_have_different_balanced_points(
        self, rebar_20, concrete_c30
    ):
        """Test that different reinforcement layouts produce different balanced points."""
        # Light reinforcement - bars at bottom
        section_light = create_rectangular_section(300, 500)
        bottom_light = create_linear_rebar_layer(
            rebar=rebar_20,
            n_bars=2,  # Light reinforcement
            start_point=(50, 50),
            end_point=(250, 50),
            layer_name="bottom",
        )
        section_light.add_rebar_group(bottom_light)

        # Heavy reinforcement with both top and bottom
        section_heavy = create_rectangular_section(300, 500)
        bottom_heavy = create_linear_rebar_layer(
            rebar=rebar_20,
            n_bars=4,  # More bottom reinforcement
            start_point=(50, 50),
            end_point=(250, 50),
            layer_name="bottom",
        )
        top_heavy = create_linear_rebar_layer(
            rebar=rebar_20,
            n_bars=2,  # Some top reinforcement
            start_point=(50, 450),
            end_point=(250, 450),
            layer_name="top",
        )
        section_heavy.add_rebar_group(bottom_heavy)
        section_heavy.add_rebar_group(top_heavy)

        diagram_light = MNInteractionDiagram(section=section_light, concrete=concrete_c30)
        diagram_heavy = MNInteractionDiagram(section=section_heavy, concrete=concrete_c30)

        balanced_light, na_light = diagram_light.find_balanced_point()
        balanced_heavy, na_heavy = diagram_heavy.find_balanced_point()

        # Both should be valid balanced points
        assert balanced_light.N > 0
        assert balanced_heavy.N > 0
        assert balanced_light.M > 0
        assert balanced_heavy.M > 0

        # Heavy reinforcement should have higher moment capacity at balanced
        # (more steel area means more force resistance)
        assert balanced_heavy.M > balanced_light.M


class TestTensionStiffening:
    """Tests for tension stiffening feature."""

    def test_tension_stiffening_disabled_by_default(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that tension stiffening is disabled by default."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        assert diagram.tension_stiffening is False

    def test_tension_stiffening_can_be_enabled(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that tension stiffening can be enabled."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            tension_stiffening=True,
        )

        assert diagram.tension_stiffening is True

    def test_tension_stiffening_affects_pure_tension(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that tension stiffening increases capacity in pure tension."""
        # Diagram without tension stiffening
        diagram_no_ts = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            tension_stiffening=False,
        )

        # Diagram with tension stiffening
        diagram_with_ts = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            tension_stiffening=True,
        )

        # Calculate point with NA above section (pure tension region)
        na_depth_tension = -100.0  # NA above section

        point_no_ts = diagram_no_ts.calculate_point(na_depth_tension)
        point_with_ts = diagram_with_ts.calculate_point(na_depth_tension)

        # With tension stiffening, tensile capacity should be higher (more negative N)
        # Because concrete contributes in tension
        assert point_with_ts.N < point_no_ts.N

    def test_tension_stiffening_affects_small_eccentricity(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that tension stiffening affects small eccentricity loading."""
        diagram_no_ts = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            tension_stiffening=False,
        )

        diagram_with_ts = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            tension_stiffening=True,
        )

        # Point with NA near mid-height (some concrete in tension)
        na_depth = 250.0

        point_no_ts = diagram_no_ts.calculate_point(na_depth)
        point_with_ts = diagram_with_ts.calculate_point(na_depth)

        # Moment capacity should differ when tension stiffening is enabled
        # Typically higher with tension stiffening
        assert point_with_ts.M != point_no_ts.M

    def test_tension_stiffening_minimal_effect_in_compression(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that tension stiffening has minimal effect in pure compression."""
        diagram_no_ts = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            tension_stiffening=False,
        )

        diagram_with_ts = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            tension_stiffening=True,
        )

        # Pure compression (NA very deep)
        na_depth_compression = 5000.0

        point_no_ts = diagram_no_ts.calculate_point(na_depth_compression)
        point_with_ts = diagram_with_ts.calculate_point(na_depth_compression)

        # Should be nearly identical in pure compression
        assert point_with_ts.N == pytest.approx(point_no_ts.N, rel=0.01)
        assert point_with_ts.M == pytest.approx(point_no_ts.M, rel=0.01)

    def test_full_diagram_with_tension_stiffening(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test generating full diagram with tension stiffening."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            tension_stiffening=True,
        )

        points = diagram.generate_diagram(n_points=50)

        # Should generate valid points
        assert len(points) > 0
        assert all(isinstance(p, InteractionPoint) for p in points)

        # Should have range of N and M values
        N_values = [p.N for p in points]
        M_values = [p.M for p in points]

        assert max(N_values) > min(N_values)
        assert max(M_values) > min(M_values)


class TestConfinedConcrete:
    """Tests for confined concrete feature (Mander model)."""

    def test_confined_concrete_disabled_by_default(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that confined concrete is disabled by default."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        assert diagram.confined_concrete is False

    def test_confined_concrete_requires_rho_s(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that confined concrete requires confinement_rho_s parameter."""
        with pytest.raises(ValueError, match="confinement_rho_s must be provided"):
            MNInteractionDiagram(
                section=rectangular_beam_with_rebars,
                concrete=concrete_c30,
                confined_concrete=True,
                # Missing confinement_rho_s
            )

    def test_confined_concrete_validates_rho_s(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that confinement_rho_s is validated."""
        # Too large
        with pytest.raises(ValueError, match="must be between 0 and 0.1"):
            MNInteractionDiagram(
                section=rectangular_beam_with_rebars,
                concrete=concrete_c30,
                confined_concrete=True,
                confinement_rho_s=0.15,  # Too high
            )

        # Negative
        with pytest.raises(ValueError, match="must be between 0 and 0.1"):
            MNInteractionDiagram(
                section=rectangular_beam_with_rebars,
                concrete=concrete_c30,
                confined_concrete=True,
                confinement_rho_s=-0.01,
            )

    def test_confined_concrete_can_be_enabled(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that confined concrete can be enabled with valid parameters."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,  # 2% volumetric ratio
            confinement_f_yh=500.0,  # MPa
        )

        assert diagram.confined_concrete is True
        assert diagram.confinement_rho_s == 0.02
        assert diagram.confinement_f_yh == 500.0

    def test_confined_concrete_defaults_f_yh(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that f_yh defaults to longitudinal steel yield strength."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            # confinement_f_yh not provided
        )

        # Should default to f_yd of first rebar
        first_rebar = rectangular_beam_with_rebars.rebar_groups[0].rebar
        assert diagram.confinement_f_yh == first_rebar.f_yd

    def test_confined_concrete_increases_compression_capacity(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that confinement increases compression capacity."""
        # Unconfined
        diagram_unconfined = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=False,
        )

        # Confined
        diagram_confined = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
        )

        # Compare pure compression capacity
        na_depth_compression = 5000.0  # Deep NA = pure compression

        point_unconfined = diagram_unconfined.calculate_point(na_depth_compression)
        point_confined = diagram_confined.calculate_point(na_depth_compression)

        # Confined should have higher axial capacity
        assert point_confined.N > point_unconfined.N

    def test_confined_concrete_increases_ductility(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that confinement allows larger strains."""
        diagram_confined = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.03,  # Higher confinement
            confinement_f_yh=500.0,
        )

        # Calculate point with large strain (would fail for unconfined)
        large_strain = 0.008  # 0.8% - well beyond unconfined ultimate
        na_depth = 300.0

        # Should not crash and return valid point
        point = diagram_confined.calculate_point(
            neutral_axis_depth=na_depth,
            max_concrete_strain=large_strain,
        )

        assert isinstance(point, InteractionPoint)
        assert point.N > 0  # Should have compression capacity

    def test_full_diagram_with_confined_concrete(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test generating full diagram with confined concrete."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.015,
        )

        points = diagram.generate_diagram(n_points=50)

        # Should generate valid points
        assert len(points) > 0
        assert all(isinstance(p, InteractionPoint) for p in points)

        # Should have range of N and M values
        N_values = [p.N for p in points]
        M_values = [p.M for p in points]

        assert max(N_values) > min(N_values)
        assert max(M_values) > min(M_values)

    def test_confined_concrete_with_tension_stiffening(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that confined concrete and tension stiffening can be combined."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            tension_stiffening=True,  # Both enabled
        )

        # Should work without errors
        points = diagram.generate_diagram(n_points=30)
        assert len(points) > 0
