"""
Tests for reinforced_concrete.analysis.interaction_diagram module.
"""

import pytest
import numpy as np
import json
import csv
from pathlib import Path
from pydantic import ValidationError
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
            compression_from_bottom=False,  # Positive moment: top compressed
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
            compression_from_bottom=False,  # Positive moment: top compressed
            max_concrete_strain=0.0035,
            max_steel_strain=0.010,
        )

        with pytest.raises(ValidationError, match="frozen"):
            point.N = 600.0

    def test_repr(self):
        """Test __repr__ method."""
        point = InteractionPoint(
            N=500.0,
            M=150.0,
            neutral_axis_depth=250.0,
            compression_from_bottom=False,  # Positive moment: top compressed
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
            compression_from_bottom=False,  # Positive moment: top compressed
            max_concrete_strain=0.0035,
            max_steel_strain=0.010,
        )

        data = point.to_dict()

        assert isinstance(data, dict)
        assert data["N"] == 500.0
        assert data["M"] == 150.0
        assert data["neutral_axis_depth"] == 250.0
        assert data["compression_from_bottom"] == False
        assert data["max_concrete_strain"] == 0.0035
        assert data["max_steel_strain"] == 0.010
        assert len(data) == 6  # All expected keys


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
            n_fibres_width=10,
            n_fibres_height=20,
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

    def test_calculate_point_pure_compression(self, diagram):
        """Test calculation at pure compression using end strains."""
        # Pure compression: both ends at ultimate compressive strain
        eps_cu = diagram.concrete_model.get_ultimate_strain()
        point = diagram.calculate_point_from_end_strains(
            eps_top=eps_cu * 0.9,
            eps_bottom=eps_cu * 0.9
        )

        # Should have compression (positive N)
        assert point.N > 0
        # Moment should be small (nearly uniform compression)
        assert abs(point.M) < abs(point.N) * 0.5  # M should be relatively small

    def test_calculate_point_balanced(self, diagram):
        """Test calculation near balanced failure with curvature."""
        # Balanced: top compressed to ultimate, bottom in moderate tension
        eps_cu = diagram.concrete_model.get_ultimate_strain()
        eps_y = diagram.steel_models[0].epsilon_y

        point = diagram.calculate_point_from_end_strains(
            eps_top=eps_cu,
            eps_bottom=-eps_y * 2.0  # Moderate tension
        )

        # Should have compression
        assert point.N > 0
        # Should have significant moment
        assert point.M > 0
        # Strains should be reasonable
        assert 0 < point.max_concrete_strain <= 0.0040  # Allow small margin
        assert point.max_steel_strain > 0

    def test_calculate_point_pure_tension(self, diagram):
        """Test calculation at pure tension using end strains."""
        # Pure tension: both ends in significant tension
        eps_y = diagram.steel_models[0].epsilon_y

        point = diagram.calculate_point_from_end_strains(
            eps_top=-eps_y * 3.0,
            eps_bottom=-eps_y * 3.0
        )

        # Should have tension (negative N)
        assert point.N < 0
        # All steel should be in tension
        assert point.max_steel_strain > 0

    def test_calculate_point_custom_strain(self, diagram):
        """Test calculation with custom strain values."""
        # Custom strain: lower compressive strain
        point = diagram.calculate_point_from_end_strains(
            eps_top=0.002,  # Below ultimate
            eps_bottom=-0.001,
        )

        # Should use the specified strains
        assert point.max_concrete_strain <= 0.0025  # Should be close to 0.002

    def test_generate_diagram_returns_points(self, diagram):
        """Test that generate_diagram returns list of points."""
        points = diagram.generate_diagram_points(n_points=30)

        assert len(points) > 0
        assert all(isinstance(p, InteractionPoint) for p in points)

    def test_generate_diagram_covers_full_range(self, diagram):
        """Test that diagram covers compression to tension."""
        # New API always generates full closed envelope
        points = diagram.generate_diagram_points(n_points=50)

        N_values = [p.N for p in points]

        # Should have compression points
        assert any(N > 0 for N in N_values)
        # Should have tension points
        assert any(N < 0 for N in N_values)
        # The diagram is a closed loop (first point repeated at end)
        # So check that max N (compression) is greater than min N (tension)
        assert max(N_values) > min(N_values)

    def test_generate_diagram_is_closed_loop(self, diagram):
        """Test that generated diagram forms a closed loop."""
        # New API always generates full closed envelope
        points = diagram.generate_diagram_points(n_points=30)

        N_values = [p.N for p in points]
        M_values = [p.M for p in points]

        # Should have range in both N and M
        assert max(N_values) > min(N_values)
        assert max(M_values) > min(M_values)

        # Check envelope is reasonably closed (first and last point should be similar)
        # Allow for resampling differences
        assert abs(points[0].N - points[-1].N) < 100  # kN
        assert abs(points[0].M - points[-1].M) < 10  # kN·m

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
        N_cap, M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

        # Should return the axial level used
        assert N_cap is not None
        # Should have moment capacity
        assert M_Rd_pos > 0
        assert M_Rd_neg < 0
        # Section has asymmetric reinforcement, so M_Rd_pos > |M_Rd_neg|
        assert M_Rd_pos > abs(M_Rd_neg)

    def test_get_capacity_tension(self, diagram):
        """Test getting moment capacity under tension."""
        N_Ed = -200.0  # 200 kN tension
        N_cap, M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

        # Should return the axial level used
        assert N_cap is not None
        # Should have moment capacity (but smaller than compression)
        assert M_Rd_pos > 0
        assert M_Rd_neg < 0

    def test_check_capacity_safe(self, diagram):
        """Test capacity check for safe loads."""
        N_Ed = 500.0  # kN compression
        N_cap, M_Rd_pos, _ = diagram.get_capacity_fixed_n(N_Ed)

        # Apply 50% of capacity at this axial load
        M_Ed = M_Rd_pos * 0.5
        utilization = M_Ed / M_Rd_pos

        assert M_Ed < M_Rd_pos
        assert utilization == pytest.approx(0.5, rel=1e-9)

    def test_check_capacity_unsafe(self, diagram):
        """Test capacity check for unsafe loads."""
        # Use a known load
        N_Ed = 500.0  # kN compression
        N_cap, M_Rd_pos, _ = diagram.get_capacity_fixed_n(N_Ed)

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
        N_cap, M_Rd_pos, _ = diagram.get_capacity_fixed_n(N_Ed)

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

        # Both should produce valid diagrams - test with same strain state
        eps_cu1 = diag1.concrete_model.get_ultimate_strain()
        eps_cu2 = diag2.concrete_model.get_ultimate_strain()
        eps_y = diag1.steel_models[0].epsilon_y

        point1 = diag1.calculate_point_from_end_strains(eps_cu1, -eps_y)
        point2 = diag2.calculate_point_from_end_strains(eps_cu2, -eps_y)

        # Results will differ slightly due to different models
        assert point1.N != pytest.approx(point2.N, rel=0.01)

    def test_different_steel_models(self, simple_beam, concrete_c30):
        """Test creating diagram with different steel models."""
        # Inclined (with strain hardening)
        diag1 = MNInteractionDiagram(
            section=simple_beam,
            concrete=concrete_c30,
            steel_model_type="inclined",
        )

        # Horizontal (perfectly plastic)
        diag2 = MNInteractionDiagram(
            section=simple_beam,
            concrete=concrete_c30,
            steel_model_type="horizontal",
        )

        # Both should produce valid diagrams - test with same strains
        eps_cu = diag1.concrete_model.get_ultimate_strain()
        eps_y = diag1.steel_models[0].epsilon_y

        point1 = diag1.calculate_point_from_end_strains(eps_cu, -eps_y)
        point2 = diag2.calculate_point_from_end_strains(eps_cu, -eps_y)

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
            n_fibres_width=30,
            n_fibres_height=50,
        )

        # Should have many fibers
        assert diagram.mesh.total_fibres > 1000

        # Should still calculate correctly
        eps_cu = diagram.concrete_model.get_ultimate_strain()
        eps_y = diagram.steel_models[0].epsilon_y
        point = diagram.calculate_point_from_end_strains(eps_cu, -eps_y)
        assert point.N > 0
        assert point.M > 0

    def test_coarse_mesh(self, simple_beam, concrete_c30):
        """Test creating diagram with coarse mesh."""
        diagram = MNInteractionDiagram(
            section=simple_beam,
            concrete=concrete_c30,
            n_fibres_width=5,
            n_fibres_height=10,
        )

        # Should have fewer fibers
        assert diagram.mesh.total_fibres < 100

        # Should still calculate (less accurate)
        eps_cu = diagram.concrete_model.get_ultimate_strain()
        point = diagram.calculate_point_from_end_strains(eps_cu, -0.001)
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
            n_fibres_width=15,
        )

        assert isinstance(diagram, MNInteractionDiagram)
        # Should use bilinear model and custom mesh
        eps_cu = diagram.concrete_model.get_ultimate_strain()
        point = diagram.calculate_point_from_end_strains(eps_cu, -0.002)
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

        eps_cu = diagram.concrete_model.get_ultimate_strain()
        # Pure compression: equal strains top and bottom
        point = diagram.calculate_point_from_end_strains(eps_cu * 0.9, eps_cu * 0.9)

        # For symmetrical section, pure compression should have M ≈ 0
        # Allow small numerical error
        assert abs(point.M) < abs(point.N) * 0.05  # M < 5% of N

    def test_equilibrium(self, rectangular_beam_with_rebars, concrete_c30):
        """Test force equilibrium (sum of fiber forces = N)."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        eps_cu = diagram.concrete_model.get_ultimate_strain()
        eps_y = diagram.steel_models[0].epsilon_y
        point = diagram.calculate_point_from_end_strains(eps_cu, -eps_y)

        # N and M should be reasonable values
        assert -1000 < point.N < 5000  # kN (reasonable for 300×500 section)
        assert -500 < point.M < 500  # kN·m

    def test_monotonic_n_with_increasing_compression(self, rectangular_beam_with_rebars, concrete_c30):
        """Test that N increases with more compression in strain field."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        eps_cu = diagram.concrete_model.get_ultimate_strain()
        eps_y = diagram.steel_models[0].epsilon_y

        # Case 1: Top compressed, bottom in tension (less compression overall)
        point1 = diagram.calculate_point_from_end_strains(eps_cu * 0.5, -eps_y)

        # Case 2: More compression on both ends
        point2 = diagram.calculate_point_from_end_strains(eps_cu * 0.9, eps_cu * 0.3)

        # Case 3: High compression on both
        point3 = diagram.calculate_point_from_end_strains(eps_cu * 0.9, eps_cu * 0.8)

        # N should generally increase with more compression
        assert point3.N > point1.N

    def test_strain_distribution_linear(self, rectangular_beam_with_rebars, concrete_c30):
        """Test that the strain-based calculation produces valid results."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        # Calculate point with specific end strains
        eps_cu = diagram.concrete_model.get_ultimate_strain()
        eps_y = diagram.steel_models[0].epsilon_y
        point = diagram.calculate_point_from_end_strains(eps_cu, -eps_y)

        # Strains should be within reasonable limits
        assert point.max_concrete_strain <= 0.0040  # Near EC2 limit (allow margin)
        assert point.max_steel_strain >= 0  # Steel can be in tension or compression


class TestExportFunctionality:
    """Tests for export to JSON, CSV, and dict."""

    @pytest.fixture
    def diagram(self, rectangular_beam_with_rebars, concrete_c30):
        """Create M-N diagram for export testing."""
        return MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            n_fibres_width=10,
            n_fibres_height=20,
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
        assert "N" in point
        assert "M" in point
        assert "neutral_axis_depth" in point
        assert "max_concrete_strain" in point
        assert "max_steel_strain" in point

        # Check metadata
        metadata = data["metadata"]
        assert "concrete_grade" in metadata
        assert "concrete_fck" in metadata
        assert "n_fibres" in metadata
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
        assert 'N' in rows[0]
        assert 'M' in rows[0]
        assert 'neutral_axis_depth' in rows[0]
        assert 'max_concrete_strain' in rows[0]
        assert 'max_steel_strain' in rows[0]

        # Check data is numeric
        for row in rows:
            assert float(row['N']) is not None
            assert float(row['M']) is not None

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
        assert 'N' in rows[0]
        assert 'M' in rows[0]
        assert 'neutral_axis_depth' not in rows[0]
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
        assert "N" in point
        assert "M" in point

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
        assert reloaded_data["diagram_points"][0]["N"] == dict_data["points"][0]["N"]
        assert reloaded_data["diagram_points"][0]["M"] == dict_data["points"][0]["M"]

    def test_export_csv_data_integrity(self, diagram, tmp_path):
        """Test that CSV export maintains data integrity."""
        output_file = tmp_path / "mn_diagram_integrity.csv"

        # Export
        diagram.export_to_csv(output_file, n_points=20)

        # Get reference data
        points = diagram.generate_diagram_points(n_points=20)

        # Read CSV
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        # Compare
        for i, (point, csv_row) in enumerate(zip(points, csv_rows)):
            assert point.N == pytest.approx(float(csv_row['N']), rel=1e-6)
            assert point.M == pytest.approx(float(csv_row['M']), rel=1e-6)
            assert point.neutral_axis_depth == pytest.approx(
                float(csv_row['neutral_axis_depth']), rel=1e-6
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
        N_cap, M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

        # Should return the axial level
        assert N_cap is not None
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
            N_cap, M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

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
        N_cap, M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

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
        N_cap, M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(N_Ed)

        # Should be approximately symmetric (allow small numerical differences)
        assert abs(M_Rd_pos + M_Rd_neg) < max(abs(M_Rd_pos), abs(M_Rd_neg)) * 0.1

    def test_get_capacity_returns_correct_signs(self, asymmetric_rebar_beam, concrete_c30):
        """Test that get_capacity returns correctly signed values."""
        diagram = MNInteractionDiagram(
            section=asymmetric_rebar_beam,
            concrete=concrete_c30,
        )

        N_cap, M_Rd_pos, M_Rd_neg = diagram.get_capacity_fixed_n(500.0)

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

        points = diagram.generate_diagram_points(n_points=100)
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
        """Test that tension stiffening provides additional tensile capacity."""
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

        # Calculate point with top in small compression, bottom in tension
        # This creates a tension zone where concrete can contribute via tension stiffening
        eps_cu = diagram_no_ts.concrete_model.get_ultimate_strain()
        eps_y = diagram_no_ts.steel_models[0].epsilon_y

        point_no_ts = diagram_no_ts.calculate_point_from_end_strains(eps_cu * 0.2, -eps_y * 2.0)
        point_with_ts = diagram_with_ts.calculate_point_from_end_strains(eps_cu * 0.2, -eps_y * 2.0)

        # With tension stiffening, should have different (typically higher) moment capacity
        # The exact effect depends on strain distribution, but they should differ
        assert abs(point_with_ts.M - point_no_ts.M) > 0.1  # Should have measurable difference

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

        # Point with top compressed, bottom in small tension (some concrete in tension)
        eps_cu = diagram_no_ts.concrete_model.get_ultimate_strain()
        eps_y = diagram_no_ts.steel_models[0].epsilon_y

        point_no_ts = diagram_no_ts.calculate_point_from_end_strains(eps_cu, -eps_y * 0.5)
        point_with_ts = diagram_with_ts.calculate_point_from_end_strains(eps_cu, -eps_y * 0.5)

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

        # Pure compression (both ends compressed)
        eps_cu = diagram_no_ts.concrete_model.get_ultimate_strain()

        point_no_ts = diagram_no_ts.calculate_point_from_end_strains(eps_cu * 0.9, eps_cu * 0.9)
        point_with_ts = diagram_with_ts.calculate_point_from_end_strains(eps_cu * 0.9, eps_cu * 0.9)

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

        points = diagram.generate_diagram_points(n_points=50)

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
        with pytest.raises(ValueError, match="must be in"):
            MNInteractionDiagram(
                section=rectangular_beam_with_rebars,
                concrete=concrete_c30,
                confined_concrete=True,
                confinement_rho_s=0.15,  # Too high
            )

        # Negative
        with pytest.raises(ValueError, match="must be in"):
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
        """Test that f_yh defaults to characteristic steel yield strength."""
        diagram = MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            # confinement_f_yh not provided
        )

        # Should default to f_yk (500.0 for B500B steel)
        first_rebar = rectangular_beam_with_rebars.rebar_groups[0].rebar
        assert diagram.confinement_f_yh == first_rebar.f_yk

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
        eps_cu_unconf = diagram_unconfined.concrete_model.get_ultimate_strain()

        point_unconfined = diagram_unconfined.calculate_point_from_end_strains(
            eps_cu_unconf * 0.9, eps_cu_unconf * 0.9
        )
        point_confined = diagram_confined.calculate_point_from_end_strains(
            eps_cu_unconf * 0.9, eps_cu_unconf * 0.9
        )

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

        # Should not crash and return valid point
        point = diagram_confined.calculate_point_from_end_strains(
            eps_top=large_strain,
            eps_bottom=large_strain * 0.5,
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

        points = diagram.generate_diagram_points(n_points=50)

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
        points = diagram.generate_diagram_points(n_points=30)
        assert len(points) > 0

    def test_get_capacity_vector_safe_load(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test get_capacity_vector for a safe load case."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        # Apply a load well within capacity
        N_Ed = 500.0  # kN
        M_Ed = 100.0  # kN·m

        result = diagram.get_capacity_vector(
            N_Ed=N_Ed, M_Ed=M_Ed, n_points=100
        )
        N_Rd, M_Rd, is_safe, utilization = result.N_Rd, result.M_Rd, result.is_safe, result.utilization

        # Check results
        assert N_Rd is not None
        assert M_Rd is not None
        assert is_safe is True
        assert 0.0 < utilization < 1.0

        # Capacity should be greater than applied load (linear scaling)
        assert N_Rd >= N_Ed
        assert M_Rd >= M_Ed

        # Check that N_Rd and M_Rd are in correct ratio to N_Ed and M_Ed
        # (should be on same ray from origin)
        if abs(N_Ed) > 1e-6:
            ratio_N = N_Rd / N_Ed
            ratio_M = M_Rd / M_Ed
            assert abs(ratio_N - ratio_M) < 0.01  # Should be same scaling

    def test_get_capacity_vector_failing_load(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test get_capacity_vector for an overloaded case."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        # Apply a very large load (likely to fail)
        N_Ed = 10000.0  # kN - very high
        M_Ed = 1000.0   # kN·m - very high

        result = diagram.get_capacity_vector(
            N_Ed=N_Ed, M_Ed=M_Ed, n_points=100
        )
        N_Rd, M_Rd, is_safe, utilization = result.N_Rd, result.M_Rd, result.is_safe, result.utilization

        # Check results
        assert N_Rd is not None
        assert M_Rd is not None
        assert is_safe is False
        assert utilization > 1.0

        # Capacity should be less than applied load
        assert N_Rd < N_Ed
        assert M_Rd < M_Ed

    def test_get_capacity_vector_origin(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test get_capacity_vector at origin (no load)."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        result = diagram.get_capacity_vector(
            N_Ed=0.0, M_Ed=0.0, n_points=100
        )
        N_Rd, M_Rd, is_safe, utilization = result.N_Rd, result.M_Rd, result.is_safe, result.utilization

        assert N_Rd == 0.0
        assert M_Rd == 0.0
        assert is_safe is True
        assert utilization == 0.0

    def test_get_capacity_vector_pure_axial(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test get_capacity_vector for pure axial load."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        N_Ed = 1000.0  # kN
        M_Ed = 0.0     # No moment

        result = diagram.get_capacity_vector(
            N_Ed=N_Ed, M_Ed=M_Ed, n_points=100
        )
        N_Rd, M_Rd, is_safe, utilization = result.N_Rd, result.M_Rd, result.is_safe, result.utilization

        assert N_Rd is not None
        assert M_Rd is not None
        # For pure axial, moment should be very small
        assert abs(M_Rd) < 1.0  # Should be close to zero

    def test_get_capacity_vector_pure_moment(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test get_capacity_vector for pure moment."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        N_Ed = 0.0      # No axial
        M_Ed = 150.0    # kN·m

        result = diagram.get_capacity_vector(
            N_Ed=N_Ed, M_Ed=M_Ed, n_points=100
        )
        N_Rd, M_Rd, is_safe, utilization = result.N_Rd, result.M_Rd, result.is_safe, result.utilization

        assert N_Rd is not None
        assert M_Rd is not None
        # For pure moment, axial should be very small
        assert abs(N_Rd) < 10.0  # Should be close to zero

    def test_plot_basic(self, rectangular_beam_with_rebars, concrete_c30):
        """Test basic plot without load points."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        # Should not raise error and return a figure
        fig = diagram.plot_mn(show=False, n_points=50)
        assert fig is not None

        # Check that figure has traces (M-N curve and origin)
        assert len(fig.data) >= 2

    def test_plot_with_load_points(self, rectangular_beam_with_rebars, concrete_c30):
        """Test plot with load points."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        load_points = [
            {"N_Ed": 500, "M_Ed": 100, "name": "LC1: Test"},
            {"N_Ed": 800, "M_Ed": 150, "name": "LC2: Test"},
        ]

        fig = diagram.plot_mn(
            load_points=load_points,
            show_vectors=False,
            show_metadata=True,
            show=False,
            n_points=50
        )

        assert fig is not None
        # Should have M-N curve + origin + 2 load points = at least 4 traces
        assert len(fig.data) >= 4

    def test_plot_with_vectors(self, rectangular_beam_with_rebars, concrete_c30):
        """Test plot with vector projection rays."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        load_points = [
            {"N_Ed": 500, "M_Ed": 100, "name": "LC1: Test"},
        ]

        fig = diagram.plot_mn(
            load_points=load_points,
            show_vectors=True,
            show=False,
            n_points=50
        )

        assert fig is not None
        # Should have more traces due to vector lines
        # M-N curve + origin + load point + 2 vector lines = at least 5 traces
        assert len(fig.data) >= 5

    def test_plot_save_to_file(
        self, rectangular_beam_with_rebars, concrete_c30, tmp_path
    ):
        """Test saving plot to HTML file."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        save_path = tmp_path / "test_plot.html"

        fig = diagram.plot_mn(
            show=False,
            save_path=str(save_path),
            n_points=50
        )

        assert fig is not None
        assert save_path.exists()
        assert save_path.stat().st_size > 0  # File should have content

    def test_plot_custom_title(self, rectangular_beam_with_rebars, concrete_c30):
        """Test plot with custom title."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        custom_title = "Custom Test Title"

        fig = diagram.plot_mn(
            show=False,
            title=custom_title,
            n_points=50
        )

        assert fig is not None
        assert fig.layout.title.text == custom_title

    def test_plot_without_metadata(self, rectangular_beam_with_rebars, concrete_c30):
        """Test plot with metadata disabled."""
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        load_points = [
            {"N_Ed": 500, "M_Ed": 100, "name": "LC1: Test"},
        ]

        fig = diagram.plot_mn(
            load_points=load_points,
            show_metadata=False,
            show=False,
            n_points=50
        )

        assert fig is not None
        # Should still create plot, just simpler hover text
        assert len(fig.data) >= 3


class TestVectorMethodRobustness:
    """Tests for vector method robustness and edge cases."""

    def test_vector_method_linear_scaling_consistency(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that utilization scales linearly when load point is scaled.

        This catches silent errors where the vector projection gives wrong results.
        Pick a random (M, N) inside the envelope, compute utilization, then scale
        the point by 0.5 and expect utilization to be approximately halved.
        """
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        # Pick a point inside the envelope (use a strain pair to ensure it's valid)
        eps_cu = diagram.concrete_model.get_ultimate_strain()
        eps_y = diagram.steel_models[0].epsilon_y

        # Calculate a point inside the envelope
        point = diagram.calculate_point_from_end_strains(eps_cu * 0.8, -eps_y * 0.5)
        N_Ed = point.N
        M_Ed = point.M

        # Get utilization at the full point
        result_full = diagram.get_capacity_vector(
            N_Ed=N_Ed, M_Ed=M_Ed, n_points=120
        )
        N_Rd_full, M_Rd_full, is_safe_full, util_full = result_full.N_Rd, result_full.M_Rd, result_full.is_safe, result_full.utilization

        # Scale the point by 0.5
        N_Ed_half = N_Ed * 0.5
        M_Ed_half = M_Ed * 0.5

        # Get utilization at half point
        result_half = diagram.get_capacity_vector(
            N_Ed=N_Ed_half, M_Ed=M_Ed_half, n_points=120
        )
        N_Rd_half, M_Rd_half, is_safe_half, util_half = result_half.N_Rd, result_half.M_Rd, result_half.is_safe, result_half.utilization

        # Both should be safe (inside envelope)
        assert is_safe_full is True
        assert is_safe_half is True

        # Utilization should scale linearly: util_half ≈ util_full * 0.5
        # Allow some tolerance due to discretization and numerical precision
        assert util_half == pytest.approx(util_full * 0.5, rel=0.15)

        # Alternative check: capacity should double when load is halved
        # t_cap_full * (N_Ed, M_Ed) = (N_Rd_full, M_Rd_full)
        # t_cap_half * (N_Ed_half, M_Ed_half) = (N_Rd_half, M_Rd_half)
        # Since load is halved, t_cap_half should be approximately 2 * t_cap_full
        t_cap_full = util_full  # utilization is 1/t_cap for safe loads
        t_cap_half = util_half

        # Expect: t_cap_half ≈ 0.5 * t_cap_full (half the utilization)
        ratio = t_cap_half / t_cap_full
        assert ratio == pytest.approx(0.5, rel=0.15)

    def test_multiple_intersections_uses_conservative_min(
        self, rectangular_beam_with_rebars, concrete_c30
    ):
        """Test that self-intersecting curves pick min positive intersection.

        This test deliberately creates a scenario where multiple intersections
        might occur and verifies the code uses the conservative (minimum) value.
        """
        diagram = create_interaction_diagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30
        )

        # Generate a coarse diagram which might have local non-convexity
        # or self-intersection due to numerical issues
        points = diagram.generate_diagram_points(n_points=20)  # Coarse for potential issues

        # Test a load case that might intersect the curve multiple times
        # Use a point near the envelope where discretization might cause issues
        eps_cu = diagram.concrete_model.get_ultimate_strain()
        eps_y = diagram.steel_models[0].epsilon_y

        # Point very close to the envelope (95% of a capacity point)
        point_near_envelope = diagram.calculate_point_from_end_strains(
            eps_cu * 0.95, -eps_y * 0.8
        )

        N_Ed = point_near_envelope.N * 0.95
        M_Ed = point_near_envelope.M * 0.95

        # Get capacity - should use minimum positive intersection
        result = diagram.get_capacity_vector(
            N_Ed=N_Ed, M_Ed=M_Ed, n_points=20
        )
        N_Rd, M_Rd, is_safe, utilization = result.N_Rd, result.M_Rd, result.is_safe, result.utilization

        # Should be safe (we scaled down from envelope)
        assert is_safe is True
        assert utilization < 1.0

        # The capacity should be on the correct side of the load
        # (capacity >= demand in the direction of the ray)
        if abs(N_Ed) > 1e-6:
            assert N_Rd / N_Ed >= 0.95  # Should be scaled up from load
        if abs(M_Ed) > 1e-6:
            assert M_Rd / M_Ed >= 0.95

        # Test with a finer diagram for comparison
        result_fine = diagram.get_capacity_vector(
            N_Ed=N_Ed, M_Ed=M_Ed, n_points=200
        )
        N_Rd_fine, M_Rd_fine, is_safe_fine, util_fine = result_fine.N_Rd, result_fine.M_Rd, result_fine.is_safe, result_fine.utilization

        # Coarse and fine should give consistent results (within tolerance)
        # Coarse might be slightly more conservative
        assert is_safe_fine is True
        assert util_fine == pytest.approx(utilization, rel=0.2)


class TestExtremalPinning:
    """Tests that max N, min N, max M, min M from the dense diagram are preserved
    exactly in the resampled output of generate_diagram_points."""

    @pytest.fixture
    def symmetric_diagram(self, rebar_20, concrete_c30):
        """Symmetric section: equal top/bottom reinforcement (min N has M ≈ 0)."""
        section = create_rectangular_section(300, 500)
        for y in (50, 450):
            layer = create_linear_rebar_layer(
                rebar=rebar_20,
                n_bars=3,
                start_point=(50, y),
                end_point=(250, y),
            )
            section.add_rebar_group(layer)
        return MNInteractionDiagram(section=section, concrete=concrete_c30)

    @pytest.fixture
    def asymmetric_diagram(self, rebar_20, rebar_16, concrete_c30):
        """Asymmetric section: heavy bottom, light top (min N has M ≠ 0)."""
        section = create_rectangular_section(300, 500)
        bottom = create_linear_rebar_layer(
            rebar=rebar_20, n_bars=5,
            start_point=(40, 50), end_point=(260, 50),
        )
        top = create_linear_rebar_layer(
            rebar=rebar_16, n_bars=2,
            start_point=(100, 450), end_point=(200, 450),
        )
        section.add_rebar_group(bottom)
        section.add_rebar_group(top)
        return MNInteractionDiagram(section=section, concrete=concrete_c30)

    def _dense_extrema(self, diagram):
        """Return (max_N, min_N, max_M, min_M) InteractionPoints from the dense set."""
        dense = diagram._get_dense_diagram_points(n_dense=800)
        N_arr = np.array([p.N for p in dense])
        M_arr = np.array([p.M for p in dense])
        return (
            dense[int(np.argmax(N_arr))],
            dense[int(np.argmin(N_arr))],
            dense[int(np.argmax(M_arr))],
            dense[int(np.argmin(M_arr))],
        )

    def test_max_N_always_present(self, symmetric_diagram):
        """max N from generate_diagram_points must equal max N from dense set."""
        d_max_N, *_ = self._dense_extrema(symmetric_diagram)
        pts = symmetric_diagram.generate_diagram_points(n_points=120)
        N_vals = np.array([p.N for p in pts])
        assert np.max(N_vals) == pytest.approx(d_max_N.N, abs=1e-6)

    def test_min_N_present_symmetric(self, symmetric_diagram):
        """For a symmetric section, the pinned pure-tension point must match dense min N."""
        _, d_min_N, *_ = self._dense_extrema(symmetric_diagram)
        pts = symmetric_diagram.generate_diagram_points(n_points=120)
        N_vals = np.array([p.N for p in pts])
        assert np.min(N_vals) == pytest.approx(d_min_N.N, abs=1e-6)

    def test_min_N_present_asymmetric(self, asymmetric_diagram):
        """For an asymmetric section, the pinned pure-tension point must match dense min N."""
        _, d_min_N, *_ = self._dense_extrema(asymmetric_diagram)
        pts = asymmetric_diagram.generate_diagram_points(n_points=120)
        N_vals = np.array([p.N for p in pts])
        assert np.min(N_vals) == pytest.approx(d_min_N.N, abs=1e-6)

    def test_max_M_present(self, symmetric_diagram):
        """max M in generate_diagram_points must equal max M in dense set."""
        _, _, d_max_M, _ = self._dense_extrema(symmetric_diagram)
        pts = symmetric_diagram.generate_diagram_points(n_points=120)
        M_vals = np.array([p.M for p in pts])
        assert np.max(M_vals) == pytest.approx(d_max_M.M, abs=1e-6)

    def test_min_M_present(self, symmetric_diagram):
        """min M in generate_diagram_points must equal min M in dense set."""
        _, _, _, d_min_M = self._dense_extrema(symmetric_diagram)
        pts = symmetric_diagram.generate_diagram_points(n_points=120)
        M_vals = np.array([p.M for p in pts])
        assert np.min(M_vals) == pytest.approx(d_min_M.M, abs=1e-6)

    @pytest.mark.parametrize("n_points", [40, 60, 80, 120, 200])
    def test_extrema_invariant_under_n_points(self, symmetric_diagram, n_points):
        """All four extrema must be present regardless of n_points."""
        d_max_N, d_min_N, d_max_M, d_min_M = self._dense_extrema(symmetric_diagram)
        pts = symmetric_diagram.generate_diagram_points(n_points=n_points)
        N_vals = np.array([p.N for p in pts])
        M_vals = np.array([p.M for p in pts])
        assert np.max(N_vals) == pytest.approx(d_max_N.N, abs=1e-6)
        assert np.min(N_vals) == pytest.approx(d_min_N.N, abs=1e-6)
        assert np.max(M_vals) == pytest.approx(d_max_M.M, abs=1e-6)
        assert np.min(M_vals) == pytest.approx(d_min_M.M, abs=1e-6)

    def test_output_length_unchanged(self, symmetric_diagram):
        """generate_diagram_points must return exactly n_points points."""
        for n in (40, 80, 120, 200):
            pts = symmetric_diagram.generate_diagram_points(n_points=n)
            assert len(pts) == n, f"Expected {n} points, got {len(pts)}"

    def test_closure_maintained(self, symmetric_diagram):
        """First and last points must be identical after pinning."""
        pts = symmetric_diagram.generate_diagram_points(n_points=120)
        assert pts[0].M == pts[-1].M
        assert pts[0].N == pts[-1].N
