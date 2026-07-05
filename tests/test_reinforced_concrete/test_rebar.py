"""
Tests for reinforced_concrete.materials.rebar module.
"""

import pytest
import math
from pydantic import ValidationError
from materials.reinforced_concrete.materials import (
    Rebar,
    ShearRebar,
    create_standard_rebar,
)


class TestRebar:
    """Tests for Rebar class."""

    def test_create_rebar(self, rebar_16):
        """Test creating a rebar."""
        assert rebar_16.diameter == 16
        assert rebar_16.grade == "B500B"

    def test_rebar_inherits_steel_properties(self, rebar_16):
        """Test that Rebar inherits ReinforcingSteel properties."""
        assert rebar_16.f_yk == 500.0
        assert rebar_16.f_yd == pytest.approx(500.0 / 1.15)
        assert rebar_16.E_s == 200_000.0

    def test_bar_area(self, rebar_16):
        """Test cross-sectional area calculation."""
        # A = π · d² / 4
        expected = math.pi * (16 ** 2) / 4.0
        assert rebar_16.area == pytest.approx(expected, rel=1e-6)

    def test_bar_area_various_diameters(self):
        """Test area for various diameters."""
        diameters = [8, 10, 12, 16, 20, 25, 32, 40]
        for d in diameters:
            bar = Rebar(diameter=d, grade="B500B")
            expected = math.pi * (d ** 2) / 4.0
            assert bar.area == pytest.approx(expected, rel=1e-6)

    def test_bar_perimeter(self, rebar_16):
        """Test perimeter calculation."""
        # P = π · d
        expected = math.pi * 16
        assert rebar_16.perimeter == pytest.approx(expected, rel=1e-6)

    def test_diameter_validation_too_small(self):
        """Test that very small diameters trigger warning."""
        with pytest.raises(ValidationError, match="outside typical range"):
            Rebar(diameter=4, grade="B500B")

    def test_diameter_validation_too_large(self):
        """Test that very large diameters trigger warning."""
        with pytest.raises(ValidationError, match="outside typical range"):
            Rebar(diameter=60, grade="B500B")

    def test_diameter_validation_negative(self):
        """Test that negative diameters are rejected."""
        with pytest.raises(ValidationError):
            Rebar(diameter=-10, grade="B500B")

    def test_str_representation(self, rebar_16):
        """Test __str__ method."""
        s = str(rebar_16)
        assert "16" in s
        assert "B500B" in s
        assert "201" in s  # Area

    def test_json_serialization(self, rebar_16):
        """Test JSON serialization."""
        json_data = rebar_16.model_dump()
        assert json_data["diameter"] == 16
        assert json_data["grade"] == "B500B"
        assert json_data["area"] == pytest.approx(math.pi * 16**2 / 4, rel=1e-6)


class TestShearRebar:
    """Tests for ShearRebar class."""

    def test_create_shear_rebar(self, shear_links):
        """Test creating shear reinforcement."""
        assert shear_links.diameter == 10
        assert shear_links.spacing == 200
        assert shear_links.n_legs == 2
        assert shear_links.angle == 90.0

    def test_shear_rebar_inherits_rebar(self, shear_links):
        """Test that ShearRebar inherits Rebar properties."""
        assert shear_links.area == pytest.approx(math.pi * 10**2 / 4)
        assert shear_links.f_yk == 500.0

    def test_total_area_per_spacing(self, shear_links):
        """Test total area calculation."""
        # A_sw = n_legs × A_bar
        expected = 2 * math.pi * 10**2 / 4
        assert shear_links.total_area_per_spacing == pytest.approx(expected, rel=1e-6)

    def test_area_per_unit_length(self, shear_links):
        """Test area per unit length."""
        # A_sw / s
        total_area = 2 * math.pi * 10**2 / 4
        expected = total_area / 200
        assert shear_links.area_per_unit_length == pytest.approx(expected, rel=1e-6)

    def test_rho_w_vertical_links(self, shear_links):
        """Test rho_w for vertical links."""
        # For α = 90°, sin(α) = 1
        total_area = 2 * math.pi * 10**2 / 4
        expected = total_area / 200  # A_sw / (s · sin(α))
        assert shear_links.rho_w == pytest.approx(expected, rel=1e-6)

    def test_rho_w_inclined_links(self):
        """Test rho_w for inclined links."""
        links = ShearRebar(
            diameter=10,
            grade="B500B",
            spacing=200,
            n_legs=2,
            angle=45.0,
        )
        total_area = 2 * math.pi * 10**2 / 4
        expected = total_area / (200 * math.sin(math.radians(45)))
        assert links.rho_w == pytest.approx(expected, rel=1e-6)

    def test_spacing_validation(self):
        """Test that spacing must be positive."""
        with pytest.raises(ValidationError):
            ShearRebar(
                diameter=10,
                grade="B500B",
                spacing=-100,
                n_legs=2,
            )

    def test_n_legs_validation(self):
        """Test that n_legs must be at least 1."""
        with pytest.raises(ValidationError):
            ShearRebar(
                diameter=10,
                grade="B500B",
                spacing=200,
                n_legs=0,
            )

    def test_angle_validation_too_small(self):
        """Test that angle must be ≥ 45°."""
        with pytest.raises(ValidationError):
            ShearRebar(
                diameter=10,
                grade="B500B",
                spacing=200,
                n_legs=2,
                angle=30.0,
            )

    def test_angle_validation_too_large(self):
        """Test that angle must be ≤ 90°."""
        with pytest.raises(ValidationError):
            ShearRebar(
                diameter=10,
                grade="B500B",
                spacing=200,
                n_legs=2,
                angle=120.0,
            )

    def test_str_representation(self, shear_links):
        """Test __str__ method."""
        s = str(shear_links)
        assert "10" in s  # diameter
        assert "200" in s  # spacing
        assert "2" in s  # n_legs
        assert "90" in s  # angle


class TestCreateStandardRebar:
    """Tests for create_standard_rebar factory function."""

    def test_create_standard_rebar_default(self):
        """Test creating standard rebar with defaults."""
        bar = create_standard_rebar(16)
        assert bar.diameter == 16
        assert bar.grade == "B500B"
        assert "ϕ16" in bar.name or "16" in bar.name

    def test_create_standard_rebar_custom_grade(self):
        """Test creating with custom grade."""
        bar = create_standard_rebar(20, grade="B500C")
        assert bar.diameter == 20
        assert bar.grade == "B500C"

    def test_create_standard_rebar_custom_name(self):
        """Test creating with custom name."""
        bar = create_standard_rebar(16, name="Main reinforcement")
        assert bar.name == "Main reinforcement"

    def test_create_all_standard_diameters(self):
        """Test creating all standard diameters."""
        diameters = [6, 8, 10, 12, 16, 20, 25, 32, 40]
        for d in diameters:
            bar = create_standard_rebar(d)
            assert bar.diameter == d
