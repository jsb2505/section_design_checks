"""
Tests for reinforced_concrete.materials.rebar module.
"""

import pytest
import math
import warnings
from pydantic import ValidationError
from materials.reinforced_concrete.materials import (
    Rebar,
    ShearRebar,
)
from materials.core.units import LENGTH_TO_MM, LengthUnit


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
        """Test that non-standard diameters produce a warning."""
        with pytest.warns(UserWarning, match="not in standard list"):
            bar = Rebar(diameter=4, grade="B500B")
        assert bar.diameter == 4
        assert not bar.is_standard

    def test_diameter_validation_too_large(self):
        """Test that non-standard diameters produce a warning."""
        with pytest.warns(UserWarning, match="not in standard list"):
            bar = Rebar(diameter=60, grade="B500B")
        assert bar.diameter == 60
        assert not bar.is_standard

    def test_diameter_validation_negative(self):
        """Test that negative diameters are rejected."""
        with pytest.raises(ValidationError):
            Rebar(diameter=-10, grade="B500B")

    def test_is_standard_no_warning(self):
        """Test that standard diameters don't warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bar = Rebar(diameter=16, grade="B500B")
            rebar_warnings = [x for x in w if "not in standard list" in str(x.message)]
        assert bar.is_standard is True
        assert len(rebar_warnings) == 0

    def test_mass_per_metre(self, rebar_16):
        """Test mass per unit length calculation."""
        mm_per_m = LENGTH_TO_MM[LengthUnit.M]
        expected = (rebar_16.area / mm_per_m ** 2) * rebar_16.density
        assert rebar_16.mass_per_metre == pytest.approx(expected, rel=1e-12)

    def test_perimeter_in_model_dump(self, rebar_16):
        """Test that perimeter appears in serialisation (computed_field)."""
        data = rebar_16.model_dump()
        assert "perimeter" in data
        assert data["perimeter"] == pytest.approx(math.pi * 16, rel=1e-6)

    def test_str_representation(self, rebar_16):
        """Test __str__ method."""
        s = str(rebar_16)
        assert "ϕ16" in s
        assert "B500B" in s
        assert "A=" in s
        assert "mm²" in s

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
        assert shear_links.link_spacing == 200
        assert shear_links.leg_spacing is None
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
        """Test a_sw_over_s_sin_alpha for vertical links."""
        # For α = 90°, sin(α) = 1
        total_area = 2 * math.pi * 10**2 / 4
        expected = total_area / 200  # A_sw / (s · sin(α))
        assert shear_links.a_sw_over_s_sin_alpha == pytest.approx(expected, rel=1e-6)

    def test_rho_w_inclined_links(self):
        """Test a_sw_over_s_sin_alpha for inclined links."""
        links = ShearRebar(
            diameter=10,
            grade="B500B",
            link_spacing=200,
            n_legs=2,
            angle=45.0,
        )
        total_area = 2 * math.pi * 10**2 / 4
        expected = total_area / (200 * math.sin(math.radians(45)))
        assert links.a_sw_over_s_sin_alpha == pytest.approx(expected, rel=1e-6)

    def test_spacing_validation(self):
        """Test that spacing must be positive."""
        with pytest.raises(ValidationError):
            ShearRebar(
                diameter=10,
                grade="B500B",
                link_spacing=-100,
                n_legs=2,
            )

    def test_n_legs_validation(self):
        """Test that n_legs must be at least 1."""
        with pytest.raises(ValidationError):
            ShearRebar(
                diameter=10,
                grade="B500B",
                link_spacing=200,
                n_legs=0,
            )

    def test_leg_spacing_validation(self):
        """Test that leg_spacing, when provided, must be positive."""
        with pytest.raises(ValidationError):
            ShearRebar(
                diameter=10,
                grade="B500B",
                link_spacing=200,
                leg_spacing=-20,
                n_legs=2,
            )

    def test_angle_validation_too_small(self):
        """Test that angle must be ≥ 45°."""
        with pytest.raises(ValidationError):
            ShearRebar(
                diameter=10,
                grade="B500B",
                link_spacing=200,
                n_legs=2,
                angle=30.0,
            )

    def test_angle_validation_too_large(self):
        """Test that angle must be ≤ 90°."""
        with pytest.raises(ValidationError):
            ShearRebar(
                diameter=10,
                grade="B500B",
                link_spacing=200,
                n_legs=2,
                angle=120.0,
            )

    def test_max_link_spacing_vertical(self, shear_links):
        """Test EC2 §9.2.2(6) for vertical links: s_l,max = 0.75d."""
        d = 500.0
        # alpha=90 => cot(alpha)=0 => 0.75 d
        assert shear_links.max_link_spacing(d) == pytest.approx(0.75 * d, rel=1e-12)

    def test_max_link_spacing_inclined(self):
        """Test EC2 §9.2.2(6) for inclined links: s_l,max = 0.75d(1+cot α)."""
        links = ShearRebar(diameter=10, grade="B500B", link_spacing=200, n_legs=2, angle=45.0)
        d = 500.0
        # cot(45)=1 => 0.75 d (1+1) = 1.5 d
        assert links.max_link_spacing(d) == pytest.approx(1.5 * d, rel=1e-12)

    def test_max_link_spacing_invalid_depth(self, shear_links):
        """Test that non-positive depth raises ValueError."""
        with pytest.raises(ValueError):
            shear_links.max_link_spacing(0.0)

    def test_max_leg_spacing(self, shear_links):
        """Test EC2 §9.2.2(8): s_t,max = min(600, 0.75d)."""
        # d=500 => 0.75*500=375 < 600 => 375
        assert shear_links.max_leg_spacing(500.0) == pytest.approx(375.0, rel=1e-12)

    def test_max_leg_spacing_large_depth(self, shear_links):
        """Test EC2 §9.2.2(8): cap at 600mm for large depths."""
        # d=1000 => 0.75*1000=750 > 600 => 600
        assert shear_links.max_leg_spacing(1000.0) == pytest.approx(600.0, rel=1e-12)

    def test_a_sw_over_s_sin_alpha_in_model_dump(self, shear_links):
        """Test that a_sw_over_s_sin_alpha appears in serialisation."""
        data = shear_links.model_dump()
        assert "a_sw_over_s_sin_alpha" in data

    def test_max_leg_spacing_invalid_depth(self, shear_links):
        """Test max leg spacing invalid depth."""
        with pytest.raises(ValueError, match="effective_depth must be > 0"):
            shear_links.max_leg_spacing(0.0)

    def test_str_representation(self, shear_links):
        """Test __str__ method."""
        s = str(shear_links)
        assert "ϕ10" in s
        assert "200" in s  # spacing
        assert "2 legs" in s
        assert "90" in s  # angle


class TestRebarInstantiation:
    """Tests for Rebar instantiation with default grade."""

    def test_rebar_default_grade(self):
        """Test creating rebar with default grade."""
        bar = Rebar(diameter=16)
        assert bar.diameter == 16
        assert bar.grade == "B500B"

    def test_rebar_custom_grade(self):
        """Test creating with custom grade."""
        bar = Rebar(diameter=20, grade="B500C")
        assert bar.diameter == 20
        assert bar.grade == "B500C"

    def test_rebar_custom_name(self):
        """Test creating with custom name."""
        bar = Rebar(diameter=16, name="Main reinforcement")
        assert bar.name == "Main reinforcement"

    def test_all_standard_diameters(self):
        """Test creating all standard diameters."""
        diameters = [6, 8, 10, 12, 16, 20, 25, 32, 40]
        for d in diameters:
            bar = Rebar(diameter=d)
            assert bar.diameter == d

