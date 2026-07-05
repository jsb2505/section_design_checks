"""
Tests for low-level NDP spacing helper functions.
"""

from __future__ import annotations

import pytest

from section_design_checks.reinforced_concrete.ndp.ndp import (
    _max_leg_spacing_ec2,
    _max_leg_spacing_eu_de,
    _max_link_spacing_ec2,
    _max_link_spacing_eu_de,
)


class TestLinkSpacingHelpers:
    """Tests for TestLinkSpacingHelpers."""
    def test_max_link_spacing_ec2_vertical_and_inclined(self):
        """Test max link spacing ec2 vertical and inclined."""
        s_vert = _max_link_spacing_ec2(
            effective_depth=500.0,
            section_depth=600.0,
            f_ck=30.0,
            V_Ed=100.0,
            V_Rd_max=300.0,
            V_Rd_c=80.0,
            link_angle_degrees=90.0,
        )
        assert s_vert == pytest.approx(0.75 * 500.0, rel=1e-12)

        s_45 = _max_link_spacing_ec2(
            effective_depth=500.0,
            section_depth=600.0,
            f_ck=30.0,
            V_Ed=100.0,
            V_Rd_max=300.0,
            V_Rd_c=80.0,
            link_angle_degrees=45.0,
        )
        assert s_45 == pytest.approx(0.75 * 500.0 * 2.0, rel=1e-12)

    def test_max_link_spacing_ec2_invalid_depth(self):
        """Test max link spacing ec2 invalid depth."""
        with pytest.raises(ValueError, match="effective_depth must be > 0"):
            _max_link_spacing_ec2(
                effective_depth=0.0,
                section_depth=600.0,
                f_ck=30.0,
                V_Ed=100.0,
                V_Rd_max=300.0,
                V_Rd_c=80.0,
                link_angle_degrees=90.0,
            )

    def test_max_link_spacing_eu_de_branches(self):
        # ratio <= 0.3 with "note b" floor to 150 mm
        """Test max link spacing eu de branches."""
        s_low = _max_link_spacing_eu_de(
            effective_depth=500.0,
            section_depth=180.0,
            f_ck=40.0,
            V_Ed=20.0,
            V_Rd_max=100.0,
            V_Rd_c=30.0,
            link_angle_degrees=90.0,
        )
        assert s_low == pytest.approx(150.0, rel=1e-12)

        # 0.3 < ratio <= 0.6
        s_mid = _max_link_spacing_eu_de(
            effective_depth=500.0,
            section_depth=500.0,
            f_ck=40.0,
            V_Ed=50.0,
            V_Rd_max=100.0,
            V_Rd_c=30.0,
            link_angle_degrees=90.0,
        )
        assert s_mid == pytest.approx(min(0.5 * 500.0, 300.0), rel=1e-12)

        # ratio > 0.6
        s_high = _max_link_spacing_eu_de(
            effective_depth=500.0,
            section_depth=1200.0,
            f_ck=60.0,
            V_Ed=90.0,
            V_Rd_max=100.0,
            V_Rd_c=30.0,
            link_angle_degrees=90.0,
        )
        assert s_high == pytest.approx(min(0.25 * 1200.0, 200.0), rel=1e-12)

    def test_max_link_spacing_eu_de_invalid_section_depth(self):
        """Test max link spacing eu de invalid section depth."""
        with pytest.raises(ValueError, match="section_depth must be > 0"):
            _max_link_spacing_eu_de(
                effective_depth=500.0,
                section_depth=0.0,
                f_ck=30.0,
                V_Ed=20.0,
                V_Rd_max=100.0,
                V_Rd_c=20.0,
                link_angle_degrees=90.0,
            )


class TestLegSpacingHelpers:
    """Tests for TestLegSpacingHelpers."""
    def test_max_leg_spacing_ec2_and_invalid_depth(self):
        """Test max leg spacing ec2 and invalid depth."""
        s = _max_leg_spacing_ec2(
            effective_depth=500.0,
            section_depth=600.0,
            f_ck=30.0,
            V_Ed=100.0,
            V_Rd_max=300.0,
            V_Rd_c=80.0,
            link_angle_degrees=90.0,
        )
        assert s == pytest.approx(min(600.0, 0.75 * 500.0), rel=1e-12)

        with pytest.raises(ValueError, match="effective_depth must be > 0"):
            _max_leg_spacing_ec2(
                effective_depth=0.0,
                section_depth=600.0,
                f_ck=30.0,
                V_Ed=100.0,
                V_Rd_max=300.0,
                V_Rd_c=80.0,
                link_angle_degrees=90.0,
            )

    def test_max_leg_spacing_eu_de_branches(self):
        # ratio <= 0.3
        """Test max leg spacing eu de branches."""
        low_fck = _max_leg_spacing_eu_de(
            effective_depth=500.0,
            section_depth=700.0,
            f_ck=40.0,
            V_Ed=20.0,
            V_Rd_max=100.0,
            V_Rd_c=30.0,
            link_angle_degrees=90.0,
        )
        assert low_fck == pytest.approx(min(700.0, 800.0), rel=1e-12)

        high_fck = _max_leg_spacing_eu_de(
            effective_depth=500.0,
            section_depth=700.0,
            f_ck=60.0,
            V_Ed=20.0,
            V_Rd_max=100.0,
            V_Rd_c=30.0,
            link_angle_degrees=90.0,
        )
        assert high_fck == pytest.approx(min(700.0, 600.0), rel=1e-12)

        # ratio > 0.3 branch
        high_ratio = _max_leg_spacing_eu_de(
            effective_depth=500.0,
            section_depth=700.0,
            f_ck=40.0,
            V_Ed=80.0,
            V_Rd_max=100.0,
            V_Rd_c=30.0,
            link_angle_degrees=90.0,
        )
        assert high_ratio == pytest.approx(min(700.0, 600.0), rel=1e-12)

    def test_max_leg_spacing_eu_de_invalid_section_depth(self):
        """Test max leg spacing eu de invalid section depth."""
        with pytest.raises(ValueError, match="section_depth must be > 0"):
            _max_leg_spacing_eu_de(
                effective_depth=500.0,
                section_depth=0.0,
                f_ck=40.0,
                V_Ed=20.0,
                V_Rd_max=100.0,
                V_Rd_c=30.0,
                link_angle_degrees=90.0,
            )
