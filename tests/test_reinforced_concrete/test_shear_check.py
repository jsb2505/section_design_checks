"""
Tests for ShearCheck with accidental limit state support.
"""

import pytest
from materials.core.geometry import Point2D
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck
from materials.reinforced_concrete.geometry import create_rectangular_section, RebarGroup
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar, Rebar


def create_test_section():
    """Create a simple rectangular section with tension reinforcement."""
    # With hook_ref=1 (default): section from (0, 0) to (300, 500)
    section = create_rectangular_section(width=300, height=500)

    # Add 2H20 bars at bottom (tension for sagging)
    # Section bounds: (0, 0) to (300, 500)
    # Place bars 50mm from bottom: y = 0 + 50 = 50
    rebar_20 = Rebar(diameter=20, grade="B500B")
    positions = [Point2D(x=100, y=50), Point2D(x=200, y=50)]
    group = RebarGroup(rebar=rebar_20, positions=positions)
    section.add_rebar_group(group)

    return section


class TestShearCheckAccidental:
    """Test accidental limit state functionality in ShearCheck."""

    def test_default_uses_design_strengths(self):
        """Test that default behavior uses design strengths (not accidental)."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")

        check = ShearCheck(
            section=section,
            concrete=concrete,
        )

        # Should use f_cd (not f_cd_accidental)
        assert check.f_cd_design == concrete.f_cd
        assert check.f_cd_design != concrete.f_cd_accidental

        # Should use gamma_c (not gamma_c_accidental)
        assert check.gamma_c_design == concrete.gamma_c
        assert check.gamma_c_design != concrete.gamma_c_accidental

    def test_accidental_uses_accidental_strengths(self):
        """Test that use_accidental=True uses accidental strengths."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")

        check = ShearCheck(
            section=section,
            concrete=concrete,
            use_accidental=True,
        )

        # Should use f_cd_accidental
        assert check.f_cd_design == concrete.f_cd_accidental
        assert check.f_cd_design != concrete.f_cd

        # Should use gamma_c_accidental
        assert check.gamma_c_design == concrete.gamma_c_accidental
        assert check.gamma_c_design != concrete.gamma_c

    def test_accidental_higher_capacity(self):
        """Test that accidental limit state gives higher capacity."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")

        check_design = ShearCheck(
            section=section,
            concrete=concrete,
            use_accidental=False,
        )

        check_accidental = ShearCheck(
            section=section,
            concrete=concrete,
            use_accidental=True,
        )

        # Accidental should have higher f_cd (lower gamma_c)
        assert check_accidental.f_cd_design > check_design.f_cd_design

        # Accidental should have lower gamma_c
        assert check_accidental.gamma_c_design < check_design.gamma_c_design

        # V_Rd_c should be higher for accidental
        # Compute required parameters for typical load case
        M_Ed, N_Ed = 50.0, 100.0  # Typical sagging moment with compression
        d_design = check_design.find_effective_depth(M_Ed, N_Ed)
        rho_l_design = check_design._find_rho_l(M_Ed, N_Ed, d_design)
        sigma_cp_design = check_design._find_sigma_cp(N_Ed)

        d_acc = check_accidental.find_effective_depth(M_Ed, N_Ed)
        rho_l_acc = check_accidental._find_rho_l(M_Ed, N_Ed, d_acc)
        sigma_cp_acc = check_accidental._find_sigma_cp(N_Ed)

        V_Rd_c_design = check_design.find_V_Rd_c(d_design, rho_l_design, sigma_cp_design)
        V_Rd_c_accidental = check_accidental.find_V_Rd_c(d_acc, rho_l_acc, sigma_cp_acc)
        assert V_Rd_c_accidental > V_Rd_c_design

    def test_shear_reinforcement_accidental(self):
        """Test that shear reinforcement also uses accidental strength."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(diameter=10, spacing=200, n_legs=2, grade="B500B")

        check_design = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_accidental=False,
        )

        check_accidental = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_accidental=True,
        )

        # f_ywd should be higher for accidental
        assert check_accidental.f_ywd_design > check_design.f_ywd_design

        # Specifically: should use f_yd vs f_yd_accidental
        assert check_design.f_ywd_design == shear_rebar.f_yd
        assert check_accidental.f_ywd_design == shear_rebar.f_yd_accidental

        # V_Rd_s should be higher for accidental
        cot_theta = 2.5
        # Compute lever arm for typical load case
        M_Ed, N_Ed = 50.0, 100.0
        d_design = check_design.find_effective_depth(M_Ed, N_Ed)
        z_design, _ = check_design.find_lever_arm(M_Ed, N_Ed, d_design)

        d_acc = check_accidental.find_effective_depth(M_Ed, N_Ed)
        z_acc, _ = check_accidental.find_lever_arm(M_Ed, N_Ed, d_acc)

        V_Rd_s_design = check_design.find_V_Rd_s(cot_theta, z_design)
        V_Rd_s_accidental = check_accidental.find_V_Rd_s(cot_theta, z_acc)
        assert V_Rd_s_accidental > V_Rd_s_design

    def test_V_Rd_max_accidental(self):
        """Test that V_Rd_max also uses accidental f_cd."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(diameter=10, spacing=200, n_legs=2, grade="B500B")

        check_design = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_accidental=False,
        )

        check_accidental = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_accidental=True,
        )

        # V_Rd_max should be higher for accidental
        cot_theta = 2.5
        # Compute required parameters
        M_Ed, N_Ed = 50.0, 100.0
        d_design = check_design.find_effective_depth(M_Ed, N_Ed)
        z_design, _ = check_design.find_lever_arm(M_Ed, N_Ed, d_design)
        sigma_cp_design = check_design._find_sigma_cp(N_Ed)

        d_acc = check_accidental.find_effective_depth(M_Ed, N_Ed)
        z_acc, _ = check_accidental.find_lever_arm(M_Ed, N_Ed, d_acc)
        sigma_cp_acc = check_accidental._find_sigma_cp(N_Ed)

        V_Rd_max_design = check_design.find_V_Rd_max(cot_theta, z_design, sigma_cp_design)
        V_Rd_max_accidental = check_accidental.find_V_Rd_max(cot_theta, z_acc, sigma_cp_acc)
        assert V_Rd_max_accidental > V_Rd_max_design

    def test_sigma_cp_limit_accidental(self):
        """Test that sigma_cp limit uses accidental f_cd."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")

        # Apply large axial force to trigger the 0.2*f_cd limit
        N_Ed = 5000  # kN (large compression)

        check_design = ShearCheck(
            section=section,
            concrete=concrete,
            N_Ed=N_Ed,
            use_accidental=False,
        )

        check_accidental = ShearCheck(
            section=section,
            concrete=concrete,
            N_Ed=N_Ed,
            use_accidental=True,
        )

        # Both should hit the 0.2*f_cd limit, but accidental has higher f_cd
        sigma_cp_design = check_design._find_sigma_cp(N_Ed)
        sigma_cp_accidental = check_accidental._find_sigma_cp(N_Ed)

        assert sigma_cp_accidental > sigma_cp_design

        # Verify the limit is 0.2*f_cd
        assert sigma_cp_design == pytest.approx(0.2 * check_design.f_cd_design)
        assert sigma_cp_accidental == pytest.approx(0.2 * check_accidental.f_cd_design)

    def test_no_shear_reinforcement_f_ywd_design(self):
        """Test f_ywd_design returns 0 when no shear reinforcement."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")

        check = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=None,
        )

        assert check.f_ywd_design == 0.0

    def test_required_shear_reinforcement_accidental(self):
        """Test that required shear reinforcement calculation uses accidental strength."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(diameter=10, spacing=200, n_legs=2, grade="B500B")

        V_Ed = 200  # kN (requires shear reinforcement)
        M_Ed, N_Ed = 50.0, 100.0  # Typical load case
        cot_theta = 2.5

        check_design = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_accidental=False,
        )

        check_accidental = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_accidental=True,
        )

        # Required A_sw/s should be lower for accidental (higher f_ywd)
        A_sw_s_design = check_design.get_required_shear_reinforcement(V_Ed, M_Ed, N_Ed, cot_theta)
        A_sw_s_accidental = check_accidental.get_required_shear_reinforcement(V_Ed, M_Ed, N_Ed, cot_theta)

        # Accidental should require less reinforcement due to higher strength
        assert A_sw_s_accidental < A_sw_s_design


class TestShearCheckBasicFunctionality:
    """Test basic ShearCheck functionality (not specific to accidental)."""

    def test_create_shear_check(self):
        """Test basic creation of ShearCheck."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")

        check = ShearCheck(
            section=section,
            concrete=concrete,
        )

        assert check.section == section
        assert check.concrete == concrete
        assert check.shear_reinforcement is None
        assert check.use_accidental is False

    def test_effective_depth_tension_bottom(self):
        """Test effective depth calculation with tension at bottom."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")

        check = ShearCheck(
            section=section,
            concrete=concrete,
            is_tension_bottom=True,
        )

        # Effective depth should be measured from top to tension steel
        M_Ed, N_Ed = 50.0, 100.0  # Typical sagging moment
        d = check.find_effective_depth(M_Ed, N_Ed)
        assert d > 0

    def test_effective_depth_tension_top(self):
        """Test effective depth calculation with tension at top."""
        # With hook_ref=1 (default): section from (0, 0) to (300, 500)
        section = create_rectangular_section(width=300, height=500)

        # Add 2H20 bars at top (tension for hogging)
        # Section bounds: (0, 0) to (300, 500)
        # Place bars 50mm from top: y = 500 - 50 = 450
        rebar_20 = Rebar(diameter=20, grade="B500B")
        positions = [Point2D(x=100, y=450), Point2D(x=200, y=450)]
        group = RebarGroup(rebar=rebar_20, positions=positions)
        section.add_rebar_group(group)

        concrete = ConcreteMaterial(grade="C30/37")

        check = ShearCheck(
            section=section,
            concrete=concrete,
            is_tension_bottom=False,
        )

        # Effective depth should be measured from bottom to top steel
        M_Ed, N_Ed = -50.0, 100.0  # Hogging moment (negative)
        d = check.find_effective_depth(M_Ed, N_Ed)
        assert d > 0
