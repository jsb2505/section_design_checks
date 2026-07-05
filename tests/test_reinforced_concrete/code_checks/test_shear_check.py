"""
Tests for ShearCheck with accidental limit state support.
"""

import warnings
import pytest
from materials.core.geometry import Point2D
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck
from materials.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase
from materials.reinforced_concrete.geometry import create_rectangular_section, RebarGroup
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar, Rebar
from materials.reinforced_concrete.ndp import CountryCode, get_ndp_context, set_ndp_context


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
        shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

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
        shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

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
        shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

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


class TestShearSpacingNDP:
    """Tests for spacing limit behavior with NDP-dependent rules."""

    def test_eu_de_spacing_exceedance_warns_and_sets_flag(self):
        """Test eu de spacing exceedance warns and sets flag."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(diameter=10, link_spacing=350, n_legs=2, grade="B500B")
        check = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_increased_nu_1=False,
        )
        load_case = LoadCase(V_Ed=250, M_Ed=50, N_Ed=100)

        old_code, old_country = get_ndp_context()
        try:
            set_ndp_context(country=CountryCode.EU_DE)
            with pytest.warns(UserWarning, match="maximum allowable spacing"):
                result = check.perform_check(load_case=load_case)
        finally:
            set_ndp_context(code=old_code, country=old_country)

        assert result.details["link_spacing_satisfied"] is False
        assert result.details["link_spacing_provided"] == pytest.approx(350.0)
        assert result.details["link_spacing_max_allowable"] is not None
        assert result.details["link_spacing_provided"] > result.details["link_spacing_max_allowable"]

    def test_eu_de_spacing_exceedance_can_suppress_warning(self):
        """Test eu de spacing exceedance can suppress warning."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(diameter=10, link_spacing=350, n_legs=2, grade="B500B")
        check = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_increased_nu_1=False,
        )
        load_case = LoadCase(V_Ed=250, M_Ed=50, N_Ed=100)

        old_code, old_country = get_ndp_context()
        try:
            set_ndp_context(country=CountryCode.EU_DE)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = check.perform_check(load_case=load_case, suppress_warnings=True)
        finally:
            set_ndp_context(code=old_code, country=old_country)

        assert len(caught) == 0
        assert result.details["link_spacing_satisfied"] is False

    def test_leg_spacing_check_runs_only_when_leg_spacing_is_provided(self):
        """Test leg spacing check runs only when leg spacing is provided."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")
        check = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_increased_nu_1=False,
        )
        load_case = LoadCase(V_Ed=250, M_Ed=50, N_Ed=100)

        old_code, old_country = get_ndp_context()
        try:
            set_ndp_context(country=CountryCode.EU_DE)
            result = check.perform_check(load_case=load_case, suppress_warnings=True)
        finally:
            set_ndp_context(code=old_code, country=old_country)

        assert result.details["leg_spacing_satisfied"] is None
        assert result.details["leg_spacing_provided"] is None
        assert result.details["leg_spacing_max_allowable"] is None

    def test_eu_de_leg_spacing_exceedance_warns_and_can_be_suppressed(self):
        """Test eu de leg spacing exceedance warns and can be suppressed."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(
            diameter=10,
            link_spacing=200,
            leg_spacing=550,
            n_legs=2,
            grade="B500B",
        )
        check = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_increased_nu_1=False,
        )
        load_case = LoadCase(V_Ed=250, M_Ed=50, N_Ed=100)

        old_code, old_country = get_ndp_context()
        try:
            set_ndp_context(country=CountryCode.EU_DE)
            with pytest.warns(UserWarning, match="leg spacing"):
                result_warn = check.perform_check(load_case=load_case)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result_suppressed = check.perform_check(load_case=load_case, suppress_warnings=True)
        finally:
            set_ndp_context(code=old_code, country=old_country)

        assert len(caught) == 0
        assert result_warn.details["leg_spacing_satisfied"] is False
        assert result_warn.details["leg_spacing_provided"] == pytest.approx(550.0)
        assert result_warn.details["leg_spacing_max_allowable"] is not None
        assert result_warn.details["leg_spacing_provided"] > result_warn.details["leg_spacing_max_allowable"]
        assert result_suppressed.details["leg_spacing_satisfied"] is False


class TestShearUncrackedVrdc:
    """Tests for optional use of uncracked V_Rd,c in ShearCheck."""

    def test_reports_both_cracked_and_uncracked_vrdc_by_default(self):
        """Test reports both cracked and uncracked vrdc by default."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        check = ShearCheck(section=section, concrete=concrete)
        load_case = LoadCase(V_Ed=80, M_Ed=40, N_Ed=100)

        result = check.perform_check(load_case=load_case, suppress_warnings=True)

        assert result.details["use_uncracked_V_Rd_c"] is False
        assert result.details["V_Rd_c"] == pytest.approx(result.details["V_Rd_c_cracked"], rel=1e-12)
        assert result.details["V_Rd_c_uncracked"] is not None

    def test_uses_uncracked_vrdc_when_requested(self):
        """Test uses uncracked vrdc when requested."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        check = ShearCheck(section=section, concrete=concrete)
        load_case = LoadCase(V_Ed=80, M_Ed=40, N_Ed=100)

        result = check.perform_check(
            load_case=load_case,
            use_uncracked_V_Rd_c=True,
            suppress_warnings=True,
        )

        assert result.details["use_uncracked_V_Rd_c"] is True
        assert result.details["V_Rd_c"] == pytest.approx(result.details["V_Rd_c_uncracked"], rel=1e-12)

    def test_reinforced_flow_continues_when_uncracked_vrdc_is_exceeded(self):
        """Test reinforced flow continues when uncracked vrdc is exceeded."""
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")
        check = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_rebar,
            use_increased_nu_1=False,
        )
        load_case = LoadCase(V_Ed=300, M_Ed=50, N_Ed=100)

        result = check.perform_check(
            load_case=load_case,
            use_uncracked_V_Rd_c=True,
            suppress_warnings=True,
        )

        assert result.details["V_Ed"] > result.details["V_Rd_c_uncracked"]
        assert result.details["V_Rd_s"] is not None
        assert result.details["V_Rd_max"] is not None
        assert result.details["governing_component"] in {"V_Rd_s", "V_Rd_max"}


class TestTensionCotThetaLimit:
    """Tests for UK NA §6.2.3(2) cot(θ) limit with external tension."""

    @pytest.fixture(autouse=True)
    def _use_uk_na(self):
        """Switch to EU_UK NDP for these tests."""
        old_code, old_country = get_ndp_context()
        set_ndp_context(country=CountryCode.EU_UK)
        yield
        set_ndp_context(code=old_code, country=old_country)

    def _make_check(self, *, apply_limit: bool = True) -> ShearCheck:
        section = create_test_section()
        concrete = ConcreteMaterial(grade="C30/37")
        links = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")
        return ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=links,
            apply_tension_cot_theta_limit=apply_limit,
        )

    def test_tension_cot_theta_clamped_to_1_25_under_eu_uk(self):
        """With EU_UK + tensile N_Ed, cot(θ) should be ≤ 1.25."""
        check = self._make_check(apply_limit=True)
        # Use tensile N_Ed (negative = tension)
        load_case = LoadCase(V_Ed=150, N_Ed=-200)
        result = check.perform_check(load_case=load_case, suppress_warnings=True)

        cot_theta = result.details["cot_theta"]
        assert cot_theta is not None
        assert cot_theta <= 1.25 + 1e-9

    def test_tension_cot_theta_not_limited_when_opt_out(self):
        """With apply_tension_cot_theta_limit=False, standard limit applies."""
        check = self._make_check(apply_limit=False)
        load_case = LoadCase(V_Ed=150, N_Ed=-200)
        result = check.perform_check(load_case=load_case, suppress_warnings=True)

        cot_theta = result.details["cot_theta"]
        assert cot_theta is not None
        # Standard EU_UK limit is 2.5 — cot_theta should be able to exceed 1.25
        # (it will be whatever the formula gives, up to 2.5)
        # We just verify it's not forcibly clamped to 1.25
        _, cot_max = check._find_cot_theta_limits(sigma_cp=-1.0, z=400.0, V_Ed=150.0)
        assert cot_max == pytest.approx(2.5, rel=1e-9)

    def test_compression_not_affected_by_tension_limit(self):
        """With compressive N_Ed, cot(θ) should NOT be clamped to 1.25."""
        check = self._make_check(apply_limit=True)
        load_case = LoadCase(V_Ed=150, N_Ed=200)
        result = check.perform_check(load_case=load_case, suppress_warnings=True)

        cot_theta = result.details["cot_theta"]
        assert cot_theta is not None
        # cot_theta may naturally be < 1.25 due to strut formula,
        # but the limit should be the standard 2.5
        _, cot_max = check._find_cot_theta_limits(sigma_cp=1.0, z=400.0, V_Ed=150.0)
        assert cot_max == pytest.approx(2.5, rel=1e-9)

    def test_base_eu_tension_not_affected(self):
        """With base EU NDP (not UK), tension should NOT trigger reduced limit."""
        old_code, old_country = get_ndp_context()
        set_ndp_context(country=CountryCode.EU)
        try:
            check = self._make_check(apply_limit=True)
            _, cot_max = check._find_cot_theta_limits(sigma_cp=-1.0, z=400.0, V_Ed=150.0)
            assert cot_max == pytest.approx(2.5, rel=1e-9)
        finally:
            set_ndp_context(code=old_code, country=old_country)


class TestUncrackedVRdcRotatedFirstMoment:
    """find_V_Rd_c_uncracked must take the first moment S about the SAME axis as
    I_eff. For a non-square section the 90° (minor-axis) result must use the
    vertical-axis first moment, not the horizontal one."""

    def test_90deg_uses_vertical_axis_first_moment(self):
        import math

        section = create_rectangular_section(width=300, height=500)  # non-square
        check = ShearCheck(section=section, concrete=ConcreteMaterial(grade="C30/37"))
        sigma_cp, b_w = 2.0, 200.0

        v0 = check.find_V_Rd_c_uncracked(sigma_cp=sigma_cp, b_w=b_w, shear_angle_deg=0.0)
        v90 = check.find_V_Rd_c_uncracked(sigma_cp=sigma_cp, b_w=b_w, shear_angle_deg=90.0)

        # Independent reference for 90°: I_yy with the vertical-axis first moment of
        # the right half of the 300-wide rectangle (A=150*500, centroid x=225, cx=150).
        _, I_yy, _ = section.get_second_moment_area()
        cx, _cy = section.get_centroid()
        S_y = (150.0 * 500.0) * (225.0 - cx)
        f_ctd = check.f_ctd_design
        v90_ref = (I_yy * b_w / S_y) * math.sqrt(f_ctd**2 + sigma_cp * f_ctd) / 1000.0

        assert v90 == pytest.approx(v90_ref, rel=1e-3)
        # For a non-square section the minor-axis result genuinely differs from 0°.
        assert v90 != pytest.approx(v0, rel=1e-2)


class TestRequiredReinforcementMz:
    """get_required_shear_reinforcement must not silently ignore Mz_Ed. With a
    non-biaxial diagram it now raises clearly instead of using a My-only state."""

    def test_mz_without_biaxial_diagram_raises(self):
        check = ShearCheck(section=create_test_section(), concrete=ConcreteMaterial(grade="C30/37"))
        with pytest.raises(ValueError, match="biaxial"):
            check.get_required_shear_reinforcement(
                V_Ed=200.0, My_Ed=50.0, N_Ed=0.0, Mz_Ed=30.0,
            )

    def test_uniaxial_unaffected(self):
        # Mz_Ed=0 keeps the original behaviour (no exception, returns a value).
        check = ShearCheck(
            section=create_test_section(),
            concrete=ConcreteMaterial(grade="C30/37"),
            shear_reinforcement=ShearRebar(grade="B500B", diameter=10, link_spacing=200, n_legs=2),
        )
        req = check.get_required_shear_reinforcement(V_Ed=200.0, My_Ed=50.0, N_Ed=0.0)
        assert req >= 0.0

