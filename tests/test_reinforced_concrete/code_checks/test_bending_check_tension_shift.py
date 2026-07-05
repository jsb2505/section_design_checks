"""
Unit tests for BendingCheck tension shift refactored methods.

These tests use stubs/mocks to test the internal logic without
needing real section/concrete/diagram objects. This makes them:
- Fast (no diagram generation overhead)
- Precise (can test specific edge cases)
- Isolated (test one method at a time)

Complements the integration tests in test_bending_check.py which
test the full stack with real objects.
"""

import math
from types import SimpleNamespace
import pytest

from materials.reinforced_concrete.code_checks.ec2_2004.bending_check import BendingCheck


# -------------------------
# Test doubles / stubs
# -------------------------

class DummyDiagram:
    """Stub for MNInteractionDiagram that returns configurable values."""

    def __init__(self, *, d_mm=500.0, z_mm=450.0, eps_top=1.0, eps_bottom=-1.0):
        self.d = d_mm
        self.z = z_mm
        self.eps_top = eps_top
        self.eps_bottom = eps_bottom
        self.find_strains_calls = 0
        self.apply_tension_shift_calls = 0

        # Capacity return is configurable per test
        self.capacity = SimpleNamespace(N_Rd=100.0, M_Rd=200.0, is_safe=True, utilization=0.5)

    def find_strains_for_MN(self, M_Ed: float, N_Ed: float):
        self.find_strains_calls += 1
        return (self.eps_top, self.eps_bottom)

    def get_effective_depth(self, *, M_Ed, N_Ed, eps_top, eps_bottom):
        return float(self.d)

    def get_lever_arm(
        self,
        *,
        M_Ed,
        N_Ed,
        d,
        eps_top,
        eps_bottom,
        prefer_rigorous,
        cap_to_09d,
        warn_on_fallback,
    ):
        # Return (z_ec2, z_mech). We only care about z_ec2.
        return (float(self.z), float(self.z))

    def get_capacity_vector(self, *, N_Ed, M_Ed, return_details=False):
        return self.capacity

    def apply_tension_shift(
        self,
        *,
        M_Ed,
        V_Ed,
        N_Ed,
        M_cap,
        shear_reinforcement,
        iterate_z=False,
        cot_theta_override=None,
        use_v_rd_s_for_cot_theta=False,
        cot_max_override=None,
    ):
        """Stub for MNInteractionDiagram.apply_tension_shift."""
        from math import copysign
        from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import TensionShiftResult

        self.apply_tension_shift_calls += 1

        # Use z for shear reinforcement case, d otherwise
        if shear_reinforcement is not None:
            # a_l = 0.5 * z * cot_theta (assuming cot_theta=2.0 from patched shear_utils)
            cot_theta = 2.0
            a_l = 0.5 * self.z * cot_theta
        else:
            # a_l = d
            cot_theta = None
            a_l = self.d

        M_add = abs(V_Ed) * (a_l / 1000.0)
        abs_M_design = abs(M_Ed) + M_add

        capped = False
        if M_cap is not None:
            if abs_M_design > abs(M_cap):
                abs_M_design = abs(M_cap)
                capped = True

        M_design = copysign(abs_M_design, M_Ed)

        return TensionShiftResult(
            M_design=M_design,
            M_add=M_add,
            shift_distance_a_l=a_l,
            cot_theta=cot_theta,
            capped_by_M_cap=capped,
            z=self.z,
            d=self.d,
        )


def make_check(*, diagram: DummyDiagram) -> BendingCheck:
    """
    Build BendingCheck without running model_post_init (no real diagram creation).

    We bypass Pydantic's normal initialization by:
    1. Creating instance with object.__new__()
    2. Setting all required attributes directly
    3. Injecting our stub diagram
    """
    from materials.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType

    concrete = SimpleNamespace(
        f_ck=30.0,
        grade="C30/37",
        f_cd=20.0,
        f_cd_accidental=20.0,
        gamma_c=1.5,
        gamma_c_accidental=1.2,
        E_cm=33_000.0,
        model_dump=lambda: {"stub": "concrete"},
    )

    section = SimpleNamespace(
        section_name="test_section",
        reinforcement_ratio=0.0123,
        model_dump=lambda: {"stub": "section"},
    )

    # Bypass Pydantic's __init__ and model_post_init entirely
    check = object.__new__(BendingCheck)

    # Set public fields that Pydantic would normally set
    object.__setattr__(check, 'section', section)
    object.__setattr__(check, 'concrete', concrete)
    object.__setattr__(check, 'concrete_model_type', ConcreteModelType.PARABOLA_RECTANGLE)
    object.__setattr__(check, 'steel_model_type', SteelModelType.INCLINED)
    object.__setattr__(check, 'n_fibres_width', 20)
    object.__setattr__(check, 'n_fibres_height', 30)
    object.__setattr__(check, 'use_accidental', False)
    object.__setattr__(check, 'apply_tension_cot_theta_limit', True)
    object.__setattr__(check, 'concrete_model_override', None)
    object.__setattr__(check, 'steel_models_override', None)

    # Snapshot-based caching: pre-set snapshot so _get_diagram() returns the stub
    snapshot = {
        "section": {"stub": "section"},
        "concrete": {"stub": "concrete"},
        "concrete_model_type": ConcreteModelType.PARABOLA_RECTANGLE,
        "steel_model_type": SteelModelType.INCLINED,
        "n_fibres_width": 20,
        "n_fibres_height": 30,
        "use_accidental": False,
    }
    object.__setattr__(check, '_diagram', diagram)
    object.__setattr__(check, '_diagram_snapshot', snapshot)
    object.__setattr__(check, '_diagram_no_comp_steel', None)
    object.__setattr__(check, '_diagram_no_comp_snapshot', None)
    object.__setattr__(check, '_A_transformed', 100_000.0)  # mm², arbitrary

    # Monkeypatchable result creator (so tests don't depend on BaseCodeCheck internals)
    def _fake_create_result(**kwargs):
        # Return raw dict so assertions are easy
        return kwargs

    check._create_result = _fake_create_result  # type: ignore[method-assign]
    return check


# -------------------------
# shear_utils patch helpers
# -------------------------

@pytest.fixture
def patch_shear_utils(monkeypatch):
    """
    Patch shear_utils functions so the tests don't depend on real implementation or RCSection geometry.
    """
    import materials.reinforced_concrete.code_checks.ec2_2004.shear_utils as shear_utils

    monkeypatch.setattr(shear_utils, "sigma_cp_from_N_and_area", lambda *, N_Ed, A_mm2: 1.0)
    monkeypatch.setattr(shear_utils, "cap_sigma_cp_upper", lambda *, sigma_cp, f_cd: sigma_cp)
    monkeypatch.setattr(shear_utils, "find_alpha_cw", lambda *, f_cd, sigma_cp: 1.0)
    monkeypatch.setattr(shear_utils, "find_nu_factor", lambda *, f_ck: 1.0)
    monkeypatch.setattr(shear_utils, "calculate_section_breadth", lambda *, section: 300.0)

    # Default cot(theta)=2.0, can override per-test by monkeypatching again
    monkeypatch.setattr(
        shear_utils,
        "find_cot_theta_for_V_Ed_from_V_Rd_max",
        lambda *, V_Ed, K, link_angle_degrees: 2.0,
    )

    return shear_utils


# -------------------------
# Tests: _apply_tension_shift
# -------------------------

class TestTensionShiftDisabled:
    """Tests for when tension shift is disabled (no M_cap provided to _check_single_case)."""

    def test_returns_original_moment(self):
        """When M_cap is None (no tension shift), M_design should equal M_Ed."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        # Test via _check_single_case with M_cap=None (tension shift disabled)
        res = check._check_single_case(
            M_Ed=120.0,
            N_Ed=0.0,
            V_Ed=None,
            M_cap=None,  # No M_cap means no tension shift
            shear_reinforcement=None,
            warning_threshold=0.95,
        )

        assert res["details"]["tension_shift_applied"] is False
        assert res["details"]["M_Ed_design"] == 120.0

    def test_details_are_none(self):
        """When tension shift is disabled, all shift details should be None."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        res = check._check_single_case(
            M_Ed=120.0,
            N_Ed=0.0,
            V_Ed=None,
            M_cap=None,  # No M_cap means no tension shift
            shear_reinforcement=None,
            warning_threshold=0.95,
        )

        d = res["details"]
        assert d["tension_shift_applied"] is False
        assert d["M_add"] is None
        assert d["z_lever_arm"] is None
        assert d["cot_theta"] is None
        assert d["shift_distance_a_l"] is None


class TestTensionShiftNoShearReinforcement:
    """Tests for tension shift enabled without shear reinforcement (a_l = d)."""

    def test_uses_al_equals_d(self):
        """Without shear reinforcement, a_l should equal effective depth d."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        shift = check._diagram.apply_tension_shift(
            M_Ed=120.0,
            N_Ed=0.0,
            V_Ed=100.0,
            M_cap=300.0,
            shear_reinforcement=None,
        )

        assert math.isclose(shift.shift_distance_a_l, 500.0, rel_tol=1e-9)
        assert shift.cot_theta is None

    def test_uses_abs_V_Ed(self):
        """M_add should use abs(V_Ed), so negative V_Ed still adds moment."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        # M_add = abs(V)*d/1000 = abs(-100) * 500/1000 = 50 kN·m
        shift = check._diagram.apply_tension_shift(
            M_Ed=120.0,
            N_Ed=0.0,
            V_Ed=-100.0,  # Negative V_Ed
            M_cap=300.0,
            shear_reinforcement=None,
        )

        assert math.isclose(shift.M_add, 50.0, rel_tol=1e-9)

    def test_applies_magnitude_cap_and_restores_sign(self):
        """Should cap the magnitude and restore original moment sign."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        # M_add = abs(V)*d/1000 = 100 * 0.5 = 50
        # abs(M_orig)=120, abs(M_cap)=160 => min(160, 120+50)=160, sign positive => 160
        shift = check._diagram.apply_tension_shift(
            M_Ed=120.0,
            N_Ed=0.0,
            V_Ed=-100.0,
            M_cap=160.0,
            shear_reinforcement=None,
        )

        assert math.isclose(shift.M_add, 50.0, rel_tol=1e-9)
        assert shift.M_design == 160.0


class TestTensionShiftWithShearReinforcement:
    """Tests for tension shift enabled with shear reinforcement (a_l = 0.5·z·cot(θ))."""

    def test_uses_half_z_cot_theta(self, patch_shear_utils):
        """With shear reinforcement, a_l = 0.5 * z * cot(θ)."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        # Diagram stub uses cot_theta = 2.0
        # a_l = 0.5 * 450 * 2 = 450 mm
        shear_reinf = SimpleNamespace(angle=90.0)

        shift = check._diagram.apply_tension_shift(
            M_Ed=100.0,
            N_Ed=0.0,
            V_Ed=100.0,
            M_cap=300.0,
            shear_reinforcement=shear_reinf,
        )

        assert math.isclose(shift.cot_theta, 2.0, rel_tol=1e-9)
        assert math.isclose(shift.shift_distance_a_l, 450.0, rel_tol=1e-9)

    def test_restores_negative_sign(self, patch_shear_utils):
        """Negative moment sign should be restored after applying shift and cap."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        # Diagram stub uses cot_theta = 2.0
        # a_l = 0.5*z*cot = 0.5*450*2 = 450 mm
        # M_add = abs(100)*0.45 = 45 kN·m
        # abs(M_orig)=80, cap=200 => min(200, 80+45)=125, sign negative => -125
        shear_reinf = SimpleNamespace(angle=90.0)

        shift = check._diagram.apply_tension_shift(
            M_Ed=-80.0,
            N_Ed=100.0,
            V_Ed=100.0,
            M_cap=200.0,
            shear_reinforcement=shear_reinf,
        )

        assert math.isclose(shift.cot_theta, 2.0, rel_tol=1e-9)
        assert math.isclose(shift.shift_distance_a_l, 450.0, rel_tol=1e-9)
        assert math.isclose(shift.M_add, 45.0, rel_tol=1e-9)
        assert shift.M_design == -125.0


class TestTensionShiftCapBehavior:
    """Tests for M_cap limiting behavior."""

    def test_cap_smaller_than_original_reduces_to_cap(self):
        """When M_cap < M_Ed, design moment should be capped."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        # abs(M_orig)=200, cap=150
        # Even with M_add, result should be min(cap, ...)=150
        shift = check._diagram.apply_tension_shift(
            M_Ed=200.0,
            N_Ed=0.0,
            V_Ed=100.0,
            M_cap=150.0,
            shear_reinforcement=None,
        )

        assert shift.M_design == 150.0

    def test_cap_larger_than_shifted_uses_shifted(self):
        """When M_cap > M_Ed + M_add, design moment is the shifted value."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        # M_add = 100 * 500/1000 = 50
        # M_Ed + M_add = 100 + 50 = 150
        # M_cap = 300 > 150, so use 150
        shift = check._diagram.apply_tension_shift(
            M_Ed=100.0,
            N_Ed=0.0,
            V_Ed=100.0,
            M_cap=300.0,
            shear_reinforcement=None,
        )

        assert shift.M_design == 150.0


class TestStrainSolving:
    """Tests for strain solving via the diagram's apply_tension_shift."""

    def test_zero_moment_uses_diagram_apply_tension_shift(self):
        """Even with small moment, the diagram's apply_tension_shift is called."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        _ = check._diagram.apply_tension_shift(
            M_Ed=1e-12,  # Below 1e-6 threshold
            N_Ed=0.0,
            V_Ed=100.0,
            M_cap=200.0,
            shear_reinforcement=None,
        )

        # The diagram's apply_tension_shift should be called
        assert diag.apply_tension_shift_calls == 1

    def test_nonzero_moment_uses_diagram_apply_tension_shift(self):
        """With non-zero moment, the diagram's apply_tension_shift is called."""
        diag = DummyDiagram(d_mm=500.0, z_mm=450.0)
        check = make_check(diagram=diag)

        _ = check._diagram.apply_tension_shift(
            M_Ed=100.0,
            N_Ed=0.0,
            V_Ed=100.0,
            M_cap=200.0,
            shear_reinforcement=None,
        )

        assert diag.apply_tension_shift_calls == 1


# -------------------------
# Tests: _check_single_case edge cases
# -------------------------

class TestCheckSingleCaseInvalidCapacity:
    """Tests for _check_single_case handling invalid capacity results."""

    def test_none_capacity_returns_inf_utilization(self):
        """When diagram returns None capacity, utilization should be inf."""
        diag = DummyDiagram()
        diag.capacity = SimpleNamespace(N_Rd=None, M_Rd=None, is_safe=False, utilization=float("inf"))
        check = make_check(diagram=diag)

        res = check._check_single_case(
            M_Ed=120.0,
            N_Ed=0.0,
            V_Ed=100.0,
            M_cap=160.0,
            shear_reinforcement=None,
            warning_threshold=0.95,
        )

        assert res["utilization"] == float("inf")

    def test_invalid_capacity_includes_shift_details(self):
        """Even with invalid capacity, shift details should be included."""
        diag = DummyDiagram()
        diag.capacity = SimpleNamespace(N_Rd=None, M_Rd=None, is_safe=False, utilization=float("inf"))
        check = make_check(diagram=diag)

        res = check._check_single_case(
            M_Ed=120.0,
            N_Ed=0.0,
            V_Ed=100.0,
            M_cap=160.0,
            shear_reinforcement=None,
            warning_threshold=0.95,
        )

        assert res["details"]["tension_shift_applied"] is True
        assert res["details"]["M_Ed_original"] == 120.0
        assert res["details"]["M_Ed_design"] == 160.0  # Capped
        assert res["details"]["M_add"] is not None

    def test_nan_utilization_treated_as_invalid(self):
        """NaN utilization should be treated as invalid (returns inf)."""
        diag = DummyDiagram()
        diag.capacity = SimpleNamespace(N_Rd=0.0, M_Rd=0.0, is_safe=True, utilization=float("nan"))
        check = make_check(diagram=diag)

        res = check._check_single_case(
            M_Ed=50.0,
            N_Ed=0.0,
            V_Ed=None,
            M_cap=None,
            shear_reinforcement=None,
            warning_threshold=0.95,
        )

        assert res["utilization"] == float("inf")
        assert "no capacity found" in res["message"].lower()


class TestPerformCheckValidation:
    """Tests for perform_check input validation."""

    def test_raises_if_mcap_without_ved(self):
        """Should raise ValueError if M_cap provided but V_Ed is None."""
        diag = DummyDiagram()
        check = make_check(diagram=diag)

        with pytest.raises(ValueError, match="V_Ed must be provided"):
            check.perform_check(M_Ed=10.0, N_Ed=0.0, M_cap=100.0, V_Ed=None)
