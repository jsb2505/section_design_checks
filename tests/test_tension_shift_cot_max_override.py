"""Regression test: apply_tension_shift must honour cot_max_override during the
iterate_z loop.

Bug: the initial calculate_tension_shift call passed cot_max_override, but the
two re-solve calls inside the iterate_z convergence loop dropped it, so the
iterated result silently reverted to the NDP default cot(theta) cap (e.g. 2.5)
instead of a supplied cap (e.g. 1.25 for the UK NA with tension).
"""


from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from materials.reinforced_concrete.geometry import (
    create_linear_rebar_layer,
    create_rectangular_section,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar, ShearRebar


def _diagram() -> MNInteractionDiagram:
    section = create_rectangular_section(300, 500, section_name="Beam")
    rebar_20 = Rebar(diameter=20, grade="B500B")
    # Bottom + top layers so the mechanical lever arm can be computed.
    section.add_rebar_group(
        create_linear_rebar_layer(rebar=rebar_20, n_bars=3, start_point=(50, 50), end_point=(250, 50))
    )
    section.add_rebar_group(
        create_linear_rebar_layer(rebar=rebar_20, n_bars=2, start_point=(50, 450), end_point=(250, 450))
    )
    return MNInteractionDiagram(section, ConcreteMaterial(grade="C30/37"))


def _cot_theta(cot_max_override):
    """cot(theta) from an iterated tension-shift solve with low shear (cot wants to be high)."""
    links = ShearRebar(grade="B500B", diameter=10, link_spacing=200, n_legs=2, angle=90.0)
    result = _diagram().apply_tension_shift(
        M_Ed=150.0,
        V_Ed=40.0,        # low shear -> natural cot(theta) tends to the upper cap
        N_Ed=0.0,
        shear_reinforcement=links,
        iterate_z=True,
        use_mechanical_lever_arm=True,
        cot_max_override=cot_max_override,
    )
    return result.cot_theta


def test_iterate_z_honours_cot_max_override():
    capped = _cot_theta(cot_max_override=1.25)
    uncapped = _cot_theta(cot_max_override=2.5)

    # The scenario must actually exercise the cap (else the test proves nothing):
    # without the tight cap, cot(theta) sits above 1.25.
    assert uncapped > 1.26, f"scenario does not bind the cap (uncapped cot={uncapped})"

    # With the 1.25 override the iterated result must respect it (was ~2.5 before the fix).
    assert capped <= 1.25 + 1e-6, f"iterate_z ignored cot_max_override (cot={capped})"
