"""Phase 5: confined-concrete (Mander) ultimate strain.

Validates that:
1. effective_ultimate_strain() uses the corrected Mander/Priestley form
   eps_cu = 0.004 + 1.4·rho_s·f_yh·eps_su / f_cc (confined f_cc, explicit eps_su)
   and is NOT the previous over-predicting 0.004 + 0.14·rho_s·f_yh / f_co.
2. The 1D diagram and the biaxial surface agree.
3. The confined value actually reaches the solver bounds, so the confined M-N
   envelope is no longer identical to the unconfined one at high axial load.

NOTE (engineering sign-off): this corrects the confined ultimate strain and so
changes confined capacities — review against a validated Mander example before use.
"""

import pytest

from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from materials.reinforced_concrete.analysis.biaxial_interaction import BiaxialMNInteractionSurface


def _section():
    sec = create_rectangular_section(400, 400, section_name="Confined Column")
    r = Rebar(diameter=20, grade="B500B")
    for x, y in [(50, 50), (350, 50), (350, 350), (50, 350)]:
        sec.add_rebar_group(create_linear_rebar_layer(rebar=r, n_bars=1, start_point=(x, y), end_point=(x, y)))
    return sec


_CONF = dict(confined_concrete=True, confinement_rho_s=0.02, confinement_f_yh=500.0, confinement_eps_su=0.10)


def test_effective_ultimate_strain_corrected_value():
    diag = MNInteractionDiagram(_section(), ConcreteMaterial(grade="C30/37"), **_CONF)
    eff = diag.effective_ultimate_strain()
    assert eff == pytest.approx(0.03182, abs=2e-4)         # corrected Mander/Priestley
    assert eff != pytest.approx(0.05067, abs=1e-3)         # not the old over-predicting form


def test_unconfined_effective_ultimate_strain_unchanged():
    diag = MNInteractionDiagram(_section(), ConcreteMaterial(grade="C30/37"))
    assert diag.effective_ultimate_strain() == pytest.approx(0.0035, abs=1e-4)


def test_biaxial_effective_ultimate_strain_matches_1d():
    surf = BiaxialMNInteractionSurface(section=_section(), concrete=ConcreteMaterial(grade="C30/37"), **_CONF)
    assert surf.effective_ultimate_strain() == pytest.approx(0.03182, abs=2e-4)


def test_eps_su_out_of_range_rejected():
    with pytest.raises(ValueError, match="eps_su"):
        MNInteractionDiagram(
            _section(), ConcreteMaterial(grade="C30/37"),
            confined_concrete=True, confinement_rho_s=0.02, confinement_f_yh=500.0,
            confinement_eps_su=0.5,
        )


def test_confined_envelope_exceeds_unconfined_at_high_axial():
    c = ConcreteMaterial(grade="C30/37")
    unconf = MNInteractionDiagram(_section(), c, confined_concrete=False)
    conf = MNInteractionDiagram(_section(), c, **_CONF)
    n_hi = 3000.0  # high compression, where confinement ductility matters
    _, m_unconf, _ = unconf.get_capacity_fixed_n(n_hi)
    _, m_conf, _ = conf.get_capacity_fixed_n(n_hi)
    assert m_unconf is not None and m_conf is not None
    # Confinement must not reduce capacity and should increase it where the solver
    # can now reach the extended confined strains.
    assert m_conf >= m_unconf
    assert m_conf > m_unconf * 1.01
