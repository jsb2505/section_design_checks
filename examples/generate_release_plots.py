"""
Generate the gallery figures used in the README (saved to examples/plots/).

Requires the ``viz`` extra plus ``kaleido`` for static PNG export:

    pip install -e ".[viz]" kaleido

Run from the repository root or the examples directory:

    python examples/generate_release_plots.py
"""

# ruff: noqa: E402  (sys.path bootstrap must precede the materials imports)
import sys
import warnings
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from section_design_checks.reinforced_concrete.analysis import create_interaction_diagram
from section_design_checks.reinforced_concrete.analysis.biaxial_interaction import (
    BiaxialMNInteractionSurface,
)
from section_design_checks.reinforced_concrete.analysis.biaxial_interaction_viewer import (
    BiaxialInteractionViewer,
)
from section_design_checks.reinforced_concrete.analysis.mn_diagram_viewer import MNDiagramViewer
from section_design_checks.reinforced_concrete.analysis.stress_strain_viewer import StressStrainViewer
from section_design_checks.reinforced_concrete.code_checks.ec2_2004 import (
    CrackingCheck,
    LoadCase,
    LoadDuration,
    ShearCheck,
)
from section_design_checks.reinforced_concrete.geometry import (
    create_linear_rebar_layer,
    create_rectangular_section,
)
from section_design_checks.reinforced_concrete.geometry.section_viewer import SectionViewer
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar, ShearRebar

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def save(fig, name: str, *, width: int | None = None, height: int | None = None) -> None:
    if width or height:
        fig.update_layout(width=width, height=height)
    path = PLOTS_DIR / name
    fig.write_image(str(path), scale=2)
    print(f"  saved {path.relative_to(project_root)}")


def make_beam_section() -> tuple:
    """300x500 beam: 4H20 bottom, 2H16 top, H10 links at 150."""
    width, height = 300.0, 500.0
    cover, link_dia = 35.0, 10.0

    section = create_rectangular_section(width=width, height=height, section_name="Beam 300x500")

    bot_bar = Rebar(diameter=20, grade="B500B")
    top_bar = Rebar(diameter=16, grade="B500B")
    side_cover = cover + link_dia
    y_bot = cover + link_dia + bot_bar.diameter / 2.0
    y_top = height - cover - link_dia - top_bar.diameter / 2.0

    section.add_rebar_group(
        create_linear_rebar_layer(
            rebar=bot_bar,
            n_bars=4,
            start_point=(side_cover + bot_bar.diameter / 2.0, y_bot),
            end_point=(width - side_cover - bot_bar.diameter / 2.0, y_bot),
            layer_name="bottom",
        )
    )
    section.add_rebar_group(
        create_linear_rebar_layer(
            rebar=top_bar,
            n_bars=2,
            start_point=(side_cover + top_bar.diameter / 2.0, y_top),
            end_point=(width - side_cover - top_bar.diameter / 2.0, y_top),
            layer_name="top",
        )
    )

    concrete = ConcreteMaterial(grade="C30/37")
    return section, concrete


def make_column_section() -> tuple:
    """400x400 column: 8H25 perimeter arrangement."""
    size, cover, link_dia = 400.0, 35.0, 10.0
    bar = Rebar(diameter=25, grade="B500B")
    edge = cover + link_dia + bar.diameter / 2.0

    section = create_rectangular_section(width=size, height=size, section_name="Column 400x400")
    for n_bars, y in ((3, edge), (2, size / 2.0), (3, size - edge)):
        section.add_rebar_group(
            create_linear_rebar_layer(
                rebar=bar,
                n_bars=n_bars,
                start_point=(edge, y),
                end_point=(size - edge, y),
            )
        )

    concrete = ConcreteMaterial(grade="C35/45")
    return section, concrete


def main() -> None:
    beam, beam_concrete = make_beam_section()
    column, column_concrete = make_column_section()

    print("1/6 Section geometry...")
    fig = SectionViewer(beam).plot(concrete=beam_concrete, show=False, title="RC Section — Beam 300x500, C30/37")
    save(fig, "section_geometry.png")

    print("2/6 Uniaxial M-N interaction diagram...")
    diagram = create_interaction_diagram(section=beam, concrete=beam_concrete)
    load_points = [
        {"N_Ed": 400.0, "M_Ed": 150.0, "name": "LC1: DL+LL"},
        {"N_Ed": 1200.0, "M_Ed": 260.0, "name": "LC2: DL+Wind"},
        {"N_Ed": -150.0, "M_Ed": 90.0, "name": "LC3: Uplift"},
    ]
    fig = MNDiagramViewer(diagram).plot(
        load_points=load_points,
        show_vectors=True,
        show=False,
        title="M-N Interaction Diagram — Beam 300x500, C30/37",
    )
    save(fig, "mn_interaction_diagram.png")

    print("3/6 Stress-strain state...")
    fig = StressStrainViewer(diagram).plot(
        My_Ed=150.0,
        N_Ed=400.0,
        show=False,
        title="Strain Compatibility Solution — My_Ed = 150 kN·m, N_Ed = 400 kN",
    )
    save(fig, "stress_strain_state.png")

    print("4/6 Biaxial M-M-N interaction surface...")
    surface = BiaxialMNInteractionSurface(section=column, concrete=column_concrete)
    biaxial_loads = [
        {"N_Ed": 1500.0, "My_Ed": 180.0, "Mz_Ed": 120.0, "name": "LC1"},
        {"N_Ed": 2500.0, "My_Ed": 120.0, "Mz_Ed": 220.0, "name": "LC2"},
    ]
    fig = BiaxialInteractionViewer(surface).plot(
        load_points=biaxial_loads,
        show_vectors=True,
        n_angles=36,
        n_axial_levels=20,
        show=False,
        title="Biaxial M-M-N Interaction Surface — Column 400x400, C35/45",
    )
    fig.update_layout(scene_camera={"eye": {"x": 1.55, "y": 1.55, "z": 0.7}})
    save(fig, "biaxial_mn_surface.png", width=1000, height=800)

    print("5/6 Shear cot(theta) study...")
    links = ShearRebar(diameter=10, link_spacing=150, n_legs=2, angle=90.0, grade="B500B")
    shear_check = ShearCheck(
        section=beam,
        concrete=beam_concrete,
        shear_reinforcement=links,
        use_mechanical_lever_arm=True,
    )
    fig = shear_check.plot_cot_theta_study(
        load_case=LoadCase(V_Ed=250.0, M_Ed=100.0, N_Ed=150.0),
        n_points=60,
        show=False,
    )
    save(fig, "shear_cot_theta_study.png")

    print("6/6 Crack width contour map...")
    cracking_check = CrackingCheck(
        section=beam,
        concrete=beam_concrete,
        w_k_limit=0.3,
        load_duration=LoadDuration.LONG_TERM,
        creep_coefficient=1.5,
    )
    crack_loads = [
        {"name": "LC1", "M_Ed": 120, "N_Ed": 0},
        {"name": "LC2", "M_Ed": 180, "N_Ed": 300},
        {"name": "LC3", "M_Ed": 90, "N_Ed": -300},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = cracking_check.plot_crack_width_contours(
            load_cases=crack_loads,
            n_grid=25,
            show=False,
            title="Crack Width w_k (mm) — M-N Contour Map, Beam 300x500",
        )
    save(fig, "crack_width_contours.png")

    print("Done.")


if __name__ == "__main__":
    main()
