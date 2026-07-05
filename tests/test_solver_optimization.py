"""
Test that the M-N solver is only invoked when M_Ed is non-zero.

Optimization: When M_Ed = 0 (pure shear or pure axial), we can skip the solver
and just assume top compression, since moment (not axial load) determines the
compression face.
"""

from unittest.mock import patch, MagicMock
from materials.core.geometry import Point2D
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck
from materials.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase
from materials.reinforced_concrete.geometry import create_rectangular_section, RebarGroup
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar, Rebar


def test_solver_not_called_when_M_zero():
    """Verify that find_strains_for_MN is NOT called when M_Ed = 0."""
    # Create section with symmetric reinforcement
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    # Bottom bars
    bottom_positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    # Top bars (needed for hogging cases)
    top_positions = [Point2D(x=50, y=450), Point2D(x=250, y=450)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

    check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=False,  # Approximate mode
    )

    print("\n=== Test: Solver NOT called when M_Ed = 0 ===")

    # Force lazy diagram creation, then patch the find_strains_for_MN method
    diagram = check._get_diagram()
    with patch.object(
        diagram, 'find_strains_for_MN', wraps=diagram.find_strains_for_MN
    ) as mock_solver:

        # Case 1: Pure shear (M=0, N=0)
        print("\nCase 1: Pure shear (M=0, N=0)")
        result1 = check.perform_check(load_case=LoadCase(V_Ed=100, M_Ed=0, N_Ed=0))
        print(f"  Status: {result1.status}, d={result1.details['d']:.1f} mm")
        print(f"  Solver called: {mock_solver.called}")
        assert not mock_solver.called, "Solver should NOT be called when M_Ed=0"

        mock_solver.reset_mock()

        # Case 2: Pure axial (M=0, N>0)
        print("\nCase 2: Pure axial (M=0, N=500)")
        result2 = check.perform_check(load_case=LoadCase(V_Ed=100, M_Ed=0, N_Ed=500))
        print(f"  Status: {result2.status}, d={result2.details['d']:.1f} mm")
        print(f"  Solver called: {mock_solver.called}")
        assert not mock_solver.called, "Solver should NOT be called when M_Ed=0 (even with N_Ed)"

        mock_solver.reset_mock()

        # Case 3: With moment (M>0, N=0)
        print("\nCase 3: With moment (M=50, N=0)")
        result3 = check.perform_check(load_case=LoadCase(V_Ed=100, M_Ed=50, N_Ed=0))
        print(f"  Status: {result3.status}, d={result3.details['d']:.1f} mm")
        print(f"  Solver called: {mock_solver.called}")
        assert mock_solver.called, "Solver SHOULD be called when M_Ed != 0"

        mock_solver.reset_mock()

        # Case 4: With moment and axial (M>0, N>0)
        print("\nCase 4: With moment and axial (M=50, N=500)")
        result4 = check.perform_check(load_case=LoadCase(V_Ed=100, M_Ed=50, N_Ed=500))
        print(f"  Status: {result4.status}, d={result4.details['d']:.1f} mm")
        print(f"  Solver called: {mock_solver.called}")
        assert mock_solver.called, "Solver SHOULD be called when M_Ed != 0"

    print("\n[OK] Solver optimization verified!")


def test_results_consistent_with_optimization():
    """Verify that results are correct with the optimization."""
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    # Add symmetric reinforcement
    bottom_positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    top_positions = [Point2D(x=50, y=450), Point2D(x=250, y=450)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

    print("\n=== Test: Results Consistent with Optimization ===")

    # Approximate mode
    check_approx = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=False,
    )

    # Rigorous mode
    check_rigorous = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=True,
    )

    # Test pure axial: M=0, N=500
    print("\nPure axial (M=0, N=500):")
    result_approx = check_approx.perform_check(load_case=LoadCase(V_Ed=100, M_Ed=0, N_Ed=500))
    result_rigorous = check_rigorous.perform_check(load_case=LoadCase(V_Ed=100, M_Ed=0, N_Ed=500))

    print(f"  Approximate: d={result_approx.details['d']:.1f} mm, z={result_approx.details['z']:.1f} mm")
    print(f"  Rigorous:    d={result_rigorous.details['d']:.1f} mm, z={result_rigorous.details['z']:.1f} mm")

    # Both should detect top compression (same d)
    assert abs(result_approx.details['d'] - result_rigorous.details['d']) < 0.1, \
        "Pure axial should give same compression face in both modes"

    # Test pure shear: M=0, N=0
    print("\nPure shear (M=0, N=0):")
    result_approx = check_approx.perform_check(load_case=LoadCase(V_Ed=100, M_Ed=0, N_Ed=0))
    result_rigorous = check_rigorous.perform_check(load_case=LoadCase(V_Ed=100, M_Ed=0, N_Ed=0))

    print(f"  Approximate: d={result_approx.details['d']:.1f} mm, z={result_approx.details['z']:.1f} mm")
    print(f"  Rigorous:    d={result_rigorous.details['d']:.1f} mm, z={result_rigorous.details['z']:.1f} mm")

    # Both should detect top compression (same d)
    assert abs(result_approx.details['d'] - result_rigorous.details['d']) < 0.1, \
        "Pure shear should give same compression face in both modes"

    print("\n[OK] Results are consistent!")


if __name__ == "__main__":
    test_solver_not_called_when_M_zero()
    test_results_consistent_with_optimization()
    print("\n=== All solver optimization tests passed! ===")

