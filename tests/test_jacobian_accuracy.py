"""
Test analytical Jacobian accuracy against numerical finite-difference.

If the analytical Jacobian has errors, the solver will take more iterations
and may not converge as efficiently.
"""

import numpy as np

from materials.core.geometry import Point2D
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from materials.reinforced_concrete.geometry import RebarGroup, create_rectangular_section
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar


def numerical_jacobian_2point(diagram, eps_top, eps_bottom, M_target, N_target, h=1e-8):
    """
    Compute Jacobian using 2-point finite difference (same as scipy).

    Args:
        diagram: MNInteractionDiagram
        eps_top, eps_bottom: Current strain state
        M_target, N_target: Target forces
        h: Step size

    Returns:
        2x2 Jacobian matrix
    """
    def residual(eps_pair):
        point = diagram.calculate_point_from_end_strains(eps_pair[0], eps_pair[1])
        return np.array([point.N - N_target, point.M - M_target])

    eps_pair = np.array([eps_top, eps_bottom])
    f0 = residual(eps_pair)

    jac = np.zeros((2, 2))

    # Column 0: derivative w.r.t. eps_top
    eps_perturbed = eps_pair.copy()
    eps_perturbed[0] += h
    f1 = residual(eps_perturbed)
    jac[:, 0] = (f1 - f0) / h

    # Column 1: derivative w.r.t. eps_bottom
    eps_perturbed = eps_pair.copy()
    eps_perturbed[1] += h
    f1 = residual(eps_perturbed)
    jac[:, 1] = (f1 - f0) / h

    return jac


def test_jacobian_accuracy_at_points():
    """Test Jacobian accuracy at various strain states."""
    print("\n=== Jacobian Accuracy Test ===\n")

    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    section.add_rebar_group(RebarGroup(rebar=rebar_20, positions=[
        Point2D(x=50, y=50), Point2D(x=250, y=50)
    ]))
    section.add_rebar_group(RebarGroup(rebar=rebar_20, positions=[
        Point2D(x=50, y=450), Point2D(x=250, y=450)
    ]))

    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(section=section, concrete=concrete, tension_stiffening=True)

    # Test at several load cases
    test_cases = [
        (50.0, 100.0, "Sagging, low axial"),
        (80.0, 200.0, "Sagging, med axial"),
        (-30.0, 150.0, "Hogging, low axial"),
        (0.0, 300.0, "Pure compression"),
    ]

    print(f"{'Case':<25} {'Max Error':<12} {'Frobenius Error':<18} {'Status':<10}")
    print("=" * 75)

    all_errors = []

    for M_target, N_target, label in test_cases:
        # Get solution
        eps_top, eps_bottom = diagram.find_strains_for_MN(My_target=M_target, N_target=N_target)

        # Analytical Jacobian
        jac_analytical = diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        # Numerical Jacobian (2-point FD like scipy)
        jac_numerical = numerical_jacobian_2point(diagram, eps_top, eps_bottom, M_target, N_target)

        # Error metrics
        error_matrix = jac_analytical - jac_numerical
        max_error = np.max(np.abs(error_matrix))
        frobenius_error = np.linalg.norm(error_matrix, 'fro')
        relative_error = frobenius_error / np.linalg.norm(jac_numerical, 'fro')

        all_errors.append(relative_error)

        status = "OK" if relative_error < 0.01 else "WARN"

        print(f"{label:<25} {max_error:11.6f}  {frobenius_error:17.6f}  {status:<10}")

        if relative_error > 0.01:
            print("  Analytical Jacobian:")
            print(f"    {jac_analytical}")
            print("  Numerical Jacobian:")
            print(f"    {jac_numerical}")
            print("  Error matrix:")
            print(f"    {error_matrix}")
            print()

    avg_relative_error = np.mean(all_errors)
    max_relative_error = np.max(all_errors)

    print("=" * 75)
    print(f"\nAverage relative error: {avg_relative_error:.6f}")
    print(f"Maximum relative error: {max_relative_error:.6f}")

    if max_relative_error < 0.01:
        print("\n[OK] Analytical Jacobian matches numerical to < 1% error")
    elif max_relative_error < 0.05:
        print("\n[WARN] Analytical Jacobian has 1-5% error - may affect convergence")
    else:
        print("\n[ERROR] Analytical Jacobian has >5% error - will hurt convergence!")

    return max_relative_error


def test_jacobian_consistency_with_residual():
    """
    Test that Jacobian correctly predicts residual changes.

    A good Jacobian should satisfy:
    residual(x + dx) ≈ residual(x) + J*dx
    """
    print("\n=== Jacobian Consistency Test ===\n")

    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    section.add_rebar_group(RebarGroup(rebar=rebar_20, positions=[
        Point2D(x=50, y=50), Point2D(x=250, y=50)
    ]))
    section.add_rebar_group(RebarGroup(rebar=rebar_20, positions=[
        Point2D(x=50, y=450), Point2D(x=250, y=450)
    ]))

    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(section=section, concrete=concrete, tension_stiffening=True)

    M_target = 50.0
    N_target = 100.0

    # Get solution
    eps_top, eps_bottom = diagram.find_strains_for_MN(My_target=M_target, N_target=N_target)

    def residual(eps_top, eps_bottom):
        point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        return np.array([point.N - N_target, point.M - M_target])

    # Current residual (should be ~0 at solution)
    r0 = residual(eps_top, eps_bottom)

    # Analytical Jacobian
    jac = diagram._compute_analytical_jacobian(eps_top, eps_bottom)

    # Test with small perturbations
    test_steps = [1e-5, 1e-4, 1e-3]

    print(f"{'Step Size':<12} {'Predicted Change':<18} {'Actual Change':<18} {'Error':<12}")
    print("=" * 65)

    for step in test_steps:
        # Perturb in direction [1, 1] (arbitrary direction)
        deps = np.array([step, step])

        # Predicted change: J * deps
        predicted_change = jac @ deps

        # Actual change
        r_new = residual(eps_top + deps[0], eps_bottom + deps[1])
        actual_change = r_new - r0

        # Error
        error = np.linalg.norm(actual_change - predicted_change)

        print(f"{step:<12.6f} {np.linalg.norm(predicted_change):<18.10f} "
              f"{np.linalg.norm(actual_change):<18.10f} {error:<12.10f}")

    print("\n[INFO] Error should decrease linearly with step size for accurate Jacobian")


if __name__ == "__main__":
    max_error = test_jacobian_accuracy_at_points()
    test_jacobian_consistency_with_residual()

    if max_error < 0.01:
        print("\n" + "="*75)
        print("CONCLUSION: Jacobian is accurate")
        print("="*75)
        print("The analytical Jacobian matches numerical FD to high precision.")
        print("The slower convergence must be due to other factors (solver tolerance,")
        print("initial guess quality, etc.), not Jacobian accuracy.")
    else:
        print("\n" + "="*75)
        print("CONCLUSION: Jacobian has errors")
        print("="*75)
        print(f"Maximum error: {max_error:.4f}")
        print("This will cause the solver to take more iterations or fail to converge.")
        print("Need to debug the analytical Jacobian implementation.")
