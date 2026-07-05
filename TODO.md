# TODO

## Jacobian convergence with tension stiffening

**Test:** `tests/test_sanity_checks.py::test_jacobian_convergence_with_options`

**Problem:** The case `(M=100, N=100)` with `tension_stiffening=True` fails to
converge — the solver lands at `(M=134.5, N=93.5)` with an error of 34.5 kN
(tolerance is 1.0 kN).

**Analysis:** This load point sits near the cracking transition where the
tension stiffening model introduces a discontinuity in the stress-strain curve.
The Newton-Raphson solver with analytical Jacobian converges to a wrong branch
or local minimum rather than the target equilibrium.

**Possible fixes:**
1. Improve the initial guess for strain states near the cracking transition
2. Add Jacobian smoothing or regularisation across the cracking strain
3. Implement a line-search or trust-region fallback when the Newton step
   overshoots across the discontinuity
4. Use a continuation/homotopy approach: solve without tension stiffening first,
   then gradually enable it using the previous solution as the initial guess
