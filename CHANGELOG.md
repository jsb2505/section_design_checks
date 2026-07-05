# Changelog

All notable changes to this project are recorded here. The project is in early
(alpha) development, so breaking changes may occur between minor versions.

## [Unreleased]

_Nothing yet._

## [0.1.0] — 2026-07-05

First public release, published to PyPI as
[`section-design-checks`](https://pypi.org/project/section-design-checks/).

### Changed — BREAKING: import package renamed `materials` → `section_design_checks`

In preparation for a PyPI listing the distribution is now named
**`section-design-checks`** (the previous distribution name `materials` is
already taken on PyPI by an unrelated project, whose import package would also
collide on disk) and the import package is renamed to match:

```python
# before
from materials.reinforced_concrete.materials import ConcreteMaterial
# after
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial
```

**Migration:** replace the `materials.` prefix with `section_design_checks.` in
imports (the `reinforced_concrete.materials` *sub*-package keeps its name), and
reinstall the environment: `pip uninstall materials` then `pip install -e .`.

### Added — PyPI release workflow

`.github/workflows/release.yml` builds and publishes to PyPI via Trusted
Publishing (OIDC) whenever a GitHub Release is published. See the comment in
the workflow for the one-time pending-publisher setup on pypi.org.

### Fixed — biaxial surface plotting crash on sparse grids

`BiaxialMNInteractionSurface._downsample_surface` now selects points via the
stored `(i_axial, j_angle)` grid metadata when some dense-grid points fail to
converge, and returns output-grid indices for downstream reshaping.
Previously a sparse dense grid was passed through unchanged with dense-grid
indices, so `_prepare_surface_matrices` (used by `plot()`) raised
`IndexError` for any non-default surface resolution. Unresolvable points now
render as NaN holes, as documented.

### Changed — BREAKING: shear-axis (Vy/Vz) convention corrected

The `LoadCase` shear subscripts now follow the standard structural convention,
where a shear subscript names the **direction the force acts**:

- **`Vz_Ed`** is the **major-axis** shear — it acts along the **vertical (z)** axis
  (previously this was `Vy_Ed`).
- **`Vy_Ed`** is the **minor-axis** shear — it acts along the **horizontal (y)** axis
  (previously this was `Vz_Ed`).
- Moments are unchanged: **`My_Ed`** (major, about the horizontal *y* axis) and
  **`Mz_Ed`** (minor, about the vertical *z* axis). The pairs are now
  `Vz_Ed ↔ My_Ed` (major) and `Vy_Ed ↔ Mz_Ed` (minor).

**Migration:** code that passed `Vz_Ed=` expecting *minor*-axis shear (or `Vy_Ed=`
expecting *major*) must swap the two. Code that only ever passed the resultant
`V_Ed=` is unaffected — see below.

### Changed — `V_Ed` and `M_Ed` are now first-class agnostic inputs

`LoadCase(V_Ed=...)` and `LoadCase(M_Ed=...)` are no longer deprecated and no
longer emit `DeprecationWarning`. They are direction-agnostic convenience inputs
that map to the **major axis** (`V_Ed → Vz_Ed`, `M_Ed → My_Ed`). Explicit
components always take precedence (`LoadCase(Vy_Ed=…, Vz_Ed=…)` is honoured as
given). The computed `LoadCase.V_Ed` read-back remains the resultant
`hypot(Vy_Ed, Vz_Ed)`.
