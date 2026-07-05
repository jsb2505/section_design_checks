# Changelog

All notable changes to this project are recorded here. The project is in early
(alpha) development, so breaking changes may occur between minor versions.

## [Unreleased]

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
