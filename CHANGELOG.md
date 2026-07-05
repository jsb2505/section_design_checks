# Changelog

All notable changes to the Materials library will be documented in this file.

## [Unreleased]

### Added - 2024-01-XX

#### M-N Interaction Diagrams (Complete Implementation)
- **Core functionality** ([#interaction_diagram.py](materials/reinforced_concrete/analysis/interaction_diagram.py))
  - `InteractionPoint` - Pydantic model for individual (N, M) points
  - `MNInteractionDiagram` - Complete fiber-based strain compatibility analysis
  - `create_interaction_diagram()` - Factory function for easy creation

- **Key methods:**
  - `calculate_point(neutral_axis_depth)` - Calculate single point using plane sections remain plane
  - `generate_diagram(n_points)` - Generate complete M-N curve from pure compression to pure tension
  - `get_capacity(N_Ed)` - Find moment capacity at given axial force
  - `check_capacity(N_Ed, M_Ed)` - Validate applied loads against capacity
  - `get_diagram_arrays(n_points)` - Export as NumPy arrays for plotting

- **Features:**
  - Fiber-based strain compatibility per EC2
  - Multiple concrete models (parabola-rectangle, bilinear, schematic)
  - Steel strain hardening options (inclined/horizontal branch)
  - Support for arbitrary section shapes via Shapely
  - Configurable mesh resolution
  - Comprehensive error handling

- **Testing** - 30 new tests, 100% passing
  - InteractionPoint validation (3 tests)
  - Diagram generation and calculation (8 tests)
  - Capacity checking and queries (6 tests)
  - Different constitutive models (2 tests)
  - Mesh resolution effects (2 tests)
  - Numerical accuracy validation (4 tests)
  - Factory function (2 tests)
  - Error handling (3 tests)

- **Documentation:**
  - [M-N_DIAGRAM_IMPLEMENTATION.md](M-N_DIAGRAM_IMPLEMENTATION.md) - Complete technical documentation
  - [examples/m_n_interaction_diagram_tutorial.ipynb](examples/m_n_interaction_diagram_tutorial.ipynb) - Interactive Jupyter tutorial
  - Updated [README.md](README.md) with M-N examples
  - Updated [GETTING_STARTED.md](GETTING_STARTED.md) with quick start guide

### Fixed
- Export `BaseConstitutiveModel` from `constitutive/__init__.py` for M-N diagram imports

### Test Summary
- **Total tests:** 240 (up from 210)
- **Pass rate:** 100%
- **Execution time:** ~1.8 seconds
- **New coverage:** M-N interaction diagrams fully tested

---

## [Initial Release] - Prior Work

### Implemented
- Core infrastructure (base materials, constitutive models, geometry, units)
- RC materials (concrete C12/15 to C90/105, steel B500A/B/C, rebar)
- Constitutive models (3 concrete, 2 steel per EC2)
- Section geometry with Shapely (arbitrary 2D polygons, rebar positioning)
- Fiber mesh generation for analysis
- Code check framework (base classes, result models)
- Comprehensive test suite (210 tests, 100% passing)
- Full Pydantic validation throughout
- JSON serialization support

### Test Coverage (Initial)
- Core abstractions: 22 tests
- RC materials: 69 tests
- Constitutive models: 57 tests
- Geometry & sections: 43 tests
- Code check framework: 19 tests
