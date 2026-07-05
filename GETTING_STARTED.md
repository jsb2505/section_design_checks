# Getting Started with Materials Library

## Installation

### 1. Install Dependencies

```bash
cd c:\Users\user\Repo\Scripts\materials
pip install -r requirements.txt
```

Required packages:
- `pydantic>=2.0.0` - Data validation and serialization
- `numpy>=1.24.0` - Numerical computations
- `shapely>=2.0.0` - 2D geometry operations

### 2. Install Package (Editable Mode)

```bash
pip install -e .
```

This allows you to import the package from anywhere and make changes during development.

### 3. Verify Installation

```python
python -c "from materials.reinforced_concrete.materials import ConcreteMaterial; print('Success!')"
```

## Quick Start Examples

### Example 1: Basic Material Properties

```python
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar

# Create concrete material
concrete = ConcreteMaterial(grade="C30/37")

# Access properties (all auto-calculated from grade)
print(f"f_ck = {concrete.f_ck} MPa")           # 30.0
print(f"f_cd = {concrete.f_cd:.1f} MPa")       # 20.0
print(f"E_cm = {concrete.E_cm:.0f} MPa")       # 32837
print(f"f_ctm = {concrete.f_ctm:.2f} MPa")     # 2.90

# Create rebar
bar = Rebar(diameter=16, grade="B500B")
print(f"Area = {bar.area:.1f} mm²")            # 201.1
print(f"f_yd = {bar.f_yd:.1f} MPa")            # 434.8
```

### Example 2: Creating an RC Section

```python
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)

# Create 300×500 beam section
section = create_rectangular_section(
    width=300,
    height=500,
    section_name="Beam B1",
)

# Add bottom reinforcement
from materials.reinforced_concrete.materials import Rebar

bar = Rebar(diameter=20, grade="B500B")
bottom_layer = create_linear_rebar_layer(
    rebar=bar,
    n_bars=3,
    start_point=(50, 50),    # Cover = 50mm
    end_point=(250, 50),
    layer_name="bottom",
)

section.add_rebar_group(bottom_layer)

# Check properties
print(section)
print(f"Reinforcement ratio: {section.reinforcement_ratio:.4f}")
print(f"Effective depth: {section.get_effective_depth('top'):.1f} mm")
```

### Example 3: Stress-Strain Analysis

```python
from materials.reinforced_concrete.constitutive import (
    create_concrete_stress_strain,
    create_steel_stress_strain,
)

# Concrete parabola-rectangle model
concrete_model = create_concrete_stress_strain(
    concrete=concrete,
    model_type="parabola-rectangle",
)

# Get stress at 2‰ strain
stress = concrete_model.get_stress(0.002)
print(f"Concrete stress: {stress:.1f} MPa")

# Steel with strain hardening
steel_model = create_steel_stress_strain(
    steel=bar,
    branch_type="inclined",
)

stress = steel_model.get_stress(0.01)  # 1% strain
print(f"Steel stress: {stress:.1f} MPa")
```

### Example 4: M-N Interaction Diagrams

```python
from materials.reinforced_concrete.analysis import create_interaction_diagram

# Create M-N interaction diagram
diagram = create_interaction_diagram(
    section=section,
    concrete=concrete,
    concrete_model_type="parabola-rectangle",  # EC2 Fig 3.3
    steel_branch_type="inclined",              # With strain hardening
    n_fibers_width=20,
    n_fibers_height=30,
)

# Generate complete M-N curve
points = diagram.generate_diagram(n_points=100)
print(f"Generated {len(points)} points on M-N curve")

# Get arrays for plotting
N, M = diagram.get_diagram_arrays(n_points=100)

# Check if applied loads are safe
N_Ed = 500  # kN compression
M_Ed = 150  # kN·m
is_safe, utilization = diagram.check_capacity(N_Ed, M_Ed)

print(f"Applied loads: N={N_Ed} kN, M={M_Ed} kN·m")
print(f"Utilization: {utilization:.1%}")
print(f"Status: {'✓ SAFE' if is_safe else '✗ UNSAFE'}")

# Get moment capacity at specific axial force
M_Rd_pos, M_Rd_neg = diagram.get_capacity(N_Ed=1000)
print(f"M_Rd at N=1000 kN: ±{M_Rd_pos:.1f} kN·m")
```

### Example 5: Visualizing M-N Diagram

```python
import matplotlib.pyplot as plt

# Generate diagram
diagram = create_interaction_diagram(section, concrete)
N, M = diagram.get_diagram_arrays(n_points=100)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(M, N, 'b-', linewidth=2.5, label='M-N Interaction Curve')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Moment M (kN·m)')
plt.ylabel('Axial Force N (kN)')
plt.title('M-N Interaction Diagram')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

### Example 6: Advanced - T-Beam Section

```python
from shapely.geometry import Polygon
from materials.reinforced_concrete.geometry import RCSection

# Define T-beam geometry
t_beam_coords = [
    (0, 0),      # Bottom left of web
    (200, 0),    # Bottom right of web
    (200, 400),  # Top of web
    (500, 400),  # Flange right
    (500, 500),  # Top right
    (0, 500),    # Top left
    (0, 400),    # Flange left
]

t_beam = RCSection(
    outline=Polygon(t_beam_coords),
    section_name="T-Beam",
)

# Add reinforcement
# ... (bottom and top layers)

# Generate M-N diagram for T-beam
diagram_t = create_interaction_diagram(t_beam, concrete)
N_t, M_t = diagram_t.get_diagram_arrays(n_points=100)
```

## Project Structure

```
materials/
├── materials/                      # Main package
│   ├── core/                       # Base abstractions
│   │   ├── base_material.py       # Material base class
│   │   ├── constitutive.py        # Stress-strain base
│   │   ├── geometry.py            # Geometry base
│   │   └── units.py               # Unit definitions
│   │
│   ├── reinforced_concrete/        # RC module (fully implemented)
│   │   ├── materials/              # Concrete, steel, rebar
│   │   ├── constitutive/           # Stress-strain models
│   │   ├── geometry/               # Sections, fibers
│   │   ├── code_checks/            # EC2 checks (base framework)
│   │   └── analysis/               # M-N diagrams (pending)
│   │
│   ├── structural_steel/           # Structural steel (scaffolding)
│   ├── timber/                     # Timber (scaffolding)
│   └── api/                        # API models (pending)
│
├── tests/                          # Pytest tests
├── examples/                       # Usage examples
│   └── rc_beam_example.py
│
├── pyproject.toml                  # Modern Python packaging
├── requirements.txt                # Dependencies
└── README.md                       # Full documentation
```

## What's Implemented

✅ **Core Infrastructure**
- Base material/constitutive/geometry abstractions
- Full Pydantic validation
- Type hints throughout

✅ **Reinforced Concrete Materials**
- Concrete grades C12/15 to C90/105 (EC2 Table 3.1)
- All strength properties (f_ck, f_cd, f_ctm, etc.)
- Elastic modulus with aggregate adjustments
- Strain parameters for all models

✅ **Reinforcing Steel**
- Grades B500A/B/C
- Characteristic and design strengths
- Ductility classes
- Bar geometries (6-40mm diameters)

✅ **Stress-Strain Models**
- Concrete: Schematic, Parabola-Rectangle, Bilinear (EC2 Figs 3.2-3.4)
- Steel: Elastic-plastic with/without strain hardening (EC2 Fig 3.8)
- Vectorized computations with NumPy

✅ **Section Geometry (Shapely-based)**
- Arbitrary 2D polygonal sections
- Helper functions: rectangular, circular
- Rebar positioning: linear layers, perimeter, custom
- Automatic validation (bars within section)
- Area, centroid, moments of inertia

✅ **Fiber Mesh Generation**
- Automatic meshing for M-N analysis
- Concrete + steel fibers
- Configurable resolution
- Integration with stress-strain models

✅ **M-N Interaction Diagrams** (NEW!)
- Fiber-based strain compatibility analysis
- Calculate individual (N, M) points
- Generate complete interaction curves
- Capacity checking for applied loads
- Multiple EC2 constitutive models
- Support for arbitrary section shapes
- Comprehensive test coverage (30 tests)

## What's Pending

🚧 **EC2 Code Checks** (framework ready)
- Shear (§6.2)
- Cracking (§7.3)
- Deflection (§7.4)

🚧 **API Layer**
- Pydantic request/response models
- FastAPI example integration
- JSON serialization utilities

🚧 **Additional Materials**
- Structural steel section database
- Timber material models

## Documentation & Examples

### 📚 Available Resources

- **[M-N_DIAGRAM_IMPLEMENTATION.md](../M-N_DIAGRAM_IMPLEMENTATION.md)** - Complete M-N diagram documentation
  - Theory and implementation details
  - API reference for all methods
  - Performance notes and limitations
  - Future enhancements

- **[examples/m_n_interaction_diagram_tutorial.ipynb](../examples/m_n_interaction_diagram_tutorial.ipynb)** - Interactive Jupyter tutorial
  - 13 comprehensive sections with markdown explanations
  - All M-N diagram functions demonstrated
  - Multiple visualization examples
  - Comparison of EC2 models
  - Performance analysis

- **[TEST_RESULTS_FINAL.md](../TEST_RESULTS_FINAL.md)** - Test coverage report
  - 240 tests, 100% passing
  - Breakdown by module
  - Test execution details

### 🚀 Quick Tutorial

For a hands-on introduction to M-N diagrams, run the Jupyter notebook:

```bash
# Install Jupyter and matplotlib
pip install jupyter matplotlib

# Navigate to examples directory
cd examples

# Launch Jupyter
jupyter notebook m_n_interaction_diagram_tutorial.ipynb
```

## Next Steps for Development

### 1. ✅ M-N Interaction Diagrams - COMPLETE

Full fiber-based strain compatibility implementation with comprehensive testing.

### 2. Implement EC2 Shear Check (§6.2)

```python
class ShearCheck(BaseCodeCheck):
    def perform_check(self, V_Ed, N_Ed=0):
        # Calculate V_Rd,c (concrete contribution)
        # Calculate V_Rd,s (shear reinforcement contribution)
        # Check V_Ed ≤ V_Rd
        # Return CheckResult
```

### 3. Implement Cracking Check (§7.3)

```python
class CrackingCheck(BaseCodeCheck):
    def perform_check(self, M_Ed, environmental_class="XC1"):
        # Calculate stress in reinforcement
        # Determine crack width w_k
        # Check against w_max
        # Return CheckResult
```

### 4. Create FastAPI Integration Example

```python
from fastapi import FastAPI
from materials.api.models import SectionRequest, CheckResponse

app = FastAPI()

@app.post("/check/bending")
def check_bending(request: SectionRequest) -> CheckResponse:
    # Create section from request
    # Run M-N diagram check
    # Return result
```

## Running Examples

```bash
# From materials directory
cd c:\Users\user\Repo\Scripts\materials

# Run beam example
python examples/rc_beam_example.py

# Or with explicit path (if not installed)
python -c "import sys; sys.path.insert(0, '.'); exec(open('examples/rc_beam_example.py').read())"
```

## Running Tests

The library has a comprehensive test suite with **240 tests, 100% passing**.

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Run specific test file
pytest tests/test_reinforced_concrete/test_interaction_diagram.py

# Run with coverage (requires pytest-cov)
pip install pytest-cov
pytest --cov=materials --cov-report=html --cov-report=term-missing

# Run only M-N diagram tests
pytest -k interaction_diagram
```

### Test Coverage Summary

```
======================== 240 passed in 1.78s =========================
```

**Module Breakdown:**
- Core abstractions: 22 tests ✅
- RC materials: 69 tests ✅
- Constitutive models: 57 tests ✅
- Geometry & sections: 43 tests ✅
- Code check framework: 19 tests ✅
- M-N interaction diagrams: 30 tests ✅

See [TEST_RESULTS_FINAL.md](../TEST_RESULTS_FINAL.md) for detailed test report.

## Design Philosophy

1. **Pydantic Everywhere**: All models validated, serializable, API-ready
2. **Shapely for Geometry**: Industry-standard 2D operations
3. **Fiber-Based Analysis**: Most accurate for M-N diagrams
4. **Type-Safe**: Full type hints for IDE support
5. **Extensible**: Easy to add new materials, codes, checks

## FEA Integration Workflow

```
FEA Results (JSON)
    ↓
Parse with Pydantic
    ↓
For each element:
    - Define section geometry
    - Add reinforcement
    - Run code checks
    ↓
Generate report (pass/fail/warnings)
```

## Support

For questions about this library, refer to:
- README.md for full documentation
- examples/ for usage patterns
- Your existing reinforced_concrete repo for reference implementations

Happy coding! 🏗️
