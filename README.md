# Materials - Structural Engineering Library

**API-ready library for structural materials and code-checked section design**

A modern Python library for structural engineering with full Pydantic validation, designed for integration with FEA post-processing workflows.

## Features

### ✅ Currently Implemented

- **🏗️ Reinforced Concrete (Eurocode 2)**
  - Complete material models (concrete grades C12/15 to C90/105, steel B500A/B/C)
  - Three EC2 stress-strain models (schematic, parabola-rectangle, bilinear)
  - Shapely-based 2D polygonal section geometry with arbitrary rebar positioning
  - Fiber mesh generation for strain compatibility analysis
  - **M-N interaction diagrams** with fiber-based strain compatibility (NEW!)
  - Pydantic validation throughout for API-ready usage

- **🔧 Core Infrastructure**
  - Base material and constitutive model abstractions
  - Unit system (mm, MPa, kN standard)
  - Comprehensive type hints
  - JSON serialization support

### 🚧 In Development

- **Code Checks (Eurocode 2)**
  - Shear (§6.2)
  - Cracking (§7.3)
  - Deflection (§7.4)

- **Additional Materials**
  - Structural steel (with database section profiles)
  - Timber (scaffolding)

- **API Layer**
  - Request/response models
  - FastAPI integration examples

## Installation

```bash
# Clone repository
cd c:\Users\user\Repo\Scripts
git clone <your-repo-url> materials
cd materials

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

## Quick Start

### Defining Materials

```python
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar

# Create concrete material
concrete = ConcreteMaterial(
    grade="C30/37",
    gamma_c=1.5,
    alpha_cc=1.0,
)

print(concrete.f_ck)  # 30.0 MPa
print(concrete.f_cd)  # 20.0 MPa
print(concrete.E_cm)  # 33000 MPa

# Create rebar
bar = Rebar(
    grade="B500B",
    diameter=16,
)

print(bar.area)  # 201.1 mm²
print(bar.f_yd)  # 434.8 MPa
```

### Creating RC Sections

```python
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.core.geometry import Point2D

# Create 300×500 mm beam section
section = create_rectangular_section(
    width=300,
    height=500,
    section_name="Beam B1",
)

# Add bottom reinforcement: 3×ϕ20
bottom_bars = create_linear_rebar_layer(
    rebar=Rebar(diameter=20, grade="B500B"),
    n_bars=3,
    start_point=(50, 50),    # 50mm cover
    end_point=(250, 50),
    layer_name="bottom",
)

section.add_rebar_group(bottom_bars)

print(section)
# RCSection('Beam B1', A_c=150000 mm², A_s=942 mm², 1 groups)

print(f"Reinforcement ratio: {section.reinforcement_ratio:.3f}")
# Reinforcement ratio: 0.006
```

### Stress-Strain Models

```python
from materials.reinforced_concrete.constitutive import (
    create_concrete_stress_strain,
    create_steel_stress_strain,
)

# Concrete parabola-rectangle model (EC2 Fig 3.3)
concrete_model = create_concrete_stress_strain(
    concrete=concrete,
    model_type="parabola-rectangle",
)

# Get stress at 2‰ strain (peak)
stress = concrete_model.get_stress(0.002)
print(f"Concrete stress at ε=0.002: {stress:.1f} MPa")

# Steel with strain hardening (EC2 Fig 3.8)
steel_model = create_steel_stress_strain(
    steel=bar,
    branch_type="inclined",  # Strain hardening
)

# Get stress at 1% strain
stress = steel_model.get_stress(0.01)
print(f"Steel stress at ε=0.01: {stress:.1f} MPa")
```

### M-N Interaction Diagrams

```python
from materials.reinforced_concrete.analysis import create_interaction_diagram

# Create M-N interaction diagram
diagram = create_interaction_diagram(
    section=section,  # RC section with rebars
    concrete=concrete,
    concrete_model_type="parabola-rectangle",  # EC2 Fig 3.3
    steel_branch_type="inclined",  # With strain hardening
    n_fibers_width=20,
    n_fibers_height=30,
)

# Generate complete M-N curve
points = diagram.generate_diagram(n_points=100)

# Get arrays for plotting
N, M = diagram.get_diagram_arrays(n_points=100)

# Check capacity for applied loads
N_Ed = 500  # kN compression
M_Ed = 150  # kN·m
is_safe, utilization = diagram.check_capacity(N_Ed, M_Ed)

print(f"Section utilization: {utilization:.1%}")
print(f"Safe: {is_safe}")

# Get moment capacity at specific axial force
M_Rd_pos, M_Rd_neg = diagram.get_capacity(N_Ed=1000)
print(f"M_Rd at N=1000 kN: ±{M_Rd_pos:.1f} kN·m")
```

### Fiber Mesh for Custom Analysis

```python
from materials.reinforced_concrete.geometry import FiberMesh

# Generate fiber mesh for custom strain compatibility analysis
mesh = FiberMesh(
    section=section,
    n_fibers_width=20,
    n_fibers_height=30,
    exclude_steel_area=True,
)

print(mesh)
# FiberMesh(concrete=585, steel=3, total=588)

# Get fiber data as numpy arrays
x, y, area, material_type, material_index = mesh.get_fiber_arrays()
```

### Complex Geometry with Shapely

```python
from shapely.geometry import Polygon
from materials.reinforced_concrete.geometry import RCSection, create_circular_perimeter_rebars

# T-beam section using custom polygon
t_beam_coords = [
    (0, 0),      # Bottom left
    (200, 0),    # Bottom right
    (200, 400),  # Start of flange
    (500, 400),  # Flange right
    (500, 500),  # Top right
    (0, 500),    # Top left
    (0, 400),    # Flange left
    (0, 0),      # Close
]

t_beam = RCSection(
    outline=Polygon(t_beam_coords),
    section_name="T-Beam",
)

# Circular column with perimeter reinforcement
from materials.reinforced_concrete.geometry import create_circular_section

column = create_circular_section(diameter=400, section_name="Column C1")

# Add 8×ϕ20 perimeter bars
perimeter_bars = create_circular_perimeter_rebars(
    rebar=Rebar(diameter=20, grade="B500B"),
    diameter=400,
    cover=40,
    n_bars=8,
)

column.add_rebar_group(perimeter_bars)
```

## Project Structure

```
materials/
├── materials/                      # Main package
│   ├── core/                       # Base abstractions
│   │   ├── base_material.py
│   │   ├── constitutive.py
│   │   ├── geometry.py
│   │   └── units.py
│   │
│   ├── reinforced_concrete/        # RC implementation
│   │   ├── materials/              # Material models
│   │   ├── constitutive/           # Stress-strain
│   │   ├── geometry/               # Sections & fibers
│   │   ├── code_checks/            # EC2 checks
│   │   └── analysis/               # M-N diagrams, etc.
│   │
│   ├── structural_steel/           # Steel (scaffolding)
│   ├── timber/                     # Timber (scaffolding)
│   └── api/                        # API models
│
├── tests/                          # Pytest tests
├── examples/                       # Usage examples
└── pyproject.toml                  # Modern packaging
```

## Design Principles

### 1. **Pydantic Validation Throughout**
All models use Pydantic for automatic validation, serialization, and API compatibility:

```python
# Invalid input raises validation error
try:
    bad_concrete = ConcreteMaterial(grade="C100/120")  # Invalid grade
except ValidationError as e:
    print(e)
```

### 2. **Shapely for Robust Geometry**
Complex 2D sections handled with industry-standard Shapely library:
- Arbitrary polygonal outlines
- Automatic area/centroid/moments calculations
- Containment checking for rebar positions

### 3. **Type Hints & Modern Python**
- Full type annotations (Python 3.10+)
- Computed fields with `@computed_field`
- Immutable where appropriate
- Clean separation of concerns

### 4. **API-Ready by Design**
- JSON serializable models
- Standard request/response patterns
- Easy integration with FastAPI/Flask
- Post-process FEA results → design checks

## FEA Integration Workflow

```python
# Pseudocode for FEA integration
import json
from materials.reinforced_concrete.geometry import create_rectangular_section
from materials.reinforced_concrete.code_checks.ec2 import BendingCheck

# 1. Load FEA results (JSON from post-processing)
with open("fea_results.json") as f:
    results = json.load(f)

# 2. Define section
section = create_rectangular_section(width=300, height=500)
# ... add reinforcement ...

# 3. Run design checks
for element in results["beam_elements"]:
    M_Ed = element["moment"]  # kN·m
    N_Ed = element["axial"]   # kN

    # Check = BendingCheck(section, concrete, steel)
    # result = check.perform_check(M_Ed=M_Ed, N_Ed=N_Ed)

    # if result.status == "fail":
    #     print(f"Element {element['id']} fails: {result}")
```

## Documentation & Examples

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Quick start guide with examples
- **[M-N_DIAGRAM_IMPLEMENTATION.md](M-N_DIAGRAM_IMPLEMENTATION.md)** - Detailed M-N diagram documentation
- **[examples/m_n_interaction_diagram_tutorial.ipynb](examples/m_n_interaction_diagram_tutorial.ipynb)** - Comprehensive Jupyter notebook tutorial
- **[TEST_RESULTS_FINAL.md](TEST_RESULTS_FINAL.md)** - Complete test coverage report

## Test Coverage

**240 tests, 100% passing**

```
======================== 240 passed in 1.78s =========================
```

- Core abstractions: 22 tests ✅
- RC materials: 69 tests ✅
- Constitutive models: 57 tests ✅
- Geometry & sections: 43 tests ✅
- Code check framework: 19 tests ✅
- M-N interaction diagrams: 30 tests ✅

All tests validate functionality, edge cases, error handling, and numerical accuracy.

## Roadmap

- [x] ~~Complete EC2 bending check with M-N interaction~~ **✅ COMPLETE**
- [x] ~~Comprehensive test suite~~ **✅ 240/240 passing**
- [ ] Implement shear, cracking, deflection checks
- [ ] Structural steel section database (UK/EU profiles)
- [ ] Timber material models
- [ ] FastAPI example application
- [ ] Documentation site
- [ ] PyPI package

## Contributing

This is a development project. Contributions welcome!

## License

MIT License

## References

- **Eurocode 2**: EN 1992-1-1:2004 - Design of concrete structures
- **Shapely**: https://shapely.readthedocs.io/
- **Pydantic**: https://docs.pydantic.dev/

## Contact

For questions or collaboration: [Your contact info]
