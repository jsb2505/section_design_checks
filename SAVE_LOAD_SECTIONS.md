# Save and Load RC Sections to JSON

## Overview

This guide shows how to save and load reinforced concrete section objects (including geometry, rebars, materials, and applied loads) to/from JSON format.

## Quick Start

```python
from example_save_load_section import RCSectionSerializer
from materials.reinforced_concrete.geometry import create_rectangular_section
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar

# Create section
section = create_rectangular_section(300, 500, section_name="My Beam")
# ... add rebars ...

# Define materials
concrete = ConcreteMaterial(grade="C30/37")

# Define applied loads
applied_loads = {
    "N_Ed_kN": 100.0,
    "M_Ed_kNm": 150.0,
    "V_Ed_kN": 80.0,
}

# Save to JSON
RCSectionSerializer.save_to_json(
    file_path="my_section.json",
    section=section,
    concrete=concrete,
    applied_loads=applied_loads,
)

# Load back later
section, concrete, loads = RCSectionSerializer.load_from_json("my_section.json")
```

## Features

### ✅ Complete Section Data
- **Geometry**: Section outline (any polygon) stored as WKT (Well-Known Text)
- **Rebars**: All rebar groups with positions, diameters, grades, and safety factors
- **Materials**: Concrete grade, safety factors (γ_c, α_cc), aggregate type
- **Applied Loads**: Any custom load dictionary (N, M, V, etc.)

### ✅ Human-Readable JSON
- Pretty-printed format
- Clear structure
- Can be edited manually if needed

### ✅ Version Control Friendly
- Text-based format
- Tracks changes easily with git
- Share sections between projects

## API Reference

### `RCSectionSerializer.save_to_json()`

Save an RC section to JSON file.

**Parameters:**
- `file_path`: Output file path (str or Path)
- `section`: RCSection object to save
- `concrete`: Optional ConcreteMaterial (default: None)
- `applied_loads`: Optional dict of loads (default: None)
- `indent`: JSON indentation (default: 2)

**Example:**
```python
RCSectionSerializer.save_to_json(
    file_path="column_c1.json",
    section=my_section,
    concrete=ConcreteMaterial(grade="C40/50", alpha_cc=0.85),
    applied_loads={
        "N_Ed_kN": 2000.0,
        "Mx_Ed_kNm": 120.0,
        "My_Ed_kNm": 80.0,
    },
    indent=2,
)
```

### `RCSectionSerializer.load_from_json()`

Load an RC section from JSON file.

**Parameters:**
- `file_path`: Input file path (str or Path)

**Returns:**
- `(section, concrete, applied_loads)` tuple
  - `section`: RCSection object
  - `concrete`: ConcreteMaterial or None
  - `applied_loads`: dict or None

**Example:**
```python
section, concrete, loads = RCSectionSerializer.load_from_json("column_c1.json")

print(section)  # RCSection with all rebars
print(concrete)  # C40/50 (f_ck=40.0 MPa, f_cd=22.7 MPa)
print(loads)  # {'N_Ed_kN': 2000.0, ...}
```

### `RCSectionSerializer.section_to_dict()`

Convert section to dictionary (for programmatic use).

**Returns:** Dictionary ready for JSON serialization

**Example:**
```python
data = RCSectionSerializer.section_to_dict(section, concrete, loads)
# Returns dict with 'version', 'section', 'concrete', 'applied_loads' keys
```

### `RCSectionSerializer.dict_to_section()`

Reconstruct section from dictionary.

**Returns:** `(section, concrete, applied_loads)` tuple

## JSON Structure

The generated JSON has this structure:

```json
{
  "version": "1.0",
  "section": {
    "outline_wkt": "POLYGON ((0 0, 300 0, 300 500, 0 500, 0 0))",
    "concrete_cover": 30.0,
    "section_name": "Beam B1",
    "rebar_groups": [
      {
        "rebar": {
          "grade": "B500B",
          "gamma_s": 1.15,
          "diameter": 20.0
        },
        "positions": [
          {"x": 50.0, "y": 50.0},
          {"x": 150.0, "y": 50.0}
        ],
        "layer_name": "bottom"
      }
    ]
  },
  "concrete": {
    "grade": "C30/37",
    "gamma_c": 1.5,
    "gamma_c_accidental": 1.2,
    "alpha_cc": 1.0
  },
  "applied_loads": {
    "N_Ed_kN": 100.0,
    "M_Ed_kNm": 150.0
  }
}
```

## Use Cases

### 1. Save Design for Later Analysis

```python
# Design phase: create and save section
section = create_rectangular_section(400, 600)
# ... add rebars ...
concrete = ConcreteMaterial(grade="C35/45")

RCSectionSerializer.save_to_json("designs/beam_b1.json", section, concrete)

# Analysis phase: load and analyze
section, concrete, _ = RCSectionSerializer.load_from_json("designs/beam_b1.json")
diagram = create_interaction_diagram(section, concrete)
```

### 2. Share Sections Across Projects

```python
# Project A: Save standard column
RCSectionSerializer.save_to_json("standards/column_400x400.json", section, concrete)

# Project B: Load standard column
section, concrete, _ = RCSectionSerializer.load_from_json("standards/column_400x400.json")
```

### 3. Version Control for Structural Designs

```bash
git add designs/*.json
git commit -m "Update column C1 reinforcement from 6Ø20 to 8Ø20"
git diff designs/column_c1.json  # See exact changes
```

### 4. Parametric Design with Load Cases

```python
# Save section once
RCSectionSerializer.save_to_json("beam.json", section, concrete)

# Analyze multiple load cases
load_cases = [
    {"N_Ed_kN": 100, "M_Ed_kNm": 150},
    {"N_Ed_kN": 200, "M_Ed_kNm": 100},
    {"N_Ed_kN": 50, "M_Ed_kNm": 200},
]

section, concrete, _ = RCSectionSerializer.load_from_json("beam.json")
for i, loads in enumerate(load_cases):
    diagram = create_interaction_diagram(section, concrete)
    is_safe, util = diagram.check_capacity(loads["N_Ed_kN"], loads["M_Ed_kNm"])
    print(f"Load case {i+1}: {'OK' if is_safe else 'FAIL'} (util={util:.1%})")
```

### 5. Applied Loads for Code Checks

```python
# Save section with comprehensive load data
applied_loads = {
    # ULS loads
    "N_Ed_kN": 1500.0,
    "M_Ed_kNm": 250.0,
    "V_Ed_kN": 180.0,

    # SLS loads
    "N_Qp_kN": 1000.0,
    "M_Qp_kNm": 180.0,

    # Load combination info
    "load_combination": "1.35G + 1.5Q",
    "design_situation": "ULS persistent",
}

RCSectionSerializer.save_to_json(
    "column_with_loads.json",
    section,
    concrete,
    applied_loads,
)

# Load and use for multiple checks
section, concrete, loads = RCSectionSerializer.load_from_json("column_with_loads.json")

# Bending check
diagram = create_interaction_diagram(section, concrete)
is_safe, util = diagram.check_capacity(loads["N_Ed_kN"], loads["M_Ed_kNm"])

# Shear check (if you have shear check implementation)
# ...
```

## Advanced: Custom Load Classes

For type safety with loads, you can create a Pydantic model:

```python
from pydantic import BaseModel

class ULSLoads(BaseModel):
    N_Ed_kN: float
    M_Ed_kNm: float
    V_Ed_kN: float
    load_combination: str = "1.35G + 1.5Q"

# Create loads
loads = ULSLoads(N_Ed_kN=1500, M_Ed_kNm=250, V_Ed_kN=180)

# Save (convert to dict)
RCSectionSerializer.save_to_json(
    "section.json",
    section,
    concrete,
    loads.model_dump(),
)

# Load back
section, concrete, loads_dict = RCSectionSerializer.load_from_json("section.json")
loads = ULSLoads.model_validate(loads_dict)  # Type-safe loads
```

## Notes

- **WKT Format**: Section outline is stored as Well-Known Text (WKT), a standard format for geometric data
- **Computed Fields**: Properties like f_yd, area, E_cm are recalculated on load (not stored)
- **Safety Factors**: All safety factors (γ_c, γ_s, α_cc) are preserved
- **National Annexes**: Different values (e.g., alpha_cc=0.85 for Germany) are saved correctly

## Integration with Analysis

Once loaded, sections work seamlessly with all analysis tools:

```python
# Load section
section, concrete, loads = RCSectionSerializer.load_from_json("my_section.json")

# M-N Diagram
from materials.reinforced_concrete.analysis import create_interaction_diagram
diagram = create_interaction_diagram(section, concrete)

# Biaxial M-M-N Surface
from materials.reinforced_concrete.analysis import create_biaxial_interaction_surface
surface = create_biaxial_interaction_surface(section, concrete)

# Code checks (if implemented)
# from materials.reinforced_concrete.code_checks.ec2 import BendingCheck
# check = BendingCheck(section, concrete, loads)
```

## Files Generated

After running the example:
- `beam_b1.json` - Rectangular beam with materials and loads
- `column_c1.json` - Circular column with biaxial loads
- `section_minimal.json` - Geometry only (no materials/loads)

All files are human-readable and can be edited manually if needed.
