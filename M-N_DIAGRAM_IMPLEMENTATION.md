# M-N Interaction Diagram Implementation

## Overview

The M-N interaction diagram implementation provides a complete fiber-based strain compatibility analysis tool for reinforced concrete sections under combined axial force and bending moment.

## Implementation Summary

### Location
- `materials/reinforced_concrete/analysis/interaction_diagram.py`

### Key Components

#### 1. `InteractionPoint` (Pydantic Model)
```python
class InteractionPoint(BaseModel):
    N: float  # Axial force in kN (positive = compression)
    M: float  # Moment about centroid in kN·m
    neutral_axis_depth: float  # Neutral axis depth from top (mm)
    max_concrete_strain: float
    max_steel_strain: float
```

Represents a single point on the M-N interaction curve.

#### 2. `MNInteractionDiagram` (Main Class)

**Key Methods:**

- `calculate_point(neutral_axis_depth, max_concrete_strain=None) -> InteractionPoint`
  - Calculates a single (N, M) point using strain compatibility
  - Uses plane sections remain plane assumption
  - Integrates forces over fiber mesh

- `generate_diagram(n_points=50, include_tension=True) -> List[InteractionPoint]`
  - Generates complete M-N curve from pure compression to pure tension
  - Returns ordered list of InteractionPoint objects

- `get_capacity(N_Ed) -> Tuple[float, float]`
  - Returns moment capacity (M_Rd_pos, M_Rd_neg) for given axial force
  - Uses 5% tolerance window to find maximum M at target N

- `check_capacity(N_Ed, M_Ed) -> Tuple[bool, float]`
  - Checks if applied loads are within capacity
  - Returns (is_safe, utilization) where utilization = demand/capacity

- `get_diagram_arrays(n_points=50) -> Tuple[NDArray, NDArray]`
  - Returns (N_array, M_array) for plotting

#### 3. `create_interaction_diagram()` (Factory Function)
Convenient factory function for creating MNInteractionDiagram instances.

## Theoretical Background

### Strain Compatibility Method

The implementation uses the fiber-based strain compatibility approach per EC2:

1. **Assume neutral axis depth** (x)
2. **Calculate strain distribution** using plane sections remain plane:
   ```
   ε(y) = ε_cu2 × (x - y) / x
   ```
   where:
   - ε_cu2 = maximum concrete compressive strain (typically 0.0035)
   - y = distance from section top to fiber
   - x = neutral axis depth from section top

3. **Get stresses from constitutive models:**
   - Concrete: Using EC2 models (parabola-rectangle, bilinear, or schematic)
   - Steel: Using EC2 stress-strain with inclined or horizontal branch

4. **Integrate forces over fibers:**
   ```
   N = Σ(σ_i × A_i)
   M = Σ(σ_i × A_i × y_i)
   ```

5. **Repeat for different neutral axis depths** to build complete diagram

### Neutral Axis Depth Range

The diagram covers the full range of failure modes:

- **Pure compression** (NA very deep, ~10×h): Uniform compression
- **Compression-controlled** (NA from 2h to 0.1h): Concrete crushing governs
- **Balanced failure** (NA around 0.4-0.6h): Simultaneous concrete crushing and steel yielding
- **Tension-controlled** (NA from 0.1h to -0.1h): Steel yielding governs
- **Pure tension** (NA above section): Tensile failure

## Features

### ✅ Complete EC2 Compliance
- Uses design strengths (f_cd, f_yd) by default
- Option to use characteristic strengths
- All strain limits per EC2 Table 3.1

### ✅ Multiple Constitutive Models
- **Concrete**: Parabola-rectangle (default), bilinear, schematic
- **Steel**: Inclined branch (with hardening) or horizontal (perfectly plastic)

### ✅ Fiber Mesh Integration
- Reuses existing `FiberMesh` class
- Configurable mesh resolution (n_fibers_width, n_fibers_height)
- Efficient force integration with `calculate_section_forces()`

### ✅ Capacity Checking
- Find moment capacity at any axial force level
- Separate positive and negative moment capacities
- Check utilization ratio for applied loads
- Handles non-monotonic M-N curves correctly
- **Non-symmetric section support**: Properly handles asymmetric reinforcement and geometry

### ✅ Numerical Robustness
- Handles special cases (pure compression, pure tension)
- Tolerant capacity lookup (5% window)
- Vectorized calculations for performance

### ✅ Advanced Optional Features
- **Balanced failure point optimization**: Find the point where concrete crushing and steel yielding occur simultaneously
  - Uses scipy.optimize with Brentq method for robust root finding
  - Returns both the balanced point and neutral axis depth
- **Tension stiffening effects**: Include concrete contribution in tension zone (EC2 average stress-strain)
  - Optional parameter `tension_stiffening=True`
  - Accounts for concrete between cracks using beta factor (0.6 for short-term loading)
  - Increases tension capacity for more accurate serviceability analysis
- **Confined concrete models**: Enhanced strength and ductility from transverse reinforcement (Mander model)
  - Optional parameter `confined_concrete=True`
  - Requires volumetric ratio `confinement_rho_s` and yield strength `confinement_f_yh`
  - Increases compression capacity and ultimate strain
  - Applicable to columns with closely-spaced ties or spirals

### ✅ Data Export & Integration
- **JSON export**: Complete diagram with metadata for data interchange
- **CSV export**: Tabular format for spreadsheets and analysis tools
- **Dictionary export**: Programmatic access for Python workflows
- **Pandas integration**: Easy conversion to DataFrames
- Configurable output (with/without metadata, strains, compact format)

## Usage Examples

### Basic Usage

```python
from materials.reinforced_concrete.materials import ConcreteMaterial
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.reinforced_concrete.analysis import create_interaction_diagram

# Create section
section = create_rectangular_section(300, 500)

# Add reinforcement
bottom_layer = create_linear_rebar_layer(
    rebar=Rebar(diameter=20, grade="B500B"),
    n_bars=3,
    start_point=(50, 50),
    end_point=(250, 50),
    layer_name="bottom",
)
section.add_rebar_group(bottom_layer)

# Create material
concrete = ConcreteMaterial(grade="C30/37")

# Generate diagram
diagram = create_interaction_diagram(
    section=section,
    concrete=concrete,
    concrete_model_type="parabola-rectangle",
    steel_branch_type="inclined",
)

# Get complete M-N curve
points = diagram.generate_diagram(n_points=100)
```

### Plotting

```python
import matplotlib.pyplot as plt

# Get arrays for plotting
N, M = diagram.get_diagram_arrays(n_points=100)

plt.figure(figsize=(10, 8))
plt.plot(M, N, 'b-', linewidth=2, label='M-N Interaction Curve')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Moment M (kN·m)')
plt.ylabel('Axial Force N (kN)')
plt.title('M-N Interaction Diagram')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

### Capacity Check

```python
# Check if applied loads are safe
N_Ed = 500  # kN compression
M_Ed = 150  # kN·m

is_safe, utilization = diagram.check_capacity(N_Ed, M_Ed)

if is_safe:
    print(f"✓ Section is safe (utilization = {utilization:.1%})")
else:
    print(f"✗ Section fails (utilization = {utilization:.1%})")
```

### Find Moment Capacity

```python
# Get moment capacity at specific axial force
N_Ed = 1000  # kN compression
M_Rd_pos, M_Rd_neg = diagram.get_capacity(N_Ed)

print(f"Moment capacity at N = {N_Ed} kN:")
print(f"  M_Rd,pos = {M_Rd_pos:.1f} kN·m (positive bending)")
print(f"  M_Rd,neg = {M_Rd_neg:.1f} kN·m (negative bending)")

# For symmetric sections: |M_Rd_pos| ≈ |M_Rd_neg|
# For non-symmetric sections: Capacities will differ
```

### Non-Symmetric Section Handling

```python
# Create section with asymmetric reinforcement
section = create_rectangular_section(300, 500)

# Heavy bottom reinforcement (tension zone for positive moment)
bottom_layer = create_linear_rebar_layer(
    rebar=Rebar(diameter=20, grade="B500B"),
    n_bars=5,
    start_point=(40, 50),
    end_point=(260, 50),
    layer_name="bottom",
)
section.add_rebar_group(bottom_layer)

# Light top reinforcement
top_layer = create_linear_rebar_layer(
    rebar=Rebar(diameter=16, grade="B500B"),
    n_bars=2,
    start_point=(100, 450),
    end_point=(200, 450),
    layer_name="top",
)
section.add_rebar_group(top_layer)

# Generate diagram
concrete = ConcreteMaterial(grade="C30/37")
diagram = create_interaction_diagram(section, concrete)

# Get capacities - will differ due to asymmetric reinforcement
N_Ed = 500.0  # kN
M_Rd_pos, M_Rd_neg = diagram.get_capacity(N_Ed)

# Positive moment (bottom steel in tension) has higher capacity
print(f"M_Rd,pos = {M_Rd_pos:.1f} kN·m")  # Higher (5 × Ø20 bars)
print(f"M_Rd,neg = {M_Rd_neg:.1f} kN·m")  # Lower (2 × Ø16 bars)
```

### Custom Constitutive Models

```python
# Use bilinear concrete model and perfectly plastic steel
diagram = create_interaction_diagram(
    section=section,
    concrete=concrete,
    concrete_model_type="bilinear",
    steel_branch_type="horizontal",
)
```

### Fine Mesh for Accuracy

```python
# Use fine mesh for high accuracy
diagram = create_interaction_diagram(
    section=section,
    concrete=concrete,
    n_fibers_width=30,
    n_fibers_height=50,
)
```

### Balanced Failure Point Optimization

```python
# Find the balanced failure point on the M-N diagram
# This is where concrete reaches ultimate strain (ε_cu)
# simultaneously with steel reaching yield strain (ε_y)

diagram = create_interaction_diagram(section, concrete)

# Find balanced point using numerical optimization
balanced_point, na_depth = diagram.find_balanced_point()

print(f"Balanced Failure Point:")
print(f"  N = {balanced_point.N:.1f} kN")
print(f"  M = {balanced_point.M:.1f} kN·m")
print(f"  Neutral axis depth = {na_depth:.1f} mm")
print(f"  Concrete strain = {balanced_point.max_concrete_strain:.6f}")
print(f"  Steel strain = {balanced_point.max_steel_strain:.6f}")

# Can also specify custom concrete strain limit
balanced_point_custom, na_custom = diagram.find_balanced_point(
    max_concrete_strain=0.003  # Custom strain limit
)
```

### Tension Stiffening Effects

```python
# Include concrete contribution in tension zone
# This accounts for concrete between cracks using EC2 average stress-strain
# Beta factor = 0.6 for short-term loading

# Create diagram WITHOUT tension stiffening (default, conservative)
diagram_no_ts = create_interaction_diagram(
    section=section,
    concrete=concrete,
    tension_stiffening=False,  # Default
)

# Create diagram WITH tension stiffening (more accurate for serviceability)
diagram_with_ts = create_interaction_diagram(
    section=section,
    concrete=concrete,
    tension_stiffening=True,  # Enable tension stiffening
)

# Compare pure tension capacity
point_no_ts = diagram_no_ts.calculate_point(-100)  # NA above section
point_with_ts = diagram_with_ts.calculate_point(-100)

print(f"Pure Tension Capacity:")
print(f"  Without tension stiffening: N = {point_no_ts.N:.1f} kN")
print(f"  With tension stiffening:    N = {point_with_ts.N:.1f} kN")
print(f"  Increase: {(point_with_ts.N / point_no_ts.N - 1) * 100:.1f}%")
```

### Confined Concrete Model

```python
# Use Mander confined concrete model for enhanced strength and ductility
# Applicable to columns with closely-spaced ties or spirals

# Typical transverse reinforcement parameters
rho_s = 0.02  # Volumetric ratio of transverse reinforcement (2%)
f_yh = 500.0  # Yield strength of transverse steel (MPa)

# Create diagram WITHOUT confinement (default)
diagram_unconfined = create_interaction_diagram(
    section=section,
    concrete=concrete,
    confined_concrete=False,  # Default
)

# Create diagram WITH confinement
diagram_confined = create_interaction_diagram(
    section=section,
    concrete=concrete,
    confined_concrete=True,
    confinement_rho_s=rho_s,  # Required if confined_concrete=True
    confinement_f_yh=f_yh,    # Optional, defaults to longitudinal steel f_yd
)

# Compare pure compression capacity
point_unconfined = diagram_unconfined.calculate_point(5000)  # Deep NA
point_confined = diagram_confined.calculate_point(5000)

print(f"Pure Compression Capacity:")
print(f"  Unconfined: N = {point_unconfined.N:.1f} kN")
print(f"  Confined:   N = {point_confined.N:.1f} kN")
print(f"  Increase: {(point_confined.N / point_unconfined.N - 1) * 100:.1f}%")

# Confinement also increases ductility (ultimate strain)
print(f"\nDuctility:")
print(f"  Unconfined ultimate strain: {diagram_unconfined.concrete_model.get_ultimate_strain():.6f}")
print(f"  Confined ultimate strain:   ~{0.004 + 0.14 * rho_s * f_yh / concrete.f_cd:.6f}")
```

### Combining Multiple Advanced Features

```python
# You can enable multiple optional features simultaneously

diagram_advanced = create_interaction_diagram(
    section=section,
    concrete=concrete,
    tension_stiffening=True,      # Enable tension stiffening
    confined_concrete=True,        # Enable confinement
    confinement_rho_s=0.02,       # 2% volumetric ratio
    confinement_f_yh=500.0,       # 500 MPa transverse steel
    n_fibers_width=30,            # Fine mesh for accuracy
    n_fibers_height=50,
)

# Generate complete diagram with all enhancements
points = diagram_advanced.generate_diagram(n_points=100)

# Find balanced point with all enhancements active
balanced_point, na_depth = diagram_advanced.find_balanced_point()
```

### Export to JSON

```python
# Export complete diagram to JSON file with metadata
diagram.export_to_json(
    file_path="mn_diagram.json",
    n_points=100,
    include_metadata=True,  # Includes section and material properties
    indent=2,  # Pretty-print JSON
)

# Export compact JSON without metadata
diagram.export_to_json(
    file_path="mn_diagram_compact.json",
    n_points=50,
    include_metadata=False,
    indent=None,  # Compact format
)
```

### Export to CSV

```python
# Export diagram to CSV with all data columns
diagram.export_to_csv(
    file_path="mn_diagram.csv",
    n_points=100,
    include_strains=True,  # Include strain columns
)

# Export simple CSV with only N and M
diagram.export_to_csv(
    file_path="mn_diagram_simple.csv",
    n_points=50,
    include_strains=False,  # Only N_kN and M_kNm columns
)
```

### Export to Dictionary

```python
# Get diagram as dictionary for programmatic use
data = diagram.to_dict(n_points=100, include_metadata=True)

# Access the data
N_values = data["N_array"]  # List of axial forces
M_values = data["M_array"]  # List of moments
points = data["points"]  # List of dicts with all point data
metadata = data["metadata"]  # Section and material info

# Use in pandas DataFrame
import pandas as pd
df = pd.DataFrame(data["points"])
print(df.head())
```

## Test Coverage

### 67 Comprehensive Tests

#### InteractionPoint Tests (4)
- Creation and immutability
- Representation
- Pydantic validation
- Dictionary export

#### MNInteractionDiagram Tests (21)
- Basic creation and properties
- Point calculation (pure compression, balanced, pure tension)
- Custom strain limits
- Diagram generation (full range, without tension)
- Array export for plotting
- Capacity queries (compression, tension)
- Capacity checking (safe, unsafe, at limit)
- Different constitutive models (concrete and steel)
- Error handling (no rebars)
- Mesh resolution (fine and coarse)

#### Factory Function Tests (2)
- Basic creation
- Custom parameters

#### Numerical Accuracy Tests (4)
- Symmetry in pure compression
- Force equilibrium
- Monotonic behavior
- Linear strain distribution

#### Export Functionality Tests (9)
- JSON export with/without metadata
- JSON compact format
- CSV export with/without strains
- Dictionary export with/without metadata
- Round-trip JSON export/reload
- CSV data integrity verification

#### Non-Symmetric Section Tests (6)
- Asymmetric reinforcement capacity separation
- Positive/negative moment capacity differences
- Bidirectional capacity checking
- Symmetric section verification (backward compatibility)
- Correct sign convention
- Full diagram with both moment directions

#### Balanced Failure Point Tests (6)
- Returns valid interaction point
- Correct strains at balanced point (ε_cu for concrete, ε_y for steel)
- Neutral axis depth is reasonable (0.2h to h)
- Balanced point lies on M-N diagram
- Works with custom concrete strain
- Different sections produce different balanced points

#### Tension Stiffening Tests (6)
- Disabled by default
- Can be enabled via parameter
- Affects pure tension capacity (increases N_tension)
- Affects small eccentricity loading
- Minimal effect in pure compression
- Full diagram generation works with tension stiffening

#### Confined Concrete Tests (9)
- Disabled by default
- Requires confinement_rho_s parameter when enabled
- Validates rho_s range (0 < rho_s < 0.1)
- Can be enabled with valid parameters
- Defaults f_yh to longitudinal steel yield strength
- Increases pure compression capacity
- Increases ductility (ultimate strain)
- Full diagram generation works with confinement
- Can be combined with tension stiffening

**All 67 tests passing ✓**

## Performance

- **Fast generation**: 100-point diagram in ~0.5 seconds
- **Vectorized calculations**: NumPy array operations throughout
- **Mesh efficiency**: Reuses fiber mesh, no regeneration needed
- **Memory efficient**: Point-based storage, minimal overhead

## Known Limitations

1. **Second-order effects**: Does not account for slenderness
   - Use separately with EC2 slenderness checks

2. **Time-dependent effects**: Creep and shrinkage not included
   - Use long-term modulus or effective modulus separately

3. **High-strength concrete**: Confined concrete model validated for normal strength
   - Mander model may need adjustment for f_ck > 50 MPa

## Future Enhancements

- [x] Biaxial M-M-N interaction surface ✅ **Implemented**
- [x] Tension stiffening effects ✅ **Implemented**
- [x] Confined concrete models ✅ **Implemented**
- [x] Non-symmetric section handling ✅ **Implemented**
- [x] Optimization for finding balanced failure point ✅ **Implemented**
- [x] Export to standard formats (CSV, JSON) ✅ **Implemented**
- [x] 3D visualization utilities ✅ **Implemented**

## Advanced Features Summary

All three optional enhancements have been fully implemented and tested:

### 1. Balanced Failure Point Optimization ✅
- **Method**: `find_balanced_point(max_concrete_strain=None)`
- **Returns**: `(InteractionPoint, neutral_axis_depth)`
- **Algorithm**: Scipy Brentq root-finding for strain compatibility
- **Use Case**: Find the transition point between compression-controlled and tension-controlled failure
- **Tests**: 6 comprehensive tests covering various scenarios

### 2. Tension Stiffening Effects ✅
- **Parameter**: `tension_stiffening: bool = False`
- **Model**: EC2 average stress-strain for cracked concrete in tension
- **Beta Factor**: 0.6 for short-term loading
- **Effect**: Increases tension capacity by accounting for concrete between cracks
- **Use Case**: More accurate serviceability analysis and tension member design
- **Tests**: 6 comprehensive tests verifying tension zone behavior

### 3. Confined Concrete Model ✅
- **Parameters**:
  - `confined_concrete: bool = False`
  - `confinement_rho_s: Optional[float] = None` (volumetric ratio, required if enabled)
  - `confinement_f_yh: Optional[float] = None` (transverse steel yield, defaults to f_yd)
- **Model**: Mander confined concrete model
- **Effects**:
  - Increased compressive strength: f_cc = f_co × (2.254√(1 + 7.94f_l/f_co) - 2f_l/f_co - 1.254)
  - Increased strain at peak: ε_cc = ε_co × (1 + 5(f_cc/f_co - 1))
  - Increased ultimate strain: ε_cu,confined = 0.004 + 0.14ρ_s·f_yh/f_co
- **Use Case**: Columns with closely-spaced ties or spiral reinforcement
- **Tests**: 9 comprehensive tests including validation and combination with tension stiffening

All enhancements are **fully optional** (disabled by default) and can be combined as needed.

## Integration with Materials Library

The M-N diagram implementation seamlessly integrates with:

- ✅ **Materials**: ConcreteMaterial, Rebar classes
- ✅ **Constitutive**: All EC2 stress-strain models
- ✅ **Geometry**: RCSection, RebarGroup with Shapely
- ✅ **Fiber Mesh**: Existing FiberMesh generation and integration
- ✅ **Pydantic**: Full validation and JSON serialization

## References

- EN 1992-1-1:2004 (Eurocode 2): Design of concrete structures
- Section 3: Materials (stress-strain relationships)
- Section 6.1: Bending with or without axial force
- Annex J: Examples of strain and stress distributions

## Status

**✅ Complete and Production-Ready**

- Full implementation with **67 passing tests**
- EC2-compliant analysis with all optional enhancements
- Efficient fiber-based method with vectorized calculations
- Comprehensive documentation with detailed usage examples
- API-ready with Pydantic models
- Three advanced optional features fully implemented:
  - ✅ Balanced failure point optimization
  - ✅ Tension stiffening effects
  - ✅ Confined concrete model (Mander)
- Export capabilities (JSON, CSV, dictionary)
- Non-symmetric section support
- Biaxial M-M-N surface implementation available
