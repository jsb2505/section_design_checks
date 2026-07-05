# Materials Library Examples

This directory contains examples and tutorials demonstrating the materials library functionality.

## Jupyter Notebooks

### [m_n_interaction_diagram_tutorial.ipynb](m_n_interaction_diagram_tutorial.ipynb)
Comprehensive tutorial covering all M-N interaction diagram functions:

1. **Setup and Imports** - Library setup and configuration
2. **Create Materials** - Concrete and reinforcing steel definitions
3. **Create RC Section** - Section geometry with reinforcement
4. **Create M-N Diagram** - Initialize the analysis
5. **Calculate Individual Points** - Single point calculations
6. **Generate Complete Diagram** - Full M-N curve generation
7. **Visualize** - Plotting and visualization
8. **Check Capacity** - Applied load verification
9. **Get Moment Capacity** - Capacity at specific axial force
10. **Compare Concrete Models** - EC2 model comparison
11. **Compare Steel Models** - Post-yield behavior comparison
12. **Mesh Resolution Effects** - Accuracy vs. speed trade-offs
13. **Advanced: T-Beam** - Non-rectangular sections

**Key features demonstrated:**
- All M-N diagram methods and functions
- Multiple EC2 constitutive models
- Capacity checking and utilization
- Visualization examples
- Performance optimization
- Complex geometries with Shapely

### [shear_viewer_demonstration.ipynb](shear_viewer_demonstration.ipynb)
Demonstrates the new ShearCheck visualization wrappers:

1. `plot_cot_theta_study`
2. `plot_cot_theta_moment_shift_study`
3. `plot_link_angle_study`
4. `plot_link_angle_moment_shift_study`
5. `plot_cot_theta_link_angle_heatmap`
6. `plot_axial_cot_theta_contour`

Includes a complete section/material setup and an example load case.

## Python Scripts

### Coming Soon
- Simple beam design example
- Column design example
- FEA post-processing workflow
- Parametric study example

## Running the Examples

### Jupyter Notebooks

```bash
# Install Jupyter
pip install jupyter matplotlib

# Navigate to examples directory
cd c:\Users\user\Repo\Scripts\materials\examples

# Start Jupyter
jupyter notebook
```

Then open `m_n_interaction_diagram_tutorial.ipynb` in your browser.

### Python Scripts

```bash
cd c:\Users\user\Repo\Scripts\materials\examples
python example_name.py
```

## Requirements

All examples require:
- Python 3.10+
- Materials library and dependencies (see `requirements.txt`)
- Plotly for interactive plotting
- Jupyter for notebook examples

## Learn More

- [M-N_DIAGRAM_IMPLEMENTATION.md](../M-N_DIAGRAM_IMPLEMENTATION.md) - Implementation details
- [README.md](../README.md) - Library overview
- [GETTING_STARTED.md](../GETTING_STARTED.md) - Quick start guide
- [TEST_RESULTS_FINAL.md](../TEST_RESULTS_FINAL.md) - Test coverage
