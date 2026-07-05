# Examples

Runnable scripts and Jupyter notebooks demonstrating the library.

## Jupyter notebooks

| Notebook | Topic |
|---|---|
| [section_example.ipynb](section_example.ipynb) | Building sections, adding reinforcement, section viewer |
| [m_n_interaction_diagram_tutorial.ipynb](m_n_interaction_diagram_tutorial.ipynb) | Uniaxial M-N interaction diagrams end to end: materials, sections, diagram generation, capacity checks, model comparisons, T-beams |
| [biaxial_mn_interaction_tutorial.ipynb](biaxial_mn_interaction_tutorial.ipynb) | Biaxial M-M-N interaction surfaces with the EC2 pivot method, 3D plotting, capacity vectors |
| [ec2_code_checks_demonstration.ipynb](ec2_code_checks_demonstration.ipynb) | Bending, shear, cracking and stress-limit checks on a worked example |
| [shear_viewer_demonstration.ipynb](shear_viewer_demonstration.ipynb) | Shear design study plots: cot θ and link-angle studies, heatmaps, force contour maps |
| [crack_width_viewer_demonstration.ipynb](crack_width_viewer_demonstration.ipynb) | Crack width visualisation: 3D stem plots and M-N contour maps |
| [circular_section_check_demonstration.ipynb](circular_section_check_demonstration.ipynb) | Circular pile/column checks following Orr (2012) |
| [circular_shear_viewer_demonstration.ipynb](circular_shear_viewer_demonstration.ipynb) | Shear study plots for circular sections |
| [circular_vs_standard_shear.ipynb](circular_vs_standard_shear.ipynb) | Circular vs rectangular shear check comparison |
| [ndp_demonstration.ipynb](ndp_demonstration.ipynb) | Nationally Determined Parameters: EU, UK and German annexes |
| [tension_shift_demonstration.ipynb](tension_shift_demonstration.ipynb) | Tension shift in the shear/bending interaction |
| [strain_state_1d_vs_2d_demonstration.ipynb](strain_state_1d_vs_2d_demonstration.ipynb) | 1D vs 2D strain-state solutions |
| [cracking_moment_vs_solver_comparison.ipynb](cracking_moment_vs_solver_comparison.ipynb) | Cracking moment: closed form vs solver |
| [shear_comparison.ipynb](shear_comparison.ipynb) | Shear check parameter comparisons |

## Python scripts

| Script | Topic |
|---|---|
| [rc_beam_example.py](rc_beam_example.py) | Simple beam design workflow |
| [beam_analysis.py](beam_analysis.py) | Beam section analysis |
| [example_biaxial_surface.py](example_biaxial_surface.py) | Biaxial surface generation and export |
| [example_export_mn_diagram.py](example_export_mn_diagram.py) | Exporting M-N diagram data |
| [example_save_load_section.py](example_save_load_section.py) | Saving and loading sections as JSON |
| [example_accidental_limit_state.py](example_accidental_limit_state.py) | Accidental limit state checks |
| [generate_mn_diagram_figure.py](generate_mn_diagram_figure.py) | Matplotlib M-N diagram figure |
| [generate_release_plots.py](generate_release_plots.py) | Regenerates the README gallery figures in [plots/](plots/) |

## Running the examples

From the repository root:

```bash
pip install -e ".[viz]"
pip install jupyter

jupyter notebook examples/          # notebooks
python examples/rc_beam_example.py  # scripts
```

The notebooks use Plotly for interactive figures. Static image export
(`generate_release_plots.py`) additionally requires `pip install kaleido`.
