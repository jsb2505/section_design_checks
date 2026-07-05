"""Diagnostic: trace d, z, sigma_cp, V_Rd_s, V_Rd_max through negative N_Ed range."""
import warnings
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\user\Repo\Scripts\section_design_checks")

from materials.reinforced_concrete.sections import create_rectangular_section, create_linear_rebar_layer
from materials.reinforced_concrete.rebar import Rebar
from materials.reinforced_concrete.concrete import ConcreteMaterial
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck
from materials.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase
from materials.reinforced_concrete.materials import ShearRebar

concrete = ConcreteMaterial(grade="C30/37")
section = create_rectangular_section(width=300, height=500)
rebar_20 = Rebar(diameter=20, f_yk=500)
bot = create_linear_rebar_layer(rebar=rebar_20, n_bars=3, start_point=(50, 55), end_point=(250, 55), layer_name="bottom")
top = create_linear_rebar_layer(rebar=rebar_20, n_bars=2, start_point=(50, 449), end_point=(250, 449), layer_name="top")
section.add_rebar_group(bot)
section.add_rebar_group(top)

shear_rebar = ShearRebar(diameter=10, f_yk=500, n_legs=2, spacing=200, angle_deg=90)
check = ShearCheck(section=section, concrete=concrete, shear_reinforcement=shear_rebar, use_mechanical_lever_arm=True)

M_Ed = 80.0
V_Ed = 220.0

print(f"{'N_Ed':>8} {'d':>6} {'z':>7} {'z/d':>6} {'sig_cp':>7} {'force_v':>7} | {'cot@util1':>10} {'V_Rd_s@1.5':>11} {'V_Rd_max@1.5':>13}")
print("-" * 110)

from materials.reinforced_concrete.analysis.shear_viewer import ShearViewer
viewer = ShearViewer(check)

n_vals = np.linspace(-1000, 200, 121)
for N_Ed_val in n_vals:
    lc = LoadCase(V_Ed=V_Ed, M_Ed=M_Ed, N_Ed=float(N_Ed_val))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = viewer._build_context(load_case=lc)
    
    d = ctx.d
    z = ctx.z
    sigma_cp = ctx.sigma_cp
    
    # Compute V_Rd_s and V_Rd_max at cot=1.5
    cot_test = 1.5
    V_Rd_s_15 = check.find_V_Rd_s(cot_theta=cot_test, z=z)
    V_Rd_max_15 = check.find_V_Rd_max(cot_theta=cot_test, z=z, sigma_cp=sigma_cp)
    
    # Find cot_theta where util=1 by bisection
    # util = V_Ed / min(V_Rd_s, V_Rd_max) = 1 => min(V_Rd_s, V_Rd_max) = V_Ed
    cot_at_util1 = None
    for cot_try in np.linspace(1.0, 2.5, 301):
        vrs = check.find_V_Rd_s(cot_theta=cot_try, z=z)
        vrm = check.find_V_Rd_max(cot_theta=cot_try, z=z, sigma_cp=sigma_cp)
        vrd = min(vrs, vrm)
        util = V_Ed / vrd if vrd > 0 else 999
        if util <= 1.0:
            cot_at_util1 = cot_try
            break
    
    cot_str = f"{cot_at_util1:.3f}" if cot_at_util1 is not None else "N/A"
    
    print(f"{N_Ed_val:8.1f} {d:6.1f} {z:7.1f} {z/d:6.3f} {sigma_cp:7.3f} {'  -   ':>7} | {cot_str:>10} {V_Rd_s_15:11.2f} {V_Rd_max_15:13.2f}")
