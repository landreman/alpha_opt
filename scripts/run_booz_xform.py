#!/usr/bin/env python

import sys
import os
import subprocess
import numpy as np
import booz_xform as bx

new_wout_filename = "wout_tmp.nc"

cwd = os.path.abspath(os.getcwd())
directory_name = os.path.basename(os.path.abspath(os.path.join(cwd, "../..")))
eval_name = os.path.basename(cwd)
extension = f"{directory_name}_{eval_name}"

# Run booz_xform
rho_values = [0.25, 0.5, 0.75, 1.0]
s_targets = [rho**2 for rho in rho_values]

b = bx.Booz_xform()
b.read_wout(new_wout_filename)

# Find half-grid surface indices closest to the target s = rho^2 values
ns_in = b.ns_in
s_half = np.array([(j + 0.5) / ns_in for j in range(ns_in)])
compute_surfs = [int(np.argmin(np.abs(s_half - s_t))) for s_t in s_targets]
print("compute_surfs:", compute_surfs)

b.mboz = 24
b.nboz = 24
b.compute_surfs = compute_surfs
b.run()

boozmn_filename = "boozmn_" + extension + "_ns51.nc"
b.write_boozmn(boozmn_filename)

script_dir = os.path.dirname(os.path.abspath(__file__))
plot_script_candidates = [
    os.path.join(script_dir, "make_boozer_plot.py"),
    os.path.join(script_dir, "make_Boozer_plot.py"),
    os.path.join(script_dir, "plot_Boozer.py"),
]
plot_script = next((p for p in plot_script_candidates if os.path.isfile(p)), None)
if plot_script is None:
    raise FileNotFoundError(
        "Could not find Boozer plot script in script directory. "
        f"Checked: {plot_script_candidates}"
    )

subprocess.run([sys.executable, plot_script, boozmn_filename], check=True)


