#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
begum@mit.edu
2026-08-26

Export decomposed quantum noise from a GWINC YAML.

Usage:
    python export_qnoise_decomp.py model.yaml
    python export_qnoise_decomp.py model.yaml output.mat

Decomposition assumes S(P) = A/P + B*P + C

Solve with factor1*P0, P0, and factor2*P0.
"""

import sys
from pathlib import Path

import numpy as np
import gwinc
from scipy.io import savemat


fmin = 10.0
fmax = 7000.0
npoints = 12000
cases = ("no_sqz", "fis", "fds")
factor1 = 0.5
factor2 = 2


def solve_abc(powers, spectra):
    p1, p2, p3 = powers

    M = np.array([
        [1 / p1, p1, 1],
        [1 / p2, p2, 1],
        [1 / p3, p3, 1],
    ])

    return np.linalg.solve(M, np.vstack(spectra))



def quantum_psd(yaml_path, freq, case, arm_power):
    budget = gwinc.load_budget(str(yaml_path), freq, bname="Quantum")
    ifo = budget.ifo

    ifo.Laser.ArmPower = arm_power

    if case == "no_sqz":
        del ifo.Squeezer.FilterCavity
        del ifo.Squeezer
    elif case == "fis":
        ifo.Squeezer.Type = "Freq Independent"
    elif case == "fds":
        ifo.Squeezer.Type = "Freq Dependent"

    return np.asarray(budget.run().psd, float)



def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python export_qnoise_decomp.py model.yaml [output.mat]"
        )

    yaml_path = Path(sys.argv[1]).expanduser().resolve()

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2]).expanduser().resolve()
    else:
        output_path = yaml_path.with_name(
            yaml_path.stem + "_qnoise_decomp_v71.mat"
        )

    freq = np.geomspace(fmin, fmax, npoints)

    budget = gwinc.load_budget(str(yaml_path), freq, bname="Quantum")
    p0 = float(budget.ifo.Laser.ArmPower)
    powers = [factor1 * p0, p0, factor2 * p0]

    output = {
        "freq_Hz": freq,
        "yaml_path": str(yaml_path),
        "P0_W": p0,
        "powers_W": np.asarray(powers),
    }

    for case in cases:
        spectra = [
            quantum_psd(yaml_path, freq, case, power)
            for power in powers
        ]

        S_tot = spectra[1]
        A, B, C = solve_abc(powers, spectra)

        S_imp = A / p0
        S_ba = B * p0
        S_x = C
        S_recon = S_imp + S_ba + S_x

        output[case] = {
            "S_tot": S_tot,
            "S_imp": S_imp,
            "S_ba": S_ba,
            "S_x": S_x,
            "S_recon": S_recon,
            "ASD_tot": np.sqrt(np.maximum(S_tot, 0)),
            "ASD_imp": np.sqrt(np.maximum(S_imp, 0)),
            "ASD_ba": np.sqrt(np.maximum(S_ba, 0)),
            "ASD_xabs": np.sqrt(np.abs(S_x)),
            "ASD_recon": np.sqrt(np.maximum(S_recon, 0)),
        }

    savemat(str(output_path), output, do_compression=True)
    print(f"Saved: {output_path}")
    


if __name__ == "__main__":
    main()
