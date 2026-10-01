"""Fit the scanned soft modes -> levels, Vhrd2/HRD deck blocks, and validation.

Reads the per-mode ANCHOR from the roadmap (never re-classifies) and dispatches on
``method``:
  * ``rigid-rotor`` -> read_theta -> rotor_to_hrd (physical B) + rotor_levels (gold DVR).
  * ``mode-scan``  -> read_rectilinear -> scan_to_hrd (Vhrd2) + sinc_dvr (gold DVR).
Validates q(HRD/DVR) at 300/700/1000 K, renders the MultiWell HRD block, and writes
``rotor_result.json`` + ``fit_validation.png/.csv``. Results go to a SEPARATE file so a
re-fit never clobbers the hand-edited roadmap.
"""
import json
from pathlib import Path

import numpy as np

from gausskit.multiwell.dos import scan_to_hrd, sinc_dvr, hrd_block, q_levels
from gausskit.utils.rotor.constants import KAPPA, DVR_EMAX, QCHECK_TEMPS
from gausskit.utils.rotor.levels import rotor_levels, rotor_to_hrd
from gausskit.utils.rotor.readback import read_theta, read_rectilinear
from gausskit.utils.rotor.figures import write_validation_figure


def _rms(V, Vfit):
    Vfit = np.asarray(Vfit, float) - np.asarray(Vfit, float).min()
    return float(np.sqrt(np.mean((Vfit - (V - V.min())) ** 2)))


def fit_rotors(plan, scan_dir, output, *, dvr_emax=DVR_EMAX, qcheck_temps=QCHECK_TEMPS,
               figure=True):
    """Fit every scanned mode in `plan` from logs under `scan_dir`. Writes `output`
    (rotor_result.json) and, if `figure`, fit_validation.png/.csv beside it. Returns the
    list of per-mode result dicts."""
    scan_dir = Path(scan_dir)
    results, plots, errors = [], [], []
    for m in plan["modes"]:
        k, method = m["mode"], m["method"]
        mdir = scan_dir / f"mode{k}"
        if method == "harmonic":
            results.append({"mode": k, "gauss_idx": m["gauss_idx"], "method": "harmonic",
                            "freq_cm": m["freq_cm"], "note": "stiff mode, kept harmonic"})
            continue
        try:
            if method == "rigid-rotor":
                th, V = read_theta(mdir)
                B = KAPPA / float(m["I_amuA2"])
                hrd = rotor_to_hrd(th, V, B)
                ref, zpe = rotor_levels(th, V, B)      # zpe = rotor ground above V minimum
                phi = np.radians(th)
                # reconstruct V(phi) exactly as the deck defines it: sum CV[n] cos(n*nsym*phi)
                Vfit = sum(hrd["CV"][n] * np.cos(n * hrd["nsym"] * phi) for n in range(len(hrd["CV"])))
                x, xlabel = th, "theta (deg)"
            else:                                                  # mode-scan
                Q, V = read_rectilinear(mdir)
                hrd = scan_to_hrd(Q, V)
                ref = sinc_dvr(Q, V, emax=dvr_emax)
                zpe = float(ref[0])                    # sinc_dvr is referenced to the V minimum
                L = Q.max() - Q.min()
                phi = 2 * np.pi * (Q - Q.min()) / L
                Vfit = sum(hrd["CV"][n] * np.cos(n * hrd["nsym"] * phi) for n in range(len(hrd["CV"])))
                x, xlabel = Q, "Q (A*amu^1/2)"
        except (ValueError, OSError) as e:
            errors.append(f"mode {k} ({method}): {e}")
            continue

        qr = {str(int(T)): round(q_levels(hrd["levels"], T) / q_levels(ref, T), 4)
              for T in qcheck_temps}
        results.append({
            "mode": k, "gauss_idx": m["gauss_idx"], "method": method,
            "freq_cm": m["freq_cm"], "zpe_cm": round(zpe, 3), "B_cm": round(hrd["B"], 6),
            "I_red_amuA2": round(hrd["I_red"], 4), "nsym": hrd["nsym"],
            "CV_Vhrd2": [round(float(c), 5) for c in hrd["CV"]],
            "levels_cm": [round(float(v), 3) for v in hrd["levels"][:40]],
            # ground-reference the DVR ladder too (sinc_dvr is referenced to the potential
            # minimum, ref[0]=ZPE; the ZPE is kept separately in zpe_cm) so both ladders share
            # an n=0 baseline and are directly comparable.
            "ref_levels_cm": [round(float(v), 3)
                              for v in (np.asarray(ref) - np.asarray(ref).min())[:40]],
            "fit_RMS_cm": round(_rms(V, Vfit), 3),
            "q_HRD_over_DVR": qr,
            "hrd_block": hrd_block(k, hrd, f"{method} soft mode {k}"),
        })
        plots.append({"mode": k, "method": method, "x": np.asarray(x), "xlabel": xlabel,
                      "V": V - V.min(), "Vfit": np.asarray(Vfit) - np.asarray(Vfit).min(),
                      "levels": np.asarray(hrd["levels"]),
                      "ref": np.asarray(ref) - np.asarray(ref).min()})

    if errors:
        raise RuntimeError("rotor fit failed for some modes (fix the pull, do not ignore):\n  "
                           + "\n  ".join(errors))

    out = Path(output)
    out.write_text(json.dumps({"log": plan["log"], "modes": results}, indent=2) + "\n")
    if figure and plots:
        # named after the result file so multi-surface loops don't overwrite each other
        write_validation_figure(plots, out.with_name(out.stem + ".validation.png"), results)
    return results
