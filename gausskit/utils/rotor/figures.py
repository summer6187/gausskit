"""Validation figure for the rotor fit — one row per scanned mode.

Left panel: the fitted potential (Vhrd2 for mode-scan, cosine rotor for rigid-rotor) over
the scanned points, x = Q or theta per the mode's model. Right panel: the HRD level ladder
against the matching gold-standard DVR ladder (sinc_dvr for mode-scan, periodic-rotor DVR
for rigid-rotor). Generalises hindered_rotor_kit/scripts/validate_hrd.py to mixed models.
Also writes fit_validation.csv (mode, method, well depth, fit RMS, q(HRD/DVR)).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_N_LEVELS = 14


def write_validation_figure(plots, path, results):
    path = Path(path)
    nrow = len(plots)
    fig, axes = plt.subplots(nrow, 2, figsize=(10.5, 3.1 * nrow), squeeze=False)
    for r, p in enumerate(plots):
        ax0, ax1 = axes[r]
        order = np.argsort(p["x"])
        ax0.plot(p["x"][order], p["V"][order], "o", ms=4, color=f"C{r}", label="scan")
        ax0.plot(p["x"][order], p["Vfit"][order], "-", lw=1.3, color=f"C{r}", label="fit")
        ax0.set_xlabel(p["xlabel"]); ax0.set_ylabel("V (cm$^{-1}$)")
        ax0.set_title(f"mode {p['mode']} ({p['method']}): potential fit")
        ax0.legend(fontsize=8)
        gold = "sinc-DVR" if p["method"] == "mode-scan" else "rotor-DVR"
        ax1.hlines(p["ref"][:_N_LEVELS], -0.32, -0.02, color="k", lw=1, label=gold)
        ax1.hlines(p["levels"][:_N_LEVELS], 0.02, 0.32, color="r", lw=1, label="HRD")
        ax1.set_xlim(-0.6, 0.6); ax1.set_xticks([])
        ax1.set_ylabel("level (cm$^{-1}$)")
        ax1.set_title(f"mode {p['mode']}: HRD vs {gold} levels")
        ax1.legend(fontsize=8)
    fig.suptitle("rotor fit validation", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=140)
    plt.close(fig)

    rows = ["mode,method,well_depth_cm,fit_RMS_cm,"
            "q_HRD_over_DVR_300,q_HRD_over_DVR_700,q_HRD_over_DVR_1000"]
    bymode = {res["mode"]: res for res in results if res["method"] != "harmonic"}
    for p in plots:
        res = bymode.get(p["mode"], {})
        q = res.get("q_HRD_over_DVR", {})
        rows.append(f"{p['mode']},{p['method']},{p['V'].max():.0f},{res.get('fit_RMS_cm','')},"
                    f"{q.get('300','')},{q.get('700','')},{q.get('1000','')}")
    path.with_suffix(".csv").write_text("\n".join(rows) + "\n")
    return path
