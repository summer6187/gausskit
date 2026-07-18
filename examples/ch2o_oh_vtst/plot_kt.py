#!/usr/bin/env python3
"""CH2O + OH capture rate: harmonic vs VPT2 vs Vhrd2 (rigid-rotor) VTST, all on one deck.

Reads the three rates/engine_*.csv produced by `make rates` and the Ali-Barker reference,
plots k_cap(T) on a log axis with a ratio-to-reference panel. Writes kt.png.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def read(path, kcol):
    p = HERE / path
    if not p.exists():
        return None, None
    rows = list(csv.DictReader(open(p)))
    if not rows or kcol not in rows[0]:
        return None, None
    return (np.array([float(r["T_K"]) for r in rows]),
            np.array([float(r[kcol]) for r in rows]))


# (file, column, label, color, linestyle, linewidth)
CURVES = [
    ("reference_capture.csv",        "reverse",           "Ali & Barker reference",         "k",         ":",  1.8),
    ("rates/engine_harmonic.csv",    "k_cap_harmonic",    "harmonic VTST",                  "tab:gray",  "--", 1.6),
    ("rates/engine_vpt2.csv",        "k_cap_anharmonic",  "VPT2 VTST (collapses)",          "tab:orange","-.", 1.6),
    ("rates/engine_modeaware.csv",   "k_cap_anharmonic",  "Vhrd2 rigid-rotor VTST",         "tab:blue",  "-",  2.2),
]

fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
Tref, kref = read("reference_capture.csv", "reverse")
for path, col, lab, c, ls, lw in CURVES:
    T, k = read(path, col)
    if T is None:
        continue
    ax.semilogy(1000.0 / T, k, ls, color=c, lw=lw, label=lab)
    if kref is not None and path != "reference_capture.csv":
        ax.scatter([], [])
        r = k / np.interp(T, Tref[::-1], kref[::-1])
        axr.plot(1000.0 / T, r, ls, color=c, lw=lw)

ax.set_ylabel(r"$k_{cap}$  (cm$^3$ molecule$^{-1}$ s$^{-1}$)")
ax.set_title("CH$_2$O + OH barrierless capture — three DOS treatments, one $\\mu$-VTST deck")
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.3, which="both")
axr.axhline(1.0, color="k", lw=0.8, ls=":")
axr.axhspan(0.5, 2.0, color="0.85", alpha=0.5, zorder=0)
axr.set_yscale("log")
axr.set_ylabel("model / ref")
axr.set_xlabel("1000 / T  (K$^{-1}$)")
axr.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(HERE / "kt.png", dpi=140)
print("wrote kt.png")

# one-line headline
for path, col, lab, *_ in CURVES[1:]:
    T, k = read(path, col)
    if T is None:
        continue
    g = np.exp(np.mean(np.log(k / np.interp(T, Tref[::-1], kref[::-1]))))
    print(f"  {lab:32s} geomean k/ref = {g:.3g}x   k(100K) = {k[0]:.3g}")
