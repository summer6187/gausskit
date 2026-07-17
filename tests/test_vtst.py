"""Tests for the gausskit-native microcanonical mu-VTST engine (gausskit.multiwell.vtst).

The reliability argument (GATE 3a): the engine's HARMONIC backend, run on the exact same
``ktools.dat`` that native KTOOLS consumes, must reproduce KTOOLS's capture rate k_cap(T) and
its variational bottleneck r_var(T).  Only the vibrational sum-of-states backend then changes for
the anharmonic run, so a passing harmonic-limit check transfers trust to the anharmonic result.

Fixtures (data/):
  ktools_ohco_deck.dat        the gausskit-generated OH+CO M06-2X entrance deck (reac + 5 trial TS + OH,CO)
  ktools_ohco_capture.canonical  native KTOOLS run of that deck (single-minimum -> FINAL == microcanonical)
  ohco_R2.10.dens             bdens coupled-VPT2 DOS for the R2.10 bottleneck surface

Runnable via pytest or as ``python test_vtst.py``.
"""
import os

import numpy as np

from gausskit.multiwell.vtst import (read_ktools_deck, read_dens_sum, capture_rate,
                                     formula_mass, q_rot_2d_quantum, b_rot)
from gausskit.multiwell.ktools import read_ktools_canonical, read_ktools_min_flux

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DECK = os.path.join(DATA, "ktools_ohco_deck.dat")
CANON = os.path.join(DATA, "ktools_ohco_capture.canonical")
DENS = os.path.join(DATA, "ohco_R2.10.dens")

GRAIN = 5.0                       # matches KTOOLS to <1%; 20 (the coarse default) drifts ~13% at 100 K


def _at(temps, values, t):
    return values[list(temps).index(t)]


# --- parsing ------------------------------------------------------------------
def test_read_deck():
    d = read_ktools_deck(DECK)
    assert (d["nreac"], d["ntts"], d["nprod"]) == (1, 5, 2)
    names = [s["name"] for s in d["structures"]]
    assert names == ["Mol1", "TS2", "TS3", "TS4", "Mol5", "Mol6", "Mol7", "Mol8"]
    ts4 = next(s for s in d["structures"] if s["name"] == "TS4")
    assert len(ts4["vib"]) == 5 and ts4["kro"] is not None and ts4["jro"] is not None
    oh = next(s for s in d["structures"] if s["name"] == "Mol7")
    assert oh["formula"] == "HO"
    # OH 2-Pi spin-orbit ladder present (139.7 cm-1, g=2)
    assert any(abs(e - 139.7) < 0.1 and g == 2 for e, g in oh["elev"])


def test_read_dens():
    E, dens, sos = read_dens_sum(DENS)
    # sum of states grows overall (bdens uses a Monte-Carlo/Wang-Landau count, so tiny local
    # non-monotonic wiggles are expected -- assert the trend, not strict monotonicity).
    assert E[0] == 0.0 and sos[0] > 0 and sos[-1] > sos[0] and np.all(np.isfinite(sos))
    assert E[-1] >= 20000.0 and np.median(np.diff(sos)) >= 0


def test_helpers():
    assert abs(formula_mass("HO") - 17.0027) < 1e-3
    assert abs(formula_mass("CO") - 27.995) < 1e-2
    assert abs(b_rot(1.0) - 16.85763) < 1e-4
    # quantum 2D rotor > 0 and rises with T
    assert q_rot_2d_quantum(1.0, 300) > q_rot_2d_quantum(1.0, 100) > 0


# --- GATE 3a: harmonic backend reproduces native KTOOLS ------------------------
def test_harmonic_reproduces_ktools():
    deck = read_ktools_deck(DECK)
    res = capture_rate(deck, grain=GRAIN)
    T_k, _fwd, rev, _keq = read_ktools_canonical(CANON, "reverse")   # FINAL == microcanonical (single min)
    ref = {t: r for t, r in zip(T_k, rev)}
    devs = []
    for T, k in zip(res["T"], res["k_cap"]):
        if T in ref:
            devs.append(abs(k / ref[T] - 1))
    assert devs, "no overlapping temperatures with the KTOOLS reference"
    assert max(devs) < 0.03, f"harmonic mu-VTST deviates {100*max(devs):.1f}% from native KTOOLS (>3%)"


def test_variational_bottleneck_matches_ktools():
    deck = read_ktools_deck(DECK)
    res = capture_rate(deck, grain=GRAIN)
    T_k, rvar_k, _ = read_ktools_min_flux(CANON, "reverse")
    ref = {t: r for t, r in zip(T_k, rvar_k)}
    for T, rv in zip(res["T"], res["r_var"]):
        if T in ref:
            assert abs(rv - ref[T]) < 0.11, f"r_var(T={T}) = {rv} vs KTOOLS {ref[T]}"


# --- GATE 3b: anharmonic backend runs and shifts the rate ----------------------
def test_anharmonic_backend():
    deck = read_ktools_deck(DECK)
    harm = capture_rate(deck, grain=GRAIN)
    anh = capture_rate(deck, grain=GRAIN, dens_map={"TS4": DENS})    # coupled-VPT2 at the bottleneck
    assert np.all(np.isfinite(anh["k_cap"])) and np.all(anh["k_cap"] > 0)
    # the large positive soft-mode anharmonicity (X33=+66, X55=+63) suppresses the DOS -> lower k
    i298 = list(harm["T"]).index(298.0)
    assert anh["k_cap"][i298] < harm["k_cap"][i298]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all vtst tests passed")
