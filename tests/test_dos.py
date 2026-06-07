"""Tests for the density-of-states utilities (gausskit.multiwell.dos).

Self-contained: a sinc-DVR check against an analytic harmonic oscillator, the
Stein-Rabinovitch convolution against brute-force state enumeration, and a
MultiWell `.dens` write/read round-trip. Runs under pytest or as
`python test_dos.py`.
"""
import os
import tempfile
import numpy as np
from gausskit.multiwell.dos import (
    sinc_dvr, stein_rabinovitch, write_dens, q_levels, q_harmonic,
    scan_to_hrd, hrd_block, CM_PER_EV, C_KIN_EV,
)


def test_sinc_dvr_recovers_harmonic():
    # Harmonic well V = 1/2 a2 Q^2 with a chosen frequency; DVR must give E0 = w/2
    # and uniform spacing w.
    w = 200.0                                   # cm^-1
    a2 = (w / CM_PER_EV) ** 2 / (2 * C_KIN_EV)  # eV/(amu A^2)
    Q = np.linspace(-5, 5, 401)
    V_cm = 0.5 * a2 * Q**2 * CM_PER_EV
    E = sinc_dvr(Q, V_cm, npts=400, emax=4000)
    assert abs(E[0] - w / 2) < 0.5
    sp = np.diff(E[:6])
    assert np.allclose(sp, w, atol=0.5)


def _brute_sos(freqs, E):
    """States with sum_i v_i*freq_i <= E, v_i >= 0 (three modes)."""
    f1, f2, f3 = freqs
    c = 0
    for v1 in range(int(E // f1) + 1):
        r1 = E - v1 * f1
        for v2 in range(int(r1 // f2) + 1):
            r2 = r1 - v2 * f2
            c += int(r2 // f3) + 1
    return c


def test_stein_rabinovitch_matches_brute_force():
    freqs = [100.0, 150.0, 250.0]
    grain = 50.0
    emax = 1500.0
    level_sets = [np.arange(0, emax + 1, f) for f in freqs]
    E, dens, sos = stein_rabinovitch(level_sets, grain, emax)
    for Etest in (300.0, 500.0, 1000.0):
        k = int(round(Etest / grain))
        assert sos[k] == _brute_sos(freqs, Etest), (Etest, sos[k], _brute_sos(freqs, Etest))
    # density is the per-grain count / grain; cumulative integral ~ sos
    assert abs(np.cumsum(dens * grain)[-1] - sos[-1]) < 1e-6


def test_density_sum_consistency():
    # sos must be the running integral of density*grain
    grain = 10.0
    lev = [np.arange(0, 5000, 120.0), np.arange(0, 5000, 333.0)]
    E, dens, sos = stein_rabinovitch(lev, grain, 5000)
    assert np.allclose(np.cumsum(dens * grain), sos, atol=1e-9)


def test_write_dens_roundtrip():
    grain, emax = 10.0, 3000.0
    lev = [np.arange(0, emax, 200.0), np.arange(0, emax, 270.0)]
    E, dens, sos = stein_rabinovitch(lev, grain, emax)
    isize = len(dens)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".dens", delete=False)
    tmp.close()
    write_dens(tmp.name, "TESTMOL", "test reference", grain, isize, emax, dens, sos)
    with open(tmp.name) as f:
        lines = f.read().splitlines()
    os.unlink(tmp.name)
    assert lines[0].strip() == "TESTMOL"
    grid = lines[2].split()
    assert float(grid[0]) == grain and int(grid[3]) == isize     # Egrain1, Isize
    # first data row: index 1, E=0, density and sum parse as floats
    first = lines[4].split()
    assert int(first[0]) == 1 and float(first[1]) == 0.0
    assert float(first[2]) > 0 and float(first[3]) > 0
    assert len(lines) == 4 + isize                                # header(4) + rows


def test_q_levels_matches_harmonic():
    # explicit harmonic ladder partition function == closed-form q_harmonic
    w = 120.0
    levels = w * np.arange(0, 400)
    for T in (300, 700, 1500):
        assert abs(q_levels(levels, T) - q_harmonic(w, T)) < 1e-6


def test_scan_to_hrd_reproduces_dvr():
    # a realistic stiffening soft-mode well (~1 eV at the +-11 edges, like the Na+CFxCly
    # scans); the scan-matched HRD must reproduce the direct DVR partition function.
    w, a4 = 30.0, 8.0e-4
    a2 = (w / CM_PER_EV) ** 2 / (2 * C_KIN_EV)
    Q = np.linspace(-11, 11, 41)
    V = (0.5 * a2 * Q**2 + a4 / 24 * Q**4) * CM_PER_EV          # cm^-1
    Edvr = sinc_dvr(Q, V, emax=12000)
    hrd = scan_to_hrd(Q, V, n_fourier=16)
    for T in (500, 800, 1200):
        r = q_levels(hrd["levels"], T) / q_levels(Edvr, T)
        assert abs(r - 1.0) < 0.01, (T, r)
    # block format: hrd line + Vhrd2 + Bhrd1
    block = hrd_block(10, hrd).splitlines()
    assert block[0].split()[1] == "hrd"
    assert block[1].strip().startswith("Vhrd2")
    assert block[2].strip().startswith("Bhrd1")
    assert abs(float(block[2].split()[-1]) - hrd["B"]) < 1e-4


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for n, f in fns:
        f(); print("PASS", n)
    print(f"\n{len(fns)} tests passed.")
