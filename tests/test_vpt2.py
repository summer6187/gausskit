"""VPT2 regression tests against a minimal, copyright-clean CF2Cl2-Na fixture.

Fixture (in data/): a gzipped minimal fchk holding only the numerical force-constant
blocks, and a JSON reference of Gaussian ground-truth values (frequencies, X-matrix
contribution blocks in this object's mode order, rotational constants, unreliable-FC
flags). Runs under pytest, or directly as `python test_vpt2.py`.
"""
import os
import json
import gzip
import atexit
import tempfile
import numpy as np
from ase.units import Bohr
from gausskit.degrees_of_freedom.vpt2 import VPT2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REF = json.load(open(os.path.join(DATA, "cf2cl2na_ref.json")))

# The fixture is gzipped only to keep the repo small; FCHK itself reads plain
# fchk/log files (production never gunzips). Decompress to a temp .fchk here.
_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".fchk", delete=False)
with gzip.open(os.path.join(DATA, "cf2cl2na_anharm.fchk.gz"), "rt") as _f:
    _tmp.write(_f.read())
_tmp.close()
FCHK = _tmp.name
atexit.register(lambda: os.path.exists(FCHK) and os.unlink(FCHK))


def _v():
    return VPT2.from_fchk(FCHK)               # minimal (plain) fchk, no log


def test_loads_anharmonic():
    v = _v()
    assert v.is_anharmonic
    assert v.n_modes == REF["n_modes"] == 12


def test_frequencies_match_gaussian():
    v = _v()
    ours = sorted(float(v.frequency(i)) for i in range(v.n_modes))
    assert np.allclose(ours, REF["frequencies_cm"], atol=0.01)


def test_rotational_constants_match_gaussian():
    v = _v()
    mss = v.masses
    R = v.fchk.coordinates * Bohr
    R = R - (mss[:, None]*R).sum(0)/mss.sum()
    It = sum(mss[A]*(np.dot(R[A], R[A])*np.eye(3) - np.outer(R[A], R[A]))
             for A in range(len(mss)))
    B = sorted(16.857629/np.linalg.eigh(It)[0], reverse=True)
    assert np.allclose(B, REF["rot_constants_cm"], atol=1e-4)


def test_coriolis_matches_gaussian_exactly():
    v = _v()
    ours = v.x_matrix_coriolis()
    G = np.array(REF["x_coriolis"])
    r = slice(1, 12)                           # exclude imaginary mode (index 0)
    assert np.allclose(ours[r, r], G[r, r], atol=1e-3)


def test_quartic_diagonal_reliable_modes():
    # 4th-deriv chi_ii = phi_iiii/16 matches Gaussian for the reliable modes
    # (exclude imaginary mode 0 and the two grid-unreliable soft modes 1,2 that
    #  Gaussian zeros).
    v = _v()
    ours = np.diag(v.x_matrix_quartic())
    G = np.diag(np.array(REF["x_4th"]))
    rel = slice(3, 12)
    assert np.allclose(ours[rel], G[rel], atol=0.02)


def test_displace_project_roundtrip():
    v = _v()
    assert np.allclose(v.displace_along_mode(2, 0.0).get_positions(), v.positions0)
    Q = v.project_onto_modes(v.displace_along_mode(3, 2.5))
    assert abs(Q[3] - 2.5) < 1e-6
    assert np.sqrt((Q**2).sum() - Q[3]**2) < 1e-9


def test_taylor_pes_self_consistency():
    v = _v()
    Qs = np.linspace(-2, 2, 5)
    for i in (1, 2, 5):
        E_1d = v.scan_mode_taylor(i, Qs, order=4)
        E_full = np.array([v.taylor_energy(v.displace_along_mode(i, Q), 4) for Q in Qs])
        assert np.allclose(E_1d, E_full, atol=1e-9)


def test_unreliable_fc_recorded():
    # Gaussian flags exactly the grid-sensitive soft / reaction-coordinate constants;
    # recorded in the reference so the test documents the irreducible artifacts.
    assert len(REF["unreliable_fc"]) == 4
    assert ["cubic", [11, 4, 12]] in REF["unreliable_fc"]


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for n, f in fns:
        f(); print("PASS", n)
    print(f"\n{len(fns)} tests passed.")
