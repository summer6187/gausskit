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
from ase.units import Bohr, Hartree
from gausskit.gaussian.fchk import FCHK as FCHKReader, parse_gaussian_fc
from gausskit.molecules import Molecules
from gausskit.degrees_of_freedom.harmonic import Harmonic
from gausskit.degrees_of_freedom.vpt2 import VPT2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
with open(os.path.join(DATA, "cf2cl2na_ref.json")) as _ref_f:
    REF = json.load(_ref_f)

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
    # Compare in the object's NATIVE mode order (not sorted) so a mode-ordering
    # regression is caught: the reference is ascending and the native order must be
    # too (index 0 = imaginary reaction coordinate, < 0).
    v = _v()
    ours = [float(v.frequency(i)) for i in range(v.n_modes)]
    assert np.allclose(ours, REF["frequencies_cm"], atol=0.01)
    assert ours[0] < 0                         # imaginary mode is first


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


def test_cubic_diagonal_reliable_modes():
    # 3rd-deriv (2nd-order cubic) chi_ii vs Gaussian for the reliable modes. This is
    # the highest-bug-density piece (Fermi-resonance branching + complex-frequency
    # imaginary-mode handling); the unreliable soft/RC modes (0,1,2) are excluded.
    v = _v()
    ours = np.diag(v.x_matrix_cubic())
    G = np.diag(np.array(REF["x_3rd"]))
    rel = slice(3, 12)
    assert np.allclose(ours[rel], G[rel], atol=1.0)


def test_total_x_matrix_diagonal_reliable_modes():
    # Combined deliverable X = quartic + cubic + Coriolis vs Gaussian's total on the
    # reliable modes -- regression-locks the full x_matrix() path end to end.
    v = _v()
    ours = np.diag(v.x_matrix())
    G = np.diag(np.array(REF["x_total"]))
    rel = slice(3, 12)
    assert np.allclose(ours[rel], G[rel], atol=1.0)


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


def test_calculate_energy_and_forces():
    # calculate() is the higher-order analogue of Harmonic.calculate: its energy minus
    # E0 is the Taylor PES, it is zero (with zero force) at the reference geometry, and
    # its analytic forces match a central finite difference of its own energy.
    v = _v()
    v.molecules.electronic_energy = 0.0                    # fixture has no energy; fix E0 = 0
    at0 = v.displace_along_mode(2, 0.0)                     # reference geometry
    r0 = v.calculate(at0)
    assert abs(r0["energy"] - v.E0) < 1e-9
    assert np.abs(r0["forces"]).max() < 1e-9
    at = v.displace_along_mode(1, 1.7)                      # displaced along a soft mode
    r = v.calculate(at)
    assert abs((r["energy"] - v.E0) - v.taylor_energy(at, 4)) < 1e-9
    x = at.get_positions().reshape(-1).copy()
    h, g_fd = 1e-5, np.zeros_like(x)
    for k in range(len(x)):
        xp = x.copy(); xp[k] += h
        xm = x.copy(); xm[k] -= h
        ep = v.calculate(at.__class__(numbers=v.molecules.numbers, positions=xp.reshape(-1, 3)))["energy"]
        em = v.calculate(at.__class__(numbers=v.molecules.numbers, positions=xm.reshape(-1, 3)))["energy"]
        g_fd[k] = (ep - em) / (2 * h)
    assert np.abs(-r["forces"] - g_fd).max() < 1e-5


def test_unreliable_fc_recorded():
    # Gaussian flags exactly the grid-sensitive soft / reaction-coordinate constants;
    # recorded in the reference so the test documents the irreducible artifacts.
    assert len(REF["unreliable_fc"]) == 4
    assert ["cubic", [11, 4, 12]] in REF["unreliable_fc"]


def _plain_harmonic():
    """A base Harmonic (not VPT2) built from the fixture, so the merged base-class
    methods run on Harmonic's OWN mass_weighted_modes basis -- the path VPT2's
    _modes override otherwise bypasses."""
    fc = FCHKReader(FCHK)
    mol = Molecules(numbers=fc.atomic_numbers, positions=fc.coordinates * Bohr)
    return Harmonic(mol, fc.hessian * (Hartree / Bohr / Bohr), fc.masses)


def test_plain_harmonic_displace_project_roundtrip():
    h = _plain_harmonic()
    L = h.mass_weighted_modes()
    assert L.shape[0] == 3 * len(h.masses) - 6      # 3N-6 vibrational modes
    i = 8                                           # a vibrational mode
    Q = h.project_onto_modes(h.displace_along_mode(i, 1.5))
    assert abs(Q[i] - 1.5) < 1e-6
    assert np.sqrt((Q**2).sum() - Q[i]**2) < 1e-9   # displacement is pure in mode i


def test_vibrational_modes_abs_retains_imaginary():
    # The TS-safe set keeps 3N-6 modes by smallest |w2|, so the imaginary reaction
    # coordinate (large negative w2) is RETAINED -- unlike the naive [6:] slice.
    h = _plain_harmonic()
    w2, modes = h.vibrational_modes_abs()
    n3 = 3 * len(h.masses)
    assert len(w2) == n3 - 6
    assert modes.shape == (n3 - 6, n3)
    assert w2.min() < 0                             # imaginary mode retained


def test_harmonic_only_fchk():
    # Strip the 3rd/4th-derivative block (the last block in the fixture) so the reader
    # hits the harmonic-only branch: cubic/quartic = None, is_anharmonic False, while
    # harmonic frequencies/masses still resolve.
    with open(FCHK) as f:
        lines = f.readlines()
    cut = next(k for k, l in enumerate(lines)
               if l.startswith("Cartesian 3rd/4th derivatives"))
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".fchk", delete=False)
    tmp.writelines(lines[:cut]); tmp.close()
    atexit.register(lambda: os.path.exists(tmp.name) and os.unlink(tmp.name))
    vh = VPT2.from_fchk(tmp.name)
    assert not vh.is_anharmonic
    assert vh.cubic is None and vh.quartic is None
    assert float(vh.frequency(0)) < 0               # harmonic frequencies still work
    assert len(vh.masses) == vh.fchk.n_atoms


def test_parse_gaussian_fc_converts_to_ev_per_ang2():
    # The consolidated converter returns the force-constant matrix in eV/Angstrom^2
    # (Hartree/Bohr^2 scaled), bit-identical to FCHK.hessian scaled; unsupported
    # file types return None.
    fc = parse_gaussian_fc(FCHK)
    expected = FCHKReader(FCHK).hessian * (Hartree / Bohr / Bohr)
    assert fc.shape == (3 * 6, 3 * 6)
    assert np.allclose(fc, expected, atol=0, rtol=0)
    assert parse_gaussian_fc("not_a_real_file.log") is None


def test_lazy_vpt2_package_export():
    # VPT2 is exported lazily from the package (to dodge the molecules<->Harmonic
    # circular import); the package surface must resolve to the same class.
    import gausskit.degrees_of_freedom as ddf
    from gausskit.degrees_of_freedom import VPT2 as V2
    assert V2 is VPT2 and ddf.VPT2 is VPT2
    try:
        ddf.NoSuchAttr
        raised = False
    except AttributeError:
        raised = True
    assert raised


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for n, f in fns:
        f(); print("PASS", n)
    print(f"\n{len(fns)} tests passed.")
