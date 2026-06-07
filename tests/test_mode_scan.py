"""Tests for the `gausskit utils mode_scan` helper (gausskit.utils.mode_scan) and the
log normal-mode reader it depends on (gausskit.gaussian.log_parser.read_normal_modes).

Reuses the copyright-clean CF2Cl2-Na fchk fixture in data/ for the fchk route, and a tiny
synthetic Gaussian log built inline for the log route (so no real .log is shipped).
Runs under pytest, or directly as `python test_mode_scan.py`.
"""
import os
import gzip
import atexit
import tempfile
import numpy as np

from gausskit.utils.mode_scan import generate_mode_scan, make_qgrid, load_ts, DEFAULT_QGRID
from gausskit.gaussian.log_parser import read_normal_modes
from gausskit.degrees_of_freedom.vpt2 import VPT2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Decompress the minimal fchk fixture to a temp .fchk (production never gunzips).
_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".fchk", delete=False)
with gzip.open(os.path.join(DATA, "cf2cl2na_anharm.fchk.gz"), "rt") as _f:
    _tmp.write(_f.read())
_tmp.close()
FCHK = _tmp.name
atexit.register(lambda: os.path.exists(FCHK) and os.unlink(FCHK))

# The two lowest REAL modes of the CF2Cl2-Na TS (the Na translational bends).
SOFT_FREQS = [26.77, 30.35]


def test_make_qgrid_shape():
    q = make_qgrid()
    assert np.isclose(q.min(), -11.0) and np.isclose(q.max(), 11.0)
    assert 0.0 in q                                    # the reference point is included
    assert np.allclose(np.sort(q), q)                  # ascending / unique
    dense = q[np.abs(q) <= 3.0]
    assert np.allclose(np.diff(dense), 0.25)           # 0.25 spacing in the dense window
    assert np.array_equal(make_qgrid(), DEFAULT_QGRID)


def test_make_qgrid_non_default_qmax():
    # the CLI exposes --qmax (make_qgrid(qmax=...)), so the wing arithmetic is a live path
    q = make_qgrid(qmax=6.0)
    assert q.max() == 6.0 and q.min() == -6.0 and 0.0 in q
    assert set(q[np.abs(q) > 3.0]) == {-6.0, -5.0, -4.0, 4.0, 5.0, 6.0}
    q2 = make_qgrid(qmax=2.0)                           # qmax < dense: clamp, no empty wings
    assert q2.max() == 2.0 and q2.min() == -2.0


def test_fchk_path_picks_soft_modes(tmp_path):
    written, picks, freqs = generate_mode_scan(FCHK, n_modes=2, outdir=str(tmp_path / "ms"))
    assert picks == [1, 2]                              # native order: 0 = imaginary RC
    assert np.allclose(freqs, SOFT_FREQS, atol=0.01)
    # structures only: one gjf per (mode, Q), a manifest, and NO submission script
    assert len(written) == 2 * len(DEFAULT_QGRID)
    assert (tmp_path / "ms" / "manifest.csv").exists()
    files = {p.name for p in (tmp_path / "ms").rglob("*")}
    assert not any(f.endswith((".sh", ".sbatch", ".slurm")) for f in files)


def test_displacement_matches_gausskit_object(tmp_path):
    """The util MUST build geometries via Harmonic.displace_along_mode, not a private copy."""
    out = tmp_path / "ms"
    generate_mode_scan(FCHK, n_modes=1, outdir=str(out))
    v = VPT2.from_fchk(FCHK)
    real = np.where(np.array([float(v.frequency(i)) for i in range(v.n_modes)]) > 0)[0]
    idx = int(real[0])                                  # lowest real mode = the first pick
    rows = [l.split(",") for l in (out / "manifest.csv").read_text().splitlines()[1:]]
    row = next(r for r in rows if r[0] == "1" and abs(float(r[3]) - 2.0) < 1e-9)
    gjf = (out / row[4]).read_text().splitlines()
    coords = np.array([[float(x) for x in l.split()[1:4]]
                       for l in gjf if l[:1].isalpha() and len(l.split()) == 4])
    ref = v.displace_along_mode(idx, 2.0).get_positions()
    assert np.allclose(coords, ref, atol=1e-8)


# ----- synthetic Gaussian log (H-Cl: 2 atoms, 2 mock modes) for the log reader -----
def _synthetic_log(with_masses=True, imaginary_first=False):
    f1 = "-500.0000" if imaginary_first else "1234.5678"   # imaginary RC vs a real soft mode
    L = [
        "                         Standard orientation:",
        " ---------------------------------------------------------------------",
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        " ---------------------------------------------------------------------",
        "      1          1           0        0.000000    0.000000    0.000000",
        "      2         17           0        0.000000    0.000000    1.275000",
        " ---------------------------------------------------------------------",
        f" Frequencies --   {f1}              2345.6789",
        " Red. masses --      1.0000                 1.0000",
        "  Atom  AN      X      Y      Z        X      Y      Z",
        "     1   1     0.00   0.00   0.99     0.99   0.00   0.00",
        "     2  17     0.00   0.00  -0.05    -0.05   0.00   0.00",
        " -------------------",
    ]
    if with_masses:
        L += ["Atom     1 has atomic number  1 and mass   1.00783",
              "Atom     2 has atomic number 17 and mass  34.96885"]
    return "\n".join(L) + "\n"


def _write_log(text):
    t = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    t.write(text); t.close()
    atexit.register(lambda: os.path.exists(t.name) and os.unlink(t.name))
    return t.name


def test_read_normal_modes_parses_geometry_and_modes():
    numbers, pos, mass, freqs, cart = read_normal_modes(_write_log(_synthetic_log()))
    assert numbers.tolist() == [1, 17]
    assert np.allclose(pos[1], [0, 0, 1.275])
    assert np.allclose(freqs, [1234.5678, 2345.6789])
    assert len(cart) == 2 and cart[0].shape == (2, 3)
    assert np.allclose(mass, [1.00783, 34.96885])      # masses parsed straight from the log


def test_mass_fallback_uses_gaussian_isotope_convention():
    """When the log has no mass lines, the fallback must match Gaussian's most-abundant-isotope
    convention (Cl = 34.96885), NOT ASE's isotope-averaged standard weight (Cl = 35.45)."""
    _, _, mass, _, _ = read_normal_modes(_write_log(_synthetic_log(with_masses=False)))
    assert np.isclose(mass[1], 34.96885, atol=1e-3)    # Cl-35, the Gaussian default
    assert not np.isclose(mass[1], 35.45, atol=0.1)    # the averaged weight would corrupt the mode


def test_input_orientation_fallback():
    """A nosymm freq log prints only 'Input orientation' (no 'Standard orientation') -- the
    reader must fall back to it instead of raising, since DEFAULT_ROUTE itself uses nosymm."""
    log = _write_log(_synthetic_log().replace("Standard orientation", "Input orientation"))
    numbers, pos, mass, freqs, cart = read_normal_modes(log)
    assert numbers.tolist() == [1, 17] and np.allclose(pos[1], [0, 0, 1.275])


def test_log_route_generate_mode_scan(tmp_path):
    """Drive generate_mode_scan through its .log branch (load_ts -> Harmonic with modes=Lg):
    the imaginary first mode is excluded, and each gjf equals displace_along_mode(modes=Lg)."""
    log = _write_log(_synthetic_log(imaginary_first=True))
    out = tmp_path / "ms"
    written, picks, freqs = generate_mode_scan(log, n_modes=1, outdir=str(out))
    assert picks == [1]                                # idx 0 (-500i) excluded; lowest real picked
    assert len(written) == len(DEFAULT_QGRID)
    # the log route passes an explicit modes=Lg override into displace_along_mode -- check parity
    harm, Lg, _ = load_ts(log)
    ref = harm.displace_along_mode(1, 2.0, modes=Lg).get_positions()
    rows = [l.split(",") for l in (out / "manifest.csv").read_text().splitlines()[1:]]
    row = next(r for r in rows if abs(float(r[3]) - 2.0) < 1e-9)
    gjf = (out / row[4]).read_text().splitlines()
    coords = np.array([[float(x) for x in l.split()[1:4]]
                       for l in gjf if l[:1].isalpha() and len(l.split()) == 4])
    assert np.allclose(coords, ref, atol=1e-8)


if __name__ == "__main__":
    import tempfile as _t
    from pathlib import Path as _P
    test_make_qgrid_shape()
    test_make_qgrid_non_default_qmax()
    test_fchk_path_picks_soft_modes(_P(_t.mkdtemp()))
    test_displacement_matches_gausskit_object(_P(_t.mkdtemp()))
    test_read_normal_modes_parses_geometry_and_modes()
    test_mass_fallback_uses_gaussian_isotope_convention()
    test_input_orientation_fallback()
    test_log_route_generate_mode_scan(_P(_t.mkdtemp()))
    print("all mode_scan tests passed")
