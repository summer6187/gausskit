"""Tests for the KTOOLS ``.canonical`` output readers (read_ktools_canonical / read_ktools_min_flux)
and the ``gausskit utils ktools_rates`` CLI. Runnable via pytest or as ``python test_ktools.py``.

Fixture ``data/ktools_kinf.canonical`` is a real 36-temperature ktools uVTST run for the OH+CO
barrierless entrance (harmonic microcanonical variational TST); the asserted values are the ones
the ohco reproduction quotes (k_cap(298 K) reverse = 3.44358e-11, r_var migrating 6.2 -> 2.4 A).
"""
import os

import numpy as np

from gausskit.multiwell.ktools import (read_ktools_canonical, read_ktools_min_flux,
                                       read_ktools_unified)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CANON = os.path.join(DATA, "ktools_kinf.canonical")                 # ohco entrance (no uk(t) column)
CH2O = os.path.join(DATA, "ktools_ch2o_capture.canonical")          # CH2O+OH capture (uk(t) column)


def _at(temps, values, t):
    return values[list(temps).index(t)]


def test_final_recommended():
    T, forward, reverse, Keq = read_ktools_canonical(CANON)
    assert len(T) == 36
    assert reverse is not None and Keq is not None            # product defined -> all columns
    assert np.isclose(_at(T, forward, 298.0), 6.30720e11, rtol=1e-4)
    assert np.isclose(_at(T, reverse, 298.0), 3.44358e-11, rtol=1e-4)
    assert np.isclose(_at(T, Keq, 298.0), 1.83158e22, rtol=1e-4)
    # detailed balance: reverse == forward / Keq
    assert np.allclose(reverse, forward / Keq, rtol=1e-3)


def test_min_flux_rvar():
    T, r_var, k_min = read_ktools_min_flux(CANON, direction="reverse")
    assert len(T) == 36
    assert np.isclose(_at(T, r_var, 5.0), 6.20)               # loose outer TS at low T
    assert np.isclose(_at(T, r_var, 77.0), 4.20)              # mid migration
    assert np.isclose(_at(T, r_var, 298.0), 2.40)             # tight inner TS
    assert np.isclose(_at(T, k_min, 298.0), 5.9826e-11, rtol=1e-4)


def test_unified_uk_column():
    # CH2O+OH capture .canonical has the uk(t) unified column -> the stable barrierless-capture rate
    T, uk = read_ktools_unified(CH2O, direction="reverse")
    assert len(T) == 11
    assert np.isclose(_at(T, uk, 100.0), 2.8453e-10, rtol=1e-4)     # reference capture at 100 K
    assert np.all(uk > 0)


def test_unified_no_uk_variant():
    # ohco entrance .canonical has NO uk(t) column (single true minimum) -> rate = trailing min k(t)
    T, uk = read_ktools_unified(CANON, direction="reverse")
    assert len(T) == 36 and np.all(uk > 0)


def test_cli_ktools_rates():
    from click.testing import CliRunner
    from gausskit.cli import cli

    result = CliRunner().invoke(cli, ["utils", "ktools_rates", CANON, "--direction", "reverse"])
    assert result.exit_code == 0, result.output
    assert "T_K,reverse,r_var_A" in result.output
    assert "298.00,3.44358e-11,2.40" in result.output


if __name__ == "__main__":
    test_final_recommended()
    test_min_flux_rvar()
    test_cli_ktools_rates()
    print("test_ktools: all passed")
