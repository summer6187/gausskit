"""Read a scanned soft-mode potential V back from the pulled Gaussian logs.

Two readers, one per model, each matching the title convention `rotor scan` wrote:
  * ``read_theta``       — rigid-rotor logs (title carries 'theta='); energy from 'SCF Done'.
                           Ported from mode_aware_toolkit/dos.py:_read_theta.
  * ``read_rectilinear`` — mode-scan logs (title carries 'Q='); energy via
                           gausskit Molecules.from_log. Ported from
                           hindered_rotor_kit/scripts/extract_scan.py.
Both return (x, V_cm) with V referenced to its own minimum (cm^-1). Both raise if the
directory has too few normally-terminated points, so ``fit`` fails loudly per mode.
"""
import io
import re
import glob
import contextlib
from pathlib import Path

import numpy as np

from gausskit.utils.rotor.constants import HARTREE_CM

_THETA = re.compile(r"theta=(-?\d+)")
_SCF = re.compile(r"SCF Done:\s+E\S+\s+=\s+(-?\d+\.\d+)")
_Q = re.compile(r"\bQ=\s*([+-]?\d+\.\d+)")
_MIN_PTS = 5


def read_theta(mdir):
    """(theta_deg, V_cm) from a rigid-rotor scan directory's *.log files."""
    pts = []
    for f in glob.glob(str(Path(mdir) / "*.log")):
        t = open(f, errors="ignore").read()
        if "Normal termination" not in t:
            continue
        th, e = _THETA.search(t), _SCF.findall(t)
        if th and e:
            pts.append((float(th.group(1)), float(e[-1])))
    if len(pts) < _MIN_PTS:
        raise ValueError(f"{mdir}: only {len(pts)} usable rigid-rotor points (need >= {_MIN_PTS})")
    pts.sort()
    th = np.array([p[0] for p in pts])
    V = (np.array([p[1] for p in pts]) - min(p[1] for p in pts)) * HARTREE_CM
    return th, V


def read_rectilinear(mdir):
    """(Q, V_cm) from a mode-scan directory's *.log files (Q in A*amu^1/2)."""
    from gausskit.molecules import Molecules
    pts = []
    for f in sorted(glob.glob(str(Path(mdir) / "*.log"))):
        t = open(f, errors="ignore").read()
        if "Normal termination" not in t:
            continue
        mq = _Q.search(t)
        if not mq:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            e = Molecules.from_log(f).electronic_energy
        if e is None:
            continue
        pts.append((float(mq.group(1)), float(e)))
    if len(pts) < _MIN_PTS:
        raise ValueError(f"{mdir}: only {len(pts)} usable mode-scan points (need >= {_MIN_PTS})")
    pts.sort()
    Q = np.array([p[0] for p in pts])
    V = (np.array([p[1] for p in pts]) - min(p[1] for p in pts)) * HARTREE_CM
    return Q, V
