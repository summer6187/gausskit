"""Auto-detect the fragments of a dividing surface from geometry alone.

Ported from ``mode_aware_toolkit/fragments.py``. At an entrance/dividing surface the
reaction bond is stretched well beyond a covalent length, so a covalent-radius bond
graph splits the complex into its fragments as connected components. A tight TS (e.g.
Na...Cl within 1.3x the covalent-sum) stays a single connected component — the caller
treats that as "no free fragment rotation" (all soft modes are rectilinear wells).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# covalent radii (Angstrom), Cordero 2008 (subset; extend as needed)
RCOV = {"H": 0.31, "He": 0.28, "Li": 1.28, "B": 0.84, "C": 0.76, "N": 0.71,
        "O": 0.66, "F": 0.57, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,
        "P": 1.07, "S": 1.05, "Cl": 1.02, "K": 2.03, "Br": 1.20, "I": 1.39}
_DEFAULT_RCOV = 0.77


def detect_fragments(symbols, xyz, tol=1.3):
    """Return 0-based atom-index lists (one per connected component), largest first.

    Two atoms are bonded if their distance < ``tol * (rcov_i + rcov_j)``.
    """
    xyz = np.asarray(xyz, float)
    n = len(symbols)
    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    A = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            cut = tol * (RCOV.get(symbols[i], _DEFAULT_RCOV) + RCOV.get(symbols[j], _DEFAULT_RCOV))
            if D[i, j] < cut:
                A[i, j] = A[j, i] = 1
    _, labels = connected_components(csr_matrix(A), directed=False)
    frags = [list(np.where(labels == k)[0]) for k in range(labels.max() + 1)]
    return sorted(frags, key=lambda f: -len(f))          # largest first


def reaction_bond(symbols, xyz, frags):
    """The closest inter-fragment atom pair (the forming/breaking contact), or None
    unless exactly 2 fragments. Returns (distance_A, atom_i, atom_j)."""
    if len(frags) != 2:
        return None
    xyz = np.asarray(xyz, float)
    best = None
    for i in frags[0]:
        for j in frags[1]:
            d = float(np.linalg.norm(xyz[i] - xyz[j]))
            if best is None or d < best[0]:
                best = (d, int(i), int(j))
    return best
