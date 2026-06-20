"""Unit tests for the SCTST/paradensum bound-state gating helpers.

These check the NumPy ports against the MultiWell-2023.1 nvmax.f / ckderiv.f logic. Runs under
pytest, or directly as `python test_sctst_gating.py`.
"""
import numpy as np

from gausskit.multiwell.dos import (
    nvmax_turnover,
    ckderiv_mask,
    bath_coupling_sum,
    boltzmann_bath,
    vpt2_thermal_levels,
)


def test_nvmax_turnover():
    # negative anharmonicity -> Birge-Sponer turnover vd = -w/(2x) - 1/2, then INT()
    assert nvmax_turnover(27.0, -4.0) == int(-27.0 / (2 * -4.0) - 0.5)   # = int(2.875) = 2
    assert nvmax_turnover(30.0, 5.0) == 10                               # x>0: njmax ceiling
    assert nvmax_turnover(30.0, 0.0) == 10                               # harmonic: njmax ceiling
    assert nvmax_turnover(-100.0, -4.0) == -1                            # unbound (w<=0)
    assert nvmax_turnover(27.0, -4.0, njmax=5) == 2                      # njmax irrelevant for x<0
    assert nvmax_turnover(30.0, 5.0, njmax=3) == 3                       # njmax honoured for x>0


def test_ckderiv_mask_matches_formula():
    w = np.array([30.0, 50.0])         # mode 0 soft (turnover ~ v=2 for x=-5)
    X = np.array([[-5.0, -0.5], [-0.5, -2.0]])
    U = np.array([[0.5, 0.5],          # ground state
                  [6.5, 0.5]])          # mode-0 excited past its turnover -> derivative goes negative
    mask = ckderiv_mask(U, w, X)
    deriv = w + np.diag(X) * U + U @ X
    assert np.array_equal(mask, np.all(deriv > 0.0, axis=1))
    assert mask[0]                     # ground state is bound
    assert not mask[1]                 # over-excited soft mode is rejected (fold-over)


def test_bath_coupling_sum():
    X = np.array([[0.0, -3.0, 2.0], [-3.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    occ = np.array([[1, 2]])           # bath modes 1, 2 occupations
    S = bath_coupling_sum(0, X, occ, np.array([1, 2]))
    assert np.isclose(S[0], (1 + 0.5) * (-3.0) + (2 + 0.5) * 2.0)


def test_boltzmann_bath_shape_and_caps():
    rng = np.random.default_rng(0)
    w = np.array([30.0, 200.0, -50.0])                 # mode 2 is unbound (w<0)
    X = np.array([[-5.0, 0.1, 0.0], [0.1, -1.0, 0.0], [0.0, 0.0, -2.0]])
    occ = boltzmann_bath(w, X, np.array([0, 1, 2]), 700.0, rng, 2000)
    assert occ.shape == (2000, 3)
    assert occ.min() >= 0
    assert occ[:, 0].max() <= nvmax_turnover(30.0, -5.0)   # soft mode capped at its turnover
    assert occ[:, 2].max() == 0                            # unbound mode -> only n=0 sampled


def test_vpt2_thermal_levels_matches_inline():
    w = np.array([27.0, 30.0, 150.0, 400.0])
    X = np.array([[-4.0, -6.0, 0.2, 0.0],
                  [-6.0, -5.0, 0.1, 0.0],
                  [0.2, 0.1, -1.0, 0.0],
                  [0.0, 0.0, 0.0, -2.0]])
    i, T, nimax, M = 0, 700.0, 6, 5000
    surv = vpt2_thermal_levels(w, X, i, T, np.random.default_rng(3), nimax, M)
    # reproduce inline from the primitives with the same rng seed/order
    rng = np.random.default_rng(3)
    nr, wi, xii = len(w), w[i], X[i, i]
    bath_idx = np.array([j for j in range(nr) if j != i])
    occ = boltzmann_bath(w, X, bath_idx, T, rng, M)
    S = bath_coupling_sum(i, X, occ, bath_idx)
    U = np.full((M, nr), 0.5)
    U[:, bath_idx] = occ + 0.5
    for ni in range(nimax + 1):
        U[:, i] = ni + 0.5
        dE = ni * (wi + xii * (ni + 1.0)) + ni * S
        assert np.array_equal(surv[ni], dE[ckderiv_mask(U, w, X) & (dE >= 0.0)])
    # the soft mode (x_ii<0) folds over -> surviving fraction falls off with ni
    fracs = [s.size / M for s in surv]
    assert fracs[1] >= fracs[nimax]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
