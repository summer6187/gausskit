"""VPT2 object: read Gaussian force constants (fchk) and do anharmonic things with
them -- 3rd/4th-order Taylor PES, reduced cubic/quartic, single-mode anharmonicity x,
mode-coupling strengths, full DVPT2 X matrix, Birge-Sponer turnover. Subclasses
Harmonic (the displacement/projection helpers live there).

Units (validated convention): Q in A*amu^(1/2) (KE = 1/2 Qdot^2), V in eV; the
fchk cubic/quartic (mixed normal x Cartesian x Cartesian legs) carry Hartree/Bohr^3
and Hartree/Bohr^4 factors. Output anharmonic constants in cm^-1.
"""
from __future__ import annotations
import numpy as np
from ase.units import Bohr, Hartree
from gausskit.molecules import Molecules
from gausskit.gaussian.fchk import FCHK
from gausskit.degrees_of_freedom.harmonic import Harmonic

# ---- physical constants (single source) ----
HBAR = 1.054571817e-34; EV = 1.602176634e-19; AMU = 1.66053906660e-27; ANG = 1.0e-10
C_KIN = HBAR**2/2/(EV*AMU*ANG**2)               # hbar^2/2 in eV*amu*A^2
EV2CM = EV/(2.99792458e10*6.62607015e-34)       # 1 eV in cm^-1


class VPT2(Harmonic):
    """Construct with VPT2.from_fchk(fchk_path, log_path)."""

    def _load_fchk(self, fchk_path, log_path=None):
        """Populate this instance from an fchk (+ optional log for full Molecules)."""
        fc = FCHK(fchk_path)
        mol = (Molecules.from_log(log_path) if log_path is not None
               else Molecules(numbers=fc.atomic_numbers, positions=fc.coordinates * Bohr))
        Harmonic.__init__(self, mol, fc.hessian * (Hartree/Bohr/Bohr), fc.masses)
        # Gaussian normal modes (mass-weighted, normalized; Gaussian order & sign) --
        # these MUST be the displacement/projection basis, since the cubic/quartic
        # tensors index their mode leg in this basis.
        m_w = np.repeat(fc.masses, 3) ** 0.5
        Lg = fc.vib_modes * m_w[None, :]
        self.gaussian_modes = Lg / np.linalg.norm(Lg, axis=1, keepdims=True)  # (nmode, 3N)
        self.w2_gaussian = np.einsum("ia,ab,ib->i", self.gaussian_modes,
                                     np.asarray(self.dynamical_matrix), self.gaussian_modes)
        try:                                                 # anharmonic tensors (freq=anharmonic)
            self.cubic = fc.cubic * Hartree / Bohr**3        # (nmode, 3N, 3N)
            self.quartic = fc.quartic * Hartree / Bohr**4    # (nmode, 3N, 3N) semi-diagonal
        except KeyError:                                     # harmonic-only fchk
            self.cubic = self.quartic = None
        self.fchk = fc
        self.log_path = log_path
        return self

    @property
    def is_anharmonic(self):
        return self.cubic is not None

    @classmethod
    def from_fchk(cls, fchk_path, log_path=None):
        return cls.__new__(cls)._load_fchk(fchk_path, log_path)

    # default displacement/projection basis = Gaussian's modes
    def _modes(self, modes):
        if modes is not None:
            return np.asarray(modes)
        return self.gaussian_modes

    @property
    def n_modes(self):
        return self.gaussian_modes.shape[0]

    # ---- force-derivative accessors (alternative tensor shapes) ---------------
    @property
    def third_deriv_array(self):
        """Alias of `cubic`: d3V/dQ_i dx_A dx_B, (n_modes, 3N, 3N)."""
        return self.cubic

    @property
    def fourth_deriv_array(self):
        """Quartic as a full (n_modes, n_modes, 3N, 3N) tensor, diagonal in the mode legs."""
        m, n3 = self.cubic.shape[0], self.cubic.shape[1]
        U = np.zeros((m, m, n3, n3))
        for i in range(m):
            U[i, i] = self.quartic[i]
        return U

    @property
    def vib_modes(self):
        """Raw Gaussian normal modes (n_modes, 3N) from the fchk Vib-Modes block."""
        return self.fchk.vib_modes

    @staticmethod
    def read_vib_modes_fchk(fchk_path):
        """Standalone Gaussian-normal-modes reader (kept for convenience)."""
        return FCHK(fchk_path).vib_modes

    # `project_onto_modes(atoms)` (inherited from Harmonic) maps a geometry to its
    # normal coordinates Q.

    # ---- diagonal Taylor coefficients along a single mode (analytic) ----------
    def _dxdQ(self, i):
        """Cartesian displacement per unit Q along mode i: dx_A/dQ_i = L_iA / sqrt(m_A)."""
        return self.gaussian_modes[i] / np.repeat(self.masses, 3) ** 0.5

    def a2(self, i):
        """d2V/dQ_i^2 = omega_i^2 (eV/(amu A^2))."""
        return self.w2_gaussian[i]

    def a3(self, i):
        """d3V/dQ_i^3 (eV/(A^3 amu^1.5))."""
        u = self._dxdQ(i)
        return np.einsum("ab,a,b->", self.cubic[i], u, u)

    def a4(self, i):
        """d4V/dQ_i^4 (eV/(A^4 amu^2))."""
        u = self._dxdQ(i)
        return np.einsum("ab,a,b->", self.quartic[i], u, u)

    def frequency(self, i):
        """Harmonic frequency of mode i in cm^-1 (signed: imaginary -> negative)."""
        a2 = self.a2(i)
        return np.sign(a2) * np.sqrt(2*C_KIN*abs(a2)) * EV2CM

    # ---- reduced (dimensionless) cubic/quartic and single-mode anharmonicity ----
    def reduced(self, i):
        """(omega, k3, k4) in cm^-1 for the reduced coordinate q = (a2/2C)^(1/4) (Q-Q0)."""
        a2, a3, a4 = self.a2(i), self.a3(i), self.a4(i)
        kappa = (a2 / (2*C_KIN)) ** 0.25
        w = np.sqrt(2*C_KIN*a2) * EV2CM
        return w, (a3/kappa**3)*EV2CM, (a4/kappa**4)*EV2CM

    def diagonal_x(self, i):
        """Single-mode VPT2 anharmonicity x_ii (cm^-1) from the 1D force constants
        ONLY: x = k4/16 - (5/48) k3^2/omega. (Note: this is NOT the full multi-mode
        chi_ii, which also has off-diagonal bath terms -- see x_matrix().)"""
        w, k3, k4 = self.reduced(i)
        return k4/16 - (5/48)*k3**2/w

    # ---- mode coupling -------------------------------------------------------
    def cubic_coupling(self, i, j, k):
        """d3V/dQ_i dQ_j dQ_k (eV): the cubic tensor's mode leg is i; contract its
        two Cartesian legs with modes j and k via dx/dQ. (Raw mixed third derivative,
        not the reduced dimensionless phi_ijk -- see cubic_normal() for that.)"""
        return np.einsum("ab,a,b->", self.cubic[i], self._dxdQ(j), self._dxdQ(k))

    def mode_coupling(self, i, j):
        """Cubic coupling strength phi_iij = d3V/dQ_i^2 dQ_j (a3-type, eV units).
        A natural measure of how strongly the bath mode j couples to mode i's
        anharmonicity (the term that drives chi_ii negative)."""
        return np.einsum("ab,a,b->", self.cubic[i], self._dxdQ(i), self._dxdQ(j))

    # ---- Taylor PES ----------------------------------------------------------
    def taylor_energy(self, displacement, order=4):
        """Taylor PES V (eV, rel. to the reference) for a Cartesian displacement
        (natoms,3) array or an Atoms, via the 2nd/3rd/4th-order force field."""
        if hasattr(displacement, "get_positions"):
            d = (np.asarray(displacement.get_positions()) - self.positions0).reshape(-1)
        else:
            d = np.asarray(displacement).reshape(-1)
        Q = (d * np.repeat(self.masses, 3) ** 0.5) @ self.gaussian_modes.T
        E = 0.5 * np.dot(self.w2_gaussian, Q**2)
        if order >= 3:
            E += (1/6.) * np.einsum("iab,a,b,i->", self.cubic, d, d, Q)
        if order >= 4:
            E += (1/24.) * np.einsum("iab,a,b,i->", self.quartic, d, d, Q**2)
        return E

    def scan_mode_taylor(self, i, Q_array, order=4):
        """1D Taylor PES along mode i: 1/2 a2 Q^2 + 1/6 a3 Q^3 + 1/24 a4 Q^4 (eV)."""
        Q = np.asarray(Q_array); a2, a3, a4 = self.a2(i), self.a3(i), self.a4(i)
        E = 0.5*a2*Q**2
        if order >= 3: E += a3/6*Q**3
        if order >= 4: E += a4/24*Q**4
        return E

    # ---- ASE-style evaluator (higher-order analogue of Harmonic.calculate) -----
    def _taylor_gradient(self, d, order=4):
        """Gradient dV/dx (eV/A, flat length 3N) of the Taylor PES that taylor_energy
        evaluates, for a Cartesian displacement d = x - x0 (flat 3N, A)."""
        mw = np.repeat(self.masses, 3) ** 0.5
        L = self.gaussian_modes                                   # (m, 3N)
        Q = (d * mw) @ L.T                                        # normal coords, dQ_i/dd_a = mw_a L_ia
        g = mw * (L.T @ (self.w2_gaussian * Q))                   # harmonic part
        if order >= 3:
            C = np.einsum("iab,a,b->i", self.cubic, d, d)        # sum_ab cubic_iab d_a d_b
            dC = np.einsum("iab,b->ia", self.cubic, d) + np.einsum("iab,a->ib", self.cubic, d)
            g = g + (mw * (L.T @ C) + np.einsum("i,ia->a", Q, dC)) / 6.0
        if order >= 4:
            U = np.einsum("iab,a,b->i", self.quartic, d, d)
            dU = np.einsum("iab,b->ia", self.quartic, d) + np.einsum("iab,a->ib", self.quartic, d)
            g = g + (2.0 * mw * (L.T @ (Q * U)) + np.einsum("i,ia->a", Q ** 2, dU)) / 24.0
        return g

    def calculate(self, atoms, order=4):
        """Anharmonic potential energy + forces at a displaced geometry -- the higher-order
        analogue of Harmonic.calculate. The energy is the Taylor force field referenced to
        the reference structure's electronic energy,

            V = E0 + 1/2 sum_i a2_i Q_i^2 + 1/6 <cubic,d,d,Q> + 1/24 <quartic,d,d,Q^2>,

        with Q the Gaussian normal coordinates of the Cartesian displacement d = x - x0.
        `order` caps the expansion (2 = harmonic, 3 = +cubic, 4 = +quartic). Returns
        {'energy': eV, 'forces': eV/A (flat 3N)}; needs the anharmonic fchk (cubic/quartic)."""
        if self.cubic is None:
            raise RuntimeError("VPT2.calculate needs the anharmonic force field "
                               "(load a freq=anharmonic fchk via VPT2.from_fchk).")
        assert np.allclose(atoms.numbers, self.molecules.numbers), \
            "New structure atom order does not match the force field!"
        d = (np.asarray(atoms.get_positions()) - self.positions0).reshape(-1)
        results = {
            "energy": self.E0 + self.taylor_energy(atoms, order=order),
            "forces": -self._taylor_gradient(d, order=order),
        }
        self.results.update(results)
        return results

    # ======================= full VPT2 X matrix ================================
    def _dxdQ_all(self):
        return self.gaussian_modes / np.repeat(self.masses, 3) ** 0.5     # (m, 3N)

    def _kappa(self):
        """Reduced-coordinate scale kappa_i = (a2_i/2C)^(1/4) (complex if imaginary)."""
        return (self.w2_gaussian.astype(complex) / (2*C_KIN)) ** 0.25

    def cubic_normal(self):
        """Reduced cubic force constants phi_ijk (cm^-1), full symmetric (m,m,m)."""
        u = self._dxdQ_all()
        T = np.einsum("iab,ja,kb->ijk", self.cubic, u, u)                 # d3V/dQi dQj dQk (eV)
        kap = self._kappa()
        phi = T / np.einsum("i,j,k->ijk", kap, kap, kap) * EV2CM
        perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
        return sum(phi.transpose(p) for p in perms) / 6.0                 # enforce symmetry

    def quartic_iijj(self):
        """Reduced semi-diagonal quartic phi_iijj (cm^-1), (m,m)."""
        u = self._dxdQ_all()
        U = np.einsum("iab,ja,jb->ij", self.quartic, u, u)               # d4V/dQi^2 dQj^2 (eV)
        kap = self._kappa()
        return U / (kap[:, None]**2 * kap[None, :]**2) * EV2CM

    def x_matrix_quartic(self):
        """4th-derivative (1st-order quartic) contribution to chi (cm^-1), (m,m):
        chi_ii = phi_iiii/16, chi_ij = phi_iijj/4."""
        P = self.quartic_iijj()
        X = P / 4.0
        X[np.diag_indices_from(X)] = np.diag(P) / 16.0
        return X.real

    def x_matrix_cubic(self, deperturb=False, thresh=50.0):
        """3rd-derivative (2nd-order cubic) contribution to chi (cm^-1), (m,m).
        Complex harmonic frequencies are used so the imaginary reaction-coordinate
        mode (omega^2<0) enters the sums correctly; the real part is returned.
        deperturb=True drops 2-1 Fermi-resonant terms (|2|w_i|-|w_k|| < thresh) and
        1-1-1 Fermi terms (|w_i +/- w_j +/- w_k| < thresh) -- standard DVPT2. It does
        NOT reproduce Gaussian's near-degenerate (1-1) treatment of the soft pair."""
        m = self.n_modes
        w = np.sqrt(2*C_KIN*self.w2_gaussian.astype(complex)) * EV2CM     # complex cm^-1
        aw = np.abs(w)
        phi = self.cubic_normal()
        X = np.zeros((m, m), complex)
        for i in range(m):                                               # diagonal
            for k in range(m):
                if deperturb and k != i and abs(2*aw[i] - aw[k]) < thresh:
                    continue
                X[i, i] += -1/16 * phi[i, i, k]**2 * (8*w[i]**2 - 3*w[k]**2) \
                    / (w[k]*(4*w[i]**2 - w[k]**2))
        for i in range(m):                                               # off-diagonal
            for j in range(m):
                if i == j:
                    continue
                for k in range(m):
                    X[i, j] += -1/4 * phi[i, i, k]*phi[j, j, k]/w[k]
                    if deperturb and (abs(aw[i]+aw[j]-aw[k]) < thresh
                                      or abs(aw[i]-aw[j]+aw[k]) < thresh
                                      or abs(aw[i]-aw[j]-aw[k]) < thresh):
                        continue
                    D = ((w[i]+w[j]+w[k])*(-w[i]+w[j]+w[k]) *
                         (w[i]-w[j]+w[k])*(w[i]+w[j]-w[k]))
                    X[i, j] += -1/2 * phi[i, j, k]**2 * w[k]*(w[i]**2+w[j]**2-w[k]**2)/D
        return X.real

    def x_matrix_coriolis(self):
        """Coriolis contribution to chi (cm^-1), (m,m); off-diagonal only:
        chi_ij = sum_a B_a (zeta_ij^a)^2 (w_i/w_j + w_j/w_i). Matches Gaussian exactly."""
        mss = self.masses
        R = self.fchk.coordinates * Bohr
        R = R - (mss[:, None]*R).sum(0)/mss.sum()                 # to COM
        It = sum(mss[A]*(np.dot(R[A], R[A])*np.eye(3) - np.outer(R[A], R[A]))
                 for A in range(len(mss)))
        Iv, Uax = np.linalg.eigh(It)
        B = 16.857629 / Iv                                       # rotational constants, cm^-1
        L = (self.gaussian_modes.reshape(self.n_modes, len(mss), 3)) @ Uax
        eps = np.zeros((3, 3, 3))
        for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            eps[a, b, c] = 1.0; eps[a, c, b] = -1.0
        zeta = np.einsum("iAa,jAb,abk->ijk", L, L, eps)          # (m,m,3)
        w = np.sqrt(2*C_KIN*self.w2_gaussian.astype(complex)) * EV2CM
        X = np.zeros((self.n_modes, self.n_modes), complex)
        for i in range(self.n_modes):
            for j in range(self.n_modes):
                if i != j:
                    X[i, j] = sum(B[a]*zeta[i, j, a]**2*(w[i]/w[j]+w[j]/w[i]) for a in range(3))
        return X.real

    def x_matrix(self, deperturb=False):
        """Computed VPT2 X matrix (cm^-1) = quartic + cubic + Coriolis. Matches Gaussian
        for all *reliable* force constants; entries that Gaussian flags Unreliable/NULL
        (grid-sensitive soft / reaction-coordinate modes) are its numerical artifacts and
        are not reconstructible from the fchk -- use x_matrix_gaussian() for those."""
        return (self.x_matrix_quartic() + self.x_matrix_cubic(deperturb=deperturb)
                + self.x_matrix_coriolis())

    # ---- authoritative Gaussian values (curated; NULL/unreliable handled) -----
    @staticmethod
    def _parse_x_block(log_path, title, n):
        lines = open(log_path).read().splitlines()
        s = next(k for k, l in enumerate(lines) if title in l)
        M = np.zeros((n, n)); k = s+1; cols = None
        while k < len(lines):
            l = lines[k]; t = l.split()
            if not t or set(l.strip()) <= set('-'):
                k += 1; continue
            if '.' not in l and 'D' not in l and all(x.lstrip('-').isdigit() for x in t):
                cols = [int(x)-1 for x in t]; k += 1; continue
            if cols is not None and t[0].isdigit():
                r = int(t[0])-1
                if r >= n:
                    break
                for jj, val in enumerate(t[1:]):
                    M[r, cols[jj]] = float(val.replace('D', 'E'))
                k += 1
            else:
                break
        M = M + np.tril(M, -1).T
        return M

    def x_matrix_gaussian(self, block="Total Anharmonic X Matrix"):
        """Gaussian's printed X-matrix block (cm^-1), returned in this object's mode
        order (ascending; index 0 = imaginary mode). Requires the log."""
        if self.log_path is None:
            raise ValueError("x_matrix_gaussian needs the Gaussian log (pass log_path).")
        G = self._parse_x_block(self.log_path, block, self.n_modes)   # Gaussian descending order
        perm = np.arange(self.n_modes)[::-1]
        return G[np.ix_(perm, perm)]

    def unreliable_force_constants(self):
        """List of (kind, indices) force constants Gaussian flagged 'Unreliable'
        (1-based Gaussian mode indices, descending order)."""
        out = []
        if self.log_path is None:
            return out
        for l in open(self.log_path):
            if "Unreliable" in l and "force constant" in l:
                kind = "cubic" if "CUBIC" in l else "quartic"
                idx = [int(p.split("=")[1]) for p in l.split(",") if "=" in p]
                out.append((kind, tuple(idx)))
        return out

    # ---- Birge-Sponer turnover of the single-mode VPT2 ladder -----------------
    def birge_sponer(self, i):
        w, _, _ = self.reduced(i); x = self.diagonal_x(i)
        out = {"omega_cm": w, "x_cm": x}
        if x < 0:
            out["v_turnover"] = -w/(2*x) - 0.5
            out["n_bound"] = int(np.floor(out["v_turnover"])) + 1
            out["dissociation_cm"] = -w**2/(4*x)
        else:
            out["v_turnover"] = np.inf; out["n_bound"] = np.inf; out["dissociation_cm"] = np.inf
        return out
