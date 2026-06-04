"""Low-lying electronic states (spin-orbit etc.) for the electronic partition function.

These states are NOT available from a standard Gaussian output -- a Gaussian log gives
only the spin multiplicity (2S+1), i.e. the ground level's degeneracy. For open-shell
radicals with a low-lying spin-orbit (or other electronic) state, that ground-only
treatment undercounts q_elec and biases the rate. The values here are spectroscopic
(NIST Atomic Spectra Database / NIST-JANAF / Huber & Herzberg), not computed.

Each species record has:
  - ``states``:   the ``(energy_cm-1, degeneracy)`` ladder, INCLUDING the ground level at
                  0.0. Only levels populated in the usual kinetics range matter, but the
                  full known ladder is kept for fidelity (high-lying states are spectators).
  - ``comments``: up to three provenance/reference lines, written verbatim as the
                  species' comment lines in the MultiWell/ktools deck.
Keyed by the ASE chemical formula (``mol.get_chemical_formula()`` -- the same key the
deck writer uses as the species name).

Trigger: reference a species as ``name(spin-orbit)`` in a PES.in topology line (e.g.
``prod: ch2o + oh(spin-orbit)``); the PES parser strips the tag, resolves ``name`` in the
database, and attaches the looked-up states + comments (see potential_energy_surface.py).
"""

from __future__ import annotations

ELECTRONIC_DATA = {
    # OH X^2Pi: a REGULAR multiplet, so ^2Pi_3/2 is the ground spin-orbit component and
    # ^2Pi_1/2 lies 139.7 cm^-1 above (each g=2). The A/B/C ^2Sigma+ states (>32000 cm^-1)
    # are spectators below ~3000 K but are listed for completeness.
    #   q_elec = 2 + 2*exp(-139.7/kT) (+ negligible) -> ~4 at high T, 2 at low T.
    "HO": {
        "states": [
            (0.0, 2),       # X^2Pi_3/2  ground spin-orbit component
            (139.7, 2),     # X^2Pi_1/2  upper spin-orbit component
            (32403.0, 2),   # A^2Sigma+
            (68372.0, 2),   # B^2Sigma+
            (89420.0, 2),   # C^2Sigma+
        ],
        # deck comment lines for this species (provenance). No enthalpy source is listed:
        # energies are a per-deck input (PES.in [Method]), not supplied by this table.
        "comments": [
            "a) NIST / JANAF 1998",
            "b) (blank comment line)",
            "c) (blank comment line)",
        ],
    },
}


def _key(mol_or_formula):
    if hasattr(mol_or_formula, "get_chemical_formula"):
        return mol_or_formula.get_chemical_formula()
    return str(mol_or_formula)


def electronic_states_for(mol_or_formula):
    """Return the ``[(E_cm-1, g), ...]`` electronic ladder, or ``None`` if untabulated.

    Args:
        mol_or_formula: an ``ase.Atoms`` (its ``get_chemical_formula()`` is the key) or a
            chemical-formula string.

    Returns:
        list[tuple[float, int]] | None  (a copy; safe to mutate)
    """
    rec = ELECTRONIC_DATA.get(_key(mol_or_formula))
    return list(rec["states"]) if rec is not None else None


def electronic_comments_for(mol_or_formula):
    """Return the provenance/reference comment lines for a species, or ``None``.

    Returns:
        list[str] | None  (a copy; safe to mutate)
    """
    rec = ELECTRONIC_DATA.get(_key(mol_or_formula))
    if rec is None or "comments" not in rec:
        return None
    return list(rec["comments"])
