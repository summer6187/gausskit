"""Tests for the electronic-states (spin-orbit) lookup table.

Self-contained; runs under pytest or directly as `python test_electronic_states.py`.
"""

from __future__ import annotations

from ase.build import molecule

from gausskit.electronic_states import electronic_states_for, electronic_comments_for

OH_STATES = [(0.0, 2), (139.7, 2), (32403.0, 2), (68372.0, 2), (89420.0, 2)]


def test_oh_electronic_ladder():
    # OH X2Pi spin-orbit doublet (0 / 139.7) + A/B/C 2Sigma+ (32403/68372/89420), all g=2
    assert electronic_states_for(molecule("OH")) == OH_STATES


def test_formula_string_key():
    # keyed by ASE chemical formula; OH -> "HO"
    assert molecule("OH").get_chemical_formula() == "HO"
    assert electronic_states_for("HO") == OH_STATES


def test_oh_reference_comments():
    comments = electronic_comments_for("HO")
    assert comments is not None and len(comments) == 3
    assert comments[0].startswith("a) NIST")
    # no enthalpy provenance: our energies are computed (CCSD(T)//BHLYP), not experimental
    assert not any("Ruscic" in c for c in comments)


def test_untabulated_returns_none():
    assert electronic_states_for(molecule("H2O")) is None
    assert electronic_states_for("CO2") is None
    assert electronic_comments_for("CO2") is None


def test_lookup_returns_a_copy():
    # mutating the returned list must not corrupt the table
    got = electronic_states_for("HO")
    got.append((99999.0, 2))
    assert electronic_states_for("HO") == OH_STATES


if __name__ == "__main__":
    test_oh_electronic_ladder()
    test_formula_string_key()
    test_oh_reference_comments()
    test_untabulated_returns_none()
    test_lookup_returns_a_copy()
    print("ALL PASS")
