# KTOOLS example — CH₂O + OH → pre-reactive complex (entrance capture)

Demonstrates gausskit's `[Ktools]` workflow: a microcanonical **variational TST**
(canonical VTST / unified statistical) treatment of the **barrierless** entrance
channel for OH addition to formaldehyde,

    CH₂O + OH  →  [CH₂O···HO]  (pre-reactive complex, "PRC")

reproducing the workflow of **Ali & Barker, *J. Phys. Chem. A* 2015, 119, 7578**.

## What it shows

`gausskit run PES.in` reads `database.json` + `PES.in` and writes a directly-runnable
`ktools.dat`. This exercises the converter pieces that were the bug-prone part of the
ktools path:

- the **9-line ktools control header** (`title / KCAL MCC / whatdo / Emax Egrain /
  Jmax Jgrain / Imax1 Isize Emax2 / Nt / temperatures / "Nreac Ntts Nprod"`),
- one **reaction-coordinate mode removed** at each trial transition state (3N−6 → 3N−7),
- **product fragment energies summed** to the true asymptote D₀ (ktools sums the
  per-fragment block energies, so each of `CH₂O` + `OH` carries D₀/2, not D₀).

## The reaction path

31 trial dividing surfaces `rp_25 … rp_55`, labelled by the
O(carbonyl)···H(hydroxyl) distance from **2.5 to 5.5 Å**, from a constrained
BHandHLYP/aug-cc-pVTZ scan. Along the path the complex bifurcates: bent **Cs** inside
~3.6 Å, and **C₂ᵥ** (OH on the C=O axis) from ~3.7 Å outward. The pre-reactive complex
is the **bent (108°) global-minimum conformer**, D₀ = 5.617 kcal/mol (Ali–Barker: 5.61).

## Level of theory

CCSD(T)/aug-cc-pVTZ // BHandHLYP/aug-cc-pVTZ (matches Ali & Barker). The composite
energy is **not** parsed automatically: gausskit/ASE reads only the SCF line from a
`CCSD(T)//BHandHLYP` log, so the CCSD(T) energies were injected as a separate `CCSDT`
method into the shipped `database.json`. A plain `gausskit output` will **not**
reproduce the `CCSDT` entries — the prebuilt `database.json` is authoritative here.

## Files

| file            | role                                                              |
|-----------------|-------------------------------------------------------------------|
| `PES.in`        | INI config: `[Method]` (CCSDT energies, BHandHLYP ZPE) + `[Ktools]` + `[PES_1]` topology |
| `database.json` | 34 species: `prc`, `ch2o`, `oh`, and 31 reaction-path surfaces `rp_25…rp_55`, CCSDT injected |
| `Makefile`      | `make` (dry, input-gen), `make solver` (also runs the ktools binary), `make clean` |

### `[Ktools]` fields

- `reac: prc` / `prod: ch2o + oh` — ktools' *forward* direction is the unimolecular
  dissociation PRC→CH₂O+OH; the *reverse* unified rate is the bimolecular capture.
- `rc_modes` — index (0-based, ascending-frequency order) of the **reaction-coordinate
  mode** removed at each trial TS. Here `0` at every surface: each rp surface is a
  first-order saddle along the dissociation coordinate, so the single imaginary mode
  (the lowest/most-negative after sorting) is the one dropped. If omitted, gausskit
  defaults to dropping mode 0 at every surface.
- `rc_distances` — the reaction-coordinate distance (Å) per surface (the
  O(carbonyl)···H(hydroxyl) length; the leading entry is the PRC, the trailing `10.0`
  the product asymptote). Used to label the reported coordinate and for the centrifugal
  fit; it does **not** enter the per-surface microcanonical rate.

> `rc_modes` / `rc_distances` may also be written in long form as
> `reaction_coordinate_modes` / `reaction_coordinate_distances`.

## Run

```sh
make            # gausskit run PES.in --verbose --dry  -> ktools_PES_1/ktools.dat
make solver     # also runs: ktools ktools.dat         (needs the MultiWell ktools binary)
make plot       # solver + compare against the MultiWell reference -> compare_kt.png
make clean
```

## Comparison vs the MultiWell reference

`ch2o+oh/` is MultiWell's own bundled `ch2o+oh` ktools example — the deck
(`ch2o-oh.dat`) and expected output (`ch2o-oh.canonical.test`), copied verbatim from the
MultiWell distribution (© Barker et al.; see `ch2o+oh/PROVENANCE.txt`). Both use the *same* 11
temperatures and 1-reactant / 31-TS / 2-product topology as this example; they differ
only in the molecular data (ours = CCSD(T)//BHandHLYP, the reference = Ali & Barker's
published values), so agreement is expected to within a small factor, not exactly.

`make plot` runs the solver and `plot_compare.py`, writing **`compare_kt.png`** — the
forward (dissociation) and reverse (capture) unified-canonical k(T) for both decks plus
ours/reference ratio panels. This example lands at **geomean ≈ 1.5× the reference for the
capture rate** (≈ 1.2× for dissociation); the residual is the CCSD(T)//BHLYP-vs-published
energy difference. (Dropping the `(spin-orbit)` tag below pushes the capture ratio back up
to ~2.1× — the missing OH spin-orbit state, a clean illustration that it biases only the
capture direction.)

## Result

The generated deck runs in ktools and gives a **barrierless** reverse-unified capture
rate of ≈ 0.3–1.1 × 10⁻⁹ cm³ molecule⁻¹ s⁻¹ over 100–2000 K, with the variational TS
moving from ~5.4 Å at 100 K inward to ~1.9 Å at high T — the correct qualitative picture.

Two corrections are now in gausskit's **native** output, so the deck needs no post-patch:

- **External rotational symmetry number** — computed from each geometry by
  `gausskit.symmetry` (σ=2 on the outer C₂ᵥ surfaces, where OH sits on the C=O axis; σ=1
  on the inner Cs ones), correct even under `nosymm`.
- **OH ²Π₁/₂ spin-orbit state** (139.7 cm⁻¹, g=2) — `prod: ch2o + oh(spin-orbit)` pulls
  OH's electronic ladder from `gausskit.electronic_states` (a small spectroscopic table;
  these states are *not* in a Gaussian log, which carries only the spin multiplicity).
  Use the bare name `oh` to write the ground state only. The state lowers the low-T
  capture by ~1.1× and the high-T by ~1.9× — bringing the capture ratio from ~2.1× down
  to ~1.5× of Ali–Barker (it does not touch the dissociation rate).

The remaining ~1.5× is the irreducible CCSD(T)//BHLYP-vs-published energy difference. The
Gaussian logs, the reference ktools deck, and the k(T)/V(s) figures live in the project's
`multiwell_ktools_example_ch2o_oh/` campaign directory.
