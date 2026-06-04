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
make clean
```

## Result

The generated deck runs in ktools and gives a **barrierless** reverse-unified capture
rate of ≈ 0.4–2.0 × 10⁻⁹ cm³ molecule⁻¹ s⁻¹ over 100–2000 K, with the variational TS
moving from ~5.4 Å at 100 K inward to ~1.9 Å at high T — the correct qualitative picture.

The external rotational symmetry number is computed **from each geometry** by
`gausskit.symmetry` (σ=2 on the outer C₂ᵥ surfaces, where OH sits on the C=O axis; σ=1
on the inner Cs ones), so the deck is correct under `nosymm` with no post-patch. One
electronic term is still outside gausskit's native output: the OH ²Π₁/₂ spin-orbit state
(139.7 cm⁻¹), which lowers the low-T rate a further ~1.1×. With it added (the σ treatment
being identical), the full campaign reproduces Ali–Barker to geomean ~1.5×, grain-
independent. That spin-orbit-corrected comparison, the Gaussian logs, the reference
ktools deck, and the k(T)/V(s) figures live in the project's
`multiwell_ktools_example_ch2o_oh/` campaign directory.
