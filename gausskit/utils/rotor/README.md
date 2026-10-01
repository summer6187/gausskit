# `gausskit utils rotor` — determine per-soft-mode rotor information

One utility that unifies the previously scattered rotor machinery (the na-cfxcly
`hindered_rotor_kit`, the ch2o+OH `mode_aware_toolkit`, and the standalone
`gausskit.multiwell.dos` primitives) behind a single **hand-editable YAML roadmap**.

## Two rotor models, chosen per mode

A soft mode is treated as one of:

| `method` | coordinate | levels | fits |
|---|---|---|---|
| `mode-scan` | rigid rectilinear eigenvector `x(Q)=x0+(L/√m)Q` | bounded-well `sinc_dvr` | stiffening tight-TS wells (na-cfxcly) |
| `rigid-rotor` | curvilinear fragment rotation about its COM | periodic-rotor DVR, physical `B=κ/I` | librations a rectilinear scan over-stiffens (ch2o+OH) |
| `harmonic` | — (not scanned) | — | stiff modes (freq ≥ `--soft-cut`) |

The eigenvector always *seeds* the rigid-rotor (`rot_decomp` on the mode yields the rotation
axis + moment `I`); the difference is only whether we then scan the straight line or the arc.

## Workflow — the roadmap is the pivot

```bash
# 1. classify + write the hand-editable roadmap  (--auto | --mode-scan | --rigid-rotor)
gausskit utils rotor detect  TS.log  -o rotor_plan.yaml  --route "#p ... nosymm"
#    ... review & edit rotor_plan.yaml: flip a mode's `method`, fix a `fragments` atom
#        list, tweak an `axis`/`I_amuA2`/`grid` ...
# 2. roadmap -> Gaussian scan inputs (NO re-classification)
gausskit utils rotor scan    rotor_plan.yaml  -o scans/
#    ... submit scans/mode*/*.gjf on any cluster, pull the *.log back ...
# 3. roadmap + logs -> levels, Vhrd2/HRD deck blocks, q(HRD/DVR) validation
gausskit utils rotor fit     rotor_plan.yaml  scans/  -o rotor_result.json
# 4. declare the fitted HRDs in a PES.in -> `gausskit run` emits the decks natively
gausskit utils rotor patch   rotor_result.json --input PES_harm.in -o PES_hrd.in
gausskit run PES_hrd.in --dry
```

## `rotor patch` — native HRD in PES.in (retires the deck patchers)

`patch` appends an `[HRD]` section to a PES.in referencing the fit artifact:

```ini
[HRD]
CF2Cl2_Na_TS_for_NaCl_bond: ../03_hrd_fit/rotor_result.json
```

`gausskit run` then emits the `hrd/Vhrd2/Bhrd1` blocks natively — the THERMO/DENSUM DOF writer
swaps the paired soft `vib` lines (matched to the fitted modes **by frequency**), and the SCTST
`write_parsctst` drops those modes from the coupled X-matrix and re-adds them as separable Vhrd2
rotors. No `patch_thermo_hrd.py` / `build_sctst_hrd.py` post-processing.

Options: `--species NAME` (else auto: the result's log stem or the unique `*TS*` species);
`--set-dir DIR` (rewrite `[Thermo] dir:` so a variant writes to its own output dir); `--set-anharm
False|True|ts`. The `[Thermo] anharm` tri-state:

- **`True`** — full VPT2 on **every** species (reactant/products via `paradensum`, TS via
  `parsctst`) plus Vhrd2 on the TS soft modes. The physically-complete anharmonic rate; Vhrd2 does
  not conflict with VPT2, it replaces exactly the TS soft modes VPT2 can't represent.
- **`ts`** — SCTST/crp for the TS only, native harmonic reactant/wells — the "isolate the TS
  treatment" comparison (fast; also fixes external-symmetry mismatches at the source). Note
  `[Method] Anharm: True` already gives every species its anharmonic ZPE, so `ts` vs `True` differ
  only in the reactant/product *vibrational partition function* — negligible for stiff reactants
  (verified ≡ to 0.1% on the Na+CFxCly kit) but worth `True` when a reactant has floppy modes.
- **`False`** — harmonic (CTST).

**KTOOLS is excluded:** native ktools segfaults on Vhrd2 (`uhrlev ev(2000)`), so `[HRD]` on a
`[Ktools]`-run PES prints a warning and is skipped in the ktools deck — use `gausskit vtst --dens`
for HRD-corrected capture rates.

The whole `gausskit run` chain prints a transparency line per step (`write`/`exec`/`skip`/`HRD`),
default-on, since it is all input preparation for the MultiWell solvers.

`detect` computes and writes *every driving number* (fragments, `axis`, `I`, `rot_frac`,
`method`), so nothing that shapes a gjf is hidden. `scan` only materialises geometry. `fit`
reads the *same* roadmap and dispatches on `method`, writing results to a **separate** file so
a re-fit never clobbers hand-edits.

`--auto` routes each soft mode by the gate `2 fragments ∧ rot_frac ≥ --rot-min → rigid-rotor,
else mode-scan`; `--rigid-rotor` / `--mode-scan` force all soft modes (stiff stays harmonic;
rigid-rotor needs 2 fragments). The classifier is the single source of truth — it collapses
the "lowest-N-real" decision that the kit rediscovered in four separate places.

## The roadmap — `rotor_plan.yaml`

```yaml
log: rp_55.log
route: "#p UBHandHLYP/aug-cc-pVTZ int=(grid=superfine,acc2e=12) scf=(xqc,tight) nosymm"
charge_mult: 0 2
fragments:
  A: [0, 1, 2, 3]                 # CH2O
  B: [4, 5]                       # OH
modes:
  - mode: 1
    gauss_idx: 1                  # anchor into the freq log — do NOT renumber
    freq_cm: 10.45
    rot_frac: 1.0
    method: rigid-rotor
    fragment: A
    I_amuA2: 14.389               # -> B = 16.857629 / I
    axis: [-0.0, 1.0, 0.0]
    grid: {theta_deg: [-90, 90], dtheta: 10}
  - mode: 5
    gauss_idx: 5
    freq_cm: 1260.76
    method: harmonic              # stiff — no scan
```

YAML is read with PyYAML when available, else a small built-in parser that handles the emitted
subset + common hand-edits (so the tool has no hard YAML dependency and stays locally runnable).

## Validation (regression against both source kits)

- **rigid-rotor path (ch2o+OH POC):** `detect` reproduces the POC rotor spec (I/B/axis) exactly;
  `fit` on the POC rotation logs feeds a mode-aware DOS of **ZPE 8111, SOS(5000) 1.714e7,
  q(300) 2187** vs the POC anchor 8111 / 1.712e7 / 2186.
- **mode-scan path (na-cfxcly CF2Cl2):** `detect` routes the 26.8/30.3 cm⁻¹ tight-TS modes to
  `mode-scan` (1-fragment fallback, not harmonic); `fit` reproduces `hrd_params.json`
  (`B_cm` to 1e-7, leading `CV_Vhrd2` identical, `q(HRD/DVR)@700` exact) and the MultiWell
  `Vhrd2`/`Bhrd1` block via the shared `dos.hrd_block`.

## Scope & caveats

- **Input is a freq `.log`** (raw Cartesian modes for the classifier). Those displacements are
  the log's 2-decimal blocks; adequate for scan-direction generation (as validated), but a
  future option could read full-precision modes from the `.fchk` for the mode-scan geometry.
- **rot_frac borderline** near `--rot-min` flips curvilinear↔rectilinear — `detect` prints it,
  the roadmap stores it; flip `method` by hand if a mode straddles.
- **Coordinate choice is physical:** a rectilinear scan of a soft rotation over-stiffens; a
  rigid-rotor of a tight-TS internal bend is undefined (no fragment to rotate). Hence the gate.
- **Stops at rotor information** — levels, HRD/Vhrd2 blocks, validation. It does not patch decks
  or build a `.dens`/rate (the existing `hindered_rotor_kit` patchers can consume `hrd_block`).
- **Deferred:** internal torsions (dihedral + Pitzer moment), mixed/coupled-geared rotors, and
  the rate endpoint (J-resolved flux + the KTOOLS `uhrlev` overflow).

## Modules

`fragments.py` (bond-graph fragmentation) · `character.py` (`rot_decomp` + `classify_modes`) ·
`plan.py` (build/dump/load the YAML roadmap) · `scan.py` (roadmap → gjf) · `readback.py`
(V(θ)/V(Q) from logs) · `levels.py` (periodic-rotor DVR + physical HRD) · `fit.py` (levels +
HRD + validation) · `figures.py` (`fit_validation.png/.csv`). Imports the DOS primitives from
`gausskit.multiwell.dos` (never copies them) and reuses `gausskit.utils.mode_scan`.
