"""`gausskit utils rotor patch` — declare scan-fitted HRDs in a PES.in.

Textually edits a PES.in (preserving everything else byte-for-byte): drops any existing
``[HRD]`` section and appends a fresh one mapping the species to the fit artifact::

    [HRD]
    CF2Cl2_Na_TS_for_NaCl_bond: ../03_hrd_fit/rotor_result.json

`gausskit run` then emits the hrd/Vhrd2/Bhrd1 blocks natively (THERMO/DENSUM decks) and
separable-HRD parsctst decks (SCTST) — no post-hoc deck patching.

Optional rewrites for the file-per-model convention:
  * ``set_dir``    -> rewrite the ``[Thermo] dir:`` line (variant gets its own output dir)
  * ``set_anharm`` -> rewrite the ``[Method] Anharm:`` + ``[Thermo] anharm:`` lines
                      (e.g. ``ts`` = SCTST for the TS, harmonic reactant)

Species resolution order: explicit ``species`` > the result's freq-log stem starts with a
PES species name > the unique PES species containing "TS" > error listing candidates.
"""
import os
import re
from pathlib import Path

from gausskit.utils.rotor.result import load_fitted_modes, result_log


def _pes_species(text):
    """All species names appearing in the [PES_*] sections of a PES.in text."""
    species, section = [], None
    for line in text.splitlines():
        s = line.strip()
        m = re.fullmatch(r"\[(.+)\]", s)
        if m:
            section = m.group(1)
            continue
        if section and section != "Method" and "PES" in section and ":" in s:
            _, _, val = s.partition(":")
            for tok in val.split():
                if tok not in ("+", "-"):
                    species.append(re.sub(r"\(.*\)$", "", tok))   # strip e.g. (spin-orbit)
    return list(dict.fromkeys(species))                            # unique, order kept


def _resolve_species(result_path, text, species=None):
    cands = _pes_species(text)
    if species:
        if species not in cands:
            raise SystemExit(f"--species {species!r} not among PES species {cands}")
        return species
    log = result_log(result_path)
    if log:
        stem = Path(log).stem
        pref = [c for c in cands if stem.startswith(c)]
        if len(pref) == 1:
            return pref[0]
    ts = [c for c in cands if "TS" in c.upper()]
    if len(ts) == 1:
        return ts[0]
    raise SystemExit(f"cannot resolve the HRD species automatically (candidates: {cands}); "
                     "pass --species NAME")


def _drop_section(text, name):
    """Remove a whole [name] section from INI text (if present)."""
    lines, out, skipping = text.splitlines(), [], False
    for line in lines:
        s = line.strip()
        if re.fullmatch(r"\[(.+)\]", s):
            skipping = s == f"[{name}]"
        if not skipping:
            out.append(line)
    return "\n".join(out).rstrip("\n") + "\n"


def _set_key(text, section, key, value):
    """Rewrite `key: value` inside [section] (case-insensitive key match); error if absent."""
    lines, insec, done = text.splitlines(), False, False
    for i, line in enumerate(lines):
        s = line.strip()
        m = re.fullmatch(r"\[(.+)\]", s)
        if m:
            insec = m.group(1).lower() == section.lower()
            continue
        if insec and re.match(rf"\s*{key}\s*[:=]", s, re.IGNORECASE):
            lines[i] = f"{key}: {value}"
            done = True
            break
    if not done:
        raise SystemExit(f"no '{key}:' line found in [{section}] to rewrite")
    return "\n".join(lines).rstrip("\n") + "\n"


def patch_pes(result_path, pes_in, out, species=None, set_dir=None, set_anharm=None):
    """Write `out` = `pes_in` + [HRD] declaration (+ optional dir/anharm rewrites).
    Returns (species, n_modes)."""
    result_path, pes_in, out = Path(result_path), Path(pes_in), Path(out)
    modes = load_fitted_modes(result_path)                 # validates the artifact
    text = pes_in.read_text()
    sp = _resolve_species(result_path, text, species)
    text = _drop_section(text, "HRD")
    if set_dir is not None:
        text = _set_key(text, "Thermo", "dir", set_dir)
    if set_anharm is not None:
        # [Method] Anharm=True SELECTS the anharmonic (VPT2) ZPE / X-matrix data as the SCTST
        # input; [Thermo] anharm carries the raw value ('ts' = SCTST for the TS only, harmonic
        # elsewhere), so the two keys are set together.
        text = _set_key(text, "Method", "Anharm", "True" if set_anharm != "False" else "False")
        text = _set_key(text, "Thermo", "anharm", set_anharm)
    rel = os.path.relpath(result_path, out.parent)
    text += f"\n[HRD]\n{sp}: {rel}\n"
    out.write_text(text)
    freqs = ", ".join(f"{m['freq_cm']:.1f}" if m["freq_cm"] is not None else "?"
                      for m in modes)
    print(f"  write  {out}  ([HRD] {sp}: {rel} — {len(modes)} modes @ {freqs} cm-1"
          + (f"; dir -> {set_dir}" if set_dir else "")
          + (f"; anharm -> {set_anharm}" if set_anharm else "") + ")")
    return sp, len(modes)
