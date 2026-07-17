"""Loader for fitted rotor results — the single reader used by PES_parser and the CLI.

Accepts BOTH formats:
  * ``rotor_result.json`` — written by ``gausskit utils rotor fit``:
      {"log": ..., "modes": [{"mode", "method", "freq_cm", "zpe_cm", "B_cm",
                              "CV_Vhrd2", "nsym", ...}, ...]}
    (``method: harmonic`` entries carry no HRD and are skipped.)
  * legacy ``hrd_params.json`` — the old fit_hrd.py output:
      {"mode1": {"B_cm", "CV_Vhrd2", "nsym", ...}, "mode2": {...}}
    (no per-mode frequency -> ``freq_cm`` is None; pairing falls back to softest-N.)

Returns a list of per-mode dicts sorted by mode number:
    [{"mode": int, "freq_cm": float|None, "B_cm": float, "CV_Vhrd2": [float],
      "nsym": int, "zpe_cm": float|None}, ...]
"""
import json
import re


def load_fitted_modes(path):
    d = json.load(open(path))
    out = []
    if isinstance(d.get("modes"), list):                      # rotor_result.json
        for m in d["modes"]:
            if m.get("method") == "harmonic":
                continue
            out.append({"mode": int(m["mode"]), "freq_cm": m.get("freq_cm"),
                        "B_cm": float(m["B_cm"]), "CV_Vhrd2": list(m["CV_Vhrd2"]),
                        "nsym": int(m.get("nsym", 1)), "zpe_cm": m.get("zpe_cm")})
    else:                                                     # legacy hrd_params.json
        for key, m in d.items():
            n = re.fullmatch(r"mode(\d+)", key)
            if not n or "CV_Vhrd2" not in m:
                continue
            out.append({"mode": int(n.group(1)), "freq_cm": None,
                        "B_cm": float(m["B_cm"]), "CV_Vhrd2": list(m["CV_Vhrd2"]),
                        "nsym": int(m.get("nsym", 1)), "zpe_cm": None})
    if not out:
        raise ValueError(f"{path}: no fitted (non-harmonic) rotor modes found")
    return sorted(out, key=lambda m: m["mode"])


def result_log(path):
    """The source freq log recorded in a rotor_result.json (None for legacy files)."""
    return json.load(open(path)).get("log")


def pair_modes_to_freqs(freqs, fitted, label=""):
    """Pair each fitted rotor mode to an index of ``freqs`` (any order).

    With per-mode frequencies: greedy closest-frequency matching (warns when the match is
    off by >max(15%, 5 cm^-1) — deck and fit logs can disagree on ultra-soft modes, e.g.
    a freq=anharmonic run flipping a near-zero mode). Legacy results without frequencies
    fall back to softest-N pairing (fitted modes ascending -> N smallest freqs ascending).

    Returns ``{index_into_freqs: fitted_mode_dict}``.
    """
    n = len(freqs)
    if len(fitted) > n:
        raise ValueError(f"{label}: {len(fitted)} fitted rotor modes but only {n} frequencies")
    pairs = {}
    if all(m.get("freq_cm") is not None for m in fitted):
        taken = set()
        for m in sorted(fitted, key=lambda m: m["freq_cm"]):
            j = min((i for i in range(n) if i not in taken),
                    key=lambda i: abs(float(freqs[i]) - m["freq_cm"]))
            taken.add(j)
            if abs(float(freqs[j]) - m["freq_cm"]) > max(0.15 * float(freqs[j]), 5.0):
                print(f"  !!     {label}: fitted mode {m['mode']} ({m['freq_cm']} cm-1) "
                      f"paired to frequency {float(freqs[j]):.1f} cm-1 -- check the pairing")
            pairs[j] = m
    else:                                               # legacy: softest-N by rank
        order = sorted(range(n), key=lambda i: float(freqs[i]))
        for rank, m in enumerate(sorted(fitted, key=lambda m: m["mode"])):
            pairs[order[rank]] = m
    return pairs
