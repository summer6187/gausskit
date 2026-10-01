"""The hand-editable rotor roadmap (`rotor_plan.yaml`) — build / dump / load.

The roadmap is the single source of truth shared by ``scan`` and ``fit``. ``build_plan``
turns a classification into the plan dict; ``dump_plan`` writes annotated, standard YAML;
``load_plan`` reads it back (with PyYAML if installed, else a small tailored parser so the
tool has no hard YAML dependency and stays locally runnable).

Plan schema (in memory)::

    {"log", "route", "charge_mult",
     "fragments": {"A": [0,1,2,3], "B": [4,5]},
     "modes": [ {"mode","gauss_idx","freq_cm","rot_frac","method",
                 # rigid-rotor: "fragment","axis","I_amuA2","grid":{theta_deg:[lo,hi],dtheta}
                 # mode-scan:   "grid":{qmax,dense,step}
                 # harmonic:    (no scan params)
                }, ... ]}
"""
import os

from gausskit.utils.mode_scan import DEFAULT_ROUTE
from gausskit.utils.rotor.character import classify_modes
from gausskit.utils.rotor import constants as C


def build_plan(log, *, method="auto", soft_cut=C.SOFT_CUT, rot_min=C.ROT_MIN,
               nmax=C.NMAX, tol=1.3, route=None, charge_mult="0 2",
               theta_grid=None, q_grid=None):
    """Classify `log` and build the roadmap dict. `method` is the flag: 'auto' uses each
    mode's classifier suggestion; 'rigid-rotor'/'mode-scan' force all SOFT modes (stiff
    stays harmonic; rigid-rotor is only honoured when 2 fragments exist). Returns
    ``(plan, warnings)``."""
    if method not in ("auto", "rigid-rotor", "mode-scan"):
        raise ValueError(f"method must be auto|rigid-rotor|mode-scan, got {method!r}")
    cls = classify_modes(log, soft_cut=soft_cut, rot_min=rot_min, nmax=nmax, tol=tol)
    two_fragments = cls["n_fragments"] == 2
    theta = theta_grid or {"theta_deg": [C.DEFAULT_THETA_MIN, C.DEFAULT_THETA_MAX],
                           "dtheta": C.DEFAULT_DTHETA}
    qg = q_grid or {"qmax": C.DEFAULT_QMAX, "dense": C.DEFAULT_DENSE, "step": C.DEFAULT_STEP}

    warnings = []
    if cls["n_fragments"] not in (1, 2):
        warnings.append(f"{cls['n_fragments']} fragments detected (expected 1 or 2); "
                        "all soft modes default to mode-scan.")

    modes = []
    for m in cls["modes"]:
        auto = m["auto_method"]
        if auto == "harmonic":
            resolved = "harmonic"
        elif method == "auto":
            resolved = auto
        elif method == "rigid-rotor":
            resolved = "rigid-rotor" if two_fragments else "mode-scan"
        else:                                   # method == "mode-scan"
            resolved = "mode-scan"
        if method == "rigid-rotor" and not two_fragments and auto != "harmonic":
            warnings.append(f"mode {m['mode']}: --rigid-rotor requires 2 fragments "
                            f"(have {cls['n_fragments']}) -> kept mode-scan.")

        entry = {"mode": m["mode"], "gauss_idx": m["gauss_idx"],
                 "freq_cm": round(m["freq"], 2), "rot_frac": round(m["rot_frac"], 3),
                 "method": resolved}
        if resolved == "rigid-rotor":
            entry["fragment"] = m["frag"]
            entry["axis"] = [round(x, 5) for x in m["axis"]]
            # single-fragment moment about the fragment COM = an auto-SEED for the reduced
            # moment (B=KAPPA/I); edit here, or replace with mominert's I_red, for a real run.
            entry["I_amuA2"] = round(m["I"], 4)
            entry["grid"] = dict(theta)
        elif resolved == "mode-scan":
            entry["grid"] = dict(qg)
        modes.append(entry)

    plan = {"log": os.path.abspath(log), "route": route or DEFAULT_ROUTE,
            "charge_mult": charge_mult,
            "fragments": {lab: f for lab, f in zip(cls["frag_labels"], cls["fragments"])},
            "modes": modes}
    return plan, warnings


# --------------------------------------------------------------------------- dump

def _scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        # quote if it contains anything that would confuse the reader
        if v == "" or any(c in v for c in ":#[]{}") or v.strip() != v:
            return '"' + v.replace('"', '\\"') + '"'
        return v
    if isinstance(v, float):
        return repr(round(v, 6))
    return str(v)


def _flow(v):
    """Inline flow collection: [a, b] or {k: v, ...}."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_flow(x) if isinstance(x, (list, dict)) else _scalar(x)}"
                               for k, x in v.items()) + "}"
    return "[" + ", ".join(_scalar(x) for x in v) + "]"


def dump_plan(plan, path):
    """Write the roadmap as annotated, standard YAML."""
    L = ["# rotor_plan.yaml — hand-editable rotor roadmap",
         "#   scan: gausskit utils rotor scan <this file> -o scans/",
         "#   method per mode: rigid-rotor | mode-scan | harmonic. Edit anything below;",
         "#   scan/fit dispatch on 'method'. (gauss_idx is the anchor — do NOT renumber.)",
         f"log: {_scalar(plan['log'])}",
         f"route: {_scalar(plan['route'])}",
         f"charge_mult: {_scalar(plan['charge_mult'])}",
         "fragments:"]
    for lab, atoms in plan["fragments"].items():
        L.append(f"  {lab}: {_flow(atoms)}")
    L.append("modes:")
    for m in plan["modes"]:
        L.append(f"  - mode: {m['mode']}")
        for k in ("gauss_idx", "freq_cm", "rot_frac", "method", "fragment", "I_amuA2"):
            if k in m:
                L.append(f"    {k}: {_scalar(m[k])}")
        if "axis" in m:
            L.append(f"    axis: {_flow(m['axis'])}")
        if "grid" in m:
            L.append(f"    grid: {_flow(m['grid'])}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------- load

def load_plan(path):
    """Read a roadmap. Uses PyYAML if importable, else a tailored fallback parser that
    handles the emitted structure (+ common hand-edits: comments, block/flow collections)."""
    text = open(path).read()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml(text)


def _strip_comment(line):
    """Drop a trailing '# ...' comment, respecting quotes and inline flow braces."""
    out, q = [], None
    for ch in line:
        if q:
            out.append(ch)
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _parse_scalar(s):
    s = s.strip()
    if s == "" or s in ("~", "null"):
        return None
    if s in ("true", "false"):
        return s == "true"
    if (s[0] == s[-1]) and s[0] in "\"'" and len(s) >= 2:
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _split_flow(body):
    """Split the inside of a [...] / {...} on top-level commas."""
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch in "[{":
            depth += 1; cur += ch
        elif ch in "]}":
            depth -= 1; cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _parse_value(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return [_parse_value(p) for p in _split_flow(s[1:-1])]
    if s.startswith("{") and s.endswith("}"):
        d = {}
        for p in _split_flow(s[1:-1]):
            k, _, v = p.partition(":")
            d[k.strip()] = _parse_value(v)
        return d
    return _parse_scalar(s)


def _mini_yaml(text):
    """Minimal YAML reader for the rotor-roadmap subset (2-space block indent; scalars,
    inline flow [ ]/{ }, a single nested `fragments:` map, a `modes:` list of flat maps)."""
    root = {}
    lines = [(_strip_comment(l)) for l in text.splitlines()]
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        if not raw.strip():
            i += 1
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        key, sep, val = line.partition(":")
        key = key.strip()
        if sep and val.strip() == "" and indent == 0:
            # a block sub-structure at column 0: gather deeper-indented lines
            block, i = [], i + 1
            while i < n and (not lines[i].strip() or (len(lines[i]) - len(lines[i].lstrip())) > 0):
                block.append(lines[i]); i += 1
            if any(b.strip().startswith("- ") for b in block):
                root[key] = _parse_seq(block)
            else:
                root[key] = _parse_map(block)
            continue
        root[key] = _parse_value(val)
        i += 1
    return root


def _parse_map(block):
    d = {}
    for raw in block:
        if not raw.strip():
            continue
        k, _, v = raw.strip().partition(":")
        d[k.strip()] = _parse_value(v)
    return d


def _parse_seq(block):
    items, cur = [], None
    for raw in block:
        if not raw.strip():
            continue
        s = raw.strip()
        if s.startswith("- "):
            if cur is not None:
                items.append(cur)
            cur = {}
            s = s[2:].strip()
        if cur is None:
            cur = {}
        k, _, v = s.partition(":")
        cur[k.strip()] = _parse_value(v)
    if cur is not None:
        items.append(cur)
    return items
