"""a wrapper for Parsctst"""
from pathlib import Path
import re
import subprocess
import time
import collections

import numpy as np

from gausskit.molecules import Molecules
from gausskit.settings import Configuration
from gausskit.gaussian.anharm import format_freq_matrix
from gausskit._defaults import bdens_setting

config = Configuration()

module_dict = {
    "p": "[Parsctst]",
    "d": "[Bdens]",
    "pd": "[Paradensum]",
}
module = collections.namedtuple("module", module_dict.keys())(**module_dict)


def _run_mpi_wl(command, cwd, label, name):
    """Run an MPI Wang-Landau job (``parsctst``/``paradensum``) with live progress.

    These programs print nothing to stdout while working -- their Wang-Landau iteration
    counter (``Iteration = N/Nf``) is written to ``Rank0.txt`` -- so a plain blocking call
    reads as a hang in ``gausskit run``. Launch the job non-blocking and echo each new
    iteration; fall back to a start/done banner if ``Rank0.txt`` never appears (e.g. the
    deck's writing flag is off). Blocks until the job finishes, like ``subprocess.call``.

    Args:
        command (str): Shell command that launches the MPI program.
        cwd (Path): Working directory where ``Rank0.txt`` is written.
        label (str): Module tag for the printed lines (e.g. ``module.pd``).
        name (str): Species/TS name (the deck's first line) for the messages.

    Returns:
        int: The process exit code.
    """
    rank0 = cwd / "Rank0.txt"
    print(f"{label:11} Wang-Landau for '{name}' (MPI) -- this can take several minutes...")
    t0 = time.time()
    proc = subprocess.Popen(command, shell=True)
    last = None
    while proc.poll() is None:
        time.sleep(2.0)
        try:
            lines = rank0.read_text().splitlines() if rank0.exists() else []
        except OSError:
            lines = []
        cur = None
        for line in reversed(lines):
            m = re.search(r"Iteration\s*=\s*(\d+)\s*/\s*(\d+)", line)
            if m:
                cur = (m.group(1), m.group(2))
                break
        if cur and cur != last:
            last = cur
            print(f"{label:11}   {name}: Wang-Landau iteration "
                  f"{cur[0]}/{cur[1]} ({int(time.time() - t0)}s elapsed)")
    rc = proc.wait()
    print(f"{label:11} '{name}' done in {int(time.time() - t0)}s (exit {rc})")
    return rc


# --- anharmonic DOS engine selection -------------------------------------------------
# Two engines compute the anharmonic vibrational density of states, with very different
# complexity:
#   * bdens      -- serial; EXACT recursive direct count below an energy switch, then
#                   Wang-Landau above. Direct-count cost ~ s * N(Emax) ~ Emax^s / s!
#                   (it enumerates every state it counts): polynomial of degree s in the
#                   energy range, but EXPONENTIAL in the mode count s. Trivial for a few
#                   modes, hopeless for many.
#   * paradensum -- parallel (MPI) pure Wang-Landau. Cost ~ s * (Emax/grain)^~1 / nranks,
#                   INDEPENDENT of the (exponential) state count: near-linear in the
#                   energy range, linear in modes, parallelisable. Statistical (~few %).
#
# The two cross near s = 6 modes, and two independent criteria agree there:
#   (1) MultiWell's own Wang-Landau cost prefactor b1 = 0.07544*s - 0.4317 is NEGATIVE
#       for s <= 5 (zero at s = 5.72): the WL cost model is undefined below ~6 modes and
#       bdens' WL stage misbehaves/crashes there -- this is exactly why a 1-mode O2
#       crashes under the default "best man 10000" hybrid. For s <= 5 one MUST direct-count.
#   (2) the enumerated state count N(Emax) ~ Emax^s / s! only blows up past s ~ 6.
#
# Vibrational modes step by 3 per atom (s = 3n-6 nonlinear, 3n-5 linear), so s = 5 is
# skipped entirely and the boundary is exactly 3 vs 4 atoms:
#   n <= 3 atoms (s <= 4: diatomics, H2O, CO2, HO2, ...)  -> bdens, pure direct count
#   n >= 4 atoms (s >= 6: H2CO, the alkoxy well, products) -> paradensum, Wang-Landau
# Any threshold in [5, 6] gives the same atom partition, so the choice is robust.
DOS_WL_MIN_MODES = 6   # >= this many real vibrational modes -> Wang-Landau (paradensum)


def n_vib_modes(mol):
    """Number of real (positive) vibrational modes of ``mol`` (= 3n-6 or 3n-5)."""
    return int(np.sum(np.asarray(mol.frequencies, dtype=float) > 0.0))


def dos_engine_for_mol(mol):
    """Pick the anharmonic DOS engine for ``mol`` by vibrational mode count.

    Returns ``'bdens'`` (exact recursive direct count) for few-mode species
    (<= 5 modes, i.e. <= 3 atoms) and ``'paradensum'`` (parallel Wang-Landau)
    for larger ones. See the module note above for the scaling rationale.
    """
    return "bdens" if n_vib_modes(mol) < DOS_WL_MIN_MODES else "paradensum"


def paradensum_configured():
    """True if a ``paradensum_command`` is set in ~/.gausskitrc.

    When it is absent the workflow falls back to (serial) ``bdens`` so the
    pipeline still runs without an MPI build configured.
    """
    return bool(config.machine.get("paradensum_command"))


def write_parsctst(
    mol:Molecules,
    fname:str = None,
    barrier:list[float]=[],
    Egrain="10   3000   4000   50000",
    separable_modes:list[int]=[],
    datfile:Path = Path("parsctst.dat"),
    verbose:bool = False,
):
    """Write the input file for the ``parsctst`` tunnelling program.

    Args:
        mol (Molecules): Transition state molecule.
        fname (str, optional): Base filename used for output.
        barrier (list[float], optional): Forward and reverse barriers in
            kcal/mol.
        Egrain (str, optional): Energy grain specification.
        separable_modes (list[int], optional): Modes treated as separable.
        datfile (Path, optional): Output file path. Defaults to
            ``parsctst.dat``.
        verbose (bool, optional): Print progress messages. Defaults to
            ``False``.

    Returns:
        None
    """

    harm_freq = np.asarray(mol.frequencies, dtype=float)
    anharm_matrix = np.asarray(mol.anharm_matrix, dtype=float)
    # mol.frequencies (parsed from Gaussian's "Fundamental Bands") and mol.anharm_X_matrix
    # (the "Total Anharmonic X Matrix") are BOTH in Gaussian's Fundamental-Bands mode order
    # (real frequencies descending, the imaginary/reaction-coordinate mode last), so they are
    # already aligned index-by-index: harm_freq[i] <-> anharm_matrix[i, i]. DO NOT reorder.
    # (Gaussian's other frequency listing -- the harmonic "Frequencies --" lines -- is the
    # opposite order, ascending with the imaginary mode first, and must not be used to index
    # the X-matrix. An earlier argsort-based reorder did exactly that and scrambled every mode.)

    img_freq = 0
    img_index = None
    ind_array = np.argsort(harm_freq)
    if harm_freq[ind_array][0] < 0:
        if np.any(harm_freq[ind_array][1:] < 0):
            print("More than one imagine freq found!!!")
            print("Please check the calculation!!!")
        else:
            img_index = ind_array[0]
            img_freq = harm_freq[img_index]
    img_nn = anharm_matrix[img_index][img_index]

    # make full anharm matrix (symmetry matrix)
    # full_anharm_matrix = (
    #     anharm_matrix + anharm_matrix.T - np.diag(np.diag(anharm_matrix))
    # )
    full_anharm_matrix = anharm_matrix  # we already have full matrix
    img_anharm_array = full_anharm_matrix[img_index]
    img_anharm_array = np.delete(img_anharm_array, img_index)

    # remake the harm_freq and anharm_matrix for sctst.dat
    harm_freq = np.delete(harm_freq, img_index)
    full_anharm_matrix = np.delete(full_anharm_matrix, img_index, axis=0)
    full_anharm_matrix = np.delete(full_anharm_matrix, img_index, axis=1)
    anharm_matrix = np.tril(full_anharm_matrix)

    # remove separable modes if given
    if separable_modes:
        sep_harm_freq = []
        print(f"Setting separable modes: {separable_modes}")
        for sep_id in separable_modes:
            sep_harm_freq.append(harm_freq[sep_id])
            if verbose:
                print(f"    Separable mode index: {sep_id}")
                print(f"    Harmonic freq: {harm_freq[sep_id]}")
                print(f"    Diagonal Anharm X matrix: {full_anharm_matrix[sep_id,sep_id]}")
                print(f"    Full Anharm X matrix: {full_anharm_matrix[sep_id,:]}")

        harm_freq = np.delete(harm_freq, separable_modes)
        full_anharm_matrix = np.delete(full_anharm_matrix, separable_modes, axis=0)
        full_anharm_matrix = np.delete(full_anharm_matrix, separable_modes, axis=1)
        anharm_matrix = np.tril(full_anharm_matrix)
        img_anharm_array = np.delete(img_anharm_array, separable_modes)

    # prepare inputfile
    lines = []
    lines.append(fname)
    lines.append(f"At {mol.method} ?? level of theory")
    lines.append(f"Anharmonicity from {mol.method} ?? level")
    lines.append(" ")

    lines.append(f'{len(harm_freq)}, {0}, {0}, "We" ')
    lines.append(" ")
    formated_lines = format_freq_matrix(harm_freq, anharm_matrix)
    lines += formated_lines
    lines.append(" ")
    lines.append(f"{len(separable_modes)}    'AMUA'")

    # formating separable mode if given
    if separable_modes:
        sep_mode_lines = []
        for n_index, harm_freq in enumerate(sep_harm_freq):
            line = f"{n_index+1}   vib  {harm_freq:.4f}  0.0  1  ! Active separable mode"
            sep_mode_lines.append(line)
        lines += sep_mode_lines

    lines.append(f"{Egrain}")
    lines.append(f"'nochekstart'  {fname}.chk")
    lines.append("VPT4A")
    if len(barrier) == 0:
        barrier_text = "<forward_barrier>  <backword_barrier>"
    else:
        barrier_text = f"{barrier[0]:.4f}  {barrier[1]:.4f}"
    lines.append(f'{barrier_text}  "kcal"')
    lines.append(f"{img_freq}  {img_nn:.5E}")
    lines += [f"{item:.5E}" for item in img_anharm_array]
    lines.append(" ")
    pardata = [
        "4       !nwalkers",                       # >=4 walkers/window for reliable Wang-Landau convergence
        "70.d0   !perc_wind_overlap",
        "0.60d0  !flatness",
        "1       !Writing enable (1) or disable (2)",
        "0       !Seed modifier",
        "cost    !Windows balance  (cost / low / high)",
    ]
    lines += pardata
    lines.append(" ")

    # write file
    if verbose:
        print(f"{module.p:11} Writing to {datfile}")
    with open(datfile, "w") as f:
        f.writelines([line + "\n" for line in lines])

    return lines

def write_bdens(
    mol:Molecules,
    fname:str = "",
    Egrain="10   3000   4000   50000",
    datfile:Path = Path("bdens.dat"),
    verbose:bool = False,
    eswitch=None,
):
    """Write the ``bdens`` input file for anharmonic densities.

    Args:
        mol (Molecules): Molecule for which densities are computed.
        fname (str, optional): Base filename prefix.
        Egrain (str, optional): Energy grain specification.
        datfile (Path, optional): Output file path. Defaults to ``bdens.dat``.
        verbose (bool, optional): Print progress information. Defaults to
            ``False``.
        eswitch (optional): direct-count/Wang-Landau switch energy (cm-1). bdens
            direct-counts below it (exact) and uses Wang-Landau above. Pass the
            calculation's Emax to force PURE direct counting (no Wang-Landau) --
            the correct, crash-free choice for few-mode species, where the WL
            cost model is invalid (see ``dos_engine_for_mol``). ``None`` keeps
            the default hybrid switch.

    Returns:
        None
    """

    harm_freq = np.asarray(mol.frequencies, dtype=float)
    anharm_matrix = np.asarray(mol.anharm_matrix, dtype=float)
    # frequencies and anharm_X_matrix are both in Gaussian Fundamental-Bands order and already
    # aligned index-by-index; do NOT reorder (see the note in write_parsctst).

    lines = []
    lines.append(fname)
    lines.append(f"At {mol.method} ?? level of theory")
    lines.append(f"Anharmonicity from {mol.method} level")
    lines.append(" ")

    lines.append(f'{len(harm_freq)}, {0}, {0}, "We" ')
    lines.append(" ")
    formated_lines = format_freq_matrix(harm_freq, anharm_matrix)
    lines += formated_lines
    lines.append(" ")
    lines.append("0    'AMUA'")
    lines.append(" ")
    if eswitch is not None:
        # KEYWORD(accuracy)  LTMODE(man)  MVAL(switch energy). MVAL=Emax => pure direct count.
        setting = f"best   man   {int(float(eswitch))}"
    else:
        setting = bdens_setting
    lines.append(f'{Egrain}  {setting} ')
    lines.append(f"'nochekstart'  {fname}.chk")
    lines.append(" ")

    # write file
    if verbose:
        print(f"{module.p:11} Writing to {datfile}")
    with open(datfile, "w") as f:
        f.writelines([line + "\n" for line in lines])

    return lines


def write_paradensum(
    mol:Molecules,
    fname:str = "",
    Egrain="10   3000   4000   50000",
    datfile:Path = Path("paradensum.dat"),
    verbose:bool = False,
):
    """Write the ``paradensum`` (parallel anharmonic DOS) input file.

    Same anharmonic data as :func:`write_bdens`, but in paradensum's deck layout: a clean
    four-field energy-grid line (no serial-bdens trial keyword) followed by the parallel
    Wang-Landau control block. paradensum reads a mandatory blank line after the checkpoint
    line, so one is emitted before the control block.

    Args:
        mol (Molecules): Molecule for which densities are computed.
        fname (str, optional): Base filename prefix.
        Egrain (str, optional): Energy grid spec ``Egrain1 imax1 Isize Emax2``.
        datfile (Path, optional): Output file path. Defaults to ``paradensum.dat``.
        verbose (bool, optional): Print progress information. Defaults to ``False``.

    Returns:
        list: The written deck lines.
    """

    harm_freq = np.asarray(mol.frequencies, dtype=float)
    anharm_matrix = np.asarray(mol.anharm_matrix, dtype=float)
    # frequencies and anharm_X_matrix are both in Gaussian Fundamental-Bands order and already
    # aligned index-by-index; do NOT reorder (see the note in write_parsctst).

    lines = []
    lines.append(fname)
    lines.append(f"At {mol.method} ?? level of theory")
    lines.append(f"Anharmonicity from {mol.method} level")
    lines.append(" ")
    lines.append(f'{len(harm_freq)}, {0}, {0}, "We" ')
    lines.append(" ")
    lines += format_freq_matrix(harm_freq, anharm_matrix)
    lines.append(" ")
    lines.append("0    'AMUA'")
    lines.append(" ")
    lines.append(f"{Egrain}")                      # four fields only; no serial-bdens trial keyword
    lines.append(f"'nochekstart'  {fname}.chk")
    lines.append(" ")                              # mandatory blank line read by paradensum
    lines += [
        "4\t!nwalkers",                            # 1 walker/window under-converges the WL DOS; >=4 converges
        "70.d0\t!perc_wind_overlap",
        "0.95d0\t!flatness",
        "1\t!writing",
        "0\t!seed modifier",
        "const\t!Windows balance",
    ]
    lines.append(" ")

    # write file
    if verbose:
        print(f"{module.pd:11} Writing to {datfile}")
    with open(datfile, "w") as f:
        f.writelines([line + "\n" for line in lines])

    return lines


def run_parsctst(
    datfile:Path = Path("parsctst.dat"),
    verbose:bool = False,
):
    """Execute the ``parsctst`` program.

    Args:
        datfile (Path): Input control file.
        verbose (bool, optional): Print the command being run. Defaults to
            ``False``.

    Returns:
        None
    """

    cwd = datfile.parent.absolute()

    _default_datfile = "parsctst.dat"

    if datfile.name != _default_datfile:
        default_datfile = cwd / _default_datfile
        if verbose:
            print(f"{module.p:11} Copy {datfile} to {default_datfile}")
        default_datfile.write_text(datfile.read_text())

    # parsctst reuses fixed scratch filenames (Rank<N>.txt, Windows_info.txt); clear any stale
    # copies (e.g. left by paradensum on the wells earlier in the same run) before launching.
    for stale in list(cwd.glob("Rank*.txt")) + [cwd / "Windows_info.txt"]:
        try:
            stale.unlink()
        except FileNotFoundError:
            pass

    command = f"cd {cwd}; " + config.machine.parsctst_command
    if verbose:
        print(f"{module.p:11} Run command: {command}")
    try:
        name = Path(datfile).read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        name = datfile.stem
    _run_mpi_wl(command, cwd, module.p, name)

def run_bdens(
    datfile:Path = Path("bdens.dat"),
    verbose:bool = False,
):
    """Execute the ``bdens`` program.

    Args:
        datfile (Path): Input control file.
        verbose (bool, optional): Print the command being run. Defaults to
            ``False``.

    Returns:
        None
    """

    cwd = datfile.parent.absolute()

    _default_datfile = "bdens.dat"

    if datfile.name != _default_datfile:
        default_datfile = cwd / _default_datfile
        if verbose:
            print(f"{module.p:11} Copy {datfile} to {default_datfile}")
        default_datfile.write_text(datfile.read_text())

    command = f"cd {cwd}; " + config.machine.bdens_command
    if verbose:
        print(f"{module.d:11} Run command: {command}")
    subprocess.call(command, shell=True)

def run_paradensum(
    datfile:Path = Path("paradensum.dat"),
    verbose:bool = False,
):
    """Execute the parallel ``paradensum`` program (Intel MPI under oneAPI).

    The ``paradensum_command`` in ``~/.gausskitrc`` is expected to launch it under an MPI
    runner, e.g. ``mpirun -n 4 /path/to/paradensum`` (oneAPI must be on the environment).

    Args:
        datfile (Path): Input control file.
        verbose (bool, optional): Print the command being run. Defaults to ``False``.

    Returns:
        None
    """

    cwd = datfile.parent.absolute()

    _default_datfile = "paradensum.dat"

    if datfile.name != _default_datfile:
        default_datfile = cwd / _default_datfile
        if verbose:
            print(f"{module.pd:11} Copy {datfile} to {default_datfile}")
        default_datfile.write_text(datfile.read_text())

    # paradensum reuses fixed scratch filenames (Rank<N>.txt, Windows_info.txt); stale copies
    # left by a previous parsctst/paradensum run in the same directory get read as garbage
    # (list-directed I/O syntax error), so clear them before launching.
    for stale in list(cwd.glob("Rank*.txt")) + [cwd / "Windows_info.txt"]:
        try:
            stale.unlink()
        except FileNotFoundError:
            pass

    command = f"cd {cwd}; " + config.machine.paradensum_command
    if verbose:
        print(f"{module.pd:11} Run command: {command}")

    try:
        species = Path(datfile).read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        species = datfile.stem
    _run_mpi_wl(command, cwd, module.pd, species)

    # paradensum's .qvib omits the KEYWORD2 field (and inserts a stray blank line) on the
    # energy-grid line that THERMO's reader requires (read_dat.f reads 4 fields there); patch
    # it to the bdens-compatible layout so THERMO can read it (otherwise its reads shift by one
    # and the "number of temperatures" read fails with a list-directed I/O syntax error).
    try:
        fname = Path(datfile).read_text().splitlines()[0].strip()
        qvib = cwd / f"{fname}.qvib"
        if qvib.exists():
            _fix_paradensum_qvib(qvib)
    except (OSError, IndexError):
        pass

def _fix_paradensum_qvib(filename):
    """Patch a paradensum ``.qvib`` to the layout THERMO expects.

    paradensum writes the energy-grid line as ``Egrain1 Emax2 zpp`` preceded by a blank line,
    but THERMO reads ``Egrain1 Emax2 zpp KEYWORD2`` from that line (and no blank before it).
    Drop the stray blank line and append the missing KEYWORD2 field.
    """
    lines = Path(filename).read_text().splitlines()
    egr = re.compile(r"^\s*[\d.]+\s+[\d.]+\s+[\d.]+\s*$")
    nsum = 0
    done = False
    out = []
    for ln in lines:
        if "INPUT DATA SUMMARY" in ln:
            nsum += 1
        if nsum >= 2 and not done and egr.match(ln):    # only the first energy-grid line after the 2nd summary
            if out and out[-1].strip() == "":
                out.pop()
            out.append(ln.rstrip() + "   BEST  ")
            done = True
        else:
            out.append(ln)
    Path(filename).write_text("\n".join(out) + "\n")


def fix_crp_file(filename, add_text="GOOD   VPT4A"):
    """Patch a parsctst ``.crp``/``.qcrp`` so THERMO can read it.

    THERMO reads ``Egrain1 Emax2 Vf Vr zpp KEYWORD2 VPTx`` from the energy-grid line of the
    qcrp (read_dat.f). The oneAPI (ifx) build of parsctst writes that line without the trailing
    ``KEYWORD2 VPTx`` fields and prepends a stray blank line, so the old fixed ``n+4`` offset
    landed on the blank line. Locate the grid line by pattern (the first all-numeric line after
    the 2nd ``INPUT DATA SUMMARY``), drop a preceding blank line, and append the missing fields.

    Args:
        filename (Path | str): File to modify.
        add_text (str, optional): Trailing fields to append. Defaults to ``"GOOD   VPT4A"``.

    Returns:
        None
    """
    lines = Path(filename).read_text().splitlines()
    egr = re.compile(r"^\s*[\d.]+(?:\s+[\d.]+){3,}\s*$")   # >=4 numeric fields, no trailing keyword
    nsum = 0
    done = False
    out = []
    for ln in lines:
        if "INPUT DATA SUMMARY" in ln:
            nsum += 1
        if nsum >= 2 and not done and egr.match(ln):
            if out and out[-1].strip() == "":
                out.pop()
            out.append(ln.rstrip() + "  " + add_text)
            done = True
        else:
            out.append(ln)
    Path(filename).write_text("\n".join(out) + "\n")
    return
