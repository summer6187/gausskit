"""a wrapper for Parsctst"""
from pathlib import Path
import re
import subprocess
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

def write_parsctst(
    mol:Molecules,
    fname:str = None,
    barrier:list[float]=[],
    Egrain="10   3000   4000   50000",
    separable_modes:list[int]=[],
    datfile:Path = Path("parsctst.dat"),
    verbose:bool = False,
    hrd_modes:list = None,
):
    """Write the input file for the ``parsctst`` tunnelling program.

    Args:
        mol (Molecules): Transition state molecule.
        fname (str, optional): Base filename used for output.
        barrier (list[float], optional): Forward and reverse barriers in
            kcal/mol.
        Egrain (str, optional): Energy grain specification.
        separable_modes (list[int], optional): Modes treated as separable
            (indices into the real-mode list after the imaginary mode is removed;
            emitted as plain ``vib`` lines).
        datfile (Path, optional): Output file path. Defaults to
            ``parsctst.dat``.
        verbose (bool, optional): Print progress messages. Defaults to
            ``False``.
        hrd_modes (list, optional): PES.in ``[HRD]``-declared fitted rotors — the
            paired soft modes leave the coupled X-matrix and are emitted as
            SEPARABLE ``hrd``/``Vhrd2``/``Bhrd1`` blocks (overrides
            ``separable_modes``).

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

    # [HRD]: pair the declared fitted rotors to real modes by frequency and pull them
    # out of the coupled X-matrix as separable Vhrd2 rotors. A large-amplitude soft mode is
    # INVALID under VPT2 -- its large negative diagonal x_ii folds the anharmonic level ladder
    # over (Birge-Sponer turnover) and truncates the DOS -- so it cannot stay in the coupled
    # VPT2 manifold; the scan-fitted Vhrd2 rotor is its correct replacement. The STIFF modes
    # keep their full VPT2 X-matrix (this does NOT replace VPT2, it fixes where VPT2 breaks).
    sep_hrd = None                                        # index-aligned with separable_modes
    if hrd_modes:
        from gausskit.utils.rotor.result import pair_modes_to_freqs
        pairs = pair_modes_to_freqs(harm_freq, hrd_modes, label=fname or mol.name)
        # ascending frequency order -> separable block indices 1..N softest-first
        separable_modes = sorted(pairs, key=lambda j: harm_freq[j])
        sep_hrd = [pairs[j] for j in separable_modes]
        for j, m in zip(separable_modes, sep_hrd):
            print(f"  HRD    {fname}: coupled mode {harm_freq[j]:.4f} cm-1 -> separable "
                  f"hrd (mode{m['mode']}, B={m['B_cm']:.6f}, {len(m['CV_Vhrd2'])} CV) "
                  "-- dropped from the X-matrix")

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

    # formating separable mode if given: hrd/Vhrd2/Bhrd1 blocks for [HRD]-declared
    # rotors, plain vib lines otherwise
    if separable_modes:
        sep_mode_lines = []
        for n_index, sep_freq in enumerate(sep_harm_freq):
            if sep_hrd is not None:
                from gausskit.multiwell.dos import hrd_block
                m = sep_hrd[n_index]
                blk = {"CV": np.array(m["CV_Vhrd2"]), "B": m["B_cm"], "nsym": m["nsym"]}
                line = hrd_block(n_index + 1, blk,
                                 f"separable scan-matched soft mode ({sep_freq:.2f} cm-1)")
            else:
                line = f"{n_index+1}   vib  {sep_freq:.4f}  0.0  1  ! Active separable mode"
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
):
    """Write the ``bdens`` input file for anharmonic densities.

    Args:
        mol (Molecules): Molecule for which densities are computed.
        fname (str, optional): Base filename prefix.
        Egrain (str, optional): Energy grain specification.
        datfile (Path, optional): Output file path. Defaults to ``bdens.dat``.
        verbose (bool, optional): Print progress information. Defaults to
            ``False``.

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
    lines.append(f'{Egrain}  {bdens_setting} ')
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
    subprocess.call(command, shell=True)

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
    subprocess.call(command, shell=True)

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
    # >=4 numeric fields, no trailing keyword. Fields may be SIGNED / scientific: a submerged or
    # near-zero barrier gives a negative forward height Vf on this grid line (e.g. CFCl3+Na, Vf=-12.21),
    # which a digits-and-dots-only pattern would miss -> the GOOD/VPTx fix silently skips it and THERMO
    # then fails to parse the qcrp. Allow a leading sign and exponent per field.
    _num = r"[-+]?[\d.]+(?:[eE][-+]?\d+)?"
    egr = re.compile(rf"^\s*{_num}(?:\s+{_num}){{3,}}\s*$")
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
