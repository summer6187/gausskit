"""CLI for gausskit with click"""

from pathlib import Path
import pickle

import click
import click_completion
import configparser

from gausskit import __version__ as gausskit_version
from gausskit.database import append_species, write_database, load_database, show_database
from gausskit.potential_energy_surface import PES_parser

click_completion.init()
complete_files = click.Path(exists=True)


def finish_line(dry:bool = False):
    click.echo("=" * 80)
    if dry:
        click.echo("Dry run! Only generate input files!")
    click.echo("Job done! Have a nice day!")


@click.group()
@click.version_option(gausskit_version, "-V", "--version")
def cli():
    """gausskit: Transition state theory code. Interfacing Gaussian09/16 and Multiwell"""
    click.echo("Welcome to gausskit!\n")
    click.echo("=" * 80)


@cli.command()
@click.argument("file", type=complete_files)
@click.pass_obj
def info(obj, file):
    filename = Path(file).resolve()
    # load database.pickle file
    click.echo(f"Dataset summary for {filename}:")
    ds = load_database(filename)

    # formated printing of database information
    show_database(ds)

    finish_line()


@cli.command()
@click.argument("directory", type=complete_files)
@click.option("-o", "--outfile", default="database.json", show_default=True)
@click.option("--force", is_flag=True, help="enfore parsing of output file")
@click.pass_obj
def output(obj, directory, outfile, force):
    outfile = Path(outfile)
    if not force and outfile.exists():
        click.echo(f"Output file {outfile} exists!")
        click.echo(".. (use --force to overwrite)")
        return

    database_path = Path(directory).resolve()
    click.echo(f"Parsing every log files in this directory: {database_path}")

    database = {}

    logfile_list = []
    for path_object in database_path.rglob("*"):
        if path_object.is_file() and path_object.suffix == ".log":
            logfile_list.append(path_object)

    for logfile in logfile_list:
        filepath = logfile.resolve()
        print(f"Parsing filepath {filepath}")
        database = append_species(database, filepath)

    # write database
    click.echo(f"Writing to {outfile}")
    filetype = outfile.suffix
    if filetype == ".pickle":
        # write database in pickle binary file
        with open(outfile, "wb") as f:
            pickle.dump(database, f)
    elif filetype == ".json":
        write_database(database, outfile)

    finish_line()


@cli.command()
@click.argument("file", type=complete_files)
@click.option("--dry", is_flag=True, help="Dry run, do not execute command.")
@click.option("--verbose", is_flag=True, help="Show verbose information.")
@click.pass_obj
def run(obj, file, dry, verbose):
    config = configparser.ConfigParser()

    filename = Path(file).resolve()
    click.echo(f"Run this PES: {filename}")
    config.read(filename)
    PES_parser(config, dry=dry, verbose=verbose)

    finish_line(dry=dry)


@cli.command()
@click.argument("deck", type=complete_files)
@click.option("--dens", "dens", multiple=True, metavar="NAME=PATH",
              help="use an anharmonic bdens .dens for trial surface NAME (repeatable).")
@click.option("--zpe-shift", "zpe_shift", multiple=True, metavar="NAME=DZPE_CM",
              help="shift species NAME's delh by dZPE = ZPE_anh - ZPE_harm in cm-1 (repeatable; "
                   "required for a consistent anharmonic run -- the .dens is anharmonic-ground-"
                   "referenced, so its placement must use the anharmonic ZPE).")
@click.option("--frag-dens", "frag_dens", multiple=True, metavar="NAME=PATH",
              help="replace fragment NAME's harmonic q_vib in Q_reac with the anharmonic q from "
                   "this bdens .dens (repeatable; keeps numerator and denominator consistent).")
@click.option("--grain", default=5.0, show_default=True, help="integration grain (cm-1).")
@click.option("-o", "--out", "out", default=None, type=click.Path(),
              help="write k_cap(T) to this CSV.")
def vtst(deck, dens, zpe_shift, frag_dens, grain, out):
    """gausskit-native microcanonical mu-VTST capture rate from a ktools.dat DECK.

    Bypasses the native KTOOLS solver: reads the SAME deck gausskit generates (``gausskit run
    --dry``) and computes the J-resolved variational capture rate k_cap(T) in Python.  Its
    harmonic backend reproduces native KTOOLS to <1%; pass ``--dens NAME=PATH`` to swap a bdens
    coupled-VPT2 (or scan-rotor) DOS in for trial surface NAME (e.g. ``--dens TS4=dens/R2.10.dens``),
    plus ``--zpe-shift NAME=DZPE`` to reference that surface at its anharmonic ZPE.
    """
    import csv as _csv
    from gausskit.multiwell.vtst import read_ktools_deck, capture_rate

    d = read_ktools_deck(deck)
    dens_map = dict(kv.split("=", 1) for kv in dens) or None
    zpe_map = ({k: float(v) for k, v in (kv.split("=", 1) for kv in zpe_shift)}
               if zpe_shift else None)
    frag_map = dict(kv.split("=", 1) for kv in frag_dens) or None
    harm = capture_rate(d, grain=grain)
    anh = (capture_rate(d, grain=grain, dens_map=dens_map, zpe_shift=zpe_map,
                        frag_dens=frag_map)
           if dens_map else None)

    click.echo(f"  {'T/K':>6} {'k_cap(harm)':>12} {'r_var':>6}"
               + (f" {'k_cap(anh)':>12} {'anh/harm':>8}" if anh else ""))
    rows = [["T_K", "k_cap_harmonic", "r_var_A"] + (["k_cap_anharmonic"] if anh else [])]
    for i, T in enumerate(harm["T"]):
        line = f"  {T:6.0f} {harm['k_cap'][i]:12.4e} {harm['r_var'][i]:6.2f}"
        row = [T, harm["k_cap"][i], harm["r_var"][i]]
        if anh:
            line += f" {anh['k_cap'][i]:12.4e} {anh['k_cap'][i]/harm['k_cap'][i]:8.3f}"
            row.append(anh["k_cap"][i])
        click.echo(line)
        rows.append(row)
    if out:
        with open(out, "w", newline="") as f:
            _csv.writer(f).writerows(rows)
        click.echo(f"wrote {out}")
    finish_line()


@cli.group()
def utils():
    """Convenience utilities / demos built on gausskit objects."""


@utils.command(name="mode_scan")
@click.argument("structure", type=complete_files)
@click.option("--log", "log", default=None, type=complete_files,
              help="freq=anharmonic .log (optional; used with a .fchk for the full Molecules)")
@click.option("-n", "--nmodes", default=2, show_default=True,
              help="number of lowest REAL modes to scan")
@click.option("-o", "--outdir", default="mode_scan", show_default=True, help="output directory")
@click.option("--qmax", default=11.0, show_default=True,
              help="outermost |Q| of the scan wings (A*amu^1/2)")
@click.option("--dense", default=3.0, show_default=True,
              help="half-width of the dense window |Q|<=dense (A*amu^1/2)")
@click.option("--step", default=0.25, show_default=True,
              help="step of the dense window (coarsen to reduce point count; ~1.0 + qmax 8 gives "
                   "~17 pts, converged to <0.1%% of the 41-pt scan-HRD -- see hrd/STAGE0_GRID_CONVERGENCE.md)")
@click.option("--charge-mult", "charge_mult", default="0 2", show_default=True,
              help="charge and spin multiplicity written into each gjf")
@click.option("--route", default=None,
              help="Gaussian route line for each gjf [default: M06-2X/def2TZVP single point]")
@click.option("--nproc", default=16, show_default=True, help="%nprocshared written into each gjf")
@click.option("--mem", default="16GB", show_default=True, help="%mem written into each gjf")
def mode_scan(structure, log, nmodes, outdir, qmax, dense, step, charge_mult, route, nproc, mem):
    """Generate rigid normal-mode scan STRUCTURES for the softest modes of a TS.

    STRUCTURE is a Gaussian .fchk (preferred) or a freq=anharmonic .log. Displaces the TS
    rigidly along each of the NMODES lowest real modes, x(Q)=x0+(L_i/sqrt(m))*Q, and writes
    <outdir>/mode{1..N}/*.gjf + manifest.csv -- input structures only; submit them however
    you like. (A demo/wrapper around gausskit Harmonic.displace_along_mode.)
    """
    from gausskit.utils.mode_scan import generate_mode_scan, make_qgrid, DEFAULT_ROUTE
    route = route or DEFAULT_ROUTE
    qgrid = make_qgrid(qmax=qmax, dense=dense, step=step)
    written, picks, freqs = generate_mode_scan(
        structure, log=log, n_modes=nmodes, qgrid=qgrid,
        route=route, charge_mult=charge_mult, outdir=outdir, nproc=nproc, mem=mem)
    click.echo(f"grid: {len(qgrid)} pts/mode (dense +-{dense}@{step} + wings to +-{qmax})")
    click.echo(f"picked {len(picks)} lowest real modes: idx {picks}  "
               f"freqs {[round(float(f), 2) for f in freqs]} cm-1")
    click.echo(f"charge/mult '{charge_mult}'  route: {route}")
    click.echo(f"wrote {len(written)} gjf structures to {outdir}/ (+ manifest.csv)")
    click.echo("Input structures only -- submit the jobs however you like.")
    finish_line()


@utils.command(name="ktools_rates")
@click.argument("canonical", type=complete_files)
@click.option("-o", "--out", "outfile", default=None, type=click.Path(),
              help="write CSV to this file (default: stdout; use -o for a clean, banner-free CSV)")
@click.option("--direction", type=click.Choice(["both", "forward", "reverse"]),
              default="both", show_default=True, help="which rate column(s) to report")
@click.option("--minflux/--no-minflux", default=True, show_default=True,
              help="include the variational-TS distance r_var(T)")
@click.option("--block", type=click.Choice(["final", "unified"]), default="final", show_default=True,
              help="'final' = FINAL RECOMMENDED table; 'unified' = unified-canonical uk(t), the "
                   "stable rate for a barrierless-capture (reverse) reproduction")
@click.option("--pretty", is_flag=True, help="aligned human-readable table instead of CSV")
def ktools_rates(canonical, outfile, direction, minflux, block, pretty):
    """Read a KTOOLS .canonical file and report its rate constants (+ variational-TS r_var).

    CANONICAL is a ktools '<base>.canonical' output. With --block final (default) emits T_K,
    forward, reverse, Keq from the FINAL RECOMMENDED table; with --block unified emits the
    unified-canonical uk(t) forward/reverse rate (use this for a barrierless CAPTURE reproduction
    — the FINAL RECOMMENDED reverse is forward/Keq and is unstable at low T). Columns filtered by
    --direction / --no-minflux; CSV to stdout or -o FILE, or --pretty. Forward = dissociation
    (s-1), reverse = capture (cm3 molecule-1 s-1). (A wrapper around gausskit.multiwell.ktools.read_*.)
    """
    import numpy as np
    from gausskit.multiwell.ktools import (read_ktools_canonical, read_ktools_min_flux,
                                           read_ktools_unified)

    dirs = ["forward", "reverse"] if direction == "both" else [direction]
    if block == "unified":
        cols = {}
        for d in dirs:
            Td, uk = read_ktools_unified(Path(canonical), direction=d)
            if Td is None:
                raise click.ClickException(f"No unified {d} block in {canonical}")
            cols.setdefault("T_K", Td); cols[d] = uk
        T = cols["T_K"]
    else:
        T, fwd, rev, Keq = read_ktools_canonical(Path(canonical))
        if T is None:
            raise click.ClickException(f"No FINAL RECOMMENDED block in {canonical}")
        cols = {"T_K": T}
        if "forward" in dirs:
            cols["forward"] = fwd
        if "reverse" in dirs:
            cols["reverse"] = rev if rev is not None else np.full_like(T, np.nan)
        if direction == "both":
            cols["Keq"] = Keq if Keq is not None else np.full_like(T, np.nan)
    if minflux:
        d = direction if direction in ("forward", "reverse") else "reverse"
        Tm, rvar, _ = read_ktools_min_flux(Path(canonical), direction=d)
        rmap = dict(zip([float(t) for t in Tm], rvar)) if Tm is not None else {}
        cols["r_var_A"] = np.array([rmap.get(float(t), np.nan) for t in T])

    headers = list(cols)

    def fmt(h, v):
        if h == "T_K":
            return f"{v:.2f}"
        if np.isnan(v):
            return ""
        return f"{v:.2f}" if h == "r_var_A" else f"{v:.5e}"

    rows = [[fmt(h, cols[h][i]) for h in headers] for i in range(len(T))]
    if pretty:
        w = [max(len(h), *(len(r[j]) for r in rows)) for j, h in enumerate(headers)]
        text = "  ".join(h.rjust(w[j]) for j, h in enumerate(headers)) + "\n"
        text += "\n".join("  ".join(r[j].rjust(w[j]) for j in range(len(headers))) for r in rows)
    else:
        text = ",".join(headers) + "\n" + "\n".join(",".join(r) for r in rows)

    if outfile:
        Path(outfile).write_text(text + "\n")
        click.echo(f"wrote {len(T)} rows to {outfile}")
    else:
        click.echo(text)
    finish_line()
