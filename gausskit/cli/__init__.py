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
@click.option("--charge-mult", "charge_mult", default="0 2", show_default=True,
              help="charge and spin multiplicity written into each gjf")
@click.option("--route", default=None,
              help="Gaussian route line for each gjf [default: M06-2X/def2TZVP single point]")
@click.option("--nproc", default=16, show_default=True, help="%nprocshared written into each gjf")
@click.option("--mem", default="16GB", show_default=True, help="%mem written into each gjf")
def mode_scan(structure, log, nmodes, outdir, qmax, charge_mult, route, nproc, mem):
    """Generate rigid normal-mode scan STRUCTURES for the softest modes of a TS.

    STRUCTURE is a Gaussian .fchk (preferred) or a freq=anharmonic .log. Displaces the TS
    rigidly along each of the NMODES lowest real modes, x(Q)=x0+(L_i/sqrt(m))*Q, and writes
    <outdir>/mode{1..N}/*.gjf + manifest.csv -- input structures only; submit them however
    you like. (A demo/wrapper around gausskit Harmonic.displace_along_mode.)
    """
    from gausskit.utils.mode_scan import generate_mode_scan, make_qgrid, DEFAULT_ROUTE
    route = route or DEFAULT_ROUTE
    written, picks, freqs = generate_mode_scan(
        structure, log=log, n_modes=nmodes, qgrid=make_qgrid(qmax=qmax),
        route=route, charge_mult=charge_mult, outdir=outdir, nproc=nproc, mem=mem)
    click.echo(f"picked {len(picks)} lowest real modes: idx {picks}  "
               f"freqs {[round(float(f), 2) for f in freqs]} cm-1")
    click.echo(f"charge/mult '{charge_mult}'  route: {route}")
    click.echo(f"wrote {len(written)} gjf structures to {outdir}/ (+ manifest.csv)")
    click.echo("Input structures only -- submit the jobs however you like.")
    finish_line()
