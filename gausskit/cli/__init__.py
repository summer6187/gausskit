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
    click.echo(f"Welcome to gausskit!\n")
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
