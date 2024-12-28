"""CLI for gausskit with click"""

from pathlib import Path
import os
import pickle

import click
import click_completion

from gausskit import __version__ as gausskit_version
from gausskit.molecules import Molecules

click_completion.init()
complete_files = click.Path(exists=True)


def finish_line():
    click.echo("=" * 80)
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
    import json
    from gausskit.database import get_name_info

    filename = Path(file).resolve()
    # load database.pickle file
    click.echo(f"Dataset summary for {filename}:")
    if filename.suffix == ".pickle":
        with open(filename, "rb") as f:
            ds = pickle.load(f)
    elif filename.suffix == ".json":
        with open(filename) as f:
            database_dict = json.load(f)
        database = {}
        for item in database_dict:
            database[item] = {}
            for method in database_dict[item]:
                database[item][method] = Molecules.from_dict(
                    database_dict[item][method]
                )
        ds = database

    for name in ds.keys():
        name_info = get_name_info(ds, name)
        print(f"{name:30}: {name_info}")

    finish_line()


@cli.command()
@click.argument("directory", type=complete_files)
@click.option("-o", "--outfile", default="database.json", show_default=True)
@click.option("--force", is_flag=True, help="enfore parsing of output file")
@click.pass_obj
def output(obj, directory, outfile, force):
    from gausskit.database import append_species, write_database

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
@click.option("--verbose", is_flag=True, help="Show verbose information.")
@click.pass_obj
def run(obj, file, verbose):
    import configparser
    from gausskit.potential_energy_surface import PES_parser

    config = configparser.ConfigParser()

    filename = Path(file).resolve()
    click.echo(f"Run this PES: {filename}")
    config.read(filename)
    PES_parser(config, verbose=verbose)

    finish_line()
