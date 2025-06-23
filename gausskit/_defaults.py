""" some default settings """

from pathlib import Path
from ase import Atom

HOME = Path().home()

DEFAULT_CONFIG_FILE = HOME / ".gausskitrc"

# SigM: Lennard-Jones σ (Å) for collider
# EpsM: Lennard-Jones ε/kB (Kelvins) for collider
# AmuM: Molecular weight (g/mole) of collider
# Amu: Molecular weight (g/mole) of reactant
colliders = {
    "O2": f"3.3920    121.74    {Atom('O').mass*2:.4f}    ",
    "N2": f"3.7047    84.942    {Atom('N').mass*2:.4f}    ",
}

trail_line = "50  'COLL'  5000     'THERMAL'   1   3   0."

bdens_setting = "best   man   10000"
