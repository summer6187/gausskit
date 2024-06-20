""" some default settings """

from pathlib import Path

HOME = Path().home()

DEFAULT_CONFIG_FILE = HOME / ".gausskitrc"

colliders = {
    "N2": "3.3920    121.74    31.98983    132.96680    ! N2 Collider"
}

trail_line = "50  'COLL'  5000     'THERMAL'   1   3   0."
