import pkg_resources

from ._defaults import DEFAULT_CONFIG_FILE
from .settings import Configuration, Settings

__version__ = str(pkg_resources.require("gausskit")[0].version)
