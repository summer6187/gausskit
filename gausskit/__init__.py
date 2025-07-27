import pkg_resources

# from gausskit._defaults import DEFAULT_CONFIG_FILE
# from gausskit.settings import Configuration, Settings

__version__ = str(pkg_resources.require("gausskit")[0].version)
