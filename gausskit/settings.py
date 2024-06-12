from jconfigparser import Config
from jconfigparser.dict import DotDict

from gausskit._defaults import DEFAULT_CONFIG_FILE

def merge(source: dict, destination: dict, dict_type=dict):
    """
    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, dict_type)
            merge(value, node, dict_type)
        else:
            destination[key] = value

    return destination

class Configuration(Config):

    def __init__(self, config_file: str = DEFAULT_CONFIG_FILE):
        """Initializer

        Args:
            config_file: Path to the configure file
        """

        super().__init__(filenames=config_file)

class Settings(Config):
    """Class to hold the settings parsed from settings.in (+ the configuration)"""

    def __init__(
        self,
        settings_file: str = None,
        config_file: str = DEFAULT_CONFIG_FILE,
        dct: dict = None,
    ):
        """Initialize Settings

        Args:
            settings_file: Path to the settings file
            config_files: Path to the configuration files
            dct: create Settings from this dictionary

        """
        # read config, then template, then user settings
        
        _dct = DotDict()

        if not dct:
            dct = {}

            if settings_file is not None:
                dct = Config(settings_file)

        _dct = merge(dct, _dct, dict_type=DotDict)
        
        if config_file is not None:
            _dct = Config(config_file)

        super().__init__()
        for key in _dct:
            self[key] = _dct[key]

