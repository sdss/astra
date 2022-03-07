import warnings
from sdsstools import (get_config, get_logger, get_package_version)

NAME = "astra"

__version__ = get_package_version(path=__file__, package_name=NAME)

# The recommended path for the user configuration file is:
# ~/.astra/astra.yml
config = get_config(NAME)

log_level = config.get("logging", {}).get("level", None)
log = get_logger(NAME, log_level=log_level)

warnings.filterwarnings(
    'ignore', 
    '.*Skipped unsupported reflection of expression-based index .*q3c.*'
)