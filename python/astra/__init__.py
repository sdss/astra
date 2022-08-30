import warnings
from sdsstools import get_config, get_logger, get_package_version

NAME = "astra"

__version__ = get_package_version(path=__file__, package_name=NAME)

# The recommended path for the user configuration file is:
# ~/.astra/astra.yml
config = get_config(NAME)

logger_kwds = {}
if "logging" in config and "level" in config["logging"]:
    logger_kwds.update(log_level=config["logging"]["level"])
log = get_logger(NAME, **logger_kwds)

warnings.filterwarnings(
    "ignore", ".*Skipped unsupported reflection of expression-based index .*q3c.*"
)
