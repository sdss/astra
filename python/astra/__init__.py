NAME = "astra"
__version__ = "0.2.3dev"

# TODO: Move these things elsewhere (e.g., utils) to speed up top-level import.
from sdsstools.configuration import get_config
from sdsstools.logger import get_logger

# The recommended path for the user configuration file is:
# ~/.astra/astra.yml
config = get_config(NAME)

logger_kwds = {}
if "logging" in config and "level" in config["logging"]:
    logger_kwds.update(log_level=config["logging"]["level"])
log = get_logger(NAME, **logger_kwds)
