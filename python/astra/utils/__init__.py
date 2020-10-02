
#from .logger import log
from sdsstools.logger import get_logger

log = get_logger(__name__.split(".")[0])
log.propagate = False

def unique_dicts(list_of_dicts):
    return [dict(y) for y in set(tuple(x.items()) for x in list_of_dicts)]

