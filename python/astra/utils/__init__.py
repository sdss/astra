
#from .logger import log
from sdsstools.logger import get_logger

log = get_logger(__name__.split(".")[0])
log.propagate = False

def unique_dicts(list_of_dicts):
    return [dict(y) for y in set(tuple(x.items()) for x in list_of_dicts)]


def batcher(iterable):
    batch_kwds = {}
    for item in iterable:
        for k, v in item.items():
            batch_kwds.setdefault(k, [])
            batch_kwds[k].append(v)
    return batch_kwds

