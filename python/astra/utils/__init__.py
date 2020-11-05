
#from .logger import log
from sdsstools.logger import get_logger

log = get_logger(__name__.split(".")[0])
log.propagate = False

def unique_dicts(list_of_dicts):
    return [dict(y) for y in set(tuple(x.items()) for x in list_of_dicts)]


def batcher(iterable, task_factory=None):
    all_kwds = {}
    for item in iterable:
        for k, v in item.items():
            all_kwds.setdefault(k, [])
            all_kwds[k].append(v)

    if task_factory is None:
        return all_kwds
    
    else:
        batch_kwds = {}
        batch_param_names = task_factory.batch_param_names()

        for param_name, values in all_kwds.items():
            if param_name in batch_param_names:
                batch_kwds[param_name] = values
            else:
                unique_values = list(set(values))
                assert len(unique_values) == 1
                batch_kwds[param_name] = unique_values[0]

        return batch_kwds


