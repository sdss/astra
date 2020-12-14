
#from .logger import log
from sdsstools.logger import get_logger

import os
import tempfile

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





def symlink_force(source, destination):
    '''
    Create a symbolic link destination pointing to source.
    Overwrites destination if it exists.
    '''

    # os.replace() may fail if files are on different filesystems
    link_dir = os.path.dirname(destination)

    while True:
        temp_destination = tempfile.mktemp(dir=link_dir)
        try:
            os.symlink(source, temp_destination)
            break
        except FileExistsError:
            pass
    try:
        os.replace(temp_destination, destination)
    except OSError:  # e.g. permission denied
        os.remove(temp_destination)
        raise