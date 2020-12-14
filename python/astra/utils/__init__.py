
#from .logger import log

import os, tempfile
from sdsstools.logger import get_logger

log = get_logger(__name__.split(".")[0])
log.propagate = False

def unique_dicts(list_of_dicts):
    return [dict(y) for y in set(tuple(x.items()) for x in list_of_dicts)]


def symlink(target, link_name, overwrite=False):
    '''
    Create a symbolic link named link_name pointing to target.
    If link_name exists then FileExistsError is raised, unless overwrite=True.
    When trying to overwrite a directory, IsADirectoryError is raised.
    '''

    if not overwrite:
        os.symlink(target, link_name)
        return

    # os.replace() may fail if files are on different filesystems
    link_dir = os.path.dirname(link_name)

    # Create link to target with temporary filename
    while True:
        temp_link_name = tempfile.mktemp(dir=link_dir)

        # os.* functions mimic as closely as possible system functions
        # The POSIX symlink() returns EEXIST if link_name already exists
        # https://pubs.opengroup.org/onlinepubs/9699919799/functions/symlink.html
        try:
            os.symlink(target, temp_link_name)
            break
        except FileExistsError:
            pass

    # Replace link_name with temp_link_name
    try:
        # Pre-empt os.replace on a directory with a nicer message
        if not os.path.islink(link_name) and os.path.isdir(link_name):
            raise IsADirectoryError(f"Cannot symlink over existing directory: '{link_name}'")
        os.replace(temp_link_name, link_name)
    except:
        if os.path.islink(temp_link_name):
            os.remove(temp_link_name)
        raise


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


