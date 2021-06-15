
#from .logger import log

import json
import os, tempfile
import logging
#from sdsstools.logger import get_logger
from time import time

import os
import tempfile

#log = get_logger(__name__.split(".")[0])
log = logging.getLogger(__name__.split(".")[0])
#log.propagate = False



def unique_dicts(list_of_dicts):
    return [dict(y) for y in set(tuple(x.items()) for x in list_of_dicts)]


def get_default(task_factory, parameter_name):
    return getattr(task_factory, parameter_name).task_value(task_factory, parameter_name)


def timer(iterable):
    for element in iterable:
        yield (time(), element)


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


from itertools import filterfalse

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def skip_incomplete(iterable, task_class):
    return filter(lambda kwd: task_class(**kwd).complete(), iterable)


def batcher(iterable, max_batch_size=None, task_factory=None, unique=False, ordered=True):

    if unique:
        if ordered:
            # Maintain an ordered set.
            iterable = list(unique_everseen(iterable, key=lambda _: frozenset(_.items())))
        else:
            iterable = [dict(s) for s in set(frozenset(d.items()) for d in iterable)]

    total = len(iterable)
    all_kwds = None
    for i, item in enumerate(iterable):
        if i == 0 or (max_batch_size is not None and (i % max_batch_size) == 0):

            all_kwds = {}
            for k in item.keys():
                all_kwds.setdefault(k, [])

        for k, v in item.items():
            all_kwds[k].append(v)
        
        if ((i + 1) == total) \
        or (max_batch_size is not None and i > 0 and ((i + 1) % max_batch_size) == 0):
            if task_factory is None:
                yield { k: (tuple(v) if isinstance(v, list) else v) for k, v in all_kwds.items() }
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

                yield { k: (tuple(v) if k in batch_param_names else v) for k, v in batch_kwds.items() }


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
