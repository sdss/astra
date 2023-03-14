"""Utilities."""

import importlib
import json
import os
from time import time
from inspect import getmodule
from importlib import import_module
import hashlib
import json

import numpy as np
from itertools import filterfalse
from sdsstools.logger import get_logger


logger_kwds = {}
#if "logging" in config and "level" in config["logging"]:
#    logger_kwds.update(log_level=config["logging"]["level"])
log = get_logger("astra", **logger_kwds)


def to_callable(string):
    module_name, func_name = string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def transfer_mask(original_wavelength, original_mask, new_wavelength):
    """
    Convert a boolean mask on some original wavelength array to a boolean mask on a
    new wavelength array.
    """
    new_mask = np.ones(new_wavelength.size, dtype=bool)
    new_mask[new_wavelength.searchsorted(original_wavelength[~original_mask])] = False
    return new_mask



def deserialize(inputs, model):
    if isinstance(inputs, str):
        inputs = json.loads(inputs)
    _inputs = []
    for each in flatten(inputs):
        if isinstance(each, model):
            _inputs.append(each)
        else:
            _inputs.append(model.get_by_id(each))
    return _inputs


def expand_path(path):
    return os.path.expandvars(os.path.expanduser(path))


def executable(name):
    module_name, class_name = name.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def serialize_executable(callable):
    module = getmodule(callable)
    return f"{module.__name__}.{callable.__name__}"


def nested_list(ijks):
    # This could be the worst code I've ever written.
    nest = []
    for i, j, k in ijks:
        while True:
            try:
                nest[i]
            except:
                nest.append([])
            else:
                try:
                    nest[i][j]
                except:
                    nest[i].append([])
                else:
                    try:
                        nest[i][j][k]
                    except:
                        nest[i][j].append([])
                    else:
                        break
    return nest


def dict_to_list(DL):
    return [dict(zip(DL, t)) for t in zip(*DL.values())]


def list_to_dict(LD):
    return {k: [dic[k] for dic in LD] for k in LD[0]}


def unique_dicts(list_of_dicts):
    return [dict(y) for y in set(tuple(x.items()) for x in list_of_dicts)]


def timer(iterable):
    for element in iterable:
        yield (time(), element)


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


def flatten(struct):
    """
    Creates a flat list of all all items in structured output (dicts, lists, items):

    .. code-block:: python

        >>> sorted(flatten({'a': 'foo', 'b': 'bar'}))
        ['bar', 'foo']
        >>> sorted(flatten(['foo', ['bar', 'troll']]))
        ['bar', 'foo', 'troll']
        >>> flatten('foo')
        ['foo']
        >>> flatten(42)
        [42]
    """
    if struct is None:
        return []
    flat = []
    if isinstance(struct, dict):
        for _, result in struct.items():
            flat += flatten(result)
        return flat
    if isinstance(struct, str):
        return [struct]

    try:
        # if iterable
        iterator = iter(struct)
    except TypeError:
        return [struct]

    for result in iterator:
        flat += flatten(result)
    return flat



def hashify(params, max_length=8):
    """
    Create a short hashed string of the given parameters.

    :param params:
        A dictionary of key, value pairs for parameters.

    :param max_length: [optional]
        The maximum length of the hashed string.
    """
    param_str = json.dumps(params, separators=(",", ":"), sort_keys=True)
    param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()
    return param_hash[:max_length]