"""Utilities."""
# from .logger import log

from dataclasses import is_dataclass
import json
import os, tempfile
import logging

# from sdsstools.logger import get_logger
from time import time
from inspect import getmodule

from importlib import import_module
import hashlib
import json

import os
import tempfile


import numpy as np
from tqdm import tqdm


def transfer_mask(original_wavelength, original_mask, new_wavelength):
    """
    Convert a boolean mask on some original wavelength array to a boolean mask on a
    new wavelength array.
    """
    new_mask = np.ones(new_wavelength.size, dtype=bool)
    new_mask[new_wavelength.searchsorted(original_wavelength[~original_mask])] = False
    return new_mask


class logarithmic_tqdm(tqdm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.total is not None and self.miniters > 0:
            self.logarithmic_iters = np.unique(
                np.logspace(0, np.log10(self.total), self.miniters, dtype=int)
            )
            F = self.miniters - self.logarithmic_iters.size
            if F > 0:
                # Remove uniques and add mid-points for later iterations.
                deltas = -0.5 * np.diff(self.logarithmic_iters)[-F:]
                self.logarithmic_iters = np.sort(
                    np.hstack(
                        [self.logarithmic_iters, self.logarithmic_iters[-F:] + deltas]
                    )
                ).astype(int)
        else:
            self.logarithmic_iters = None
        # Keep a decoy internal counter and only update the real counter when we want.
        self._n = self.n
        return None

    def update(self, n=1):
        if self.disable:
            return

        if self.logarithmic_iters is not None:
            self._n += n
            index = np.where(self.logarithmic_iters == self._n)[0]
            if index.size > 0:
                numer = 1 + index[-1]
                # tqdm does something on the final refresh that changes .miniters
                denom = self.miniters if self.miniters > numer else numer
                self.set_postfix(dict(log_update=f"{numer}/{denom}"), refresh=False)
                return super().update(self._n - self.n)
        else:
            return super().update(n)


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


def estimate_relative_cost(bundle):
    """
    Estimate the relative cost of executing a bundle.

    The cost of executing a bundle depends on the kind of task, the bundle parameters, the
    number of data products provided, and the number of spectra in each data product. We
    can only really estimate the 'relative' cost, and so here we will estimate the cost
    based on the:
    - number of tasks
    - number of data products
    - size of each data product
    - parameters in the task bundle
    """

    example_task = bundle.tasks.first()
    klass = executable(example_task.name)
    try:
        factors = klass.estimate_relative_cost_factors(example_task.parameters)
    except:
        # By default we will scale cost relative to the size of data products.
        # (which defaults to the number of data products, if size is unavailable)
        cost = bundle.count_input_data_products_size()
    else:
        A = np.array(
            [
                bundle.count_tasks(),
                bundle.count_input_data_products(),
                bundle.count_input_data_products_size(),
            ]
        )
        cost = np.sum(A * factors)
    return cost


def bundler(tasks, dry_run=False, as_primary_keys=False):
    """
    Create bundles for tasks that can be executed together.

    :param tasks:
        The tasks to bundle.
    """

    from astra.base import Parameter
    from astra.database.astradb import database, Task, TaskBundle, Bundle

    if isinstance(tasks, str):
        # They will be primary keys.
        ids = list(map(int, json.loads(tasks)))
        tasks = list(Task.select().where(Task.id.in_(ids)))

    # We need to bundle tasks by their name, and what bundled parameters are the same.
    # We only know the bundle parameters once we load the executable class.
    executables = {name: executable(name) for name in {task.name for task in tasks}}
    bundled_parameter_names = {}
    for task_name, task_executable in executables.items():
        bundled_parameter_names[task_name] = []
        for parameter_name, parameter in task_executable.__dict__.items():
            if isinstance(parameter, Parameter) and parameter.bundled:
                bundled_parameter_names[task_name].append(parameter_name)

    # Group tasks together with the same name and bundled parameters.
    hashes = {}
    hashed_sets = {}
    for task in tasks:
        hash_args = [task.name]
        for parameter_name in bundled_parameter_names[task.name]:
            try:
                value = task.parameters[parameter_name]
            except:
                # Get the default value.
                value = getattr(executables[task.name], parameter_name).default

            # TODO: If we have a bundled parameter with 1.0 and 1 in two different tasks,
            #       they won't be bundled together. We'd need typed parameters to do this
            #       properly, I think. And we might even want to have typed parameters
            #       for other reasons.
            hash_args.append(f"{value}")

        task_hash = hash("|".join(hash_args))
        hashes.setdefault(task_hash, [])
        hashed_sets[task_hash] = (
            task.name,
            dict(zip(bundled_parameter_names[task.name], hash_args[1:])),
        )
        hashes[task_hash].append(task)

    if dry_run:
        return (hashes, hashed_sets)

    bundled = []
    with database.atomic() as txn:
        for _, group in hashes.items():
            bundle = Bundle.create()
            for task in group:
                TaskBundle.create(task=task, bundle=bundle)
            bundled.append(bundle)

    if as_primary_keys:
        return [bundle.id for bundle in bundled]
    return bundled


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


class Timer:
    def __enter__(self):
        self.t_enter = time()

    def __exit__(self, type, value, traceback):
        self.t_exit = time()

    @property
    def time(self):
        return self.t_exit - self.t_enter


timer = Timer()


def get_scratch_dir():
    dirname = os.path.join(get_base_output_path(), "scratch")
    os.makedirs(dirname, exist_ok=True)
    return dirname


def get_base_output_path(version=None):
    """
    Get the base output path for Astra.

    :param version: [optional]
        The version of Astra. If `None` is given then the current version (`astra.__version__`)
        will be used.
    """
    if version is None:
        from astra import __version__ as version

        version = version.split("-")[0]
    return os.path.join(os.path.expandvars(f"$MWM_ASTRA/{version}"))


def unique_dicts(list_of_dicts):
    return [dict(y) for y in set(tuple(x.items()) for x in list_of_dicts)]


def get_default(task_factory, parameter_name):
    return getattr(task_factory, parameter_name).task_value(
        task_factory, parameter_name
    )


def timer(iterable):
    for element in iterable:
        yield (time(), element)


def symlink(target, link_name, overwrite=False):
    """
    Create a symbolic link named link_name pointing to target.
    If link_name exists then FileExistsError is raised, unless overwrite=True.
    When trying to overwrite a directory, IsADirectoryError is raised.
    """

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
            raise IsADirectoryError(
                f"Cannot symlink over existing directory: '{link_name}'"
            )
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


def skip_incomplete(iterable, task_class):
    return filter(lambda kwd: task_class(**kwd).complete(), iterable)


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


def batcher(
    iterable, max_batch_size=None, task_factory=None, unique=False, ordered=True
):

    if unique:
        if ordered:
            # Maintain an ordered set.
            iterable = list(
                unique_everseen(iterable, key=lambda _: frozenset(_.items()))
            )
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

        if ((i + 1) == total) or (
            max_batch_size is not None and i > 0 and ((i + 1) % max_batch_size) == 0
        ):
            if task_factory is None:
                yield {
                    k: (tuple(v) if isinstance(v, list) else v)
                    for k, v in all_kwds.items()
                }
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

                yield {
                    k: (tuple(v) if k in batch_param_names else v)
                    for k, v in batch_kwds.items()
                }


def symlink_force(source, destination):
    """
    Create a symbolic link destination pointing to source.
    Overwrites destination if it exists.
    """

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
