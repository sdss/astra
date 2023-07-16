import os
from sdsstools.logger import get_logger
import warnings
from importlib import import_module
from time import time

logger_kwds = {}
#if "logging" in config and "level" in config["logging"]:
#    logger_kwds.update(log_level=config["logging"]["level"])
log = get_logger("astra", **logger_kwds)


def get_config_paths():
    from sdsstools.configuration import DEFAULT_PATHS
    return [f"{default_path.format(name='astra')}.yml" for default_path in DEFAULT_PATHS]


class Timer(object):

    def __init__(self, iterable, frequency=None, callback=None, attribute_name=None):
        """

        :param iterable:
            An iterable to record performance time across.
        
        :param frequency: [optional]
            The frequency (in seconds) between database check points.
        
        :param callback: [optional]
            A callback function to execute when the timer is complete.
        
        :param attribute_name: [optional]
            The attribute name to use to store the elapsed time per object. If `None`
            is given then this will not be stored.
        """
        self._iterable = iter(iterable)
        self._time_paused = 0
        self._start = time()
        self._n_check_points = 0
        self._last_check_point = time()
        self._frequency = frequency
        self._callback = callback
        self._attribute_name = attribute_name

    def __enter__(self):
        self._time_last = time()
        return self

    def __exit__(self, *args):
        if callable(self._callback):
            self._callback(time() - self._time_last)
        return None
    
    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._iterable)
        self.interval = time() - self._time_last - self._time_paused
        # If the pipeline has set their own t_elapsed, do not overwrite it.
        if self._attribute_name is not None and getattr(item, self._attribute_name, None) is None:
            try:
                # Note the time it took to analyse this object using the named attribute, 
                # but don't assume it exists.
                setattr(item, self._attribute_name, self.interval)
            except:
                warnings.warn(f"Could not store elapsed time on attribute name `{self._attribute_name}`")

        self._time_paused = 0
        self._time_last = time()
        return item

    @property
    def elapsed(self):
        return time() - self._start    

    @property
    def check_point(self):
        if self._frequency is None:
            return False
        
        is_check_point = (time() - self._last_check_point) > self._frequency
        if is_check_point:
            self._last_check_point = time()
            self._n_check_points += 1
        return is_check_point

    # Make it so we can use `with timer.paused(): ... `
    def pause(self):
        def callback(time_paused):
            self._time_paused += time_paused
        return self.__class__([], None, callback=callback)
    

def callable(input_callable):
    if isinstance(input_callable, str):
        try_prefixes = ("", "astra.")
        for prefix in try_prefixes:
            try:                
                module_name, func_name = (prefix + input_callable).rsplit(".", 1)
                module = import_module(module_name)
                return getattr(module, func_name)
            except:
                continue        
        raise ImportError(f"Cannot resolve input callable `{input_callable}`")
    else:
        return input_callable


    
def expand_path(path):
    """
    Expand a given path to its full path, including environment variables and user home directory.
    
    :param path:
        A short-hand path to expand.
    """
    return os.path.expandvars(os.path.expanduser(path))


def dict_to_list(DL):
    """
    Convert a dictionary with lists as values to a list of dictionaries.
    """
    return [dict(zip(DL, t)) for t in zip(*DL.values())]


def list_to_dict(LD):
    """
    Convert a list of dictionaries to a dictionary with lists as values.
    """
    return {k: [dic[k] for dic in LD] for k in LD[0]}


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