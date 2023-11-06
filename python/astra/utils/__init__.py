import os
from sdsstools.logger import get_logger as _get_logger, StreamFormatter
import warnings
from importlib import import_module
from time import time

def get_logger(kwargs=None):
    logger = _get_logger("astra", **(kwargs or {}))
    # https://stackoverflow.com/questions/6729268/log-messages-appearing-twice-with-python-logging
    # This used to cause lots of problems where logs would appear twice in Airflow / stdout.
    # But by setting it to false, I don't get *any* logs in Airflow.
    logger.propagate = True
    handler, *_ = logger.handlers
    handler.setFormatter(
        StreamFormatter("%(asctime)s %(message)s")
    )
    return logger

log = get_logger()

def executable(name):
    module_name, class_name = name.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def get_config_paths():
    from sdsstools.configuration import DEFAULT_PATHS
    return [f"{default_path.format(name='astra')}.yml" for default_path in DEFAULT_PATHS]


class Timer(object):

    def __init__(
            self, 
            iterable, 
            frequency=None, 
            callback=None, 
            attr_t_elapsed=None,
            attr_t_overhead=None,
            skip_result_callable=lambda x: x is Ellipsis
        ):
        """

        :param iterable:
            An iterable to record performance time across.
        
        :param frequency: [optional]
            The frequency (in seconds) between database check points.
        
        :param callback: [optional]
            A callback function to execute when the timer is complete.
        
        :param attr_t_elapsed: [optional]
            The attribute name to use to store the elapsed time per object. If `None`
            is given then this will not be stored.

        :param attr_t_overhead: [optional]
            The attribute name to use to store the mean overhead time per object. If `None`
            is given then this will not be stored.

        :param skip_result_callable: [optional]
            A callable that returns `True` if a result should be considered for timing.
            Usually all results are timed, but if this callable returns `False`, then
            the interval time will be added to the common overheads for these tasks.

            Usually we recommend using the `Ellipsis` (`yield ...`) to indicate that
            the interval time spent is related to overheads, and not related to the
            calculations for a single result.
        """
        self.start = time()
        self.frequency = frequency
        self.callback = callback
        self.attr_t_elapsed = attr_t_elapsed
        self.attr_t_overhead = attr_t_overhead
        self.skip_result_callable = skip_result_callable
        self.interval = 0
        self.overheads = 0
        self._iterable = iter(iterable)
        self._time_paused = 0
        self._n_check_points = 0
        self._n_results = 0
        self._last_check_point = time()        

    def __enter__(self):
        self._time_last = time()
        return self

    def __exit__(self, *args):
        self.stop = time()
        if callable(self.callback):
            self.callback(time() - self._time_last)
        
        # Add the time between last `yield` and the timer's completion to overheads
        self.overheads += self.stop - self._time_last - self._time_paused
        return None
    
    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._iterable)
        interval = time() - self._time_last - self._time_paused

        if self.skip_result_callable(item):
            self.overheads += interval        
        else:                
            self._n_results += 1
            # If the pipeline has set their own t_elapsed, do not overwrite it.
            if self.attr_t_elapsed is not None and getattr(item, self.attr_t_elapsed, None) is None:
                try:
                    # Note the time it took to analyse this object using the named attribute, 
                    # but don't assume it exists.
                    setattr(item, self.attr_t_elapsed, interval)
                except:
                    warnings.warn(f"Could not store elapsed time on attribute name `{self.attr_t_elapsed}`")

        self._time_paused = 0
        self._time_last = time()
        return item

    def add_overheads(self, items):
        o = self.mean_overhead_per_result
        for item in items:
            try:
                v = getattr(item, self.attr_t_elapsed, 0)
                setattr(item, self.attr_t_elapsed, v + o)
                setattr(item, self.attr_t_overhead, o)
            except:
                continue
        return None

    @property
    def elapsed(self):
        return time() - self.start        

    @property
    def mean_overhead_per_result(self):
        try:
            return self.overheads / self._n_results
        except ZeroDivisionError:
            return 0

    @property
    def check_point(self):
        if self.frequency is None:
            return False
        
        is_check_point = (time() - self._last_check_point) > self.frequency
        if is_check_point:
            self._last_check_point = time()
            self._n_check_points += 1
        return is_check_point

    # Make it so we can use `with timer.pause(): ... `
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