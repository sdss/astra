import inspect
import importlib
from ast import literal_eval
from astropy.time import Time

from sdss_access import SDSSPath

from astra.database import (astradb, session)
from astra.database.utils import deserialize_pks
from astra.tools.spectrum import Spectrum1D
from astra.utils import log


def get_data_model_path(instance, trees=None, full_output=False):
    release = instance.parameters["release"]
    trees = trees or dict()
    tree = trees.get(release, None)
    if tree is None:
        trees[release] = tree = SDSSPath(release=release)

    # Monkey-patch BOSS Spec paths.
    try:
        path = tree.full(**instance.parameters)
    except:
        if instance.parameters["filetype"] == "spec":
            from astra.utils import monkey_patch_get_boss_spec_path
            path = monkey_patch_get_boss_spec_path(**instance.parameters)
        else:
            raise
    
    return (path, trees) if full_output else path


def prepare_data(pks):
    """
    Return the task instance, data model path, and spectrum for each given primary key,
    and apply any spectrum callbacks to the spectrum as it is loaded.

    :param pks:
        Primary keys of task instances to load data products for.

    :returns:
        Yields a four length tuple containing the task instance, the spectrum path, the
        original spectrum, and the modified spectrum after any spectrum callbacks have been
        executed. If no spectrum callback is executed, then the modified spectrum will be
        `None`.
    """
    
    trees = {}

    for pk in deserialize_pks(pks, flatten=True):
        q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk == pk)
        instance = q.one_or_none()

        if instance is None:
            log.warning(f"No task instance found for primary key {pk}")
            path = spectrum = None

        else:
            release = instance.parameters["release"]
            tree = trees.get(release, None)
            if tree is None:
                trees[release] = tree = SDSSPath(release=release)

            # Monkey-patch BOSS Spec paths.
            try:
                path = tree.full(**instance.parameters)
            except:
                if instance.parameters["filetype"] == "spec":
                    from astra.utils import monkey_patch_get_boss_spec_path
                    path = monkey_patch_get_boss_spec_path(**instance.parameters)
                else:
                    raise

            try:
                spectrum = Spectrum1D.read(path)
            except:
                log.exception(f"Unable to load Spectrum1D from path {path} on task instance {instance}")
                spectrum = None
            else:
                # Are there any spectrum callbacks?
                spectrum_callback = instance.parameters.get("spectrum_callback", None)
                if spectrum_callback is not None:
                    spectrum_callback_kwargs = instance.parameters.get("spectrum_callback_kwargs", "{}")
                    try:
                        spectrum_callback_kwargs = literal_eval(spectrum_callback_kwargs)
                    except:
                        log.exception(f"Unable to literally evalute spectrum callback kwargs for {instance}: {spectrum_callback_kwargs}")
                        raise

                    try:
                        func = string_to_callable(spectrum_callback)

                        spectrum = func(
                            spectrum=spectrum,
                            path=path,
                            instance=instance,
                            **spectrum_callback_kwargs
                        )

                    except:
                        log.exception(f"Unable to execute spectrum callback '{spectrum_callback}' on {instance}")
                        raise
                                        
        yield (instance, path, spectrum)
    

def parse_as_mjd(mjd):
    """
    Parse Modified Julian Date, which might be in the form of an execution date
    from Apache Airflow (e.g., YYYY-MM-DD), or as a MJD integer. The order of
    checks here is:

        1. if it is not a string, just return the input
        2. if it is a string, try to parse the input as an integer
        3. if it is a string and cannot be parsed as an integer, parse it as
           a date time string

    :param mjd:
        the Modified Julian Date, in various possible forms.
    
    :returns:
        the parsed Modified Julian Date
    """
    if isinstance(mjd, str):
        try:
            mjd = int(mjd)
        except:
            return Time(mjd).mjd
    return mjd


def callable_to_string(function):
    if callable(function):
        module = inspect.getmodule(function)
        return f"{module.__name__}.{function.__name__}"
    elif isinstance(function, str):
        return function
    else:
        raise TypeError(f"function must be a callable, or a string representation of a callable (not {type(function)}: {function})")

def string_to_callable(function_string):
    if callable(function_string):
        return function_string
    elif isinstance(function_string, str):
        mod_name, func_name = function_string.rsplit('.', 1)
        module = importlib.import_module(mod_name)
        return getattr(module, func_name)
    else:
        raise TypeError(f"function must be a callable, or a string representation of a callable (not {type(function_string)}: {function_string})s")
    