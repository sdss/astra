import importlib
import numpy as np
from ast import literal_eval
from astropy.time import Time
from datetime import datetime

from sdss_access import SDSSPath

from astra.database import (astradb, session)
from astra.database.utils import deserialize_pks
from astra.tools.spectrum import Spectrum1D
from astra.utils import log


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
            path = spectrum = modified_spectrum = None

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
                    spectrum_callback_kwargs = literal_eval(instance.parameters.get("spectrum_callback_kwargs", "{}"))

                    try:
                        mod_name, func_name = spectrum_callback.rsplit('.',1)
                        module = importlib.import_module(mod_name)
                        func = getattr(module, func_name)

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


def healpix(obj, nside=128):
    """
    Return the healpix number given the object name.
    
    :param obj:
        The name of the object in form '2M00034301-7717269'.
    :type obj: str
    :param nside:
        The number of sides in the healpix (default: 128).
    :type nside: int
    """
    
    # The logic for this function was copied directly from the sdss/apogee_drp repository, and
    # written by David Nidever.

    # apogeetarget/pro/make_2mass_style_id.pro makes these
    # APG-Jhhmmss[.]ss±ddmmss[.]s
    # http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec1_8a.html

    # Parse 2MASS-style name
    #  2M00034301-7717269
    name = obj[-16:]  # remove any prefix by counting from the end  
    # RA: 00034301 = 00h 03m 43.02s
    ra = np.float64(name[0:2]) + np.float64(name[2:4])/60. + np.float64(name[4:8])/100./3600.
    ra *= 15     # convert to degrees
    # DEC: -7717269 = -71d 17m 26.9s
    dec = np.float64(name[9:11]) + np.float64(name[11:13])/60. + np.float64(name[13:])/10./3600.
    dec *= np.float(name[8]+'1')  # dec sign

    import healpy as hp
    return hp.ang2pix(nside, ra, dec, lonlat=True)


def infer_release(context):
    """
    Infer the SDSS release to use, if none is given, based on the execution context.
    
    :param context:
        The Airflow context dictionary.
    
    :returns:
        Either 'sdss5' or 'DR17' based on the execution date.
    """
    start_date = datetime.strptime(context["ds"], "%Y-%m-%d")
    sdss5_start_date = datetime(2020, 10, 24)
    release = "sdss5" if start_date >= sdss5_start_date else "DR16"
    return release


def fulfil_defaults_for_data_model_identifiers(
        data_model_identifiers, 
        context
    ):
    """
    Intelligently set default entries for partially specified data model identifiers.
    
    :param data_model_identifiers:
        An list (or iterable) of dictionaries, where each dictionary contains keyword arguments
        to specify a data model product.

    :param context:
        The Airflow context dictionary. This is only used to infer the 'release' context,
        if it is not given, based on the execution date.

    :returns:
        A list of data model identifiers, where all required parameters are provided.
    
    :raises RuntimeError:
        If all data model identifiers could not be fulfilled.
    """

    try:
        default_release = infer_release(context)
    except:
        log.exception(f"Could not infer release from context {context}")
        default_release = None 

    trees = {}

    defaults = {
        "sdss5": {
            "apStar": {
                "apstar": "stars",
                "apred": "daily",
                "telescope": lambda obj, **_: "apo25m" if "+" in obj else "lco25m",
                "healpix": lambda obj, **_: str(healpix(obj)),
            }
        }
    }
    
    for dmi in data_model_identifiers:

        try:
            filetype = dmi["filetype"]
        except KeyError:
            raise KeyError(f"no filetype given for data model identifiers {dmi} "
                           f"-- set 'filetype': 'full' and use 'full': <PATH> to set explicit path")
        except:
            raise TypeError(f"data model identifiers must be dict-like object (not {type(dmi)}: {dmi}")
        
        source = dmi.copy()
        release = source.setdefault("release", default_release)
        
        try:
            tree = trees[release]
        except KeyError:
            trees[release] = tree = SDSSPath(release=release)
        
        missing_keys = set(tree.lookup_keys(filetype)).difference(dmi)
        for missing_key in missing_keys:
            try:
                default = defaults[release][filetype][missing_key]
            except KeyError:
                raise RuntimeError(f"no default function found for {missing_key} for {release} / {filetype}")
            
            if callable(default):
                default = default(**source)

            log.warning(f"Filling '{missing_key}' with default value '{default}' for {source}")
            source[missing_key] = default
        
        yield source
