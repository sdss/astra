import numpy as np
from tqdm import tqdm
from sdss_access import SDSSPath

from astra.database import astradb, session
from astra.database.utils import (deserialize_pks, create_task_output)
from astra.utils import log

# TODO: Move this to astra/contrib
try:
    import doppler
except:
    log.exception(f"Cannot import `doppler` module!")


def estimate_radial_velocity(
        pks, 
        verbose=True, 
        mcmc=False,
        figfile=None,
        cornername=None,
        retpmodels=False,
        plot=False,
        tweak=True,
        usepeak=False,
        maxvel=[-1000, 1000]
    ):
    """
    Estimate radial velocities for the sources that are identified by the task instances
    of the given primary keys.

    :param pks:
        The primary keys of task instances to estimate radial velocities for, which includes
        parameters to identify the source SDSS data model product.

    See `doppler.rv.fit` for more information on other keyword arguments.
    """

    pks = deserialize_pks(pks, flatten=True)
    N = len(pks)

    log.info(f"Estimating radial velocities for {N} task instances")

    trees = {} # if only it were so easy to grow trees

    for pk in tqdm(pks):
        
        q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk == pk)
        instance = q.one_or_none()

        if instance is None:
            log.warning(f"No instance found for primary key {pk}")
            continue
        
        parameters = instance.parameters
        tree = trees.get(parameters["release"], None)
        if tree is None:
            trees[parameters["release"]] = tree = SDSSPath(release=parameters["release"])
        
        path = tree.full(**parameters)

        try:
            spectrum = doppler.read(path)
            summary, model_spectrum, modified_input_spectrum = doppler.rv.fit(
                spectrum,
                verbose=verbose,
                mcmc=mcmc,
                figfile=figfile,
                cornername=cornername,
                retpmodels=retpmodels,
                plot=plot,
                tweak=tweak,
                usepeak=usepeak,
                maxvel=maxvel
            )

        except:
            log.exception(f"Exception occurred on Doppler on {path} with task instance {instance}")
            raise 
        
        else:
            # Write the output to the database.
            results = prepare_results(summary)

            create_task_output(
                instance,
                astradb.Doppler,
                **results
            )


def prepare_results(summary_table):
    """
    Prepare outputs for the database from the Doppler summary table. 
    
    :param summary_table:
        The output summary table provided by Doppler.

    :returns:
        A dictionary that can be directly written to the astra database.
    """
    _translate = {
        "vrelerr": "u_vrel",
        "tefferr": "u_teff",
        "loggerr": "u_logg",
        "feh": "fe_h",
        "feherr": "u_fe_h"
    }
    results = {}
    for column_name in summary_table.dtype.names:
        key = _translate.get(column_name, column_name)
        results[key] = tuple(np.array(summary_table[column_name]).astype(float))
    return results
