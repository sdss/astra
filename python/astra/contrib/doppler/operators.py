import numpy as np

from astra.database import astradb
from astra.database.utils import create_task_output
from astra.utils import log

from astra.operators import (ApVisitOperator, BossSpecOperator)
from astra.operators.utils import prepare_data


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

    # TODO: Move this to astra/contrib
    import doppler

    log.info(f"Estimating radial velocities for {len(pks)} task instances")

    failures = []
    for instance, path, spectrum in prepare_data(pks):
        if spectrum is None: continue

        log.debug(f"Running Doppler on {instance} from {path}")

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
            failures.append(instance.pk)
            continue

        else:
            # Write the output to the database.
            results = prepare_results(summary)

            create_task_output(
                instance,
                astradb.Doppler,
                **results
            )

    if len(failures) > 0:
        log.warning(f"There were {len(failures)} Doppler failures out of a total {len(pks)} executions.")
        log.warning(f"Failed primary keys include: {failures}")

        log.warning(f"Raising last exception to indicate failure in pipeline.")
        raise
    

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


class BaseDopplerOperator:

    bash_command_prefix = "astra run doppler"
    python_callable = estimate_radial_velocity

    
class DopplerApVisitOperator(BaseDopplerOperator, ApVisitOperator):
    pass

class DopplerBossSpecOperator(BaseDopplerOperator, BossSpecOperator):
    pass

class DopplerOperator(BaseDopplerOperator, ApVisitOperator, BossSpecOperator):
    pass


    
