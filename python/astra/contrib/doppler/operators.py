import numpy as np
import os
from tqdm import tqdm
from sdss_access import SDSSPath

from astra.database import astradb, session
from astra.database.utils import (serialize_pks_to_path, deserialize_pks, create_task_output)
from astra.utils import log, get_scratch_dir

from astra.new_operators import (ApVisitOperator, BossSpecOperator)

# TODO: Move this to astra/contrib
import doppler


class BaseDopplerOperator:

    def __init__(
        self,
        *,
        slurm_kwargs=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # TODO: Allow to pass on any parameters?
        self.slurm_kwargs = slurm_kwargs or dict()
        return None


    def execute(self, context):

        if self.slurm_kwargs:
            # Write the primary keys to a path that is accessible by all nodes.
            pks_path = serialize_pks_to_path(
                self.pks,
                dir=get_scratch_dir()
            )
            bash_command = f"astra run doppler {pks_path}"
            
            self.execute_by_slurm(
                context,
                bash_command,
            )
            
            # Remove the temporary file.
            os.unlink(pks_path)

        else:
            # Run it in Python.
            estimate_radial_velocity(self.pks)
            
        return None
    
class DopplerApVisitOperator(BaseDopplerOperator, ApVisitOperator):
    pass

class DopplerBossSpecOperator(BaseDopplerOperator, BossSpecOperator):
    pass

class DopplerOperator(BaseDopplerOperator, ApVisitOperator, BossSpecOperator):
    pass


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

    failed_pks = []
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
        
        if parameters["filetype"] == "spSpec":
            log.warning(f"Using monkey-patch for filetype {parameters['filetype']} on pk {pk}")
            path = os.path.expandvars(
                "$BOSS_SPECTRO_REDUX/{run2d}/{plate}p/coadd/{mjd}/spSpec-{plate}-{mjd}-{identifier:0>11}.fits".format(
                    identifier=max(int(parameters["fiberid"]), int(parameters["catalogid"])),
                    **parameters
                )
            )
        else:
            path = tree.full(**parameters)

        log.debug(f"Running Doppler on {instance} at {path}")

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
            failed_pks.append(pk)
            continue

        else:
            # Write the output to the database.
            results = prepare_results(summary)

            create_task_output(
                instance,
                astradb.Doppler,
                **results
            )

    if len(failed_pks) > 0:
        log.warning(f"There were {len(failed_pks)} Doppler failures out of a total {len(pks)} executions.")
        log.warning(f"Failed primary keys include: {failed_pks}")

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
