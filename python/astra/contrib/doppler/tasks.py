import numpy as np
from time import time
from luigi.task import flatten
from astra.database import astradb, session, catalogdb
from astra.tasks import BaseTask
from astra.tasks.io.sdss5 import SpecFile
from astra.tasks.targets import DatabaseTarget
from astra.utils import log, timer

import doppler

# TODO: Slurm-ify this task.

class Doppler(BaseTask):

    """ Base task for Doppler. """

    def requires(self):
        raise NotImplementedError("this should be over-written by sub-classes")


    def prepare_outputs(self, summary_table):
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

    #@slurmify
    def run(self):
        """ Run this task. """
        
        for t_init, task in timer(self.get_batch_tasks()):
            if task.complete(): continue

            try:
                spectrum = doppler.read(task.input()["observation"].path)
                summary, model_spectrum, modified_input_spectrum = doppler.rv.fit(
                    spectrum,
                    verbose=True,
                    figfile=None
                )
            
            except:
                log.exception(f"Exception occurred on task {task}:")
                task.trigger_event_failed()

            else:
                # Write output(s).
                task.output()["database"].write(self.prepare_outputs(summary))
                
                # TODO: Write the model_spectrum and modified_input_spectrum to some AstraSource object?
                # TODO: Write a figure as output?
                task.trigger_event_processing_time(time() - t_init, cascade=True)                


    def output(self):
        """ Outputs of this task. """
        if self.is_batch_mode:
            return (task.output() for task in self.get_batch_tasks())
        return dict(database=DatabaseTarget(astradb.Doppler, self))


class DopplerGivenSpecFile(Doppler, SpecFile):

    """ Run Doppler given an SDSS-V BHM Spec file. """

    def requires(self):
        """ Requirements of this task. """
        return dict(observation=self.clone(SpecFile))


class DopplerOnAllMWMSpecFiles(Doppler):

    """ A wrapper task that runs Doppler on all SDSS-V BHM Spec files that are of interest to MWM. """


    def requires(self):
        """ Query all MWM-like targets from the BOSS database table and generate tasks. """
        results = (
            session.query(catalogdb.SDSSVBossSpall)\
                   .filter(catalogdb.SDSSVBossSpall.firstcarton.like("%mwm%")) \
                   .filter(catalogdb.SDSSVBossSpall.catalogid > 0)
        ).all()

        """
        kwds = dict(run2d=[], plate=[], mjd=[], catalogid=[])
        for result in results:
            for key in kwds.keys():
                kwds[key].append(getattr(result, key))
        
        return DopplerGivenSpecFile(**kwds)
        """
        for result in results:
            yield DopplerGivenSpecFile(
                run2d=result.run2d,
                plate=result.plate,
                mjd=result.mjd,
                catalogid=result.catalogid
            )

    def complete(self):
        return all(r.complete() for r in flatten(self.requires()))
