
import astra
from astra.utils import batcher
from astra.tasks.io import ApStarFile
from astra.tasks.daily import (GetDailyApStarFiles, get_visits, get_visits_given_star, get_stars)
import astra.contrib.apogeenet.tasks as apogeenet
import astra.contrib.ferre.tasks as ferre
import astra.contrib.classifier.tasks.test as classifier
# TODO: revise
import astra.contrib.thecannon.tasks.sdss5 as thecannon
import astra.contrib.thepayne.tasks as thepayne
import astra.contrib.wd.tasks as wd

from astra.tasks.targets import DatabaseTarget
from sqlalchemy import Boolean, Column


from luigi.mock import MockTarget

class DistributeAnalysisGivenApStarFileResult(DatabaseTarget):

    """ A database row indicating we distributed analysis tasks for that object. """

    pass


class DistributeAnalysisGivenApStarFile(ApStarFile):

    def requires(self):
        """ This task requires classifications of individual sources. """
        return classifier.ClassifySourceGivenApStarFile(
            **self.get_common_param_kwargs(classifier.ClassifySourceGivenApStarFile)
        )


    def run(self):
        """ Execute the task. """

        conditions = [
            # Young stellar objects.
            (lambda classification: classification["lp_yso"] > 0.5, [
                apogeenet.EstimateStellarParametersGivenApStarFile
            ]),
            # FGKM stars (less probable)
            (lambda classification: classification["lp_fgkm"] > 0.0, [
                thecannon.EstimateStellarLabelsGivenApStarFile,
                thepayne.EstimateStellarLabelsGivenApStarFile,
            ]),
            # FGKM stars
            #(lambda classification: classification["lp_fgkm"] > 0.9, [
            #    ferre.IterativeEstimateOfStellarParametersGivenApStarFile
            #]),    
            # White Dwarf classifications
            #(lambda classification: True, [
            #    wd.ClassifyWhiteDwarfGivenSpecFile
            #])
        ]

        distributed_tasks = {}
        for requirement in self.input():

            classification = requirement.read(as_dict=True)

            for condition, factories in conditions:
                if condition(classification):
                    for factory in factories:
                        distributed_tasks.setdefault(factory, [])
                        distributed_tasks[factory].append(
                            requirement.task.get_common_param_kwargs(factory)
                        )
            
        for task_factory, rows in distributed_tasks.items():
            # I would have thought that it would be better to batch these together and yield one
            # task, but there are some deep issues with representing and parsing batch parameters
            # and so the better thing to do is to yield a list of tasks

            # Note to future self: This caused a *lot* of deep problems where it seemed that
            # luigi would loop over the same number of pending tasks forever, and never actually
            # run any. If that happens you *have* to batch things so that you yield as few tasks
            # as possible.
            
            # It also caused other problems where the batch parameters needed to be serialised,
            # and that wasn't working very well. A hack exists, in astra.tasks.BaseTask, but a
            # more robust solution is wanting.
            yield task_factory(**batcher(rows, task_factory=task_factory))
        
        # Mark all tasks as being done.
        for task in self.get_batch_tasks():
            task.output().write()


    def output(self):
        """ Outputs of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        return DistributeAnalysisGivenApStarFileResult(self)
        


# TODO: Grab these from database when we get back on-sky!
mjds = [
    59146,
    59159,
    59163,
    59164,
    59165,
    59166,
    59167,
    59168,
    59169
]
# Note that many parameters are set in sdss5.cfg


tasks = []
for mjd in mjds:
    tasks.append(
        DistributeAnalysisGivenApStarFile(use_remote=True, **batcher(get_stars(mjd=mjd)))
    )

astra.build(
    tasks,
    local_scheduler=True
)
