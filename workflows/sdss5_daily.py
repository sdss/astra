
import astra
from luigi.task import WrapperTask, flatten
from astra.utils import batcher
from astra.tasks.io import ApStarFile
from astra.tasks.targets import LocalTarget
from astra.tasks.daily import (GetDailyApStarFiles, get_visits, get_visits_given_star, get_stars)
from astra.contrib.classifier.tasks.test import (
    ClassifySourceGivenApVisitFile,
    ClassifySourceGivenApStarFile
)
import astra.contrib.apogeenet.tasks as apogeenet
#import astra.contrib.thecannon.tasks as thecannon
import astra.contrib.thepayne.tasks as thepayne
import astra.contrib.ferre.tasks.aspcap as aspcap

from astra.tasks.targets import DatabaseTarget
from sqlalchemy import Boolean, Column


class DistributeAnalysisGivenApStarFileResult(DatabaseTarget):

    """ A database row indicating we distributed analysis tasks for that object. """

    pass


class DistributeAnalysisGivenApStarFile(ApStarFile):

    def requires(self):
        """ This task requires classifications of individual sources. """
        return ClassifySourceGivenApStarFile(**self.get_common_param_kwargs(ClassifySourceGivenApStarFile))


    def run(self):
        """ Execute the task. """

        conditions = [
            # Young stellar objects.
            (lambda classification, task: classification["lp_yso"] > 0.7, [
                apogeenet.EstimateStellarParametersGivenApStarFile
            ]),
            # FGKM stars
            (lambda classification, task: classification["lp_fgkm"] > 0.7, [
                thepayne.EstimateStellarParametersGivenNormalisedApStarFile,
                aspcap.IterativeEstimateOfStellarParametersGivenApStarFile
            ])
        ]

        distributed_tasks = {}
        for requirement in self.input():

            classification = requirement.read(as_dict=True)

            for condition, factories in conditions:
                if condition(classification, requirement.task):
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



# [X] Outputs from APOGEENet to database, and [-] saved pkl file.
# [X] Execute FERRE... outputs to database, and saved pkl file.
# [ ] Execute TC... outputs to database, and saved pkl file.
# [X] Execute TP... outputs to database, and saved pkl file.

#

# Note that many parameters are set in sdss5.cfg
mjd = 59146



# [ ] Put everything into a fits table: apVisit outputs per MJD?
# [ ] Put everything into a fits table: apStar outputs per MJD?
star_kwds = batcher(list(get_stars(mjd=mjd))[:101])

#task = aspcap.IterativeEstimateOfStellarParametersGivenApStarFile(**star_kwds)
task = DistributeAnalysisGivenApStarFile(**star_kwds)

astra.build(
    [task],
    local_scheduler=True
)
raise a

task = DistributeAnalysisGivenApStarFile(**star_kwds)

astra.build(
    [task],
)


raise a


