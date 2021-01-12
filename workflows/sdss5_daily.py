
import astra
from astra.utils import batcher, get_default
from astra.tasks.io import ApStarFile
from astra.tasks.daily import (GetDailyApStarFiles, get_visits, get_visits_given_star, get_stars)
import astra.contrib.apogeenet.tasks as apogeenet
import astra.contrib.ferre.tasks.aspcap as aspcap
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

    table_name = "distribute_apstar"






class DistributeAnalysisGivenApStarFile(ApStarFile):

    def requires(self):
        """ This task requires classifications of individual sources. """
        return self.clone(classifier.ClassifySourceGivenApStarFile)


    def run(self):
        """ Execute the task. """

        # We only want to distribute ApStar files to ASPCAP if the stellar parameter estimates
        # from Doppler are within the boundary of *any* grid.
        # Otherwise we spend a lot of time upstream implementing hacks to ignore these edge cases.
        in_bounds = aspcap.doppler_estimate_in_bounds_factory(
            release=self.release,
            public=self.public,
            mirror=self.mirror
        )
        distribute_to_aspcap = lambda c, r: in_bounds({ k: getattr(r.task, k) for k in r.task.batch_param_names()})
        
        conditions = [
            #(distribute_to_aspcap, [
            #    aspcap.EstimateChemicalAbundancesGivenApStarFile
            #]),
            # Everything on everything
            (lambda r, c: True, [
                apogeenet.EstimateStellarParametersGivenApStarFile,
            #    thecannon.EstimateStellarLabelsGivenApStarFile,
                thepayne.EstimateStellarLabelsGivenApStarFile,
            ]),
            # Young stellar objects.
            #(lambda classification: True, [# classification["lp_yso"] > 0.0, [
            #    apogeenet.EstimateStellarParametersGivenApStarFile
            #]),
            # FGKM stars (less probable)
            #(lambda classification: True, [#classification["lp_fgkm"] > 0.0, [
            #    thecannon.EstimateStellarLabelsGivenApStarFile,
            #    thepayne.EstimateStellarLabelsGivenApStarFile,
            #]),
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
                if condition(classification, requirement):
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
        
        # Create summary files for all the tasks.
        #for condition, factories in conditions:
        #    #for factory in factories:
                

        # Mark all tasks as being done.
        for task in self.get_batch_tasks():
            task.output().write()


    def output(self):
        """ Outputs of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        return DistributeAnalysisGivenApStarFileResult(self)
        

'''
sdss5db=> select distinct(mjdend) from apogee_drp.star order by mjdend asc;
 mjdend 
'''

mjds = [
  59146,
  59159,
  59163,
  59164,
  59165,
  59166,
  59167,
  59168,
  59169,
  59186,
  59187,
  59188,
  59189,
  59190,
  59191,
  59192,
  59193,
  59195,
  59198,
  59199,
  59200,
  59201,
  59202,
  59203,
  59204,
  59205,
  59206,
  59207,
  59208,
  59209,
  59210,
  59211,
  59212,
  59214,
  59215,
  59216,
  59127,
]


# Note that many parameters are set in sdss5.cfg
if False:
    # For only running ApStar files where the initial estimate is within bounds of the grid.
    kwds = []
    for mjd in mjds:
        kwds.extend(list(get_stars(mjd=mjd)))

    in_bounds = aspcap.doppler_estimate_in_bounds_factory(release="sdss5", public=False, mirror=False)

    # Remove duplicates.
    kwds = [dict(s) for s in set(frozenset(d.items()) for d in kwds)]

    # Only run things in bound.
    kwds = [kwd for kwd in kwds if in_bounds(kwd)]

    task = aspcap.EstimateChemicalAbundancesGivenApStarFile(**batcher(kwds))

    raise a

kwds = []
for mjd in sorted(mjds):
    kwds.extend(list(get_stars(mjd=mjd)))

before = len(kwds)
print(f"Checking for missing ApStar files...")

# Ignore missing ApStar files.
import os
kwds = list(filter(lambda kwd: os.path.exists(ApStarFile(**kwd).local_path), kwds))
print(f"There were {before - len(kwds)} missing ApStar files that we ignored")

# Remove duplicates.
#kwds = [dict(s) for s in set(frozenset(d.items()) for d in kwds)]
task_kwds = batcher(kwds, unique=True)
task = DistributeAnalysisGivenApStarFile(**task_kwds)


#task = apogeenet.EstimateStellarParametersGivenApStarFile(**task_kwds)

#raise a

#task = apogeenet.EstimateStellarParametersGivenApStarFile(**batcher(kwds))
raise a
astra.build(
    [task],
    local_scheduler=True
)

