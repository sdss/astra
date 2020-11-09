
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
            # FGKM stars
            (lambda classification: classification["lp_fgkm"] > 0.9, [
                ferre.IterativeEstimateOfStellarParametersGivenApStarFile
            ]),
            # FGKM stars (less probable)
            (lambda classification: classification["lp_fgkm"] > 0.1, [
                thecannon.EstimateStellarParametersGivenApStarFile,
                thepayne.EstimateStellarParametersGivenNormalisedApStarFile,
            ])
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
        #return DistributeAnalysisGivenApStarFileResult(self)
        return MockTarget(self.task_id)



# [X] Outputs from APOGEENet to database, and [-] saved pkl file.
# [X] Execute FERRE... outputs to database, and saved pkl file.
# [X] Execute TC... outputs to database, and saved pkl file.
# [X] Execute TP... outputs to database, and saved pkl file.

#

# Note that many parameters are set in sdss5.cfg
mjd = 59146



# [ ] Put everything into a fits table: apVisit outputs per MJD?
# [ ] Put everything into a fits table: apStar outputs per MJD?
foo = list(get_stars(mjd=mjd))
star_kwds = batcher(foo)

"""
task = ferre.IterativeEstimateOfStellarParametersGivenApStarFile(**foo[388])

astra.build(
    [task],
    local_scheduler=True
)
"""


task = DistributeAnalysisGivenApStarFile(**star_kwds)

astra.build(
    [task],
    local_scheduler=True
)


import numpy as np
import sqlalchemy
def get_all_star_targets(star_kwds, join_with_star=True):

    # We want to get additional information...

    task = DistributeAnalysisGivenApStarFile(**star_kwds)

    output = task.output()[0]
    engine = output.engine
    table_names = engine.table_names()
    batch_param_names = task.batch_param_names() 
   
    result = output.read(as_dict=True)

    rows = {}
    column_names = {}

    # Get star table.
    star_table = astra.tasks.daily.star_table
    s = sqlalchemy.select([star_table])
    r = astra.tasks.daily.engine.execute(s).fetchall()

    rows[star_table.name] = r
    column_names[star_table.name] = [c.name for c in star_table.c]

    for table_name in table_names:

        metadata = sqlalchemy.MetaData()
        table = sqlalchemy.Table(
            table_name, 
            metadata,
            autoload=True,
            autoload_with=engine
        )

        rows[table_name] = []
        column_names[table_name] = [c.name for c in table.c]

        # Check that this is an allStar set.
        if not all(pn in column_names[table_name] for pn in batch_param_names):
            print(f"Skipping {table_name}")
            continue

        for output in task.output():
            result = output.read(as_dict=True)

            s = sqlalchemy.select([table]).where(
                sqlalchemy.and_(
                    *[getattr(table.c, pn) == result[pn] for pn in batch_param_names]
                )
            )

            r = engine.execute(s).fetchall()

            print(table_name, result["obj"], r)
            rows[table_name].extend(r)


    return (rows, column_names)




rows, column_names = get_all_star_targets(star_kwds)

import astropy.table

_translate = {
    "apogee_id": "obj",
    "apred_vers": "apred"

}
names = [_translate.get(k, k) for k in column_names["star"]]

star_table = astropy.table.Table(
    rows=rows["star"], 
    names=names
)



for table_name, v in rows.items():
    if len(v) == 0:
        print(f"Skipping {table_name}")
        continue

    t = astropy.table.Table(rows=v, names=column_names[table_name])

    if table_name != "star":
        t = astropy.table.join(
            t, 
            star_table,
            table_names=[table_name.split(".")[0], "star"]
        )

    filename = f"DailyStar-{mjd}-{table_name}.csv"
    t.write(filename)

    print(filename)







raise a

task = DistributeAnalysisGivenApStarFile(**star_kwds)

astra.build(
    [task],
)


raise a


