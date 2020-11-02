
import astra
from luigi.task import WrapperTask
from astra.utils import batcher
from astra.tasks.io import ApStarFile
from astra.tasks.targets import LocalTarget
from astra.tasks.daily import (GetDailyApStarFiles, get_visits, get_visits_given_star, get_stars)
from astra.contrib.classifier.tasks.test import (
    ClassifySourceGivenApVisitFile,
    ClassifySourceGivenApStarFile
)
from astra.contrib.apogeenet.tasks import EstimateStellarParametersGivenApStarFile
from astra.contrib.thepayne.tasks import EstimateStellarParametersGivenNormalisedApStarFile

# Note that many parameters are set in sdss5.cfg
mjd = 59146


# Batch all apVisits together.
apvisit_kwds = batcher(get_visits(mjd))

# Classify task.
classify = ClassifySourceGivenApVisitFile(**apvisit_kwds)


star_kwds = batcher(list(get_stars(mjd=mjd))[2:3])

foo = EstimateStellarParametersGivenNormalisedApStarFile(**star_kwds)

raise a


class DistributeAnalysisGivenApStarFile(ApStarFile):

    def requires(self):
        return ClassifySourceGivenApStarFile(**self.get_common_param_kwargs(ClassifySourceGivenApStarFile))
    

    def run(self):
        for task in self.get_batch_tasks():
            
            print(f"Working on {task}")

            # Get the classification
            classification = ClassifySourceGivenApStarFile(
                    **task.get_common_param_kwargs(ClassifySourceGivenApStarFile)
                ).output().read(as_dict=True)

            print(f"Has classification {classification}")

            task_factories = []
            if classification["lp_yso"] > 0.7:
                task_factories.append(EstimateStellarParametersGivenApStarFile)
               
            if classification["lp_hotstar"] > 0.7:
                None
            
            if classification["lp_fgkm"] > 0.7:
                task_factories.append(EstimateStellarParametersGivenNormalisedApStarFile)

            
            for task_factory in task_factories:
                r = yield task_factory(**task.get_common_param_kwargs(task_factory))


    def output(self):
        return LocalTarget(self.task_id)


# [X] Outputs from APOGEENet to database, and [-] saved pkl file.
# [ ] Execute FERRE... outputs to database, and saved pkl file.
# [ ] Execute TC... outputs to database, and saved pkl file.
# [X] Execute TP... outputs to database, and saved pkl file.

# [ ] Put everything into a fits table: apVisit outputs per MJD?
# [ ] Put everything into a fits table: apStar outputs per MJD?

t = DistributeAnalysisGivenApStarFile(**star_kwds)

astra.build(
    [t],
    local_scheduler=True
)


raise a


for star_kwds in get_stars(mjd=mjd):

    visit_kwds = list(get_visits_given_star(star_kwds["obj"], star_kwds["apred"]))

    raise a




