
from astra.tasks.io.base import BaseTask, LocalTargetTask, SDSSDataModelTask
from astra.tasks.io.sdss5 import (ApVisitFile, ApStarFile, SDSS5DataModelTask)
from astra.tasks.io.sdss4 import (
    ApVisitFile as SDSS4ApVisitFile,
    ApStarFile as SDSS4ApStarFile,
    SpecFile as SDSS4SpecFile,
    AllStarFile as SDSS4AllStarFile,
    AllVisitSum as SDSS4AllVisitSum
)