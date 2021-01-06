
import os
from astra.tasks.io.sdss4 import ApStarFile
from astra.tasks.targets import (LocalTarget, AstraSource)
from astra.contrib.ferre.tasks.aspcap import (
    ApStarMixinBase,
    FerreGivenApStarFileBase,
    InitialEstimateOfStellarParametersGivenApStarFileBase,
    EstimateStellarParametersGivenApStarFileBase
)
from astra.contrib.ferre.tasks.targets import SDSS4FerreResult as FerreResult





class ApStarMixin(ApStarMixinBase, ApStarFile):

    def requires(self):
        """ The requirements for this task. """
        # If we are running in batch mode then the ApStar keywords will all be tuples, and we would have to
        # add the requirement for every single ApStarFile. That adds overhead, and we don't need to do it:
        # Astra will manage the batches to be expanded into individual tasks.
        if self.is_batch_mode:
            return []
        return dict(observation=self.clone(ApStarFile))


    def output(self):
        """ Outputs of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        requirements = {
            "database": FerreResult(self),
            "AstraSource": AstraSource(self)
        }
        if not self.write_source_output:
            requirements.pop("AstraSource")
        return requirements

    
class FerreGivenApStarFile(ApStarMixin, FerreGivenApStarFileBase):

    """ A task to execute FERRE on a SDSS-IV ApStar file. """

    pass


class InitialEstimateOfStellarParametersGivenApStarFile(ApStarMixin, InitialEstimateOfStellarParametersGivenApStarFileBase):

    """ Estimate the stellar parameters of a source given an SDSS-IV ApStar file. """

    pass




class CreateMedianFilteredApStarFile(CreateMedianFilteredApStarFileBase, ApStarMixin):


    def requires(self):
        return {
            "observation": self.clone(ApStarFile),
            "initial_estimate": self.clone(InitialEstimateOfStellarParametersGivenApStarFile)
        }


    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        # TODO: To be defined by SDSS5/SDSS4 mixin
        new_path = AstraSource(self).path.replace("/AstraSource", "/ApStar")
        return LocalTarget(new_path)


class EstimateStellarParametersGivenApStarFile(ApStarMixin, EstimateStellarParametersGivenApStarFileBase):

    def requires(self):
        """
        The requirements of this task include a median-filtered ApStar file, and an initial
        estimate of the stellar parameters (based on a series of previous FERRE executions).
        """
        return {
            "observation": self.clone(CreateMedianFilteredApStarFile),
            "initial_estimate": self.clone(InitialEstimateOfStellarParametersGivenApStarFile)
        }

    