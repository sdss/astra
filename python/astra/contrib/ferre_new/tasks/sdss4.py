
import multiprocessing as mp
from astra.utils import log
from astra.tasks.io.sdss4 import SDSS4ApStarFile
from astra.contrib.ferre_new.tasks.aspcap import (
    ApStarMixinBase, 
    FerreGivenApStarFileBase, 
    InitialEstimateOfStellarParametersGivenApStarFileBase,
    CreateMedianFilteredApStarFileBase, 
    EstimateStellarParametersGivenApStarFileBase,
    EstimateChemicalAbundanceGivenApStarFileBase, 
    CheckRequirementsForChemicalAbundancesGivenApStarFileBase,
    EstimateChemicalAbundancesGivenApStarFileBase
)


class SDSS4ApStarMixin(ApStarMixinBase, SDSS4ApStarFile):

    """ A mix-in class for SDSS-V ApStar file. """
    
    @property
    def ferre_task_factory(self):
        return FerreGivenSDSS4ApStarFile
    
    @property
    def observation_task_factory(self):
        return SDSS4ApStarFile
    
    @property
    def initial_estimate_task_factory(self):
        return InitialEstimateOfStellarParametersGivenSDSS4ApStarFile
    
    @property
    def stellar_parameters_task_factory(self):
        return EstimateStellarParametersGivenSDSS4ApStarFile

    @property
    def chemical_abundance_task_factory(self):
        return EstimateChemicalAbundanceGivenSDSS4ApStarFile



class FerreGivenSDSS4ApStarFile(SDSS4ApStarMixin, FerreGivenApStarFileBase):

    """ Execute FERRE given an SDSS-V ApStar file. """

    pass


class InitialEstimateOfStellarParametersGivenSDSS4ApStarFile(InitialEstimateOfStellarParametersGivenApStarFileBase, SDSS4ApStarMixin):

    """ Get an initial estimate of stellar parameters given an SDSS-IV ApStar file. """

    pass



class CreateMedianFilteredSDSS4ApStarFile(CreateMedianFilteredApStarFileBase, SDSS4ApStarMixin):

    """ Create a median-filtered continuum-normalized spectrum for an SDSS-V ApStar file. """

    pass


class EstimateStellarParametersGivenSDSS4ApStarFile(EstimateStellarParametersGivenApStarFileBase, SDSS4ApStarMixin):

    """ Estimate stellar parameters given a SDSS-V ApStar file. """

    # Here we overwrite the observation_task_factory (from ApStarFile) so that
    # we use the median-filtered ApStarFile for stellar parameter determination.
    @property
    def observation_task_factory(self):
        return CreateMedianFilteredSDSS4ApStarFile


class EstimateChemicalAbundanceGivenSDSS4ApStarFile(EstimateChemicalAbundanceGivenApStarFileBase, SDSS4ApStarMixin):

    # Here we overwrite the observation_task_factory (from ApStarFile) so that
    # we use the median-filtered  ApStarFile for chemical abundance determination.
    @property
    def observation_task_factory(self):
        return CreateMedianFilteredSDSS4ApStarFile


class CheckRequirementsForChemicalAbundancesGivenSDSS4ApStarFile(CheckRequirementsForChemicalAbundancesGivenApStarFileBase, SDSS4ApStarMixin):

    """ Check requirements for estimating chemical abundances given an  ApStar file. """

    @property
    def observation_task_factory(self):
        return CreateMedianFilteredSDSS4ApStarFile

     

def _async_run_ferre_given_apstar_file(kwds):
    try:
        t = FerreGivenSDSS4ApStarFile(**kwds)
        if not t.complete():
            t.run()
    
    except:
        log.exception(f"Exception failed when trying to run {t}: {kwds}")
        raise


class EstimateChemicalAbundancesGivenSDSS4ApStarFile(EstimateChemicalAbundancesGivenApStarFileBase, SDSS4ApStarMixin):

    """ Estimate chemical abundances given ApStar file. """

    # Here we overwrite the observation_task_factory (from ApStarFile) so that
    # we use the median-filtered  ApStarFile for chemical abundance determination.
    @property
    def observation_task_factory(self):
        return CreateMedianFilteredSDSS4ApStarFile


    def requires(self):
        return self.clone(CheckRequirementsForChemicalAbundancesGivenSDSS4ApStarFile)
    

    def submit_jobs(self, submit_kwds):
        if self.use_slurm:
            with mp.Pool(self.max_asynchronous_slurm_jobs) as p:
                p.map(_async_run_ferre_given_apstar_file, submit_kwds)
        else:
            _ = list(map(_async_run_ferre_given_apstar_file, submit_kwds))
