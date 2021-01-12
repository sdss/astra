
import multiprocessing as mp
from astra.tasks.io.sdss5 import ApStarFile
from astra.tasks.targets import AstraSource
from astra.contrib.ferre.tasks.targets import FerreResult

from astra.contrib.ferre.tasks.aspcap_refactor import (
    ApStarMixinBase, 
    FerreGivenApStarFileBase, 
    InitialEstimateOfStellarParametersGivenApStarFileBase,
    CreateMedianFilteredApStarFileBase, 
    EstimateStellarParametersGivenApStarFileBase,
    EstimateChemicalAbundanceGivenApStarFileBase, 
    CheckRequirementsForChemicalAbundancesGivenApStarFileBase,
    EstimateChemicalAbundancesGivenApStarFileBase
)


class SDSS5ApStarMixin(ApStarMixinBase, ApStarFile):

    """ A mix-in class for SDSS-V ApStar file. """
    
    @property
    def ferre_task_factory(self):
        return FerreGivenSDSS5ApStarFile
    
    @property
    def observation_task_factory(self):
        return ApStarFile
    
    @property
    def initial_estimate_task_factory(self):
        return InitialEstimateOfStellarParametersGivenSDSS5ApStarFile
    
    @property
    def stellar_parameters_task_factory(self):
        return EstimateStellarParametersGivenSDSS5ApStarFile

    @property
    def chemical_abundance_task_factory(self):
        return EstimateChemicalAbundanceGivenSDSS5ApStarFile
    

    def output(self):
        """ Outputs of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        requirements = {
            "database": FerreResult(self, table_name="ferre"),
            "AstraSource": AstraSource(self)
        }
        if not self.write_source_output:
            requirements.pop("AstraSource")
        return requirements


class FerreGivenSDSS5ApStarFile(SDSS5ApStarMixin, FerreGivenApStarFileBase):

    """ Execute FERRE given an SDSS-V ApStar file. """

    pass




class InitialEstimateOfStellarParametersGivenSDSS5ApStarFile(InitialEstimateOfStellarParametersGivenApStarFileBase, SDSS5ApStarMixin):

    """ Get an initial estimate of stellar parameters given an SDSS-IV ApStar file. """

    output_table_name = "ferre_apstar_initial"



class CreateMedianFilteredSDSS5ApStarFile(CreateMedianFilteredApStarFileBase, SDSS5ApStarMixin):

    """ Create a median-filtered continuum-normalized spectrum for an SDSS-V ApStar file. """

    pass


class EstimateStellarParametersGivenSDSS5ApStarFile(EstimateStellarParametersGivenApStarFileBase, SDSS5ApStarMixin):

    """ Estimate stellar parameters given a SDSS-V ApStar file. """

    output_table_name = "ferre_apstar"
    # Here we overwrite the observation_task_factory (from ApStarFile) so that
    # we use the median-filtered  ApStarFile for stellar parameter determination.
    observation_task_factory = CreateMedianFilteredSDSS5ApStarFile


class EstimateChemicalAbundanceGivenSDSS5ApStarFile(EstimateChemicalAbundanceGivenApStarFileBase, SDSS5ApStarMixin):

    # Here we overwrite the observation_task_factory (from ApStarFile) so that
    # we use the median-filtered  ApStarFile for chemical abundance determination.
    observation_task_factory = CreateMedianFilteredSDSS5ApStarFile
    output_table_name = "ferre_apstar_abundances"


class CheckRequirementsForChemicalAbundancesGivenSDSS5ApStarFile(CheckRequirementsForChemicalAbundancesGivenApStarFileBase, SDSS5ApStarMixin):

    """ Check requirements for estimating chemical abundances given an  ApStar file. """

    observation_task_factory = CreateMedianFilteredSDSS5ApStarFile
     

def _async_run_ferre_given_apstar_file(kwds):
    try:
        t = FerreGivenSDSS5ApStarFile(**kwds)
        if not t.complete():
            t.run()
    
    except:
        log.exception(f"Exception failed when trying to run {t}: {kwds}")
        raise


class EstimateChemicalAbundancesGivenSDSS5ApStarFile(EstimateChemicalAbundancesGivenApStarFileBase, SDSS5ApStarMixin):

    """ Estimate chemical abundances given ApStar file. """

    # Here we overwrite the observation_task_factory (from ApStarFile) so that
    # we use the median-filtered  ApStarFile for chemical abundance determination.
    observation_task_factory = CreateMedianFilteredSDSS5ApStarFile
    output_table_name = "ferre_apstar_abundance"

    def requires(self):
        return self.clone(CheckRequirementsForChemicalAbundancesGivenSDSS5ApStarFile)
    
    def submit_jobs(self, submit_kwds):
        if self.use_slurm:
            with mp.Pool(self.max_asynchronous_slurm_jobs) as p:
                p.map(_async_run_ferre_given_apstar_file, submit_kwds)
        else:
            _ = list(map(_async_run_ferre_given_apstar_file, submit_kwds))
