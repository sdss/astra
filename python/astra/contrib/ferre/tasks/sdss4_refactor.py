
import multiprocessing as mp
from astra.tasks.io.sdss4 import ApStarFile
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


class SDSS4ApStarMixin(ApStarMixinBase, ApStarFile):

    """ A mix-in class for SDSS-V ApStar file. """
    
    @property
    def ferre_task_factory(self):
        return FerreGivenSDSS4ApStarFile
    
    @property
    def observation_task_factory(self):
        return ApStarFile
    
    @property
    def initial_estimate_task_factory(self):
        return InitialEstimateOfStellarParametersGivenSDSS4ApStarFile
    
    @property
    def stellar_parameters_task_factory(self):
        return EstimateStellarParametersGivenSDSS4ApStarFile

    @property
    def chemical_abundance_task_factory(self):
        return EstimateChemicalAbundanceGivenSDSS4ApStarFile

    def output(self):
        """ Outputs of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        requirements = {
            "database": FerreResult(self, table_name="sdss4_ferre"),
            "AstraSource": AstraSource(self)
        }
        if not self.write_source_output:
            requirements.pop("AstraSource")
        return requirements


class FerreGivenSDSS4ApStarFile(SDSS4ApStarMixin, FerreGivenApStarFileBase):

    """ Execute FERRE given an SDSS-V ApStar file. """

    pass


class InitialEstimateOfStellarParametersGivenSDSS4ApStarFile(InitialEstimateOfStellarParametersGivenApStarFileBase, SDSS4ApStarMixin):

    """ Get an initial estimate of stellar parameters given an SDSS-IV ApStar file. """

    output_table_name = "sdss4_ferre_apstar_initial"



class CreateMedianFilteredSDSS4ApStarFile(CreateMedianFilteredApStarFileBase, SDSS4ApStarMixin):

    """ Create a median-filtered continuum-normalized spectrum for an SDSS-V ApStar file. """

    pass


class EstimateStellarParametersGivenSDSS4ApStarFile(EstimateStellarParametersGivenApStarFileBase, SDSS4ApStarMixin):

    """ Estimate stellar parameters given a SDSS-V ApStar file. """

    output_table_name = "sdss4_ferre_apstar"
    # Here we overwrite the observation_task_factory (from ApStarFile) so that
    # we use the median-filtered  ApStarFile for stellar parameter determination.
    observation_task_factory = CreateMedianFilteredSDSS4ApStarFile


class EstimateChemicalAbundanceGivenSDSS4ApStarFile(EstimateChemicalAbundanceGivenApStarFileBase, SDSS4ApStarMixin):

    # Here we overwrite the observation_task_factory (from ApStarFile) so that
    # we use the median-filtered  ApStarFile for chemical abundance determination.
    observation_task_factory = CreateMedianFilteredSDSS4ApStarFile
    output_table_name = "sdss4_ferre_apstar_abundances"


class CheckRequirementsForChemicalAbundancesGivenSDSS4ApStarFile(CheckRequirementsForChemicalAbundancesGivenApStarFileBase, SDSS4ApStarMixin):

    """ Check requirements for estimating chemical abundances given an  ApStar file. """

    observation_task_factory = CreateMedianFilteredSDSS4ApStarFile
     

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
    observation_task_factory = CreateMedianFilteredSDSS4ApStarFile
    output_table_name = "sdss4_ferre_apstar_abundance"

    def requires(self):
        return self.clone(CheckRequirementsForChemicalAbundancesGivenSDSS4ApStarFile)
    
    def submit_jobs(self, submit_kwds):
        if self.use_slurm:
            with mp.Pool(self.max_asynchronous_slurm_jobs) as p:
                p.map(_async_run_ferre_given_apstar_file, submit_kwds)
        else:
            _ = list(map(_async_run_ferre_given_apstar_file, submit_kwds))
