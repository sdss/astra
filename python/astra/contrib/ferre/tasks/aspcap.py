
""" Tasks for reproducing the functionality of ASPCAP in SDSS-IV. """

import numpy as np
import os
import pickle
from astropy.io.fits import getheader
from glob import glob
from tqdm import tqdm
from luigi import WrapperTask
from luigi.task import flatten

from sdss_access import SDSSPath
from astropy.nddata.nduncertainty import InverseVariance

import astra
from astra.utils import log
from astra.tools.spectrum import Spectrum1D
from astra.contrib.ferre import utils
from astra.contrib.ferre.tasks.mixin import (
    ApStarMixin, BaseFerreMixin, DispatcherMixin, FerreMixin, SPECLIB_DIR
)
from astra.contrib.ferre.tasks.ferre import EstimateStellarParametersGivenApStarFile
from astra.contrib.ferre.continuum import median_filtered_correction
from astra.contrib.ferre.tasks import ApStarFile
from astra.tasks.targets import LocalTarget
from astra.contrib.ferre.tasks.targets import FerreResult


class DispatchFerreTasks(DispatcherMixin):
    
    """ A dispatcher task to distribute ApStarFiles across multiple grids. """

    def dispatcher(self):
        """ A generator that yields grid keywords that are suitable for the given ApStarFile. """
        common_kwds = self.get_common_param_kwargs(self.task_factory)
        with tqdm(desc="Dispatching", total=self.get_batch_size()) as pbar:
            for iteration, kwds in self._dispatcher():
                pbar.update(iteration)
                yield { **common_kwds, **kwds }
            

    def _dispatcher(self):

        # Header stuff first.
        date_str = self.grid_creation_date.strftime("%y%m%d") if self.grid_creation_date is not None else "*"
        header_paths = glob(
            os.path.join(
                SPECLIB_DIR,
                self.radiative_transfer_code,
                self.model_photospheres,
                self.isotopes,
                f"t{self.gd}{self.spectral_type}_{date_str}_lsf{self.lsf}_{self.aspcap}",
                f"p_apst{self.gd}{self.spectral_type}_{date_str}_lsf{self.lsf}_{self.aspcap}_012_075.hdr"
            )
        )

        # Get grid limits.
        grid_limits = utils.parse_grid_limits(header_paths)

        # Common keywords.
        common_kwds = self.get_common_param_kwargs(self.task_factory)
        
        sdss_paths = {}
        for i, batch_kwds in enumerate(self.get_batch_task_kwds(include_non_batch_keywords=False)):
            try:
                sdss_path = sdss_paths[self.release]
            except KeyError:
                sdss_paths[self.release] = sdss_path = SDSSPath(
                    release=self.release,
                    public=self.public,
                    mirror=self.mirror,
                    verbose=self.verbose
                )

            path = sdss_path.full("apStar", **batch_kwds)

            try:
                header = getheader(path)
                header_dict = {
                    "MEANFIB": header["MEANFIB"],
                    "RV_TEFF": utils.safe_read_header(header, ("RV_TEFF", "RVTEFF")),
                    "RV_LOGG": utils.safe_read_header(header, ("RV_LOGG", "RVLOGG")),
                    "RV_FEH": utils.safe_read_header(header, ("RV_FEH", "RVFEH"))
                }

            except:
                log.exception("Exception:")
                continue

            else:
                teff = utils.safe_read_header(header, ("RV_TEFF", "RVTEFF"))
                logg = utils.safe_read_header(header, ("RV_LOGG", "RVLOGG"))
                fe_h = utils.safe_read_header(header, ("RV_FEH", "RVFEH"))

                batch_kwds.update(
                    initial_parameters={
                        "TEFF": teff,
                        "LOGG": logg,
                        "METALS": fe_h,
                        "C": 0,
                        "N": 0,
                        "LOG10VDOP": utils.approximate_log10_microturbulence(logg),
                        "O Mg Si S Ca Ti": 0,
                        "LGVSINI": 0
                    }
                )

                any_suitable_grids = False
                for path, grid_kwds in utils.yield_suitable_grids(header_dict, grid_limits):
                    any_suitable_grids = True
                    # We yield an integer so we can see progress of unique objects.
                    yield (i, { **common_kwds, **batch_kwds, **grid_kwds })

                if not any_suitable_grids:
                    log.warn(f"No suitable grids found for apStar keywords: {batch_kwds} with headers {header_dict}")



    def requires(self):
        """ Requirements of this task. """
        try:
            self._requirements
        except AttributeError:
            self._requirements = [self.task_factory(**kwds) for kwds in self.dispatcher()]
            
        for requirement in self._requirements:
            yield requirement


    def on_success(self):
        # Overwrite the inherited method that will mark this wrapper task as done and never re-run it.
        pass





class EstimateStellarParametersGivenMedianFilteredApStarFile(EstimateStellarParametersGivenApStarFile):

    """
    Estimate the stellar parameters given a median filtered ApStarFile. 
    
    This task requires a previous estimate of stellar parameters, and uses the model spectrum from that
    estimate to perform a correction to the continuum normalisation. This largely reproduces the ASPCAP
    functionality that was used for SDSS-IV Data Release 16.

    This task requires the same parameters as `EstimateStellarParametersGivenApStarFile`,
    and the following parameters:

    :param median_filter_width: (optional)
        The median width of the filter (default: 151).
    
    :param bad_minimum_flux: (optional)
        The minimum flux value before considering a pixel as having erroneous flux (default: 0.01).

    :param non_finite_err_value: (optional)
        The value to assign to pixels with non-finite fluxes or errors.
    """

    median_filter_width = astra.IntParameter(default=151)
    bad_minimum_flux = astra.FloatParameter(default=0.01)
    non_finite_err_value = astra.FloatParameter(default=1e10)
    
    def requires(self):
        """ The requirements of this task, which include the previous estimate. """
        requirements = super(EstimateStellarParametersGivenMedianFilteredApStarFile, self).requires()
        requirements.update(
            previous_estimate=EstimateStellarParametersGivenApStarFile(**self.get_common_param_kwargs(EstimateStellarParametersGivenApStarFile))
        )
        return requirements
        
    
    def read_input_observations(self):
        """ Read the input observations and return median-filtered-corrected spectra. """
        spectra = []

        # TODO: dont do this.
        with open("/home/ubuntu/data/sdss/astra-components/wavelength_mask.pkl", "rb") as fp:
            mask = pickle.load(fp)

        for task in self.get_batch_tasks():
            observation = Spectrum1D.read(
                task.input()["observation"].path,
                data_slice=(slice(0, 1), slice(None))
            )

            # Re-normalise using previous estimate.
            with open(task.input()["previous_estimate"]["spectrum"].path, "rb") as fp:
                result = pickle.load(fp)
            
            wavelength = observation.wavelength.value[mask]
            flux = observation.flux.value[0][mask]

            continuum = flux / result["normalized_observed_flux"]
            normalised_observed_flux_error = observation.uncertainty.array[0][mask]**-0.5 / continuum

            mfc = median_filtered_correction(
                observation.wavelength.value[mask],
                normalised_observed_flux=result["normalized_observed_flux"],
                normalised_observed_flux_err=normalised_observed_flux_error,
                normalised_model_flux=result["model_spectrum"],
                width=self.median_filter_width,
                bad_minimum_flux=self.bad_minimum_flux,
                non_finite_err_value=self.non_finite_err_value
            )
            '''
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            #ax.plot(wavelength, result["normalized_observed_flux"], c="k")
            ax.plot(wavelength, flux, c="k")
            ax.plot(wavelength, flux / mfc[0][0], c="tab:blue")
            raise a
            '''
            observation.flux[0, mask] /= mfc[0][0]
            v = observation.uncertainty.array
            v[0, mask] *= mfc[0][0] * mfc[0][0]
            observation.uncertainty = InverseVariance(v)
            spectra.append(observation)

        return spectra



class InitialEstimateOfStellarParametersGivenApStarFile(DispatchFerreTasks):

    """
    A task that dispatches an ApStarFile to multiple FERRE grids, in a similar way done by ASPCAP in SDSS-IV.
    """

    task_factory = EstimateStellarParametersGivenApStarFile

    def dispatcher(self):
        """ Dispatch sources to multiple FERRE grids. """

        # We want to add some custom logic for the initial dispatching.
        # Specifically, we want to set LOG10VDOP to be frozen for all grids,
        # and for dwarf grids we want to fix C and N to be zero.
        for kwds in super(InitialEstimateOfStellarParametersGivenApStarFile, self).dispatcher():
            kwds.update(frozen_parameters=dict(LOG10VDOP=None))
            if kwds["gd"] == "d":
                kwds["frozen_parameters"].update(C=0.0, N=0.0)
            yield kwds


    def run(self):
        """ Execute this task. """

        # Get the best task among the initial estimates.
        uid = lambda task: "_".join([f"{getattr(task, pn)}" for pn in ApStarFile.batch_param_names()])

        best_tasks = {}
        for task in self.requires():
            key = uid(task)
            best_tasks.setdefault(key, (np.inf, None))

            output = task.output()["database"].read(as_dict=True)

            log_chisq_fit = output["log_chisq_fit"]

            # Penalise chi-sq in the same way they did for DR16.
            # See github.com/sdss/apogee/python/apogee/aspcap/aspcap.py#L492
            if task.spectral_type == "GK" and output["TEFF"] < 3985:
                # \chi^2 *= 10
                log_chisq_fit += np.log(10)

            if log_chisq_fit < best_tasks[key][0]:

                kwds = task.param_kwargs.copy()
                # Set C, N to be frozen if this is a dwarf grid.
                frozen_parameters = None if kwds["gd"] == "g" else dict(C=0.0, N=0.0)
                # Update with initial estimate from previous task.
                kwds.update(
                    frozen_parameters=frozen_parameters,
                    # TODO: This is bad practice...
                    initial_parameters={ k: v for k, v in output.items() if k.upper() == k }
                )

                best_tasks[key] = (log_chisq_fit, kwds)

        # Write outputs for individual tasks
        for task in self.get_batch_tasks():
            key = uid(task)

            try:
                result = best_tasks[key]

            except KeyError:
                # No result for this source. 
                # Initial value from header is likely outside all grid bounds.
                result = (+np.inf, {})

            finally:
                with open(task.output().path, "wb") as fp:
                    pickle.dump(result, fp)

        return None


    def output(self):
        """ The outputs produced by this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        path = os.path.join(
            self.output_base_dir,
            # For SDSS-V:
            f"star/{self.telescope}/{int(self.healpix/1000)}/{self.healpix}/",
            # For SDSS-IV:
            #f"star/{self.telescope}/{self.field}/",
            f"apStar-{self.apred}-{self.telescope}-{self.obj}-{self.task_id}.pkl"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        return LocalTarget(path)




class IterativeEstimateOfStellarParametersGivenApStarFile(BaseFerreMixin, ApStarFile):

    """ 
    (Nearly) reproduce the steps used to estimate stellar parameters for the DR16 data release. 
    
    This task will read the headers of an ApStarFile, search the SPECLIB for grid header files,
    and generate tasks to estimate stellar parameters using all grids where the initial estimate
    of stellar parameters falls within the boundaries of the grid. 
    
    This could yield several estimates of stellar parameters, each with different grids. 
    Then this task will find the best among those estimates (based on a \chi^2 score), and
    and will use the result from the best estimate to correct the continuum normalisation.
    With the continuum-corrected spectra, a new task will estimate the stellar parameters
    from the grid where the best initial estimate was found, using the previous estimate
    and the continuum-corrected spectra.

    NOTE: Although Astra will schedule tasks and batch them together to be efficient,
          you should run this task with *many* ApStar keywords (i.e., batch together many
          ApStar objects at once). This will make sure that the final step of parameter
          estimation is efficient.
    """

    def requires(self):
        """ This task requires the initial estimates of stellar parameters from many grids. """
        return InitialEstimateOfStellarParametersGivenApStarFile(
            **self.get_common_param_kwargs(InitialEstimateOfStellarParametersGivenApStarFile)
        )


    def run(self):
        """ Execute the task. """

        tasks = []
        for previous_estimate in self.requires().output():
            with open(previous_estimate.path, "rb") as fp:
                log_chisq_fit, kwds = pickle.load(fp)

            if np.isfinite(log_chisq_fit):
                tasks.append(
                    EstimateStellarParametersGivenMedianFilteredApStarFile(**kwds)
                )

        yield tasks

        for task in tasks:

            # Get the result from the median filtered run.
            result = task.output()["database"].read(as_dict=True)
            ignore = task.get_param_names() + ["task_id"]

            result = { k: v or np.nan for k, v in result.items() if k not in ignore }

            for key, value in task.output().items():
                if key == "database":
                    # Write result to a symbolic task.
                    symbolic_task = self.__class__(**task.get_common_param_kwargs(self))
                    symbolic_task.output()[key].write(result)

                else:
                    # Write symbolic link.
                    source = value.path
                    destination = symbolic_task.output()[key].path
                    if os.path.exists(destination):
                        os.unlink(destination)
                    
                    os.symlink(source, destination)
                    
        return None


    def output(self):
        """ The outputs of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]

        path = os.path.join(
            self.output_base_dir,
            # For SDSS-V:
            f"star/{self.telescope}/{int(self.healpix/1000)}/{self.healpix}/",
            # For SDSS-IV:
            #f"star/{self.telescope}/{self.field}/",
            f"apStar-{self.apred}-{self.telescope}-{self.obj}-{self.task_id}.pkl"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        return {
            "database": FerreResult(self),
            "spectrum": LocalTarget(path)
        }

