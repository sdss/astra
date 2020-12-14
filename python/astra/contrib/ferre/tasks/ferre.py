import os
import numpy as np
import pickle
from astra import Parameter
from astropy import units as u
from astropy.table import Table
from collections import OrderedDict

from astra.tasks.targets import LocalTarget
from astra.tasks.io.sdss5 import ApStarFile
from astra.tools.spectrum import Spectrum1D
from astra.tools.spectrum.writers import create_astra_source
from astra.utils import log

from astra.contrib.ferre.core import (Ferre, FerreQueue)
from astra.contrib.ferre import utils
from astra.contrib.ferre.tasks.mixin import (FerreMixin, BaseFerreMixin, ApStarMixin, GridHeaderFileMixin)
from astra.contrib.ferre.tasks.targets import (FerreResult, GridHeaderFile)

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter





class EstimateStellarParametersGivenApStarFileBase(FerreMixin, GridHeaderFileMixin):

    """ Use FERRE to estimate stellar parameters given a single spectrum. """

    max_batch_size = 5000

    def run(self):
        """ Execute the task. """
        N = self.get_batch_size()
        log.info(f"Running {N} task{('s in batch mode' if N > 1 else '')}: {self}")

        # Load spectra.
        spectra = self.read_input_observations()
        
        # Directory keywords. By default use a scratch location.
        directory_kwds = self.directory_kwds or {}
        directory_kwds.setdefault(
            "dir",
            os.path.join(self.output_base_dir, "scratch")
        )
        
        FerreProcess = FerreQueue if self.use_queue else Ferre

        # Load the model.
        model = FerreProcess(
            grid_header_path=self.input()["grid_header"].path,
            interpolation_order=self.interpolation_order,
            init_algorithm_flag=self.init_algorithm_flag,
            error_algorithm_flag=self.error_algorithm_flag,
            continuum_flag=self.continuum_flag,
            continuum_order=self.continuum_order,
            continuum_reject=self.continuum_reject,
            continuum_observations_flag=self.continuum_observations_flag,
            full_covariance=self.full_covariance,
            pca_project=self.pca_project,
            pca_chi=self.pca_chi,
            frozen_parameters=self.frozen_parameters,
            optimization_algorithm_flag=self.optimization_algorithm_flag,
            wavelength_interpolation_flag=self.wavelength_interpolation_flag,
            lsf_shape_flag=self.lsf_shape_flag,
            input_weights_path=self.input_weights_path,
            input_lsf_path=self.input_lsf_path,
            # TODO: Consider what is most efficient. Consider removing as a task parameter.
            use_direct_access=self.use_direct_access,
            #(True if N <= self.max_batch_size_for_direct_access else False),
            n_threads=self.n_threads,
            debug=self.debug,
            directory_kwds=directory_kwds,
            executable=self.ferre_executable,
            # Be able to overwrite FERRE keywords directly.
            ferre_kwds=self.ferre_kwds               
        )

        # Star names to monitor output.
        names = [f"{i:.0f}_{telescope}_{obj}" for i, (telescope, obj) in enumerate(zip(self.telescope, self.obj))]

        # Get initial parameter estimates.
        p_opt, p_opt_err, meta = results = model.fit(
            spectra,
            initial_parameters=self.initial_parameters,
            full_output=True,
            names=names
        )

        for i, (task, spectrum) in enumerate(zip(self.get_batch_tasks(), spectra)):
            # Write result to database.
            result = dict(zip(model.parameter_names, p_opt[i]))
            result.update(
                log_snr_sq=meta["log_snr_sq"][i],
                log_chisq_fit=meta["log_chisq_fit"][i],
            )
            task.output()["database"].write(result)

            # Write AstraSource object.
            # TODO: What if the continuum was not done by FERRE?
            #       We would get back an array of ones but in truth it was normalized..
            continuum = meta["continuum"][i]
            
            task.output()["AstraSource"].write(
                spectrum,
                normalized_flux=spectrum.flux / continuum,
                normalized_ivar=continuum * spectrum.uncertainty.array * continuum,
                continuum=continuum,
                model_flux=meta["model_flux"][i],
                model_ivar=None,
                results_table=Table(rows=[result])
            )

        # We are done with this model.
        model.teardown()

        return results




class EstimateAbundanceGivenStellarParametersAndApStarFileBase(FerreGivenApStarFileBase):

    """ Use FERRE to estimate stellar parameters given a single spectrum. """

    element = Parameter()
    element_masks_dir = Parameter(default="/home/andy/data/sdss/astra-component-data/FERRE")

    thawed_dimension_name = Parameter(default="O Mg Si S Ca Ti")

    def requires(self):
        raise NotImplementedError("this should be implemented by sub-classes and include a 'stellar_parameters' key")

    

    def read_input_observations(self):
        """ Read the input observations. """

        raise a
        # Since apStar files contain the combined spectrum and all individual visits, we are going
        # to supply a data_slice parameter to only return the first spectrum.        
        spectra = []

        for task in self.get_batch_tasks():
            spectra.append(
                Spectrum1D.read(
                    task.input()["observation"].path,
                    data_slice=(slice(0, 1), slice(None))
                )
            )
        
        return spectra


    def run(self):
        """ Execute the task. """
        N = self.get_batch_size()
        log.info(f"Running {N} task{('s in batch mode' if N > 1 else '')}: {self}")

        # Load spectra.
        spectra = self.read_input_observations()
        
        # Directory keywords. By default use a scratch location.
        directory_kwds = self.directory_kwds or {}
        directory_kwds.setdefault(
            "dir",
            os.path.join(self.output_base_dir, "scratch")
        )
        
        FerreProcess = FerreQueue if self.use_queue else Ferre

        # We are going to OVERWRITE the input_weights_path here!
        input_weights_path = os.path.join(
            self.element_masks_dir,
            f"{self.element}.mask"
        )
        
        # We are going to IGNORE the frozen_parameters here!
        stellar_parameters = self.requires()["stellar_parameters"].output()["database"]
        
        #self.thawed_dimension_name

        # We are going to IGNORE init_algorithm_flag

        # Load the model.
        model = FerreProcess(
            grid_header_path=self.input()["grid_header"].path,
            interpolation_order=self.interpolation_order,
            #init_algorithm_flag=self.init_algorithm_flag,
            error_algorithm_flag=self.error_algorithm_flag,
            continuum_flag=self.continuum_flag,
            continuum_order=self.continuum_order,
            continuum_reject=self.continuum_reject,
            continuum_observations_flag=self.continuum_observations_flag,
            full_covariance=self.full_covariance,
            pca_project=self.pca_project,
            pca_chi=self.pca_chi,
            
            optimization_algorithm_flag=self.optimization_algorithm_flag,
            wavelength_interpolation_flag=self.wavelength_interpolation_flag,
            lsf_shape_flag=self.lsf_shape_flag,
            input_lsf_path=self.input_lsf_path,
            # TODO: Consider what is most efficient. Consider removing as a task parameter.
            use_direct_access=self.use_direct_access,
            #(True if N <= self.max_batch_size_for_direct_access else False),
            n_threads=self.n_threads,
            debug=self.debug,
            directory_kwds=directory_kwds,
            executable=self.ferre_executable,

            # Abundance-specific things:
            n_ties=0,
            type_tie=1,
            input_weights_path=input_weights_path,
            init_flag=0,
            init_algorithm_flag=None
        )

        # Star names to monitor output.
        names = [f"{i:.0f}_{telescope}_{obj}" for i, (telescope, obj) in enumerate(zip(self.telescope, self.obj))]

        # Get initial parameter estimates.
        p_opt, p_opt_err, meta = results = model.fit(
            spectra,
            # No initial parameters given for abundance determination.
            # TODO: Is this right?
            #initial_parameters=self.initial_parameters,
            full_output=True,
            names=names
        )

        for i, (task, spectrum) in enumerate(zip(self.get_batch_tasks(), spectra)):
            # Write result to database.
            result = dict(zip(model.parameter_names, p_opt[i]))
            result.update(
                log_snr_sq=meta["log_snr_sq"][i],
                log_chisq_fit=meta["log_chisq_fit"][i],
            )

            raise a

            task.output()["database"].write(result)

            # Write AstraSource object.
            # TODO: What if the continuum was not done by FERRE?
            #       We would get back an array of ones but in truth it was normalized..
            continuum = meta["continuum"][i]
            
            task.output()["AstraSource"].write(
                spectrum,
                normalized_flux=spectrum.flux / continuum,
                normalized_ivar=continuum * spectrum.uncertainty.array * continuum,
                continuum=continuum,
                model_flux=meta["model_flux"][i],
                model_ivar=None,
                results_table=Table(rows=[result])
            )

        # We are done with this model.
        model.teardown()

        return results





'''
class EstimateElementAbundanceGivenStellarParametersAndApStarFile(FerreMixin):

    element = 

    def requires(self):
        """ The requirements for this task. """
        requirements = dict(grid_header=GridHeaderFile(**self.get_common_param_kwargs(GridHeaderFile)))

        # If we are running in batch mode then the ApStar keywords will all be tuples, and we would have to
        # add the requirement for every single ApStarFile. That adds overhead, and we don't need to do it:
        # Astra will manage the batches to be expanded into individual tasks.
        if not self.is_batch_mode:
            requirements.update(observation=self.observation_task(**self.get_common_param_kwargs(self.observation_task)))
        return requirements


    def read_input_observations(self):
        """ Read the input observations. """
        # Since apStar files contain the combined spectrum and all individual visits, we are going
        # to supply a data_slice parameter to only return the first spectrum.        
        spectra = []

        # TODO: Consider a better way to do this.
        keys = ("pseudo_continuum_normalized_observation", "observation")
        for task in self.get_batch_tasks():
            for key in keys:
                try:
                    spectrum = Spectrum1D.read(
                        task.input()[key].path,
                        data_slice=(slice(0, 1), slice(None))
                    )
                except KeyError:
                    continue

                else:
                    spectra.append(spectrum)
                    break
            
            else:
                raise
            
        return spectra
    
    

    def run(self):
        """ Execute the task. """
        N = self.get_batch_size()
        log.info(f"Running {N} task{('s in batch mode' if N > 1 else '')}: {self}")

        # Load spectra.
        spectra = self.read_input_observations()
        
        # Directory keywords. By default use a scratch location.
        directory_kwds = self.directory_kwds or {}
        directory_kwds.setdefault(
            "dir",
            os.path.join(self.output_base_dir, "scratch")
        )
        
        FerreProcess = FerreQueue if self.use_queue else Ferre

        # Load the model.
        model = FerreProcess(
            grid_header_path=self.input()["grid_header"].path,
            interpolation_order=self.interpolation_order,
            init_algorithm_flag=self.init_algorithm_flag,
            error_algorithm_flag=self.error_algorithm_flag,
            continuum_flag=self.continuum_flag,
            continuum_order=self.continuum_order,
            continuum_reject=self.continuum_reject,
            continuum_observations_flag=self.continuum_observations_flag,
            full_covariance=self.full_covariance,
            pca_project=self.pca_project,
            pca_chi=self.pca_chi,
            frozen_parameters=self.frozen_parameters,
            optimization_algorithm_flag=self.optimization_algorithm_flag,
            wavelength_interpolation_flag=self.wavelength_interpolation_flag,
            lsf_shape_flag=self.lsf_shape_flag,
            input_weights_path=self.input_weights_path,
            input_lsf_path=self.input_lsf_path,
            # TODO: Consider what is most efficient. Consider removing as a task parameter.
            use_direct_access=self.use_direct_access,
            #(True if N <= self.max_batch_size_for_direct_access else False),
            n_threads=self.n_threads,
            debug=self.debug,
            directory_kwds=directory_kwds,
            executable=self.ferre_executable,
            n_ties=0,
            type_tie=1
        )

        # Star names to monitor output.
        names = [f"{i:.0f}_{telescope}_{obj}" for i, (telescope, obj) in enumerate(zip(self.telescope, self.obj))]

        # Get initial parameter estimates.
        p_opt, p_opt_err, meta = results = model.fit(
            spectra,
            initial_parameters=self.initial_parameters,
            full_output=True,
            names=names
        )

        for i, (task, spectrum) in enumerate(zip(self.get_batch_tasks(), spectra)):
            # Write result to database.
            result = dict(zip(model.parameter_names, p_opt[i]))
            result.update(
                log_snr_sq=meta["log_snr_sq"][i],
                log_chisq_fit=meta["log_chisq_fit"][i],
            )
            task.output()["database"].write(result)

            # Write AstraSource object.
            # TODO: What if the continuum was not done by FERRE?
            #       We would get back an array of ones but in truth it was normalized..
            continuum = meta["continuum"][i]
            
            task.output()["AstraSource"].write(
                spectrum,
                normalized_flux=spectrum.flux / continuum,
                normalized_ivar=continuum * spectrum.uncertainty.array * continuum,
                continuum=continuum,
                model_flux=meta["model_flux"][i],
                model_ivar=None,
                results_table=Table(rows=[result])
            )

        # We are done with this model.
        model.teardown()

        return resultsle(FerreMixin):

    

    def requires(self):
        """ The requirements for this task. """
        requirements = dict(grid_header=GridHeaderFile(**self.get_common_param_kwargs(GridHeaderFile)))

        if not self.is_batch_mode:
            requirements.update(observation=self.observation_task(**self.get_common_param_kwargs(self.observation_task)))
        return requirements


    def read_input_observations(self):
        """ Read the input observations. """
        # Since apStar files contain the combined spectrum and all individual visits, we are going
        # to supply a data_slice parameter to only return the first spectrum.        
        spectra = []

        # TODO: Consider a better way to do this.
        keys = ("pseudo_continuum_normalized_observation", "observation")
        for task in self.get_batch_tasks():
            for key in keys:
                try:
                    spectrum = Spectrum1D.read(
                        task.input()[key].path,
                        data_slice=(slice(0, 1), slice(None))
                    )
                except KeyError:
                    continue

                else:
                    spectra.append(spectrum)
                    break
            
            else:
                raise
            
        return spectra
    
    

    def run(self):
        """ Execute the task. """
        N = self.get_batch_size()
        log.info(f"Running {N} task{('s in batch mode' if N > 1 else '')}: {self}")

        # Load spectra.
        spectra = self.read_input_observations()
        
        # Directory keywords. By default use a scratch location.
        directory_kwds = self.directory_kwds or {}
        directory_kwds.setdefault(
            "dir",
            os.path.join(self.output_base_dir, "scratch")
        )
        
        FerreProcess = FerreQueue if self.use_queue else Ferre

        # Load the model.
        model = FerreProcess(
            grid_header_path=self.input()["grid_header"].path,
            interpolation_order=self.interpolation_order,
            init_algorithm_flag=self.init_algorithm_flag,
            error_algorithm_flag=self.error_algorithm_flag,
            continuum_flag=self.continuum_flag,
            continuum_order=self.continuum_order,
            continuum_reject=self.continuum_reject,
            continuum_observations_flag=self.continuum_observations_flag,
            full_covariance=self.full_covariance,
            pca_project=self.pca_project,
            pca_chi=self.pca_chi,
            frozen_parameters=self.frozen_parameters,
            optimization_algorithm_flag=self.optimization_algorithm_flag,
            wavelength_interpolation_flag=self.wavelength_interpolation_flag,
            lsf_shape_flag=self.lsf_shape_flag,
            input_weights_path=self.input_weights_path,
            input_lsf_path=self.input_lsf_path,
            # TODO: Consider what is most efficient. Consider removing as a task parameter.
            use_direct_access=self.use_direct_access,
            #(True if N <= self.max_batch_size_for_direct_access else False),
            n_threads=self.n_threads,
            debug=self.debug,
            directory_kwds=directory_kwds,
            executable=self.ferre_executable                 
        )

        # Star names to monitor output.
        names = [f"{i:.0f}_{telescope}_{obj}" for i, (telescope, obj) in enumerate(zip(self.telescope, self.obj))]

        # Get initial parameter estimates.
        p_opt, p_opt_err, meta = results = model.fit(
            spectra,
            initial_parameters=self.initial_parameters,
            full_output=True,
            names=names
        )

        for i, (task, spectrum) in enumerate(zip(self.get_batch_tasks(), spectra)):
            # Write result to database.
            result = dict(zip(model.parameter_names, p_opt[i]))
            result.update(
                log_snr_sq=meta["log_snr_sq"][i],
                log_chisq_fit=meta["log_chisq_fit"][i],
            )
            task.output()["database"].write(result)

            # Write AstraSource object.
            # TODO: What if the continuum was not done by FERRE?
            #       We would get back an array of ones but in truth it was normalized..
            continuum = meta["continuum"][i]
            
            task.output()["AstraSource"].write(
                spectrum,
                normalized_flux=spectrum.flux / continuum,
                normalized_ivar=continuum * spectrum.uncertainty.array * continuum,
                continuum=continuum,
                model_flux=meta["model_flux"][i],
                model_ivar=None,
                results_table=Table(rows=[result])
            )

        # We are done with this model.
        model.teardown()

        return results

'''