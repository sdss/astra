import os
import numpy as np
import pickle
from astropy import units as u
from astropy.table import Table
from collections import OrderedDict

from astra.tasks.targets import LocalTarget
from astra.tools.spectrum import Spectrum1D
from astra.tools.spectrum.writers import create_astra_source
from astra.utils import log

from astra.contrib.ferre.core import (Ferre, FerreQueue)
from astra.contrib.ferre import utils
from astra.contrib.ferre.tasks.mixin import FerreMixin
from astra.contrib.ferre.tasks.targets import (FerreResult, GridHeaderFile)

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

class EstimateStellarParametersGivenApStarFileBase(FerreMixin):

    """ Use FERRE to estimate stellar parameters given a single spectrum. """

    max_batch_size = 1000
    max_batch_size_for_direct_access = 25 # TODO: Currently ignored.

    def requires(self):
        """ The requirements for this task. """
        requirements = dict(grid_header=GridHeaderFile(**self.get_common_param_kwargs(GridHeaderFile)))

        # If we are running in batch mode then the ApStar keywords will all be tuples, and we would have to
        # add the requirement for every single ApStarFile. That adds overhead, and we don't need to do it:
        # Astra will manage the batches to be expanded into individual tasks.
        if not self.is_batch_mode:
            requirements.update(observation=self.observation_task(**self.get_common_param_kwargs(self.observation_task)))
        return requirements


    def output(self):
        """ The output of the task. """
        raise RuntimeError("this should be over-written by the sub-classses")


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


    def plot(self, model, model_flux):

        width_ratios = np.array([np.ptp(wl.value) for wl in model.wavelength])
        width_ratios = width_ratios/np.min(width_ratios)

        fig, axes = plt.subplots(
            ncols=3, nrows=2, figsize=(40, 5),
            gridspec_kw=dict(width_ratios=width_ratios, height_ratios=[1, 5])
        )

        diff_axes = axes[0]
        plot_axes = axes[1]

        P = 0
        for i, (ax, ax_diff, wavelength) in enumerate(zip(plot_axes, diff_axes, model.wavelength)):
            p = wavelength.size
            ax.plot(
                wavelength, 
                model_flux[0][P:P+p],
                c="tab:red",
                lw=1,
            )

            if "normalized_input_flux" in meta:
                ax.plot(
                    wavelength,
                    meta["normalized_input_flux"][0][P:P+p],
                    c="#000000",
                    lw=1,
                )

                ax_diff.plot(
                    wavelength,
                    meta["normalized_input_flux"][0][P:P+p] - model_flux[0][P:P+p],
                    c="#000000",
                    lw=1,
                )

                s_index, e_index = spectrum.wavelength.value.searchsorted(wavelength[[0, -1]]) 
                
                continuum = spectrum.flux.value[0, s_index:e_index + 1] / meta["normalized_input_flux"][0][P:P+p]

                sigma = spectrum.uncertainty.array[0, s_index:e_index + 1]**-0.5 / continuum

                ax.fill_between(
                    wavelength, 
                    meta["normalized_input_flux"][0][P:P+p] - sigma,
                    meta["normalized_input_flux"][0][P:P+p] + sigma,
                    facecolor="#CCCCCC",
                    edgecolor="#CCCCCC",
                    zorder=-1
                )

                ax_diff.fill_between(
                    wavelength,
                    -sigma,
                    +sigma,
                    facecolor="#CCCCCC",
                    edgecolor="#CCCCCC",
                    zorder=-1
                )


            else:
                ax.plot(
                    spectrum.wavelength,
                    spectrum.flux.value[0, :],
                    c="#000000"
                )
        

            ax_diff.set_ylim(-0.1, 0.1)
            ax.set_ylim(0.5, 1.1)

            ax_diff.axhline(0, c="#666666", ls=":", zorder=-10)
            

            for ax_ in (ax_diff, ax):

                ax_.set_xlim(*wavelength.value[[0, -1]])

                if not ax.is_last_row():
                    ax.set_xticks([])

                if len(model.wavelength) > 0:
                    if not ax_.is_last_col():
                        ax_.spines["right"].set_visible(False)
                        ax_.tick_params(right=False, which="both")
                        
                    if not ax_.is_first_col():
                        ax_.spines["left"].set_visible(False)
                        ax_.tick_params(left=False, which="both")
                        ax_.yaxis.set_major_formatter(NullFormatter())


                # Plot cut-out markers
                cutOutkwargs = dict(transform=ax_.transAxes,color='k',
                                    clip_on=False)

                nregions = len(model.wavelength)
                dx = np.ones(nregions)
                skipdx = 0.015

                d = .015 # how big to make the diagonal lines in axes coordinates
                #d = 0.015 if ax.is_last_row() else 0.015/5
                skipdx = 0.015
                slope= 1./(dx[i]+0.2*skipdx)/3.
                slope = slope if ax.is_last_row() else 5 * slope 
                if i == 0 and not nregions == 1:
                    ax_.plot((1-slope*d,1+slope*d),(-d,+d), **cutOutkwargs)
                    ax_.plot((1-slope*d,1+slope*d),(1-d,1+d), **cutOutkwargs)
                elif i == (nregions-1) and not nregions == 1:
                    ax_.plot((-slope*d,+slope*d),(-d,+d), **cutOutkwargs)
                    ax_.plot((-slope*d,+slope*d),(1-d,1+d), **cutOutkwargs)
                elif not nregions == 1:
                    ax_.plot((1-slope*d,1+slope*d),(-d,+d), **cutOutkwargs)
                    ax_.plot((1-slope*d,1+slope*d),(1-d,1+d), **cutOutkwargs)
                    ax_.plot((-slope*d,+slope*d),(-d,+d), **cutOutkwargs)
                    ax_.plot((-slope*d,+slope*d),(1-d,1+d), **cutOutkwargs)

            P += p
        
        fig.tight_layout()
        return fig
