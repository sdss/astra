
import os
import numpy as np
from time import time
from astra.utils import log
from astra.tasks.slurm import slurmify
from astra.contrib.ferre_new.core import Ferre
from astra.contrib.ferre_new.tasks.mixin import (FerreMixin, SourceMixin)
from astra.contrib.ferre_new.utils import sanitise_parameter_names
from astropy.table import Table


class FerreBase(FerreMixin, SourceMixin):

    def output(self):
        raise NotImplementedError("this should be over-written by sub-classes")


    def requires(self):
        raise NotImplementedError("this should be over-written by sub-classes")
    

    def get_directory_kwds(self):
        """ Get the keywords for creating a directory for FERRE to run in. """
        return dict(dir=os.path.join(self.output_base_dir, "scratch", self.task_id.split("_")[1]))
        
    
    def get_ferre_kwds(self):
        """ Return human-readable keywords that will be used with FERRE. """
        return {
            "grid_header_path": self.grid_header_path,
            "interpolation_order": self.interpolation_order,
            "init_algorithm_flag": self.init_algorithm_flag,
            "error_algorithm_flag": self.error_algorithm_flag,
            "continuum_flag": self.continuum_flag,
            "continuum_order": self.continuum_order,
            "continuum_reject": self.continuum_reject,
            "continuum_observations_flag": self.continuum_observations_flag,
            "full_covariance": self.full_covariance,
            "pca_project": self.pca_project,
            "pca_chi": self.pca_chi,
            "frozen_parameters": self.frozen_parameters,
            "optimization_algorithm_flag": self.optimization_algorithm_flag,
            "wavelength_interpolation_flag": self.wavelength_interpolation_flag,
            "lsf_shape_flag": self.lsf_shape_flag,
            "input_weights_path": self.input_weights_path,
            "input_lsf_path": self.input_lsf_path,
            "use_direct_access": self.use_direct_access,
            "n_threads": self.n_threads,
            "debug": self.debug,
            "directory_kwds": self.get_directory_kwds(),
            "ferre_kwds": self.ferre_kwds
        }


    @property
    def frozen_parameters(self):
        frozen_parameters = []
        for key, is_frozen, value in self._ferre_parameters:
            if is_frozen:
                frozen_parameters.append(key)
        return tuple(frozen_parameters)
    

    @property
    def initial_parameters(self):
        initial_parameters = {}
        for key, is_frozen, value in self._ferre_parameters:
            initial_parameters[key] = value
        return initial_parameters


    @property
    def _ferre_parameters(self):
        return [
            ("TEFF", self.frozen_teff, self.initial_teff),
            ("LOGG", self.frozen_logg, self.initial_logg),
            ("METALS", self.frozen_metals, self.initial_metals),
            ("LOG10VDOP", self.frozen_log10vdop, self.initial_log10vdop),
            ("O Mg Si S Ca Ti", self.frozen_o_mg_si_s_ca_ti, self.initial_o_mg_si_s_ca_ti),
            ("LGVSINI", self.frozen_lgvsini, self.initial_lgvsini),
            ("C", self.frozen_c, self.initial_c),
            ("N", self.frozen_n, self.initial_n)
        ]


    def get_source_names(self, spectra):
        """ Return a list of source names for convenience in FERRE. """
        N = sum([spectrum.flux.shape[0] for spectrum in spectra])
        return list(map(str, range(N)))
        

    def read_input_observations(self):
        raise NotImplementedError("this should be implemented by the sub-classes")


    def execute(self):
        """ Execute FERRE. """

        N = self.get_batch_size()
        log.info(f"Running {N} task{('s in batch mode' if N > 1 else '')}: {self}")

        spectra = self.read_input_observations()
    
        model = Ferre(**self.get_ferre_kwds())
        
        results = model.fit(
            spectra,
            initial_parameters=self.initial_parameters,
            names=self.get_source_names(spectra),
            full_output=True
        )

        return (model, spectra, results)


    def prepare_results(self, model, all_spectra, p_opt, p_err, meta):

        spn = list(map(sanitise_parameter_names, model.parameter_names))
        frozen_kwds = dict([(f"frozen_{pn}", getattr(self, f"frozen_{pn}")) for pn in spn])

        si = 0
        for (task, spectra, N) in zip(self.get_batch_tasks(), all_spectra, meta["n_spectra_per_source"]):
            # One 'task' and one 'spectrum' actually can include many visits.
            sliced = slice(si, si + N)

            # Prepare results dictionary.
            snr = spectra.meta["snr"]
            if not task.analyse_individual_visits:
                snr = [snr[0]]

            results = dict(
                snr=snr,
                log_snr_sq=meta["log_snr_sq"][sliced],
                log_chisq_fit=meta["log_chisq_fit"][sliced]
            )
            # Initial values.
            results.update(
                dict([(f"initial_{pn}", [getattr(task, f"initial_{pn}")] * N) for pn in spn])
            )
            # Optimized values.
            results.update(dict(zip(spn, p_opt[sliced].T)))
            # Errors.
            results.update(dict(zip([f"u_{pn}" for pn in spn], p_err[sliced].T)))
            # Frozen values.
            results.update(frozen_kwds)

            yield (task, spectra, results, sliced)

            si += N


    @slurmify
    def run(self):
        """ Run this task. """

        t_init = time()
        model, all_spectra, (p_opt, p_err, meta) = self.execute()

        # Get processing times.
        times = model.get_processing_times()

        for task, spectra, results, sliced in self.prepare_results(model, all_spectra, p_opt, p_err, meta):
            
            # Write to database as required.
            if "database" in task.output():
                task.output()["database"].write(results)

            # Write an AstraSource object as required.
            if "AstraSource" in task.output():
                # Get the continuum used by FERRE.
                continuum = meta["continuum"][sliced]
                task.output()["AstraSource"].write(
                    spectra,
                    normalized_flux=spectra.flux.value / continuum,
                    normalized_ivar=continuum * spectra.uncertainty.array * continuum,
                    continuum=continuum,
                    model_flux=meta["model_flux"][sliced],
                    model_ivar=None,
                    results_table=Table(
                        data={ k: v for k, v in results.items() if not k.startswith("frozen_") }
                    )
                )
            
            # Trigger the task as complete.
            task.trigger_event_processing_time(
                sum(times["time_per_ordered_spectrum"][sliced]),
                cascade=True
            )
        
        model.teardown()
        self.trigger_event_processing_time(time() - t_init, cascade=True)
        
        return None


