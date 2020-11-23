
import os
import luigi
import pickle
import numpy as np
from astropy.table import Table
from astra.tasks.base import BaseTask
from astra.utils import log
from astra.tools.spectrum import Spectrum1D
from astra.tools.spectrum.writers import create_astra_source
from astra.tasks.io import ApStarFile
from tqdm import tqdm

import astra.contrib.thecannon as tc

from astra.contrib.thecannon.tasks.base import TheCannonMixin
from astra.contrib.thecannon.tasks.train import TrainTheCannonGivenTrainingSetTarget

class TestTheCannon(TheCannonMixin):

    # TODO: Allow for explicit initial values?
    N_initialisations = luigi.IntParameter(default=10)
    use_derivatives = luigi.BoolParameter(default=True)
    
    def requires(self):
        raise NotImplementedError(
            "You must overwrite the `requires()` function in the `astra_thecannon.tasks.Test` class "
            "so that it provides a dictionary with keys 'model' and 'observation', and tasks as values."
        )


    def output(self):
        raise NotImplementedError("This should be provided by the sub-classes")


    def read_observation(self):
        return Spectrum1D.read(self.input()["observation"].path)


    def resample_observation(self, dispersion, spectrum=None):
        if spectrum is None:
            spectrum = self.read_observation()

        o_x = spectrum.wavelength.value
        o_f = np.atleast_2d(spectrum.flux.value)
        o_i = np.atleast_2d(spectrum.uncertainty.quantity.value)

        N, P = shape = (o_f.shape[0], dispersion.size)
        flux = np.empty(shape)
        ivar = np.empty(shape)

        for i in range(N):
            flux[i] = np.interp(dispersion, o_x, o_f[i])
            ivar[i] = np.interp(dispersion, o_x, o_i[i])

        return (flux, ivar)


    def read_model(self):
        return tc.CannonModel.read(self.input()["model"].path)


    def run(self):

        model = self.read_model()

        # This can be run in batch mode.
        for task in tqdm(self.get_batch_tasks()):
            flux, ivar = task.resample_observation(model.dispersion)
            labels, cov, metadata = model.test(
                flux,
                ivar,
                initialisations=task.N_initialisations,
                use_derivatives=task.use_derivatives
            )
            
            # TODO: Write outputs somewhere!
            log.warn("Not writing outputs anywhere!")




class EstimateStellarParametersGivenApStarFileBase(TrainTheCannonGivenTrainingSetTarget, TestTheCannon):

    def requires(self):
        return dict(model=TrainTheCannonGivenTrainingSetTarget(**self.get_common_param_kwargs(TrainTheCannonGivenTrainingSetTarget)))

    def run(self):

        model = self.read_model()

        # We can run The Cannon in batch mode (many stars at once).

        # We have to do a little bit of work here because we need to keep count of how many spectra
        # per ApStar file (or per task).
        data = []
        task_meta = []
        task_spectra = []
        for task in tqdm(self.get_batch_tasks(), total=self.get_batch_size()):
            spectrum = task.read_observation()
            flux, ivar = task.resample_observation(model.dispersion, spectrum=spectrum)
            data.append((flux, ivar))

            N_spectra = flux.shape[0]
            N_visits = spectrum.meta["header"]["NVISITS"]
            snr_combined = spectrum.meta["header"]["SNR"]
            snr_visits = [spectrum.meta["header"][f"SNRVIS{i}"] for i in range(1, 1 + N_visits)]
            if N_visits > 1:
                assert len(snr_visits) == N_visits

            task_meta.append(dict(
                N_visits=N_visits,
                N_spectra=N_spectra,
                snr_visits=snr_visits,
                snr_combined=snr_combined
            ))
            task_spectra.append(spectrum)
            
        flux, ivar = map(np.vstack, zip(*data))

        labels, cov, op_meta = model.test(
            flux,
            ivar,
            initialisations=self.N_initialisations,
            use_derivatives=self.use_derivatives,
            threads=self.threads
        )

        si = 0
        for i, (task, spectrum, meta) in enumerate(zip(self.get_batch_tasks(), task_spectra, task_meta)):
            
            N_spectra = meta["N_spectra"]
            result = dict(
                N_visits=meta["N_visits"],
                N_spectra=meta["N_spectra"],
                snr_visits=meta["snr_visits"],
                snr_combined=meta["snr_combined"],
                labels=labels[si:si + N_spectra],
                cov=cov[si:si + N_spectra],
                metadata=op_meta[si:si + N_spectra],
                label_names=model.vectorizer.label_names
            )

            # Create results table.
            results_table = Table(
                data=labels[si:si + N_spectra],
                names=model.vectorizer.label_names
            )
            results_table["cov"] = cov[si:si + N_spectra]
            keys = ("snr", "r_chi_sq", "chi_sq", "x0")
            for key in keys:
                results_table[key] = [row[key] for row in op_meta[si:si + N_spectra]]

            # This is a shit way to have to get the continuum...
            # TODO: Consider a more holistic way that will work for any intermediate step.
            _cont = Spectrum1D.read(task.requires()["observation"].output().path)
            _orig = Spectrum1D.read(task.requires()["observation"].requires().local_path)
            continuum = (_orig.flux / _cont.flux).value

            # Write astraSource target.
            task.output()["astraSource"].write(
                spectrum=spectrum,
                normalized_flux=flux[si:si + N_spectra],
                normalized_ivar=ivar[si:si + N_spectra],
                continuum=continuum,
                model_flux=np.array([ea["model_flux"] for ea in op_meta[si:si + N_spectra]]),
                # TODO: Project uncertainties to flux space and include here.
                model_ivar=None,
                results_table=results_table
            )
                        
            # Write database result.
            if "database" in task.output():

                # TODO: Here we are just writing the result from the stacked spectrum.
                #       Consider including the results from individual visits to the database.
                #       (These are still accessible through the 'etc' target output.)

                database_result = dict(zip(
                    map(str.lower, model.vectorizer.label_names),
                    labels[si]
                ))
                database_result.update(dict(zip(
                    (f"u_{ln.lower()}" for ln in model.vectorizer.label_names),
                    np.sqrt(np.diag(cov[si]))
                )))

                task.output()["database"].write(database_result)
                
            si += N_spectra


    def output(self):
        raise RuntimeError("this should be over-written by parent classes")

