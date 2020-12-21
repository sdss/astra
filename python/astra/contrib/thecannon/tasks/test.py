
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
        """
        Read the input observation, and if continuum is a requirement, normalize it.
        """
        observation = Spectrum1D.read(self.input()["observation"].path)
        if "continuum" in self.input():
            with open(self.input()["continuum"].path, "rb") as fp:
                continuum = pickle.load(fp)
        else:
            continuum = 1

        return (observation, continuum)


    def prepare_observation(self, dispersion, spectrum=None, continuum=None):
        if spectrum is None:
            spectrum, continuum_ = self.read_observation()
            if continuum is None:
                continuum = continuum_

        o_x = spectrum.wavelength.value
        o_f = np.atleast_2d(spectrum.flux.value)
        o_i = np.atleast_2d(spectrum.uncertainty.quantity.value)

        # Continuum normalize:
        o_f /= continuum
        o_i *= continuum * continuum

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
            flux, ivar = task.prepare_observation(model.dispersion)
            labels, cov, metadata = model.test(
                flux,
                ivar,
                initialisations=task.N_initialisations,
                use_derivatives=task.use_derivatives
            )
            
            # TODO: Write outputs somewhere!
            log.warn("Not writing outputs anywhere!")




class EstimateStellarLabelsGivenApStarFileBase(TrainTheCannonGivenTrainingSetTarget, TestTheCannon):

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
            if task.complete():
                task_spectra.append(None)
                task_meta.append(None)
                continue
            spectrum, cont = task.read_observation()
            flux, ivar = task.prepare_observation(
                model.dispersion, 
                spectrum=spectrum, 
                continuum=cont
            )
            data.append((flux, ivar, cont))

            N_spectra = flux.shape[0]
            task_meta.append(dict(
                N_spectra=N_spectra,
                snr=spectrum.meta["snr"],
            ))
            task_spectra.append(spectrum)
            
        flux, ivar, cont = map(np.vstack, zip(*data))

        labels, cov, op_meta = model.test(
            flux,
            ivar,
            initialisations=self.N_initialisations,
            use_derivatives=self.use_derivatives,
            threads=self.threads
        )

        si = 0
        for i, (task, spectrum, meta) in enumerate(zip(self.get_batch_tasks(), task_spectra, task_meta)):
            if spectrum is None and meta is None:
                continue 

            sliced = slice(si, si + meta["N_spectra"])
            
            result = dict(
                labels=labels[sliced],
                cov=cov[sliced],
            )
            for key in  ("snr", "r_chi_sq", "chi_sq", "x0"):
                result[key] = [row[key] for row in op_meta[sliced]]
                            
            if "AstraSource" in task.output():
                # Write AstraSource target.
                task.output()["AstraSource"].write(
                    spectrum=spectrum,
                    normalized_flux=flux[sliced],
                    normalized_ivar=ivar[sliced],
                    continuum=cont[sliced],
                    model_flux=np.array([ea["model_flux"] for ea in op_meta[sliced]]),
                    # TODO: Project uncertainties to flux space and include here.
                    model_ivar=None,
                    results_table=Table(data=result)
                )
            
            if "database" in task.output():
                # Write database rows.
                L = len(model.vectorizer.label_names)
                rows = dict(zip(
                    map(str.lower, model.vectorizer.label_names),
                    result["labels"].T
                ))
                rows.update(dict(zip(
                    (f"u_{ln.lower()}" for ln in model.vectorizer.label_names),
                    np.sqrt(result["cov"][:, np.arange(L), np.arange(L)]).T
                )))
                for key in ("snr", "r_chi_sq", "chi_sq"):
                    rows.update({ key: result[key] })

                task.output()["database"].write(rows)
                
            si += meta["N_spectra"]


    def output(self):
        raise RuntimeError("this should be over-written by parent classes")

