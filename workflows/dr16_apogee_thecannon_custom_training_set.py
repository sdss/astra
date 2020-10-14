
import numpy as np
import pickle
import astropy.table 
from sqlalchemy import Column, Float
from tqdm import tqdm

import astra
from astra.tasks.base import BaseTask
from astra.tasks.targets import (LocalTarget, DatabaseTarget, MockTarget)
from astra.tasks.io import (ApStarFile, ApVisitFile)
from astra.tasks.continuum import Sinusoidal
from astra.tools.spectrum import Spectrum1D

from astra.contrib.thecannon.tasks import (TrainTheCannon, TestTheCannon)

# CONTINUUM NORMALISATION

class ContinuumNormalizeIndividualVisitSpectrum(Sinusoidal, ApStarFile):

    """
    A pseudo-continuum normalisation task for individual visit spectra 
    in ApStarFiles using a sum of sines and cosines to model the continuum.
    """

    # Row 0 is individual pixel weighting
    # Row 1 is global pixel weighting
    # Row 2+ are the individual visits.
    # We will just analyse them all because it's cheap.

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))


class ContinuumNormalizePixelWeightedSpectrum(Sinusoidal, ApStarFile):

    """
    A pseudo-continuum normalisation task for stacked spectra in ApStarFiles that 
    uses sums of sines and cosines to model the continuum.
    """

    # For training we only want to take the first spectrum, which
    # is stacked by individual pixel weighting.
    spectrum_kwds = dict(data_slice=(slice(0, 1), slice(None)))

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))




# ESTIMATE STELLAR PARAMETERS ON INDIVIDUAL VISIT SPECTRA

@astra.inherits(TrainTheCannon, ContinuumNormalizeIndividualVisitSpectrum)
class StellarParameters(TestTheCannon):

    max_batch_size = 1000

    def requires(self):
        return {
            "model": TrainTheCannon(**self.get_common_param_kwargs(TrainTheCannon)),
            "observation": ContinuumNormalizeIndividualVisitSpectrum(**self.get_common_param_kwargs(ContinuumNormalizeIndividualVisitSpectrum))
        }
        

    def run(self):

        model = self.read_model()

        # We can run The Cannon in batch mode (many stars at once).

        # We have to do a little bit of work here because we need to keep count of how many spectra
        # per ApStar file (or per task).
        data = []
        task_meta = []
        for task in tqdm(self.get_batch_tasks()):
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
            
        flux, ivar = map(np.vstack, zip(*data))

        labels, cov, op_meta = model.test(
            flux,
            ivar,
            initialisations=self.N_initialisations,
            use_derivatives=self.use_derivatives,
            threads=self.threads
        )

        si = 0
        for i, (task, meta) in enumerate(zip(self.get_batch_tasks(), task_meta)):
            
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

            # Write result.
            with open(task.output().path, "wb") as fp:
                pickle.dump(result, fp)
            
            si += N_spectra


    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        return LocalTarget(path=f"{self.task_id}")



if __name__ == "__main__":

    # The training set file (stored in `training_set_path`) should be a binary pickle file that contains a dictionary
    # with the following keys:
    #   wavelength: an array of shape (P, ) where P is the number of pixels
    #   flux: an array of flux values with shape (N, P) where N is the number of observed spectra and P is the number of pixels
    #   ivar: an array of inverse variance values with shape (N, P) where N is the number of observed spectra and P is the number of pixels
    #   labels: an array of shape (L, N) where L is the number of labels and N is the number observed spectra
    #   label_names: a tuple of length L that describes the names of the labels

    common_kwds = dict(
        # Training set path.
        training_set_path="my_training_set.pkl",
        # Continuum regions.
        continuum_regions_path="python/astra/contrib/thecannon/etc/continuum-regions.list",
        # Note that the 'label names' you give here are expected to be in your training set!
        label_names=("TEFF", "LOGG", "FE_H"),
        # Polynomial order of cannon model.
        order=2,
        # Download SDSS data products if we don't have them.
        use_remote=True
    )

    # Now we just need to supply some keywords so that astra knows what ApStar files
    # you want to analyse. The keywords we need are:
    #   apred (e.g., 'r12')
    #   apstar (e.g., 'stars')
    #   prefix (e.g., 'ap' or 'as')
    #   telescope (e.g., 'apo25m')
    #   field
    #   obj

    # You can provide a single value, or a tuple of multiple values. If you supply a tuple
    # then you just need to make sure you supply the same length tuple for apred, apstar,
    # prefix, telescope, field, and obj.
    single_star_kwds = {
        "apred": "r12",
        "apstar": "stars",
        "prefix": "ap",
        "telescope": "apo1m",
        "field": "calibration",
        "obj": "2M01054129+8719084",
    }
    
    multiple_star_kwds = {
        "apred": ("r12", "r12"),
        "apstar": ("stars", "stars"),
        "prefix": ("ap", "ap"),
        "telescope": ("apo1m", "apo1m"),
        "field": ("calibration", "calibration"),
        "obj": ("2M01054129+8719084", "HD_102328")
    }
    
    task = StellarParameters(**{ **common_kwds, **multiple_star_kwds })

    astra.build(
        [task],
        local_scheduler=True
    )
    
    # We can access the outputs of the task in task.output():
    with open(task.output()[0].path, "rb") as fp:
        result = pickle.load(fp)
    print(result)

