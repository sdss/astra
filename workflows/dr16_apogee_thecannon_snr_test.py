
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

from astra.contrib.thecannon.tasks import (TrainTheCannonBase, TestTheCannon)

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


# CREATE THE TRAINING SET

class CreateTrainingSet(Sinusoidal):

    label_names = astra.ListParameter()
    calibration_set_path = astra.Parameter()

    minimum_ivar = astra.FloatParameter(default=1e-6)
    minimum_edge_fraction = astra.FloatParameter(default=0.98)
    
    def requires(self):
        self.calibration_set, all_batch_kwds = read_calibration_set(self.calibration_set_path)
        
        common_kwds = self.get_common_param_kwargs(ContinuumNormalizePixelWeightedSpectrum)
        return ContinuumNormalizePixelWeightedSpectrum(**{ **common_kwds, **all_batch_kwds })
            

    def run(self):
        # The training set file should be a binary pickle file that contains a dictionary
        # with the following keys:
        #   wavelength: an array of shape (P, ) where P is the number of pixels
        #   flux: an array of flux values with shape (N, P) where N is the number of observed spectra and P is the number of pixels
        #   ivar: an array of inverse variance values with shape (N, P) where N is the number of observed spectra and P is the number of pixels
        #   labels: an array of shape (L, N) where L is the number of labels and N is the number observed spectra
        #   label_names: a tuple of length L that describes the names of the labels

        initial_spectrum = Spectrum1D.read(self.input()[0].path)
        P = initial_spectrum.flux.size
        S = len(self.calibration_set)

        training_set = dict(
            wavelength=initial_spectrum.wavelength.value,
            label_names=self.label_names,
            labels=np.array([
                self.calibration_set[label_name] for label_name in self.label_names
            ]),
        )
        
        flux = np.ones((S, P))
        ivar = np.zeros_like(flux)
        
        for i, output in enumerate(tqdm(self.input())):
            spectrum = Spectrum1D.read(output.path)
            flux[i] = spectrum.flux.value.flatten()
            ivar[i] = spectrum.uncertainty.quantity.value.flatten()

        # Check for finite-ness.
        finite = np.isfinite(flux) * np.isfinite(ivar)
        flux[~finite] = 1
        ivar[~finite] = 0

        # Check for edge effects.
        fraction = np.sum(ivar > self.minimum_ivar, axis=0) / S
        on_edge = fraction <= self.minimum_edge_fraction
        flux[:, on_edge] = 1
        ivar[:, on_edge] = 0

        training_set.update(flux=flux, ivar=ivar)
        
        with open(self.output().path, "wb") as fp:
            pickle.dump(training_set, fp)


    def output(self):
        return LocalTarget(path=f"{self.task_id}.pkl")


# TRAIN THE MODEL

@astra.inherits(CreateTrainingSet)
class TrainTheCannon(TrainTheCannonBase):

    """
    The standard `astra.contrib.thecannon.tasks.TrainTheCannon` requires that you
    provide a training_set_path. Here instead we want to create a training set as needed.

    Here we will override that and say that it requires the CreateTrainingSet task
    to be complete (which produces the training set file).
    """

    def requires(self):
        return CreateTrainingSet(**self.get_common_param_kwargs(CreateTrainingSet))
    
    def output(self):
        return LocalTarget(path=f"{self.task_id}")


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


    # Normally the astra.contrib.thecannon.tasks.TrainTheCannon task only requires
    # that the training set file exists (e.g., it assumes it was made somewhere else).
    # That's fine, but if the training set file doesn't exist then the TrainTheCannon
    # task will fail because it doesn't know how to create the file.

    def read_calibration_set(path, N_max=None, verbose=False):
        calibration_set = astropy.table.unique(
            astropy.table.Table.read(calibration_set_path), 
            ("TELESCOPE", "FIELD", "APOGEE_ID")
        )

        batch_kwds = {}
        for parameter_name in ApStarFile.batch_param_names():
            batch_kwds[parameter_name] = []

        for i, row in enumerate(calibration_set, start=1):
            kv_pair = {
                "apred": "r12",
                "apstar": "stars",
                "prefix": row["FILE"][:2],
                "telescope": row["TELESCOPE"],
                "field": row["FIELD"],
                "obj": row["APOGEE_ID"],
            }
            for k, v in kv_pair.items():
                batch_kwds[k].append(v)

            if N_max is not None and i >= N_max:
                break

        if N_max is not None:
            calibration_set = calibration_set[:N_max]

        if verbose:
            batch_kwds = [{k: batch_kwds[k][i] for k in batch_kwds} for i in range(len(batch_kwds["obj"]))]

        return (calibration_set, batch_kwds)


    # The allCal file does not have TEFF, LOGG, FE_H, etc.
    # (It's dumb that we have to do this.)
    calibration_set_path = "allCal-r12-l33-subset.fits"
    original_calibration_set_path = "/home/andy/data/sdss/apogeework/apogee/spectro/aspcap/r12/l33/allCal-r12-l33.fits"
    t = astropy.table.Table.read(original_calibration_set_path)
    t["TEFF"] = t["FPARAM"][:, 0]
    t["LOGG"] = t["FPARAM"][:, 1]
    t["FE_H"] = t["FPARAM"][:, 3]

    # Restrict it to some good-ish things.
    mask = (t["ASPCAPFLAG"] == 0) \
         * (t["TEFF"] < 6500) \
         * (t["SNR"] > 200)
    t = t[mask]
    t.write(calibration_set_path, overwrite=True)


    # Now let's get 1000 stars (each with many visits) to run analysis on
    expected, all_apStar_kwds = read_calibration_set(original_calibration_set_path, 1000)

    common_kwds = dict(
        label_names=("TEFF", "LOGG", "FE_H"),
        order=2,
        calibration_set_path=calibration_set_path,
        continuum_regions_path="python/astra/contrib/thecannon/etc/continuum-regions.list",
        threads=10,
        use_remote=True
    )

    # Create the task.
    task = StellarParameters(**common_kwds, **all_apStar_kwds)

    # Tell astra to run the task, and all of it's dependencies.
    astra.build(
        [task],
        local_scheduler=True
    )


    # Let's collect the results and make a plot.
    # (We could do this in a task, but it is useful to see how to access the outputs)
    x = []
    y_actual = {}
    y_expected = {}
    for row, output in tqdm(zip(expected, task.output())):

        with open(output.path, "rb") as fp:
            result = pickle.load(fp)
        
        N_visits = result["N_visits"]
        if N_visits == 1:
            x.append(result["snr_combined"])
            mask = np.array([True])
        else:
            # We will just consider the pixel-weighted stack and the individual visits.
            # In other words: we will ignore the global-weighted stack!
            x.extend([result["snr_combined"]] + result["snr_visits"])
            mask = np.ones(N_visits + 2, dtype=bool)
            mask[1] = False # This is the global-weighted stack we will ignore.
        
        for i, label_name in enumerate(result["label_names"]):
            for y in (y_expected, y_actual):
                y.setdefault(label_name, [])
            
            y_actual[label_name].extend(result["labels"][mask, i])
            y_expected[label_name].extend(np.repeat(row[label_name], mask.sum()))



    # Array-ify.
    x = np.array(x)
    for y in (y_expected, y_actual):
        for label_name in y.keys():
            y[label_name] = np.array(y[label_name])

    # Make plots.
    import matplotlib.pyplot as plt
    L = len(task.label_names)
    fig, axes = plt.subplots(L, 1, figsize=(3, 3 * L))
    
    for l, (ax, label_name) in enumerate(zip(axes, task.label_names)):
        ax.scatter(
            x,
            np.abs(y_expected[label_name] - y_actual[label_name]),
            s=1,
            c="#000000"
        )
        ax.axhline(0, c="#666666", lw=0.5, ls=":", zorder=-1)
        ax.set_ylabel(f"|Delta({label_name}|)")
        
        if ax.is_last_row():
            ax.set_xlabel("S/N")
        
        else:
            ax.set_xticks([])

    fig.tight_layout()
    fig.savefig("ness_workflow.png")
    
