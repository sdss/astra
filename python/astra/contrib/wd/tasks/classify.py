import astra
import numpy as np
import pickle
import sqlalchemy
import warnings

from luigi.parameter import ListParameter, Parameter, IntParameter

from itertools import cycle
from tqdm import tqdm
from astra.tasks import BaseTask
from astra.tasks.targets import DatabaseTarget, LocalTarget
from astra.database import astradb
from astra.tasks.io import LocalTargetTask
from astra.tasks.io.sdss4 import SDSS4SpecFile
from astra.tools.spectrum import Spectrum1D

from astra.contrib.wd.utils import line_features

warnings.filterwarnings("ignore", category=np.RankWarning) 


class ClassifyWhiteDwarfMixin(BaseTask):

    """
    Mix-in class for classifying white dwarfs.
    """

    model_path = Parameter()

    wavelength_regions = ListParameter(
        default=[
            [3860, 3900], # Balmer line
            [3950, 4000], # Balmer line
            [4085, 4120], # Balmer line
            [4320, 4360], # Balmer line
            [4840, 4880], # Balmer line
            [6540, 6580], # Balmer line
            [3880, 3905], # He I/II line
            [3955, 3975], # He I/II line
            [3990, 4056], # He I/II line
            [4110, 4140], # He I/II line
            [4370, 4410], # He I/II line
            [4450, 4485], # He I/II line
            [4705, 4725], # He I/II line
            [4900, 4950], # He I/II line
            [5000, 5030], # He I/II line
            [5860, 5890], # He I/II line
            [6670, 6700], # He I/II line
            [7050, 7090], # He I/II line
            [7265, 7300], # He I/II line
            [4600, 4750], # Molecular C absorption band
            [5000, 5160], # Molecular C absorption band
            [3925, 3940], # Ca H/K line
            [3960, 3975], # Ca H/K line
        ]
    )

    polyfit_order = IntParameter(default=5)
    polyfit_regions = ListParameter(
        default=[
            [3850, 3870],
            [4220, 4245],
            [5250, 5400],
            [6100, 6470],
            [7100, 9000]
        ]
    )


class ClassifyWhiteDwarfGivenSDSS4SpecFile(ClassifyWhiteDwarfMixin, SDSS4SpecFile):

    """
    Classify a white dwarf given a SDSS4 BOSS SpecFile.

    :param model_path:
        The path to a file where the model is stored.
    
    :param wavelength_regions: (optional)
        A list of two-length tuples that contains the start and end wavelength around line features to use for classification. Note that if you change these then you should probably re-train a classifier to use the same features.
    
    :param polyfit_order: (optional)
        The order of the polynomial to use for continuum fitting around absorption lines (default: 5).
    
    :param polyfit_regions: (optional)
        A list of two-length tuples that contain the start end end wavelengths of regions to use when fitting the continuum.
    """

    max_batch_size = 10000

    def requires(self):
        """ The requirements for this task. """
        requirements = dict(model=LocalTargetTask(path=self.model_path))
        if not self.is_batch_mode:
            requirements.update(observation=self.clone(SDSS4SpecFile))
        return requirements


    def run(self):
        """ Execute this task. """

        # Load the model.
        with open(self.input()["model"].path, "rb") as fp:
            classifier = pickle.load(fp)

        # This can be run in batch mode.
        tasks = list(self.get_batch_tasks())
        spectra = [Spectrum1D.read(t.input()["observation"].path) for t in tasks]

        features = np.empty((len(tasks), len(self.wavelength_regions)))
        for i, (task, spectrum) in enumerate(tqdm(zip(tasks, spectra))):
            features[i, :] = line_features(
                spectrum,
                wavelength_regions=self.wavelength_regions,
                polyfit_regions=self.polyfit_regions,
                polyfit_order=self.polyfit_order
            )

        # Check for non-finite features.
        if not np.all(np.isfinite(features)):
            # TODO: Here we are just setting non-finite entries to be the mean of other finite entries
            #       *in this batch*. This is the wrong thing to do but I await on the WD team to do
            #       something else.
            mean_features = np.nanmean(features, axis=0)
            non_finite = np.where(~np.isfinite(features))
            for i, j in zip(*non_finite):
                features[i, j] = mean_features[j]

            flags = np.zeros(len(tasks), dtype=bool)
            flags[non_finite[0]] = True
        else:
            flags = cycle([False])

        # Write results to database.
        for task, wd_class, flag in zip(tasks, classifier.predict(features), flags):
            task.output()["database"].write(dict(wd_class=wd_class, flag=flag))
            

    def output(self):
        """ The output of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        return dict(database=DatabaseTarget(astradb.WDClassification, self))
    
