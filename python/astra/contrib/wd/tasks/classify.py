
import astra
import luigi
import numpy as np
import pickle
import sqlalchemy
import warnings

from itertools import cycle
from time import time
from tqdm import tqdm
from astra.tasks import BaseTask
from astra.tasks.targets import DatabaseTarget, LocalTarget
from astra.tasks.io import LocalTargetTask, SpecFile
from astra.tools.spectrum import Spectrum1D
from astra.utils import log

from astra_whitedwarf.utils import line_features

warnings.filterwarnings("ignore", category=np.RankWarning) 


class ClassifyWhiteDwarfMixin(BaseTask):
    model_path = luigi.Parameter()

    wavelength_regions = luigi.ListParameter(
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

    polyfit_order = luigi.IntParameter(default=5)
    polyfit_regions = luigi.ListParameter(
        default=[
            [3850, 3870],
            [4220, 4245],
            [5250, 5400],
            [6100, 6470],
            [7100, 9000]
        ]
    )


class WDClassification(DatabaseTarget):
    results_schema = [
        sqlalchemy.Column("wd_class", sqlalchemy.String(2)),
        sqlalchemy.Column("flag", sqlalchemy.Boolean())
    ]



class ClassifyWhiteDwarf(ClassifyWhiteDwarfMixin, SpecFile):

    max_batch_size = 10000

    def requires(self):
        requirements = dict(model=LocalTargetTask(path=self.model_path))
        if not self.is_batch_mode:
            requirements.update(observation=SpecFile(**self.get_common_param_kwargs(SpecFile)))
        return requirements


    def run(self):

        # Load the model.
        with open(self.input()["model"].path, "rb") as fp:
            classifier = pickle.load(fp)

        # This can be run in batch mode.
        tasks = list(self.get_batch_tasks())
    
        features = np.empty((len(tasks), len(self.wavelength_regions)))
        for i, task in enumerate(tqdm(tasks)):
            features[i, :] = line_features(
                Spectrum1D.read(task.input()["observation"].path),
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

        for task, wd_class, flag in zip(tasks, classifier.predict(features), flags):
            task.output().write(dict(wd_class=wd_class, flag=flag))


    def output(self):
        return WDClassification(self)



if __name__ == "__main__":

    import yaml
    with open("examples.yml", "r") as fp:
        examples = yaml.load(fp)

    expected = [example.pop("expected_class") for example in examples]
    tasks = [ClassifyWhiteDwarf(model_path="../../../data/training_file", **kwds) for kwds in examples]
    
    # Just do a random subset.
    N_subset = len(tasks)
    np.random.seed(7)
    indices = np.random.choice(len(tasks), N_subset, replace=False)

    tasks = [task for i, task in enumerate(tasks) if i in indices]
    expected = [each for i, each in enumerate(expected) if i in indices]
    astra.build(
        tasks,
        local_scheduler=True
    )
    
    class_names = expected + [task.output().read(as_dict=True)["wd_class"] for task in tasks]

    # Restrict to two first chars.
    max_chars = 2
    class_names = list(set([ea.upper()[:max_chars] for ea in class_names]))
    
    M = len(class_names)

    confusion_matrix = np.zeros((M, M))

    for i, (task, expected_class) in enumerate(zip(tasks, expected)):

        actual_class = task.output().read(as_dict=True)["wd_class"]

        j = class_names.index(expected_class.upper())
        k = class_names.index(actual_class.upper()[:max_chars])

        confusion_matrix[j, k] += 1
    
