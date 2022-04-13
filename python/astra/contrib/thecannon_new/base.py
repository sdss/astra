
import numpy as np
import pickle
from astra import log, __version__
from astra.base import ExecutableTask

class BaseCannonExecutableTask(ExecutableTask):

    def _load_training_set(self):
        task = self.context["tasks"][0]
        training_set, *_ = self.context["input_data_products"]

        with open(training_set.path, "rb") as fp:
            training_set = pickle.load(fp)
        
        if self.label_names is None:
            label_names = list(training_set["labels"].keys())
        else:
            label_names = self.label_names
            missing = set(self.label_names).difference(training_set["labels"].keys())
            if missing:
                raise ValueError(f"Missing labels from training set: {missing}")
        
        labels = np.array([training_set["labels"][name] for name in label_names])

        return (task, training_set, labels, label_names)

