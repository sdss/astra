

import numpy as np
import os
import pickle
from astra import log, __version__
from astra.base import ExecutableTask, Parameter
from astra.database.astradb import DataProduct, TaskInputDataProducts, database, TaskOutputDataProducts
from astropy.table import Table

from astra.tools.spectrum import Spectrum1D
from astra.utils import executable, expand_path
from sdss_access import SDSSPath
from tqdm import tqdm

from astra.contrib.thecannon_new.model import CannonModel
from astra.contrib.thecannon_new.base import BaseCannonExecutableTask


class TrainTheCannon(BaseCannonExecutableTask):

    # The input data product is the training set, which contains:
    # - training labels
    # - flux
    # - inverse variance of the flux
    # - label names
    # - dispersion
    
    # If no label names given, use everything in training set.
    label_names = Parameter("label_names", default=None)
    regularization = Parameter("regularization", default=1e-26)
    n_threads = Parameter("n_threads", default=-1)

    tol = Parameter("tol", default=1e-4)
    precompute = Parameter("precompute", default=True)
    max_iter = Parameter("max_iter", default=10_000)

    release = Parameter("release", default="sdss5")

    def execute(self):

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
        
        labels = np.array([training_set["labels"][name] for name in label_names]).T

        model = CannonModel(
            labels,
            training_set["data"]["flux"], 
            training_set["data"]["ivar"],
            label_names,
            # TODO: assuming single-array for dispersion, otherwise we have to resample
            dispersion=training_set["data"]["wavelength"].flatten(),
            regularization=self.regularization,
            n_threads=self.n_threads
        )

        model.train(
            tol=self.tol,
            precompute=self.precompute,
            max_iter=self.max_iter
        )
        return model


    def post_execute(self):
        # Write Cannon model to disk.
        task = self.context["tasks"][0]
        path = expand_path(f"$MWM_ASTRA/{__version__}/thecannon/model-{task.id}.pkl")
        self.result.write(path)

        # Create output data product.
        data_product = DataProduct.create(
            release=self.release,
            filetype="full",
            kwargs=dict(full=path)
        )
        TaskOutputDataProducts.create(task=task, data_product=data_product)
        log.info(f"Created data product {data_product} and assigned it as output to task {task}")

