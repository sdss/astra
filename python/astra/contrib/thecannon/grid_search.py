import numpy as np
import pickle
from astra import log, __version__
from astra.base import Parameter
from astra.database.astradb import DataProduct, TaskOutputDataProducts
from astra.contrib.thecannon.model import CannonModel

from astra.utils import expand_path


def grid_search(
    train_labels,
    train_flux,
    train_ivar,
    validation_labels,
    validation_flux,
    validation_ivar,
    label_names=None,
    max_alpha=1e-1, 
    min_alpha=0, 
    step=5,
    **kwargs
):

    # Train a model at the minimum alpha first. This will be the baseline.
    if label_names is None:
        N, L = train_labels.shape
        label_names = list(map(str, range(L)))
    args = (train_labels, train_flux, train_ivar, label_names)
    baseline = CannonModel(*args, regularization=min_alpha)
    baseline.train(**kwargs)

    baseline_validation_chisq = np.sum((baseline.predict(validation_labels) - validation_flux)**2 * validation_ivar)

    # Now let's start from the strongest regularization and work our way down.
    # (It's faster that way)
    min_log_alpha = int(np.log10(max(min_alpha, 1e-12)))
    max_log_alpha = int(np.log10(max_alpha))
    alphas = np.logspace(min_log_alpha, max_log_alpha, (max_log_alpha - min_log_alpha) * step + 1 )

    models = {}
    validation_chisq = {}

    for i, alpha in enumerate(alphas[::-1]):

        model = CannonModel(*args, regularization=alpha)
        model.train(**kwargs)
        validation_chisq[alpha] = np.sum((model.predict(validation_labels) - validation_flux)**2 * validation_ivar)

        models[alpha] = model

        print(i, alpha, validation_chisq[alpha]/baseline_validation_chisq)
        if (validation_chisq[alpha] <= baseline_validation_chisq):
            break
    
    meta = dict(models=models, validation_chisq=validation_chisq)
    return (model, baseline, meta)





'''

class HyperparameterGridSearch(ExecutableTask):

    # input data products is the (training set + validation set)
    _lower, _upper, _step = (-10, 0, 5)
    regularization = Parameter(
        "regularization",
        default=tuple(np.logspace(_lower, _upper, (_upper - _lower) * _step + 1)),
    )

    label_names = Parameter("label_names", default=None)
    n_threads = Parameter("n_threads", default=-1)

    # Training params
    tol = Parameter("tol", default=1e-4)
    precompute = Parameter("precompute", default=True)
    max_iter = Parameter("max_iter", default=10_000)

    def execute(self):

        validation_set_dp, test_set_dp = self.input_data_products

        R = len(self.regularization)
        log.info(f"Starting grid search with {R} regularization values")

        time_train = np.zeros(R)
        sparsity = np.zeros(R)
        # types: linear terms, quadratic terms, cross terms
        sparsity_by_term_type = np.zeros((R, 3))
        chisq = np.zeros(R)
        chisq_test = np.zeros(R)
        train_warning = np.zeros(R)

        # load data sets
        with open(validation_set_dp.path, "rb") as fp:
            validation_set = pickle.load(fp)

        with open(test_set_dp.path, "rb") as fp:
            test_set = pickle.load(fp)

        if self.label_names is None:
            label_names = list(validation_set["labels"].keys())
        else:
            label_names = self.label_names
            missing = set(self.label_names).difference(validation_set["labels"].keys())
            if missing:
                raise ValueError(f"Missing labels from training set: {missing}")

        validation_labels = np.array(
            [validation_set["labels"][name] for name in label_names]
        ).T
        validation_flux, validation_ivar = (
            validation_set["data"]["flux"],
            validation_set["data"]["ivar"],
        )

        test_labels = np.array([test_set["labels"][name] for name in label_names]).T
        test_flux, test_ivar = (test_set["data"]["flux"], test_set["data"]["ivar"])

        task = self.context["tasks"][0]

        for i, regularization in enumerate(self.regularization):
            log.info(f"At {i+1}/{R} (index {i}): {regularization:.3e}")

            model = CannonModel(
                validation_labels,
                validation_flux,
                validation_ivar,
                label_names,
                regularization=regularization,
                n_threads=self.n_threads,
            )

            model.train(
                tol=self.tol, precompute=self.precompute, max_iter=self.max_iter
            )

            # Summary statistics.
            time_train[i] = model.meta["t_train"]
            chisq[i] = model.chi_sq(
                validation_labels, validation_flux, validation_ivar, aggregate=np.median
            )
            chisq_test[i] = model.chi_sq(
                test_labels, test_flux, test_ivar, aggregate=np.median
            )
            train_warning[i] = np.mean(model.meta["train_warning"])

            sparsity[i] = np.sum(model.theta == 0) / model.theta.size
            for j, idx in enumerate(model.term_type_indices):
                sparsity_by_term_type[i, j] = (
                    np.sum(model.theta[idx] == 0) / model.theta[idx].size
                )

            log.info(f"Completed {i+1}/{R} (index {i}): {regularization:.3e}")
            log.info(f"  chisq: {chisq[i]:.3e}")
            log.info(f"  chisq_test: {chisq_test[i]:.3e}")
            log.info(f"  train_warning: {train_warning[i]:.3e}")
            log.info(f"  sparsity: {sparsity[i]:.3e}")
            log.info(f"  sparsity_by_term_type: {sparsity_by_term_type[i, :]}")

            path = expand_path(
                f"$MWM_ASTRA/{__version__}/thecannon/{self.__class__.__name__}-{task.id}-model-{i}-{regularization:.3e}.pkl"
            )
            model.write(path)
            log.info(f"Saved model to {model}")

        path = expand_path(
            f"$MWM_ASTRA/{__version__}/thecannon/{self.__class__.__name__}-{task.id}-statistics.pkl"
        )
        with open(path, "wb") as fp:
            pickle.dump(
                dict(
                    regularization=self.regularization,
                    time_train=time_train,
                    chisq=chisq,
                    chisq_test=chisq_test,
                    train_warning=train_warning,
                    sparsity=sparsity,
                    sparsity_by_term_type=sparsity_by_term_type,
                ),
                fp,
            )
        log.info(f"Wrote summary statistics to {path}")
'''