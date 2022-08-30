import numpy as np
import pickle
from astra import log, __version__
from astra.base import ExecutableTask, Parameter
from astra.utils import expand_path, flatten
from astra.database.astradb import DataProduct, TaskOutputDataProducts
from astra.contrib.thecannon_new.model import CannonModel
from astra.contrib.thecannon_new.plot import (
    plot_labels,
    plot_theta,
    plot_gridsearch_chisq,
    plot_gridsearch_sparsity,
)


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


def _load_data_set(data_set):
    log.info(f"Loading data set from {data_set}")
    with open(data_set.path, "rb") as fp:
        contents = pickle.load(fp)

    flux = contents["data"]["flux"]
    ivar = contents["data"]["ivar"]

    N, P = flux.shape
    # Exclude pixels that are not fully sampled.
    pixel_mask = ~np.all(ivar > 0, axis=0)
    ivar = np.copy(ivar)
    ivar[:, pixel_mask] = 0

    label_names = list(contents["labels"].keys())
    labels = np.array([contents["labels"][k] for k in label_names]).T
    return (label_names, labels, flux, ivar, pixel_mask, contents)


class TrainTheCannon(ExecutableTask):

    release = Parameter(default="sdss5")
    regularization = Parameter(default=0)
    n_threads = Parameter(default=-1)

    def execute(self):

        # Load training set.
        training_set, *other_sets = flatten(self.input_data_products)
        (
            label_names,
            train_labels,
            train_flux,
            train_ivar,
            train_pixel_mask,
            train_dataset,
        ) = _load_data_set(training_set)

        # Prepare output path.
        (task,) = self.context["tasks"]
        model_path = expand_path(
            f"$MWM_ASTRA/{__version__}/thecannon/{task.id}-model.pkl"
        )

        model = CannonModel(
            train_labels,
            train_flux,
            train_ivar,
            label_names,
            regularization=self.regularization,
            n_threads=self.n_threads,
        )
        model.train()  # TODO: put n_threads as an argument for model.train(), not model()
        model.write(model_path)

        # Create output data product for the model.
        output_data_product = DataProduct.create(
            release=self.release, filetype="full", kwargs=dict(full=model_path)
        )
        TaskOutputDataProducts.create(task=task, data_product=output_data_product)
        return model

    def post_execute(self):

        model = self.context["execute"]
        (task,) = self.context["tasks"]

        # Plot theta.
        figure_path = expand_path(
            f"$MWM_ASTRA/{__version__}/thecannon/{task.id}-theta.png"
        )
        fig = plot_theta(model.theta, model.term_type_indices[0])
        fig.savefig(figure_path)
        log.info(f"Created figure {figure_path}")

        # Run the test step on the training data.
        training_set, *other_sets = flatten(self.input_data_products)
        (
            label_names,
            train_labels,
            train_flux,
            train_ivar,
            train_pixel_mask,
            train_dataset,
        ) = _load_data_set(training_set)

        opt_train = model.fit_spectrum(train_flux, train_ivar, train_labels)
        figure_path = expand_path(
            f"$MWM_ASTRA/{__version__}/thecannon/{task.id}-train.png"
        )
        opt_labels = np.array([each[0] for each in opt_train])
        fig = plot_labels(train_labels, opt_labels, model.label_names)
        fig.savefig(figure_path)
        log.info(f"Created figure {figure_path}")

        # Do validation step?
        if len(flatten(self.input_data_products)) > 1:
            _, validation_set = flatten(self.input_data_products)
            log.info(f"Running validation step with {validation_set}")

            (
                _,
                validation_labels,
                validation_flux,
                validation_ivar,
                validation_pixel_mask,
                validation_dataset,
            ) = _load_data_set(validation_set)

            opt_validation = model.fit_spectrum(
                validation_flux, validation_ivar, validation_labels
            )
            opt_validation_labels = np.array([each[0] for each in opt_validation])
            fig = plot_labels(
                validation_labels,
                opt_validation_labels,
                model.label_names,
                scatter_kwds=dict(c="tab:red"),
            )
            figure_path = expand_path(
                f"$MWM_ASTRA/{__version__}/thecannon/{task.id}-validation.png"
            )
            fig.savefig(figure_path)
            log.info(f"Created figure {figure_path}")


class TheCannonRegularizationGridSearch(ExecutableTask):

    release = Parameter(default="sdss5")
    n_threads = Parameter(default=-1)

    lower = Parameter(default=-10)
    upper = Parameter(default=-2)
    spacing = Parameter(default=3)

    @property
    def alphas(self):
        return np.logspace(
            self.lower, self.upper, self.spacing * (self.upper - self.lower) + 1
        )

    def execute(self):
        training_tasks = []
        for i, regularization in enumerate(self.alphas):

            log.info(
                f"Training model with regularization {regularization} ({i+1}/{regularization.size})"
            )
            task = TrainTheCannon(
                self.input_data_products,
                regularization=regularization,
                release=self.release,
                n_threads=self.n_threads,
            )
            task.execute()
            training_tasks.append(task)

        return training_tasks

    def post_execute(self):
        N = self.alphas.size
        train_chisq = np.nan * np.ones(N)
        validation_chisq = np.nan * np.ones(N)
        train_chisq_median = np.nan * np.ones(N)
        validation_chisq_median = np.nan * np.ones(N)

        sparsity = np.nan * np.ones(N)
        sparsity_by_feature_type = np.nan * np.ones((N, 3))

        # Load training and validation data
        training_set, validation_set = flatten(self.input_data_products)
        __, train_labels, train_flux, train_ivar, *_ = _load_data_set(training_set)
        __, validation_labels, validation_flux, validation_ivar, *_ = _load_data_set(
            validation_set
        )

        # for i, task in enumerate(training_tasks):
        #    model = task.context["execute"]

        aggregate = lambda _: np.median(np.median(_, axis=1))

        for i, task in enumerate(self.context["execute"]):
            model = task.context["execute"]
            # from glob import glob
            # for i, path in enumerate(sorted(glob(expand_path(f"$MWM_ASTRA/{__version__}/thecannon/*-model.pkl")))):
            #    model = CannonModel.read(path)
            sparsity[i] = np.sum(model.theta == 0) / model.theta.size

            for j, idx in enumerate(model.term_type_indices):
                sparsity_by_feature_type[i, j] = (
                    np.sum(model.theta[idx] == 0) / model.theta[idx].size
                )

            train_chisq[i] = model.chi_sq(
                train_labels, train_flux, train_ivar, aggregate=aggregate
            )
            validation_chisq[i] = model.chi_sq(
                validation_labels, validation_flux, validation_ivar, aggregate=aggregate
            )

            train_chisq_median[i] = model.chi_sq(
                train_labels, train_flux, train_ivar, aggregate=np.median
            )
            validation_chisq_median[i] = model.chi_sq(
                validation_labels, validation_flux, validation_ivar, aggregate=np.median
            )

        task, *_ = self.context["tasks"]

        fig = plot_gridsearch_sparsity(self.alphas, sparsity, sparsity_by_feature_type)
        figure_path = expand_path(
            f"$MWM_ASTRA/{__version__}/thecannon/{task.id}-sparsity.png"
        )
        fig.savefig(figure_path)
        log.info(f"Created figure {figure_path}")

        fig = plot_gridsearch_chisq(
            self.alphas,
            train_chisq,
            validation_chisq,
        )
        plot_gridsearch_chisq(
            self.alphas,
            train_chisq_median,
            validation_chisq_median,
            ax=fig.axes[0],
            ls=":",
        )
        fig.axes[0].set_ylim(0.95, 1.05)
        chisq_figure_path = expand_path(
            f"$MWM_ASTRA/{__version__}/thecannon/{task.id}-chisq-validation.png"
        )
        fig.savefig(chisq_figure_path)
        log.info(f"Created figure {chisq_figure_path}")


'''
        # TODO: Create HTML page
        import os
        html_content = f"""<html><body>
Chi-sq
<img src="{os.path.basename(chisq_figure_path)}" width=500 />
<br />
<br />

Sparsity
<img src="{os.path.basename(sparsity_figure_path)}" width=500 />
<br />
<br />

Training set with initial guess

<table>
    <tr>
        <td>0</td>
</tr>
    <tr>
        <td><img src="20220523-0-train.png" width=200 /></td>
    </tr>
</table>

Validation set, with initial guess
<table>
    <tr>
        <td>0</td>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
        <td>5</td>
        <td>6</td>
        <td>7</td>
        <td>8</td>
        <td>9</td>
        <td>10</td>
        <td>11</td>
        <td>12</td>
        <td>13</td>
        <td>14</td>
        <td>15</td>
        <td>16</td>
        <td>17</td>
        <td>18</td>
    </tr>
    <tr>
        <td><img src="20220523-0-validation.png" width=200 /></td>
        <td><img src="20220523-1-validation.png" width=200 /></td>
        <td><img src="20220523-2-validation.png" width=200 /></td>
        <td><img src="20220523-3-validation.png" width=200 /></td>
        <td><img src="20220523-4-validation.png" width=200 /></td>
        <td><img src="20220523-5-validation.png" width=200 /></td>
        <td><img src="20220523-6-validation.png" width=200 /></td>
        <td><img src="20220523-7-validation.png" width=200 /></td>
        <td><img src="20220523-8-validation.png" width=200 /></td>
        <td><img src="20220523-9-validation.png" width=200 /></td>
        <td><img src="20220523-10-validation.png" width=200 /></td>
        <td><img src="20220523-11-validation.png" width=200 /></td>
        <td><img src="20220523-12-validation.png" width=200 /></td>
        <td><img src="20220523-13-validation.png" width=200 /></td>
        <td><img src="20220523-14-validation.png" width=200 /></td>
        <td><img src="20220523-15-validation.png" width=200 /></td>
        <td><img src="20220523-16-validation.png" width=200 /></td>
        <td><img src="20220523-17-validation.png" width=200 /></td>
        <td><img src="20220523-18-validation.png" width=200 /></td>
    </tr>
</table>

<table>
    <tr>
        <td>0</td>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
        <td>5</td>
        <td>6</td>
        <td>7</td>
        <td>8</td>
        <td>9</td>
        <td>10</td>
        <td>11</td>
        <td>12</td>
        <td>13</td>
        <td>14</td>
        <td>15</td>
        <td>16</td>
        <td>17</td>
        <td>18</td>
    </tr>
    <tr>
        <td><img src="20220523-0-theta.png" width=500 /></td>
        <td><img src="20220523-1-theta.png" width=500 /></td>
        <td><img src="20220523-2-theta.png" width=500 /></td>
        <td><img src="20220523-3-theta.png" width=500 /></td>
        <td><img src="20220523-4-theta.png" width=500 /></td>
        <td><img src="20220523-5-theta.png" width=500 /></td>
        <td><img src="20220523-6-theta.png" width=500 /></td>
        <td><img src="20220523-7-theta.png" width=500 /></td>
        <td><img src="20220523-8-theta.png" width=500 /></td>
        <td><img src="20220523-9-theta.png" width=500 /></td>
        <td><img src="20220523-10-theta.png" width=500 /></td>
        <td><img src="20220523-11-theta.png" width=500 /></td>
        <td><img src="20220523-12-theta.png" width=500 /></td>
        <td><img src="20220523-13-theta.png" width=500 /></td>
        <td><img src="20220523-14-theta.png" width=500 /></td>
        <td><img src="20220523-15-theta.png" width=500 /></td>
        <td><img src="20220523-16-theta.png" width=500 /></td>
        <td><img src="20220523-17-theta.png" width=500 /></td>
        <td><img src="20220523-18-theta.png" width=500 /></td>
    </tr>
</table>

</body>
</html>

"""

'''
