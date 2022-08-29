import numpy as np
import os
import pickle
from astra import log, __version__
from astra.base import ExecutableTask, Parameter
from astra.database.astradb import (
    DataProduct,
    TaskOutput,
    Output,
    TheCannonOutput,
    TaskInputDataProducts,
    database,
    TaskOutputDataProducts,
)
from astropy.table import Table

from astra.tools.spectrum import Spectrum1D
from astra.utils import executable, expand_path
from sdss_access import SDSSPath
from tqdm import tqdm

from astra.contrib.thecannon_new.model import CannonModel


def result_dict(p_opt, p_cov, label_names):
    """
    Return a dictionary of results that can be supplied to the database.
    """
    uncertainty_prefix, rho_prefix = "u_", "rho_"
    result = dict(zip(label_names, p_opt))
    result.update(
        dict(
            zip(
                [f"{uncertainty_prefix}{name}" for name in label_names],
                np.sqrt(np.diag(p_cov)),
            )
        )
    )

    # correlation coefficients
    L = len(label_names)
    for i, j in zip(*np.tril_indices(L, k=-1)):
        key = f"{rho_prefix}{label_names[i]}_{label_names[j]}"
        result[key] = p_cov[i, j] / np.sqrt(p_cov[i, i] * p_cov[j, j])

    return result


class TestTheCannon(ExecutableTask):

    model_path = Parameter("model_path", bundled=True)

    normalization_method = Parameter("normalization_method", default=None, bundled=True)
    normalization_kwds = Parameter("normalization_kwds", default=None, bundled=True)
    slice_args = Parameter("slice_args", default=None, bundled=True)

    def execute(self):

        model = CannonModel.read(self.model_path)
        model.n_threads = 1

        total = len(self.context["tasks"])
        for i, (task, data_product, _) in enumerate(tqdm(self.iterable(), total=total)):
            spectrum = self.slice_and_normalize_spectrum(data_product[0])
            flux = np.atleast_2d(spectrum.flux.value)
            ivar = np.atleast_2d(spectrum.uncertainty.array)

            with database.atomic():
                for j, (p_opt, p_cov, meta) in enumerate(
                    model.fit_spectrum(flux, ivar, tqdm_kwds=dict(disable=True))
                ):
                    result = result_dict(p_opt, p_cov, model.label_names)
                    result.update(snr=spectrum.meta["snr"][j], **meta)

                    output = Output.create()
                    TaskOutput.create(task=task, output=output)
                    TheCannonOutput.create(
                        task=task,
                        output=output,
                        **result,
                    )

            # if i in Ns_cumsum:
            #    results.append([])
            # result = result_dict(p_opt, p_cov, model.label_names)
            # result.update(meta)
            # results[-1].append(result)
            """
            N, P = flux.shape
            #x0 = np.repeat(model.offsets, N).reshape((N, -1))

            task_results = []
            for p_opt, p_cov, meta in model.fit_spectrum(flux, ivar, tqdm_kwds=dict(disable=True)):
                result = result_dict(p_opt, p_cov, model.label_names)
                result.update(meta)
                task_results.append(result)
            results.append(task_results)
            """
        return None

    """
    def post_execute(self):
        for (task, data_product, _), results in zip(self.iterable(), tqdm(self.result)):
            for result_dict in results:
                output = Output.create()
                TaskOutput.create(task=task, output=output)
                TheCannonOutput.create(
                    task=task,
                    output=output,
                    **result_dict,
                )
    """

    def slice_and_normalize_spectrum(self, data_product):

        spectrum = Spectrum1D.read(data_product.path)
        if self.slice_args is not None:
            slices = tuple([slice(*args) for args in self.slice_args])
            spectrum._data = spectrum._data[slices]
            spectrum._uncertainty.array = spectrum._uncertainty.array[slices]
            for key in ("bitmask", "snr"):
                try:
                    spectrum.meta[key] = np.array(spectrum.meta[key])[slices]
                except:
                    log.exception(
                        f"Unable to slice '{key}' metadata with {self.slice_args} on {data_product}"
                    )

        if self.normalization_method is not None:
            try:
                self._normalizer
            except AttributeError:
                klass = executable(self.normalization_method)
                kwds = self.normalization_kwds or dict()
                self._normalizer = klass(spectrum, **kwds)
            else:
                self._normalizer.spectrum = spectrum
            finally:
                return self._normalizer()
        else:
            return spectrum
