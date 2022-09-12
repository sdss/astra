from astra.utils import expand_path, executable
from astra.base import Parameter, TaskInstance, DictParameter
from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.database.astradb import database, ThePayneOutput

from astra.contrib.thepayne_new.utils import read_mask, read_model
from astra.contrib.thepayne_new.model import estimate_labels

from astra.sdss.datamodels.mwm import get_hdu_index
from astra.sdss.datamodels.pipeline import create_pipeline_product


class ThePayne(TaskInstance):

    """Estimate stellar labels using a single-layer neural network."""

    model_path = Parameter(
        default="$MWM_ASTRA/component_data/ThePayne/payne_apogee_nn.pkl", bundled=True
    )
    mask_path = Parameter(default=None, bundled=True)

    opt_tolerance = Parameter(default=5e-4)
    v_rad_tolerance = Parameter(default=0)
    initial_labels = Parameter(default=None)

    data_slice = Parameter(default=[0, 1])  # only relevant for ApStar data products
    continuum_method = Parameter(
        default="astra.tools.continuum.Chebyshev", bundled=True
    )
    continuum_kwargs = DictParameter(
        default=dict(
            deg=4,
            regions=[(15_100.0, 15_793.0), (15_880.0, 16_417.0), (16_499.0, 17_000.0)],
            mask="$MWM_ASTRA/component_data/ThePayne/cannon_apogee_pixels.npy",
        ),
        bundled=True,
    )

    def execute(self):

        model = read_model(expand_path(self.model_path))

        if self.continuum_method is not None:
            f_continuum = executable(self.continuum_method)(**self.continuum_kwargs)
        else:
            f_continuum = None

        mask = (
            None if self.mask_path is None else read_mask(expand_path(self.mask_path))
        )
        # Here we are assuming the mask is the same size as the number of model pixels.

        args = [
            model[k]
            for k in (
                "weights",
                "biases",
                "x_min",
                "x_max",
                "wavelength",
                "label_names",
            )
        ]

        for task, data_products, parameters in self.iterable():
            data_slice = parameters.get("data_slice", None)
            for data_product in data_products:
                results = []
                for spectrum in SpectrumList.read(
                    data_product.path, data_slice=data_slice
                ):
                    if spectrum_overlaps(spectrum, model["wavelength"]):
                        if f_continuum is not None:
                            f_continuum.fit(spectrum)
                            continuum = f_continuum(spectrum)
                        else:
                            continuum = None

                        results.append(
                            estimate_labels(
                                spectrum,
                                *args,
                                mask=mask,
                                initial_labels=parameters["initial_labels"],
                                v_rad_tolerance=parameters["v_rad_tolerance"],
                                opt_tolerance=parameters["opt_tolerance"],
                                continuum=continuum,
                            )
                        )
                    else:
                        results.append(None)

                # Create or update rows.
                label_results = []
                for spectrum_results in results:
                    if spectrum_results is None:
                        continue
                    for labels, model_spectrum, meta in spectrum_results:
                        label_results.append(labels)

                with database.atomic():
                    task.create_or_update_outputs(ThePayneOutput, label_results)

                # Most input data products will be mwmVisit/mwmStar files, so len(results) == 4 always.
                # If it's an ApStar file, then len(results) == 1. So we should pad the results list.
                # TODO: Put this somewhere common.
                if data_product.filetype in ("apStar", "apStar-1m"):
                    _results = [None, None, None, None]
                    index = get_hdu_index(
                        data_product.filetype, data_product.kwargs["telescope"]
                    )
                    # This gives us the index including the primary index, so we need to subtract 1.
                    _results[index - 1] = results[0]
                    results = _results

                # Create astraStar/astraVisit data product and link it to this task.
                create_pipeline_product(
                    task, data_product, results, pipeline=self.__class__.__name__
                )
