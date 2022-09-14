from astra import log
from astra.utils import executable
from astra.base import Parameter, TaskInstance, DictParameter
from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.database.astradb import database, ThePayneOutput

from astra.contrib.thepayne.utils import read_mask, read_model
from astra.contrib.thepayne.model import estimate_labels

from astra.sdss.datamodels.pipeline import create_pipeline_product


class ThePayne(TaskInstance):

    """Estimate stellar labels using a single-layer neural network."""

    model_path = Parameter(
        default="$MWM_ASTRA/component_data/ThePayne/payne_apogee_nn.pkl", bundled=True
    )
    # Note: mask should be relative to the model pixels, not the observed pixels.
    # TODO: document this somewhere, or better yet, put it in the model file itself.
    mask_path = Parameter(
        default="$MWM_ASTRA/component_data/ThePayne/payne_apogee_mask.npy", bundled=True
    )

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

        model = read_model(self.model_path)

        if self.continuum_method is not None:
            f_continuum = executable(self.continuum_method)(**self.continuum_kwargs)
        else:
            f_continuum = None

        mask = None if self.mask_path is None else read_mask(self.mask_path)
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
                    if spectrum_results is not None:
                        for labels, model_spectrum, meta in spectrum_results:
                            label_results.append(labels)

                with database.atomic():
                    task.create_or_update_outputs(ThePayneOutput, label_results)

                # Create astraStar/astraVisit data product and link it to this task.
                try:
                    create_pipeline_product(
                        task, data_product, results, pipeline=self.__class__.__name__
                    )
                except:
                    log.exception(
                        f"Failed to create pipeline product for task {task} and data product {data_product}:"
                    )
