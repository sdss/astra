import numpy as np
from collections import OrderedDict
from typing import Iterable, Optional
from peewee import FloatField, IntegerField

from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.database.astradb import DataProduct, SDSSOutput
from astra.base import task
from astra.utils import executable, flatten

from astra.contrib.thepayne.model import estimate_labels
from astra.contrib.thepayne.utils import read_mask, read_model
from astra.sdss.datamodels.base import get_extname
from astra.sdss.datamodels.pipeline import create_pipeline_data_product


LABEL_NAMES = ("teff", "logg", "v_turb", "c_h", "n_h", "o_h", "na_h", "mg_h", "al_h", "si_h", "p_h", "s_h", "k_h", "ca_h", "ti_h", "v_h", "cr_h", "mn_h", "fe_h", "co_h", "ni_h", "cu_h", "ge_h", "c12_c13", "v_macro")
V3_LABEL_NAMES = ('teff', 'logg', 'fe_h', 'v_micro', 'v_macro', 'c_h', 'n_h', 'o_h', 'na_h', 'mg_h', 'al_h', 'si_h', 'p_h', 's_h', 'k_h', 'ca_h', 'ti_h', 'v_h', 'cr_h', 'mn_h', 'co_h', 'ni_h', 'cu_h', 'ge_h', 'ce_h')

class ThePayne(SDSSOutput):

    v_rad = FloatField(null=True)

    chi_sq = FloatField()
    reduced_chi_sq = FloatField()
    bitmask_flag = IntegerField(default=0)


for field_name in LABEL_NAMES:
    ThePayne._meta.add_field(field_name, FloatField())
    ThePayne._meta.add_field(f"e_{field_name}", FloatField())
    ThePayne._meta.add_field(f"bitmask_{field_name}", IntegerField(default=0))

for i, field_name_i in enumerate(LABEL_NAMES):
    for field_name_j in LABEL_NAMES[i+1:]:
        ThePayne._meta.add_field(f"rho_{field_name_i}_{field_name_j}", FloatField())


class ThePayneV3(SDSSOutput):

    v_rad = FloatField(null=True)

    chi_sq = FloatField()
    reduced_chi_sq = FloatField()
    bitmask_flag = IntegerField(default=0)


for field_name in V3_LABEL_NAMES:
    ThePayneV3._meta.add_field(field_name, FloatField())
    ThePayneV3._meta.add_field(f"e_{field_name}", FloatField())
    ThePayneV3._meta.add_field(f"bitmask_{field_name}", IntegerField(default=0))

for i, field_name_i in enumerate(V3_LABEL_NAMES):
    for field_name_j in V3_LABEL_NAMES[i+1:]:
        ThePayneV3._meta.add_field(f"rho_{field_name_i}_{field_name_j}", FloatField())


@task
def the_payne(
    data_product: Iterable[DataProduct],
    model_path: str = "$MWM_ASTRA/component_data/ThePayne/payne_apogee_nn.pkl", 
    mask_path: str = "$MWM_ASTRA/component_data/ThePayne/payne_apogee_mask.npy", 
    opt_tolerance: Optional[float] = 5e-4,
    v_rad_tolerance: Optional[float] = 0,
    initial_labels: Optional[float] = None,
    continuum_method: str = "astra.tools.continuum.Chebyshev",
    continuum_kwargs: dict = dict(
        deg=4,
        regions=[(15_100.0, 15_793.0), (15_880.0, 16_417.0), (16_499.0, 17_000.0)],
        mask="$MWM_ASTRA/component_data/ThePayne/cannon_apogee_pixels.npy",
    ),
) -> Iterable[ThePayne]:

    model = read_model(model_path)
    mask = read_mask(mask_path)
    
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

    for data_product in flatten(data_product):
        outputs, results = ([], {})
        for spectrum in SpectrumList.read(data_product.path):
            if not spectrum_overlaps(spectrum, model["wavelength"]):
                continue
            
            if continuum_method is not None:
                f_continuum = executable(continuum_method)(**continuum_kwargs)
                f_continuum.fit(spectrum)
                continuum = f_continuum(spectrum)
                continuum = np.atleast_2d(continuum)
            else:
                continuum = None

            # With SpectrumList, we should only ever have 1 spectrum
            (result, ), (meta, ) = estimate_labels(
                spectrum,
                *args,
                mask=mask,
                initial_labels=initial_labels,
                v_rad_tolerance=v_rad_tolerance,
                opt_tolerance=opt_tolerance,
                continuum=continuum,
                data_product=data_product
            )

            output = ThePayne(data_product=data_product, spectrum=spectrum, **result)

            result_kwds = result.copy()
            result_kwds.update(
                OrderedDict([
                    ("model_flux", meta["model_flux"]),
                    ("continuum", meta["continuum"]),
                    ("task_id", output.task.id)
                ])
            )

            extname = get_extname(spectrum, data_product)
            results.setdefault(extname, [])
            results[extname].append(result_kwds)
            outputs.append(output)

        # Create pipeline product.
        if outputs:
            # TODO: capiTAlzIE?
            common_header_groups = [
                ("TEFF", "STELLAR PARAMETERS"),
                ("RHO_TEFF_LOGG", "CORRELATION COEFFICIENTS"),
                ("SNR", "SUMMARY STATISTICS"),
                ("MODEL_FLUX", "MODEL SPECTRA"),
            ]
            output_data_product = create_pipeline_data_product(
                "Payne",
                data_product,
                results,
                header_groups={ k: common_header_groups for k in results.keys() }
            )

            # Add this output data product to the database outputs
            for output in outputs:
                output.output_data_product = output_data_product

            yield outputs


@task
def the_payne_v3(
    data_product: Iterable[DataProduct],
    model_path: str = "$MWM_ASTRA/component_data/ThePayne/payne_apogee_nn_marcs_v3_norm.pkl",
    mask_path: str = "$MWM_ASTRA/component_data/ThePayne/payne_apogee_mask_marcs_v3_norm.npy", 
    opt_tolerance: Optional[float] = 5e-4,
    v_rad_tolerance: Optional[float] = 0,
    initial_labels: Optional[float] = None,
    continuum_method: str = "astra.tools.continuum.Chebyshev",
    continuum_kwargs: dict = dict(
        deg=4,
        regions=[(15_100.0, 15_793.0), (15_880.0, 16_417.0), (16_499.0, 17_000.0)],
        mask="$MWM_ASTRA/component_data/ThePayne/cannon_apogee_pixels.npy",
    ),
) -> Iterable[ThePayneV3]:

    model = read_model(model_path)
    mask = read_mask(mask_path)
    
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

    for data_product in flatten(data_product):
        outputs, results = ([], {})
        for spectrum in SpectrumList.read(data_product.path):
            if not spectrum_overlaps(spectrum, model["wavelength"]):
                continue
            
            if continuum_method is not None:
                f_continuum = executable(continuum_method)(**continuum_kwargs)
                f_continuum.fit(spectrum)
                continuum = f_continuum(spectrum)
                continuum = np.atleast_2d(continuum)
            else:
                continuum = None

            # With SpectrumList, we should only ever have 1 spectrum
            (result, ), (meta, ) = estimate_labels(
                spectrum,
                *args,
                mask=mask,
                initial_labels=initial_labels,
                v_rad_tolerance=v_rad_tolerance,
                opt_tolerance=opt_tolerance,
                continuum=continuum,
                data_product=data_product
            )

            output = ThePayneV3(data_product=data_product, spectrum=spectrum, **result)

            result_kwds = result.copy()
            result_kwds.update(
                OrderedDict([
                    ("model_flux", meta["model_flux"]),
                    ("continuum", meta["continuum"]),
                    ("task_id", output.task.id)
                ])
            )

            extname = get_extname(spectrum, data_product)
            results.setdefault(extname, [])
            results[extname].append(result_kwds)
            outputs.append(output)

        # Create pipeline product.
        if outputs:
            # TODO: capiTAlzIE?
            common_header_groups = [
                ("TEFF", "STELLAR PARAMETERS"),
                ("RHO_TEFF_LOGG", "CORRELATION COEFFICIENTS"),
                ("SNR", "SUMMARY STATISTICS"),
                ("MODEL_FLUX", "MODEL SPECTRA"),
            ]
            output_data_product = create_pipeline_data_product(
                "Payne",
                data_product,
                results,
                header_groups={ k: common_header_groups for k in results.keys() }
            )

            # Add this output data product to the database outputs
            for output in outputs:
                output.output_data_product = output_data_product

            yield outputs
