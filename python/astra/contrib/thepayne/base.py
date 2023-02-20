
from typing import Iterable, Optional
from peewee import FloatField, IntegerField

from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.database.astradb import DataProduct, SDSSOutput
from astra.base import task_decorator
from astra.utils import executable

from astra.contrib.thepayne.model import estimate_labels
from astra.contrib.thepayne.utils import read_mask, read_model


LABEL_NAMES = ("teff", "logg", "v_turb", "c_h", "n_h", "o_h", "na_h", "mg_h", "al_h", "si_h", "p_h", "s_h", "k_h", "ca_h", "ti_h", "v_h", "cr_h", "mn_h", "fe_h", "co_h", "ni_h", "cu_h", "ge_h", "c12_c13", "v_macro")

class ThePayneOutput(SDSSOutput):

    v_rad = FloatField(null=True)

    chi_sq = FloatField()
    reduced_chi_sq = FloatField()
    bitmask_flag = IntegerField(default=0)


for field_name in LABEL_NAMES:
    ThePayneOutput._meta.add_field(field_name, FloatField())
    ThePayneOutput._meta.add_field(f"e_{field_name}", FloatField())
    ThePayneOutput._meta.add_field(f"bitmask_{field_name}", IntegerField(default=0))

for i, field_name_i in enumerate(LABEL_NAMES):
    for field_name_j in LABEL_NAMES[i+1:]:
        ThePayneOutput._meta.add_field(f"rho_{field_name_i}_{field_name_j}", FloatField())


@task_decorator
def the_payne(
    data_product: DataProduct,
    model_path: str = "$MWM_ASTRA/component_data/ThePayne/payne_apogee_nn.pkl",
    mask_path: str = "$MWM_ASTRA/component_data/ThePayne/payne_apogee_mask.npy",
    opt_tolerance: Optional[float] = 5e-4,
    v_rad_tolerance: Optional[float] = 0,
    initial_labels: Optional[Iterable[float]] = None,
    continuum_method: str = "astra.tools.continuum.Chebyshev",
    continuum_kwargs: dict = dict(
        deg=4,
        regions=[(15_100.0, 15_793.0), (15_880.0, 16_417.0), (16_499.0, 17_000.0)],
        mask="$MWM_ASTRA/component_data/ThePayne/cannon_apogee_pixels.npy",
    ),
) -> Iterable[ThePayneOutput]:

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

    for spectrum in SpectrumList.read(data_product.path):
        if not spectrum_overlaps(spectrum, model["wavelength"]):
            continue
        
        if continuum_method is not None:
            f_continuum = executable(continuum_method)(**continuum_kwargs)
            f_continuum.fit(spectrum)
            continuum = f_continuum(spectrum)
        else:
            continuum = None

        labels, meta = estimate_labels(
            spectrum,
            *args,
            mask=mask,
            initial_labels=initial_labels,
            v_rad_tolerance=v_rad_tolerance,
            opt_tolerance=opt_tolerance,
            continuum=continuum,
            data_product=data_product
        )
        assert len(labels) == 1, "Expected 1 spectrum"
        result = ThePayneOutput(
            data_product=data_product,
            spectrum=spectrum,
            **labels[0]
        )

        print(f"Write out astraStar product")
        yield result
        # TODO: Write out astrastar product

'''
                hdu_results = list_to_dict(labels)
                hdu_results.update(list_to_dict(meta))
                results[extname] = hdu_results
                header_groups[extname] = [
                    ("TEFF", "STELLAR LABELS"),
                    ("RHO_TEFF_LOGG", "CORRELATION COEFFICIENTS"),
                    ("SNR", "SUMMARY STATISTICS"),
                    ("MODEL_FLUX", "MODEL SPECTRA")
                ]

            with database.atomic():
                task.create_or_update_outputs(ThePayneOutput, database_results)

            # Create astraStar/astraVisit data product and link it to this task.
            create_pipeline_product(task, data_product, results, header_groups=header_groups)
'''