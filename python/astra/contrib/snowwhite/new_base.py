import numpy as np
import pickle
from functools import cache
from astropy import units as u
from astra.base import task_decorator
from astra.database.astradb import DataProduct
from astra.database.astradb import SDSSOutput
from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.utils import expand_path, flatten
from typing import Iterable
from peewee import FloatField, TextField


POLYFIT_REGIONS = [
    [3850, 3870],
    [4220, 4245],
    [5250, 5400],
    [6100, 6470],
    [7100, 9000],    
]

DEFAULT_LINE_RATIO_REGIONS = (
    [3860, 3900],  # Balmer line
    [3950, 4000],  # Balmer line
    [4085, 4120],  # Balmer line
    [4320, 4360],  # Balmer line
    [4840, 4880],  # Balmer line
    [6540, 6580],  # Balmer line
    [3880, 3905],  # He I/II line
    [3955, 3975],  # He I/II line
    [3990, 4056],  # He I/II line
    [4110, 4140],  # He I/II line
    [4370, 4410],  # He I/II line
    [4450, 4485],  # He I/II line
    [4705, 4725],  # He I/II line
    [4900, 4950],  # He I/II line
    [5000, 5030],  # He I/II line
    [5860, 5890],  # He I/II line
    [6670, 6700],  # He I/II line
    [7050, 7090],  # He I/II line
    [7265, 7300],  # He I/II line
    [4600, 4750],  # Molecular C absorption band
    [5000, 5160],  # Molecular C absorption band
    [3925, 3940],  # Ca H/K line
    [3960, 3975],  # Ca H/K line
)

@cache
def get_region_mask_for_log_uniform_wavelengths(crval, cdelt, npixels):
    wavelength = 10**(crval + cdelt * np.arange(npixels))
    region_mask = np.zeros(npixels, dtype=bool)
    for start, end in POLYFIT_REGIONS:
        region_mask += (end > wavelength) * (wavelength > start)
    return region_mask

@cache
def read_classifier(model_path):
    with open(expand_path(model_path), "rb") as fp:
        classifier = pickle.load(fp)
    return classifier


# TODO: Put this to utilities elsewhere
def _get_first_keyword(spectrum, *keys):
    for key in keys:
        if key in spectrum.meta:
            return (key, spectrum.meta[key])
    raise KeyError(f"None of the keywords {keys} found in spectrum header")
    
def _get_wavelength_keywords(spectrum):
    _, crval = _get_first_keyword(spectrum, "CRVAL", "CRVAL1")
    _, cdelt = _get_first_keyword(spectrum, "CDELT", "CDELT1")
    _, npixels = _get_first_keyword(spectrum, "NPIXELS", "NAXI1")
    return (crval, cdelt, npixels)


class LineRatioOutput(SDSSOutput):
    line_ratio = FloatField()


class WhiteDwarfClassifierOutput(SDSSOutput):
    wd_type = TextField()


@task_decorator
def measure_line_ratio(
    data_product: DataProduct,
    wavelength_lower: float,
    wavelength_upper: float,
    polyfit_order: int = 5,
) -> Iterable[LineRatioOutput]:

    for spectrum in SpectrumList.read(data_product.path):
        if not spectrum_overlaps(spectrum, 5000 * u.Angstrom):
            # Skip non-BOSS spectra.
            continue

        wavelength = spectrum.wavelength.value
        N, P = spectrum.flux.shape
        assert N == 1, "Expected 1 spectrum from SpectrumList"

        crval, cdelt, npixels = _get_wavelength_keywords(spectrum)

        region_mask = get_region_mask_for_log_uniform_wavelengths(crval, cdelt, npixels)
        finite_mask = np.isfinite(spectrum.flux).flatten()

        line_mask = (wavelength_upper > wavelength) * (wavelength > wavelength_lower) * finite_mask

        continuum_mask = region_mask * finite_mask

        func_poly = np.polyfit(
            spectrum.wavelength.value[continuum_mask], 
            spectrum.flux.value.flatten()[continuum_mask], 
            polyfit_order
        )

        p = np.poly1d(func_poly)
        
        line_ratio = np.mean(spectrum.flux.value.flatten()[line_mask]) / np.mean(p(wavelength[line_mask]))

        yield LineRatioOutput(
            data_product=data_product,
            spectrum=spectrum,
            line_ratio=line_ratio
        )


@task_decorator
def classify_white_dwarf(
    data_product,
    model_path: str = "$MWM_ASTRA/component_data/wd/training_file",
    polyfit_order: int = 5,
) -> Iterable[WhiteDwarfClassifierOutput]:

    # Measure features for all spectra in this data product.
    features = flatten(
        measure_line_ratio(data_product, wl_lower, wl_upper, polyfit_order) \
            for wl_lower, wl_upper in DEFAULT_LINE_RATIO_REGIONS
    )

    classifier = read_classifier(model_path)

    for spectrum in SpectrumList.read(data_product.path):

        line_ratios = np.array([feature.line_ratio for feature in features if feature.spectrum == spectrum])
        if np.all(np.isfinite(line_ratios)):
            wd_type, = classifier.predict(line_ratios[None])
        else:
            wd_type = "??"
        
        yield WhiteDwarfClassifierOutput(
            data_product=data_product,
            spectrum=spectrum,
            wd_type=wd_type
        )