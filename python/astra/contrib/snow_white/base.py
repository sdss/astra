import scipy.optimize as op
import numpy as np
import pickle
import os
from collections import OrderedDict
from functools import cache
from astropy import units as u
from astropy.nddata import StdDevUncertainty, InverseVariance
from astra.base import task
from astra.database.astradb import DataProduct
from astra.database.astradb import SDSSOutput
from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.utils import log, expand_path, flatten
from typing import Iterable
from peewee import FloatField, TextField
from astra.sdss.datamodels.base import get_extname

from astra.sdss.datamodels.pipeline import create_pipeline_data_product

from astra.contrib.snowwhite.fitting import (
    norm_spectra,
    fit_line,
    fit_func,
    err_func,
    hot_vs_cold,
)

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


class SnowWhiteLineRatios(SDSSOutput):
    line_ratio = FloatField()


class SnowWhiteClassification(SDSSOutput):
    wd_type = TextField()


class SnowWhite(SDSSOutput):
    wd_type = TextField()
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)

    v_rel = FloatField(null=True)
    chi_sq = FloatField(null=True)
    reduced_chi_sq = FloatField(null=True)

    conditioned_on_parallax = FloatField(null=True)
    conditioned_on_phot_g_mean_mag = FloatField(null=True)    


@task
def measure_line_ratio(
    data_product: DataProduct,
    wavelength_lower: float,
    wavelength_upper: float,
    polyfit_order: int = 5,
) -> Iterable[SnowWhiteLineRatios]:

    for spectrum in SpectrumList.read(data_product.path):
        if not spectrum_overlaps(spectrum, 5000 * u.Angstrom):
            # Skip non-BOSS spectra.
            continue

        wavelength = spectrum.wavelength.value
        N, P = np.atleast_2d(spectrum.flux).shape
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

        yield SnowWhiteLineRatios(
            data_product=data_product,
            spectrum=spectrum,
            line_ratio=line_ratio
        )


@task
def classify_white_dwarf(
    data_product,
    model_path: str = "$MWM_ASTRA/component_data/wd/training_file",
    polyfit_order: int = 5,
) -> Iterable[SnowWhiteClassification]:

    classifier = read_classifier(model_path)

    # Measure features for all spectra in this data product.
    # TODO: This is slow, but it's not run often.
    features = flatten(
        measure_line_ratio(data_product, wl_lower, wl_upper, polyfit_order) \
            for wl_lower, wl_upper in DEFAULT_LINE_RATIO_REGIONS
    )
    line_ratios = np.array([feature.line_ratio for feature in features])
    # This is a little clunky.
    # TODO: consider refactor
    index, N_line_ratios = (0, len(DEFAULT_LINE_RATIO_REGIONS))

    for spectrum in SpectrumList.read(data_product.path):
        if not spectrum_overlaps(spectrum, 5000 * u.Angstrom):
            # Skip non-BOSS spectra.
            continue        
        
        si, ei = (index, index + N_line_ratios)
        
        if np.all(np.isfinite(line_ratios[si:ei])):
            wd_type, = classifier.predict(line_ratios[si:ei][None])
        else:
            wd_type = "??"

        si += N_line_ratios

        yield SnowWhiteClassification(
            data_product=data_product,
            spectrum=spectrum,
            wd_type=wd_type
        )


@task
def snow_white(data_product: DataProduct, model_grid: str = "$MWM_ASTRA/component_data/wd/da2014") -> SnowWhite:
    """
    Estimate effective temperature and surface gravity for DA-type white dwarfs.
    """

    model_grid = expand_path(model_grid)
    data_dir = os.path.dirname(expand_path(model_grid))
    model_basename = os.path.basename(model_grid)
    wd_type = ''.join([i for i in os.path.basename(model_grid).upper() if not i.isdigit()])

    # Do the classification first.
    if not any((c.wd_type == wd_type) for c in classify_white_dwarf(data_product)):
        # TODO: put in the spectrum-level meta from one of the classifications?
        yield SnowWhite(
            data_product=data_product,
            wd_type=wd_type
        )

    outputs, results = ([], {})
    for spectrum in SpectrumList.read(data_product.path):
        if not spectrum_overlaps(spectrum, 5000 * u.Angstrom):
            # Skip non-BOSS spectra.
            continue
    
        parallax = spectrum.meta.get("PLX", None)
        phot_g_mean_mag = spectrum.meta.get("G_MAG", None)
        if parallax is None or phot_g_mean_mag is None:
            log.warning(f"Missing PLX or G_MAG in spectrum header of {data_product}")
            continue

        wl = spectrum.wavelength.value
        data = np.vstack([
            wl,
            spectrum.flux.value,
            spectrum.uncertainty.represent_as(StdDevUncertainty).array
        ]).T

        data = data[
            (np.isnan(data[:, 1]) == False) & (data[:, 0] > 3500) & (data[:, 0] < 8000)
        ]

        # normalize data
        spec_n, cont_flux = norm_spectra(data, type=wd_type)

        # crop the  relevant lines for initial guess
        line_crop = np.loadtxt(data_dir + "/line_crop.dat")
        l_crop = line_crop[(line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())]

        # fit entire grid to find good starting point
        best = fit_line(
            spec_n, l_crop, model_in=None, quick=True, model=model_basename
        )
        first_T, first_g = best[2][0], best[2][1]
        all_chi, all_TL = best[4], best[3]

        # Teff values determine what line ranges to use precise fit
        if first_T >= 16000 and first_T <= 40000:
            line_crop = np.loadtxt(data_dir + "/line_crop.dat")
            l_crop = line_crop[
                (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
            ]
        elif first_T >= 8000 and first_T < 16000:
            line_crop = np.loadtxt(data_dir + "/line_crop_cool.dat")
            l_crop = line_crop[
                (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
            ]
        elif first_T < 8000:
            line_crop = np.loadtxt(data_dir + "/line_crop_vcool.dat")
            l_crop = line_crop[
                (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
            ]
        elif first_T > 40000:
            line_crop = np.loadtxt(data_dir + "/line_crop_hot.dat")
            l_crop = line_crop[
                (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
            ]

        # fit the spectrum
        teff_bounds, logg_bounds = [(6000, 89000), (650, 899)]
        # ARC notes:
        # if the first_T and first_g are on the edge of the grid, then the optimization fails.
        # clip it to be within some amount
        first_T = np.clip(first_T, teff_bounds[0] + 100, teff_bounds[1] - 100)
        first_g = np.clip(first_g, logg_bounds[0] + 10, logg_bounds[1] - 10)
        
        new_best = op.minimize(
            fit_func,
            (first_T, first_g, 10.0),
            bounds=[teff_bounds, logg_bounds, (None, None)],
            args=(spec_n, l_crop, model_basename, 0),
            method="L-BFGS-B",
        )
        other_T = op.minimize(
            err_func,
            (first_T, first_g),
            bounds=(teff_bounds, logg_bounds),
            args=(new_best.x[2], new_best.fun, spec_n, l_crop, model_basename),
            method="L-BFGS-B",
        )
        Teff, Teff_err, logg, logg_err, rv = (
            new_best.x[0],
            abs(new_best.x[0] - other_T.x[0]),
            new_best.x[1],
            abs((new_best.x[1] - other_T.x[1]) / 100),
            new_best.x[2],
        )

        # Repeat everything for second solution
        if first_T <= 13000.0:
            other_TL, other_chi = (
                all_TL[all_TL[:, 0] > 13000.0],
                all_chi[all_TL[:, 0] > 13000.0],
            )
            other_sol = other_TL[other_chi == np.min(other_chi)]
        elif first_T > 13000.0:
            other_TL, other_chi = (
                all_TL[all_TL[:, 0] <= 13000.0],
                all_chi[all_TL[:, 0] <= 13000.0],
            )
            other_sol = other_TL[other_chi == np.min(other_chi)]

        if other_sol[0][0] >= 16000 and other_sol[0][0] <= 40000:
            line_crop = np.loadtxt(data_dir + "/line_crop.dat")
            l_crop_s = line_crop[
                (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
            ]
        elif other_sol[0][0] >= 8000 and other_sol[0][0] < 16000:
            line_crop = np.loadtxt(data_dir + "/line_crop_cool.dat")
            l_crop_s = line_crop[
                (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
            ]
        elif other_sol[0][0] < 8000:  # first_T<other_sol[0][0]:
            line_crop = np.loadtxt(data_dir + "/line_crop_vcool.dat")
            l_crop_s = line_crop[
                (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
            ]
        elif other_sol[0][0] > 40000:  # first_T>other_sol[0][0]:
            line_crop = np.loadtxt(data_dir + "/line_crop_hot.dat")
            l_crop_s = line_crop[
                (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
            ]

        sec_best = op.minimize(
            fit_func,
            (other_sol[0][0], other_sol[0][1], new_best.x[2]),
            bounds=((6000, 89000), (650, 899), (None, None)),
            args=(spec_n, l_crop, model_basename, 0),
            method="L-BFGS-B",
        )  # xtol=1. ftol=1.
        other_T2 = op.minimize(
            err_func,
            (other_sol[0][0], other_sol[0][1]),
            bounds=((6000, 89000), (650, 899)),
            args=(new_best.x[2], sec_best.fun, spec_n, l_crop, model_basename),
            method="L-BFGS-B",
        )

        Teff2, Teff_err2, logg2, logg_err2 = (
            sec_best.x[0],
            abs(sec_best.x[0] - other_T2.x[0]),
            sec_best.x[1],
            abs((sec_best.x[1] - other_T2.x[1]) / 100),
        )

        # Use Gaia parallax and photometry to determine best solution
        final_T, final_T_err, final_g, final_g_err = hot_vs_cold(
            Teff,
            Teff_err,
            logg,
            logg_err,
            Teff2,
            Teff_err2,
            logg2,
            logg_err2,
            parallax,
            phot_g_mean_mag,
            model=model_basename,
        )

        lines_s, lines_m, mod_n = fit_func(
            (final_T, 100 * final_g, rv), spec_n, l_crop, models=model_basename, mode=1
        )
        spec_w = data[:, 0]
        mod_n[np.isnan(mod_n)] = 0.0
        check_f_spec = data[:, 1][(spec_w > 4500.0) & (spec_w < 4700.0)]
        check_f_model = mod_n[:, 1][(mod_n[:, 0] > 4500.0) & (mod_n[:, 0] < 4700.0)]
        continuum = np.average(check_f_model) / np.average(check_f_spec)    

        # Resample model flux to match spectrum
        resampled_model_flux = np.interp(
            spectrum.wavelength.value,
            mod_n[:, 0] * (rv + 299792.458) / 299792.458,
            mod_n[:, 1] / continuum,
            left=np.nan,
            right=np.nan
        )
        
        # Let's only consider chi-sq between say 3750 - 8000
        chisq_mask = (8000 >= spectrum.wavelength.value) * (spectrum.wavelength.value >= 3750)
        ivar = spectrum.uncertainty.represent_as(InverseVariance).array.flatten()
        pixel_chi_sq = (resampled_model_flux - spectrum.flux.value.flatten())**2 * ivar
        chi_sq = np.nansum(pixel_chi_sq[chisq_mask])
        reduced_chi_sq = (chi_sq / (np.sum(np.isfinite(pixel_chi_sq[chisq_mask])) - 3))

        result_kwds = OrderedDict([
            ("wd_type", wd_type),
            ("teff", final_T),
            ("e_teff", final_T_err),
            ("logg", final_g),
            ("e_logg", final_g_err),
            ("v_rel", rv),
            ("conditioned_on_parallax", parallax),
            ("conditioned_on_phot_g_mean_mag", phot_g_mean_mag),
            ("snr", spectrum.meta["SNR"][0]),
            ("chi_sq", chi_sq),
            ("reduced_chi_sq", reduced_chi_sq),
        ])

        output = SnowWhite(data_product=data_product, spectrum=spectrum, **result_kwds)
        outputs.append(output)

        # Put in results for pipeline product.
        extname = get_extname(spectrum, data_product)
        results.setdefault(extname, [])
        result_kwds.update(OrderedDict([
            ("model_flux", resampled_model_flux),
            ("continuum", continuum * np.ones_like(resampled_model_flux)),
            ("task_id", output.task.id)
        ]))
        results[extname].append(result_kwds)

    # Create pipeline product.
    if outputs:
        # TODO: capiTAlzIE?
        common_header_groups = [
            ("WD_TYPE", "STELLAR PARAMETERS"),
            ("SNR", "SUMMARY STATISTICS"),
            ("MODEL_FLUX", "MODEL SPECTRA"),
        ]
        output_data_product = create_pipeline_data_product(
            "SnowWhite",
            data_product,
            results,
            header_groups={ k: common_header_groups for k in results.keys() }
        )

        # Add this output data product to the database outputs
        for output in outputs:
            output.output_data_product = output_data_product
        
    # send back the results
    yield from outputs
    