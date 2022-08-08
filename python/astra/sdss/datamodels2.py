import datetime
import os
import numpy as np
from typing import Union, List, Callable, Optional, Dict
from astropy.io import fits
from astropy.constants import c
from astropy import units as u
from healpy import ang2pix
from functools import partial

from astra import (log, __version__ as astra_version)
from astra.utils import flatten, list_to_dict
from astra.database.astradb import Source, DataProduct
from astra.base import ExecutableTask, Parameter

from scipy import interpolate
from scipy.ndimage.filters import median_filter, gaussian_filter

from astra.sdss.apogee_bitmask import PixelBitMask # TODO: put elsewhere

from .catalog import get_sky_position

C_KM_S = c.to(u.km / u.s).value
BLANK_CARD = (None, None, None)
FILLER_CARD = (FILLER_CARD_KEY, *_) = ("TTYPE0", "Water cuggle", None)

GLOSSARY = {
    #           "*****************************************************"
    "INSTRMNT": "Instrument name",
    "TELESCOP": "Telescope name",
    # Observing conditions
    "ALT": "Telescope altitude [deg]",
    "AZ": "Telescope azimuth [deg]",
    "EXPTIME":  "Total exposure time [s]",
    "NEXP":     "Number of exposures taken",
    "AIRMASS": "Mean airmass",
    "AIRTEMP": "Air temperature [C]",
    "DEWPOINT": "Dew point temperature [C]",
    "HUMIDITY": "Humidity [%]",
    "PRESSURE": "Air pressure [inch Hg?]", # TODO
    "MOON_PHASE_MEAN": "Mean phase of the moon",
    "MOON_DIST_MEAN": "Mean sky distance to the moon [deg]",
    "SEEING": "Median seeing conditions [arcsecond]",
    "GUSTD": "Wind gust direction [deg]",
    "GUSTS": "Wind gust speed [km/s]",
    "WINDD": "Wind direction [deg]",
    "WINDS": "Wind speed [km/s]",
    "TAI-BEG": "MJD (TAI) at start of integrations [s]",
    "TAI-END": "MJD (TAI) at end of integrations [s]",
    "NGUIDE": "Number of guider frames during integration",

    # Stacking
    "V_HELIO": "Heliocentric velocity correction [km/s]",
    "V_SHIFT": "Relative velocity shift used in stack [km/s]",
    "IN_STACK": "Was this spectrum used in the stack?",

    # Metadata related to sinc interpolation and stacking
    "NRES":     "Sinc bandlimit [pixel/resolution element]",
    "FILTSIZE": "Median filter size for pseudo-continuum [pixel]",
    "NORMSIZE": "Gaussian width for pseudo-continuum [pixel]",
    "CONSCALE": "Scale by pseudo-continuum when stacking",
    "STACK_VRAD": "Radial velocity used when stacking spectra [km/s]",

    # BOSS data reduction pipeline
    "V_BOSS": "Version of the BOSS ICC",
    "VJAEGER": "Version of Jaeger",
    "VKAIJU": "Version of Kaiju",
    "VCOORDIO": "Version of coordIO",
    "VCALIBS": "Version of FPS calibrations",
    "VERSREAD": "Version of idlspec2d for processing raw data",
    "VERSIDL": "Version of IDL",
    "VERSUTIL": "Version of idlutils",
    "VERS2D": "Version of idlspec2d for 2D reduction",
    "VERSCOMB": "Version of idlspec2d for combining exposures",
    "VERSLOG": "Version of SPECLOG product",
    "VERSFLAT": "Version of SPECFLAT product",
    "DIDFLUSH": "Was CCD flushed before integration",
    "CARTID": "Cartridge identifier",
    "PSFSKY":   "Order of PSF sky subtraction",
    "PREJECT":  "Profile area rejection threshold",
    "LOWREJ":   "Extraction: low rejection",
    "HIGHREJ":  "Extraction: high rejection",
    "SCATPOLY": "Extraction: Order of scattered light polynomial",
    "PROFTYPE": "Extraction profile: 1=Gaussian",
    "NFITPOLY": "Extraction: Number of profile parameters",
    "RDNOISE0": "CCD read noise amp 0 [electrons]",
    "SKYCHI2": "Mean \chi^2 of sky subtraction",
    "SCHI2MIN": "Minimum \chi^2 of sky subtraction",
    "SCHI2MAX": "Maximum \chi^2 of sky subtraction",

    # APOGEE data reduction pipeline
    "DATE-OBS": "Observation date (UTC)",
    "JD-MID": "Julian date at mid-point of visit",
    "UT-MID": "Date at mid-point of visit",
    "FLUXFLAM": "ADU to flux conversion factor [ergs/s/cm^2/A]",
    "NPAIRS": "Number of dither pairs combined",

    # XCSAO
    "V_REL_XCSAO": "Relative velocity from XCSAO [km/s]",
    "E_V_REL_XCSAO": "Error in Relative velocity from XCSAO [km/s]",
    #"XCSAO_RXC": 
    "TEFF_XCSAO": "Effective temperature from XCSAO [K]",
    "E_TEFF_XCSAO": "Error in effective temperature from XCSAO [K]",
    "LOGG_XCSAO": "Surface gravity from XCSAO",
    "E_LOGG_XCSAO": "Error in surface gravity from XCSAO",
    "FEH_XCSAO": "Metallicity from XCSAO",
    "E_FEH_XCSAO": "Error in metallicity from XCSAO",

    # Data things
    "FLUX": "Source flux",
    "E_FLUX": "Standard deviation of source flux",
    "SNR": "Mean signal-to-noise ratio",

    # Wavelength solution
    "CRVAL1": "Log(10) wavelength of first pixel [Angstrom]",
    "CDELT1": "Log(10) delta wavelength per pixel [Angstrom]",
    "CRPIX1": "Pixel offset from the first pixel",
    "CTYPE1": "Wavelength solution description",
    "DC-FLAG": "Wavelength solution flag",

    # BOSS data model keywords
    "RUN2D":    "Spectro-2D reduction name",
    "FIELDID": "Field identifier",
    "MJD": "Modified Julian Date of the observations",
    "CATALOGID": "SDSS-V catalog identifier",
    "ISPLATE": "Whether the data were taken with plates",

    # APOGEE data model keywords
    "FIBER": "Fiber number",
    "TELESCOPE": "Telescope name",
    "PLATE": "Plate number",
    "FIELD": "Field number",
    "APRED": "APOGEE reduction tag",

    # Radial velocity keys (common to any code)
    "V_BARY": "Barycentric radial velocity [km/s]",
    "V_BC": "Barycentric velocity correction applied [km/s]",
    "V_REL": "Relative velocity [km/s]",
    "E_V_REL": "Error in relative velocity [km/s]",
    "JD": "Julian date at mid-point of visit",

    # Doppler keys
    "TEFF_DP": "Effective temperature from DOPPLER [K]",
    "E_TEFF_DP": "Error in effective temperature from DOPPLER [K]",
    "LOGG_DP": "Surface gravity from DOPPLER",
    "E_LOGG_DP": "Error in surface gravity from DOPPLER",
    "FEH_DP": "Metallicity from DOPPLER",
    "E_FEH_DP": "Error in metallicity from DOPPLER",
    "RCHISQ": "Reduced \chi-squared of model fit",
    "VISIT_PK": "Primary key in `apogee_drp.visit` table",
    "RV_VISIT_PK": "Primary key in `apogee_drp.rv_visit` table",
    "N_RV_COMPONENTS": "Number of detected RV components",
    "RV_COMPONENTS": "Relative velocity of detected components [km/s]",
    "V_REL_XCORR": "Relative velocity from XCORR [km/s]",
    "E_V_REL_XCORR": "Error in relative velocity from XCORR [km/s]",
    "V_BARY_XCORR": "Barycentric radial velocity from XCORR [km/s]",
}
for key, comment in GLOSSARY.items():
    if len(comment) > 80:
        log.warning(
            f"Glossary term {key} has a comment that is longer than 80 characters. "
            f"It will be truncated from:\n{comment}\nTo:\n{comment[:80]}"
        )


_filter_data_products = lambda idp, filetype, telescope=None: filter(
    lambda dp: (dp.filetype == filetype) and ((telescope is None) or (telescope == dp.kwargs["telescope"])),
    idp
)


def create_mwm_data_product(source):

    hdu_descriptions = [
        "Source information only",
        "BOSS spectra from Apache Point Observatory",
        "BOSS spectra from Las Campanas Observatory",
        "APOGEE spectra from Apache Point Observatory",
        "APOGEE spectra from Las Campanas Observatory"
    ]

    primary_hdu = create_primary_hdu(source, hdu_descriptions)


    # BOSS
    boss_south = create_empty_hdu("lco25m", "BOSS")
    
    boss_north, boss_stack_args = create_boss_visits_hdu(
        list(_filter_data_products(source.data_products, "specLite")),
    )
    
    boss_north_stack = create_star_hdu(
        *boss_stack_args,
        telescope="apo25m",
        instrument="BOSS"
    )

    # APOGEE
    apogee_north, apo25m_resampled = create_apogee_visits_hdu(
        list(_filter_data_products(source.data_products, "apVisit", "apo25m")),
    )
    apogee_south, lco25m_resampled = create_apogee_visits_hdu(
        list(_filter_data_products(source.data_products, "apVisit", "lco25m")),
    )

    if apo25m_resampled is None:
        apogee_north_star = create_empty_hdu("apo25m", "APOGEE")
    else:
        apogee_north_star = create_star_hdu(
            *apo25m_resampled,
            telescope="apo25m",
            instrument="APOGEE"
        )

    if lco25m_resampled is None:
        apogee_south_star = create_empty_hdu("lco25m", "APOGEE")
    else:
        apogee_south_star = create_star_hdu(
            *lco25m_resampled,
            telescope="lco25m",
            instrument="APOGEE"
        )

    hdulist_visits = fits.HDUList([
        primary_hdu,
        boss_north,
        boss_south,
        apogee_north,
        apogee_south,
    ])

    
    hdulist_stack = fits.HDUList([
        primary_hdu,
        boss_north_stack,
        boss_south,
        apogee_north_star,
        apogee_south_star
    ])

    return (hdulist_visits, hdulist_stack)


def pixel_weighted_spectrum(flux, flux_error, continuum, bitmask):

    # Pixel-by-pixel weighted average
    resampled_ivar = 1.0 / flux_error**2
    resampled_ivar[flux_error == 0] = 0

    pixel_snr_clipped = np.clip(flux * np.sqrt(resampled_ivar), 0, np.inf)
    #estimate_snr = np.mean(pixel_snr_clipped, axis=1)

    cont = np.median(continuum, axis=0) # TODO: is this right?
    stacked_ivar = np.sum(resampled_ivar, axis=0)
    stacked_flux = np.sum(flux * resampled_ivar, axis=0) / stacked_ivar * cont
    stacked_flux_error = np.sqrt(1.0/stacked_ivar) * cont

    return (stacked_flux, stacked_flux_error, bitmask)


def create_empty_hdu(telescope: str, instrument: str) -> fits.BinTableHDU:
    """
    Create an empty HDU to use as a filler.
    """
    cards = [
        BLANK_CARD,
        ("", "METADATA"),
        ("TELESCOP", telescope),
        ("INSTRUME", instrument),
    ]
    return fits.BinTableHDU(header=fits.Header(cards))


def create_star_hdu(
    flux: np.ndarray,
    flux_error: np.ndarray,
    continuum: Optional[np.ndarray],
    bitmask: np.ndarray,
    meta: Dict,
    telescope: Optional[str] = None,
    instrument: Optional[str] = None,
) -> fits.BinTableHDU:
    """
    Create a binary table HDU containing a stacked spectrum.
    """

    cards = [
        BLANK_CARD,
        ("", "METADATA"),
        ("TELESCOP", telescope),
        ("INSTRUME", instrument),
    ]

    # Data reduction pipeline keywords..
    # TODO: allow cards

    cards.extend([
        BLANK_CARD,
        ("", "SPECTRUM SAMPLING AND STACKING"),
        ("NRES",        meta["num_pixels_per_resolution_element"]),
        ("FILTSIZE",    meta["median_filter_size"]),
        ("NORMSIZE",    meta["gaussian_filter_size"]),
        ("CONSCALE",    meta["scale_by_pseudo_continuum"]),

        BLANK_CARD,
        ("", "WAVELENGTH INFORMATION (VACUUM)", None),
        ("CRVAL1", meta["crval"], None),
        ("CDELT1", meta["cdelt"], None),
        ("CTYPE1", "LOG-LINEAR", None),
        ("CRPIX1", 1, None),
        ("DC-FLAG", 1, None),
        ("NAXIS1", meta["num_pixels"], "Number of pixels per spectrum"),
        BLANK_CARD,
        ("", "SPECTRAL DATA", None),
        FILLER_CARD
    ])
    
    stacked_flux, stacked_flux_error, stacked_bitmask = pixel_weighted_spectrum(
        flux, flux_error, continuum, bitmask
    )

    # TODO: handle bitmask stack
    header = fits.Header(cards)
    mappings = [
        ("FLUX", stacked_flux),
        ("E_FLUX", stacked_flux_error),
        ("BITMASK", bitmask),
    ]
    columns = []
    for key, values in mappings:
        columns.append(
            fits.Column(
                name=key,
                array=values,
                unit=None, 
                **fits_column_kwargs(values)
            )
        )
    
    hdu = fits.BinTableHDU.from_columns(columns, header=header)        
    # Add explanatory things from glossary
    for key in hdu.header.keys():
        hdu.header.comments[key] = GLOSSARY.get(key, None)
    for i, key in enumerate(hdu.data.dtype.names, start=1):
        hdu.header.comments[f"TTYPE{i}"] = GLOSSARY.get(key, None)
    if FILLER_CARD_KEY is not None:
        try:
            del hdu.header[FILLER_CARD_KEY]
        except:
            None
        
    return hdu


def create_boss_visits_hdu(data_products: List[Union[DataProduct, str]], telescope=None, **kwargs):
    """
    Create a binary table HDU containing all BOSS visit spectra.

    :param data_products:
        A list of SpecLite data products.
    """
    cards = [
        BLANK_CARD,
        ("", "METADATA"),
        ("TELESCOP", telescope), 
        ("INSTRMNT", "BOSS"),
    ]    
    if len(data_products) == 0:
        return (fits.BinTableHDU(header=fits.Header(cards)), None)

    flux, flux_error, cont, bitmask, meta = resampled = resample_boss_visit_spectra(
        data_products,
        **kwargs
    )
    snr = np.nanmean(flux / flux_error, axis=1)


    cards.extend([
        BLANK_CARD,
        ("",   "BOSS DATA REDUCTION PIPELINE"),
    ])
    drp_keys = (
        "V_BOSS", 
        "VJAEGER", 
        "VKAIJU",
        "VCOORDIO",
        "VCALIBS",
        "VERSIDL",
        "VERSUTIL",
        "VERSREAD",
        "VERS2D",
        "VERSCOMB",
        "VERSLOG",
        "VERSFLAT",
        "DIDFLUSH", "CARTID", 
        "PSFSKY", "PREJECT", "LOWREJ", "HIGHREJ", "SCATPOLY", "PROFTYPE", "NFITPOLY",
        "SKYCHI2", "SCHI2MIN", "SCHI2MAX",
        ("HELIO_RV", "V_HELIO"),
        "RDNOISE0"
    )

    with fits.open(data_products[0].path) as image:
        for key in drp_keys:
            if isinstance(key, tuple):
                old_key, new_key = key
            else:
                old_key = new_key = key
            try:
                value = image[0].header[old_key]
                comment = image[0].header.comments[old_key]
            except KeyError:
                log.warning(f"No {old_key} header of HDU 0 in {data_products[0].path}")
                value = comment = None

            cards.append((
                new_key, 
                value, 
                GLOSSARY.get(new_key, comment)
            ))

    cards.extend([
        BLANK_CARD,
        ("", "SPECTRUM SAMPLING AND STACKING"),
        ("NRES",        meta["num_pixels_per_resolution_element"]),
        ("FILTSIZE",    meta["median_filter_size"]),
        ("NORMSIZE",    meta["gaussian_filter_size"]),
        ("CONSCALE",    meta["scale_by_pseudo_continuum"]),

        BLANK_CARD,
        ("", "WAVELENGTH INFORMATION (VACUUM)", None),
        ("CRVAL1", meta["crval"], None),
        ("CDELT1", meta["cdelt"], None),
        ("CTYPE1", "LOG-LINEAR", None),
        ("CRPIX1", 1, None),
        ("DC-FLAG", 1, None),
        ("NAXIS1", meta["num_pixels"], "Number of pixels per spectrum"),
        FILLER_CARD
    ])

    header = fits.Header(cards)
    mappings = [
        # key, function
        ("SPECTRAL DATA", None),
        ("FLUX", flux),
        ("FLUX_ERROR", flux_error),
        ("BITMASK", bitmask),
        ("SNR", snr),

        ("INPUT DATA MODEL KEYWORDS", None),   
        # https://stackoverflow.com/questions/6076270/lambda-function-in-list-comprehensions
        *[(k.upper(), partial(lambda dp, image, _k: dp.kwargs[_k], _k=k)) for k in data_products[0].kwargs.keys()],
        ("OBSERVING CONDITIONS", None),
        ("ALT", lambda dp, image: image[0].header["ALT"]),
        ("AZ", lambda dp, image: image[0].header["AZ"]),
        ("SEEING", lambda dp, image: image[2].data["SEEING50"][0]),
        ("AIRMASS", lambda dp, image: image[2].data["AIRMASS"][0]),
        ("AIRTEMP", lambda dp, image: image[0].header["AIRTEMP"]),
        ("DEWPOINT", lambda dp, image: image[0].header["DEWPOINT"]),
        ("HUMIDITY", lambda dp, image: image[0].header["HUMIDITY"]),
        ("PRESSURE", lambda dp, image: image[0].header["PRESSURE"]),
        ("GUSTD", lambda dp, image: image[0].header["GUSTD"]),
        ("GUSTS", lambda dp, image: image[0].header["GUSTS"]),
        ("WINDD", lambda dp, image: image[0].header["WINDD"]),
        ("WINDS", lambda dp, image: image[0].header["WINDS"]),
        ("MOON_DIST_MEAN", lambda dp, image: np.mean(list(map(float, image[2].data["MOON_DIST"][0][0].split(" "))))),
        ("MOON_PHASE_MEAN", lambda dp, image: np.mean(list(map(float, image[2].data["MOON_PHASE"][0][0].split(" "))))),
        ("EXPTIME", lambda dp, image: image[2].data["EXPTIME"][0]),
        ("NEXP", lambda dp, image: image[2].data["NEXP"][0]),
        ("NGUIDE", lambda dp, image: image[0].header["NGUIDE"]),
        ("TAI-BEG", lambda dp, image: image[0].header["TAI-BEG"]),
        ("TAI-END", lambda dp, image: image[0].header["TAI-END"]),


        ("RADIAL VELOCITIES (XCSAO)", None),
        ("V_REL_XCSAO", lambda dp, image: image[2].data["XCSAO_RV"][0]),
        ("E_V_REL_XCSAO", lambda dp, image: image[2].data["XCSAO_ERV"][0]),
        ("RXC_XCSAO", lambda dp, image: image[2].data["XCSAO_RXC"][0]),
        ("TEFF_XCSAO", lambda dp, image: image[2].data["XCSAO_TEFF"][0]),
        ("E_TEFF_XCSAO", lambda dp, image: image[2].data["XCSAO_ETEFF"][0]),
        ("LOGG_XCSAO", lambda dp, image: image[2].data["XCSAO_LOGG"][0]),
        ("E_LOGG_XCSAO", lambda dp, image: image[2].data["XCSAO_ELOGG"][0]),
        ("FEH_XCSAO", lambda dp, image: image[2].data["XCSAO_FEH"][0]),
        ("E_FEH_XCSAO", lambda dp, image: image[2].data["XCSAO_EFEH"][0]),

        ("SPECTRUM SAMPLING AND STACKING", None),
        ("V_SHIFT",     meta["v_shift"]),
        # TODO: have some criteria for whether we use all in the stack or not
        ("IN_STACK",    np.ones(len(data_products), dtype=bool)),
        
    ]

    table_category_headers = []
    values = {} 
    for j, data_product in enumerate(data_products):
        with fits.open(data_product.path) as image:
            for i, (key, function) in enumerate(mappings):
                if j == 0 and function is None:
                    table_category_headers.append((mappings[i + 1][0], key))
                else:
                    values.setdefault(key, [])
                    if callable(function):
                        try:
                            value = function(data_product, image)
                        except KeyError:
                            log.warning(f"No {key} found in {data_product.path}")
                            value = None
                        values[key].append(value)
                    else:
                        values[key] = function

    columns = []
    for key, function in mappings:
        if function is None: continue
        columns.append(
            fits.Column(
                name=key,
                array=values[key],
                unit=None, 
                **fits_column_kwargs(values[key])
            )
        )
    
    hdu = fits.BinTableHDU.from_columns(columns, header=header)
    for dtype_name, category_header in table_category_headers:
        index = 1 + hdu.data.dtype.names.index(dtype_name)
        key = f"TTYPE{index}"
        hdu.header.insert(key, BLANK_CARD)
        hdu.header.insert(key, ("", category_header))
        
    # Add explanatory things from glossary
    for key in hdu.header.keys():
        hdu.header.comments[key] = GLOSSARY.get(key, None)
    for i, key in enumerate(hdu.data.dtype.names, start=1):
        hdu.header.comments[f"TTYPE{i}"] = GLOSSARY.get(key, None)
    if FILLER_CARD_KEY is not None:
        try:
            del hdu.header[FILLER_CARD_KEY]
        except:
            None
        
    return (hdu, resampled)




def create_apogee_visits_hdu(
    data_products: List[Union[DataProduct, str]], 
    telescope: Optional[str] = None,
    **kwargs
):
    """
    Create a binary table HDU containing all given APOGEE visit spectra.

    :param data_products:
        A list of ApVisit data products.
    """
    
    cards = [
        BLANK_CARD,
        ("", "METADATA"),
        ("TELESCOP", telescope), 
        ("INSTRMNT", "APOGEE"),
    ]

    if len(data_products) == 0:
        return (fits.BinTableHDU(header=fits.Header(cards)), None)
    
    # First get the velocity information from the APOGEE data reduction pipeline database.
    velocity_meta = list_to_dict(tuple(map(get_apogee_visit_radial_velocity, data_products)))

    flux, flux_error, cont, bitmask, meta = resampled = resample_apogee_visit_spectra(
        data_products,
        radial_velocities=velocity_meta["V_REL"],
        **kwargs
    )

    snr = np.nanmean(flux / flux_error, axis=1)
    

    cards.extend([
        BLANK_CARD,
        ("",   "APOGEE DATA REDUCTION PIPELINE"),
    ])
    drp_keys = (
        "V_APRED", 
    )
    # MEANFIB and SIGFIB are in ApStar, but we are using ApVisits..

    with fits.open(data_products[0].path) as image:
        for key in drp_keys:
            if isinstance(key, tuple):
                old_key, new_key = key
            else:
                old_key = new_key = key
            try:
                value = image[0].header[old_key]
                comment = image[0].header.comments[old_key]
            except KeyError:
                log.warning(f"No {old_key} header of HDU 0 in {data_products[0].path}")
                value = comment = None

            cards.append((
                new_key, 
                value, 
                GLOSSARY.get(new_key, comment)
            ))

    cards.extend([
        BLANK_CARD,
        ("", "SPECTRUM SAMPLING AND STACKING"),
        ("NRES",        " ".join(map(str, meta["num_pixels_per_resolution_element"]))),
        ("FILTSIZE",    meta["median_filter_size"]),
        ("NORMSIZE",    meta["gaussian_filter_size"]),
        ("CONSCALE",    meta["scale_by_pseudo_continuum"]),

        # Since DOPPLER uses the same Cannon model for the final fit of all individual visits,
        # we include it here instead of repeating the information many times in the data table.
        BLANK_CARD,
        ("", "DOPPLER STELLAR PARAMETERS", None),
        *[(f"{k}_D", velocity_meta[f"{k}_DOPPLER"][0]) for k in (
            "TEFF", "E_TEFF",
            "LOGG", "E_LOGG",
            "FEH", "E_FEH",
        )],

        BLANK_CARD,
        ("", "WAVELENGTH INFORMATION (VACUUM)", None),
        ("CRVAL1", meta["crval"], None),
        ("CDELT1", meta["cdelt"], None),
        ("CTYPE1", "LOG-LINEAR", None),
        ("CRPIX1", 1, None),
        ("DC-FLAG", 1, None),
        ("NAXIS1", meta["num_pixels"], "Number of pixels per spectrum"),
                        
        FILLER_CARD
    ])

    header = fits.Header(cards)

    mappings = [
        # key, function
        ("SPECTRAL DATA", None),
        ("FLUX", flux),
        ("FLUX_ERROR", flux_error),
        ("BITMASK", bitmask),
        ("SNR", snr),

        ("INPUT DATA MODEL KEYWORDS", None),   
        # https://stackoverflow.com/questions/6076270/lambda-function-in-list-comprehensions
        *[(k.upper(), partial(lambda dp, image, _k: dp.kwargs[_k], _k=k)) for k in data_products[0].kwargs.keys()],
        ("OBSERVING CONDITIONS", None),
        #TODO: DATE_OBS? OBS_DATE? remove entirely since we have MJD?
        # DATE-OBS looks to be start of observation, since it is different from UT-MID
        ("DATE-OBS", lambda dp, image: image[0].header["DATE-OBS"]), 
        ("EXPTIME", lambda dp, image: image[0].header["EXPTIME"]),
        # TODO: homogenise with TAI-BEG, TAI-END with BOSS?
        #("JD-MID", lambda dp, image: image[0].header["JD-MID"]),
        #("UT-MID", lambda dp, image: image[0].header["UT-MID"]),
        ("FLUXFLAM", lambda dp, image: image[0].header["FLUXFLAM"]),
        ("NPAIRS", lambda dp, image: image[0].header["NPAIRS"]),
        # TODO Is NCOMBINE same as NEXP for BOSS? --> make consistent naming?
        ("NEXP", lambda dp, image: image[0].header["NCOMBINE"]),

        ("SPECTRUM SAMPLING AND STACKING", None),
        ("V_SHIFT",     meta["v_shift"]),
        # TODO: have some criteria for whether we use all in the stack or not
        ("IN_STACK",    np.ones(len(data_products), dtype=bool)),

        ("RADIAL VELOCITIES (DOPPLER)", None),
        *((k, velocity_meta[k]) for k in (
            "JD",
            "V_BARY", "V_REL", "E_V_REL", "V_BC",  
            "RCHISQ",
        )),
        
        ("RADIAL VELOCITIES (CROSS-CORRELATION)", None),
        *((k, velocity_meta[k]) for k in (
            "V_BARY_XCORR", "V_REL_XCORR", "E_V_REL_XCORR",
            "N_RV_COMPONENTS", "RV_COMPONENTS",
        )),

        ("DATABASE PRIMARY KEYS", None),
        ("VISIT_PK", velocity_meta["VISIT_PK"]),
        ("RV_VISIT_PK", velocity_meta["RV_VISIT_PK"]),
    ]

    table_category_headers = []
    values = {} 
    for j, data_product in enumerate(data_products):
        with fits.open(data_product.path) as image:
            for i, (key, function) in enumerate(mappings):
                if j == 0 and function is None:
                    table_category_headers.append((mappings[i + 1][0], key))
                else:
                    values.setdefault(key, [])
                    if callable(function):
                        try:
                            value = function(data_product, image)
                        except KeyError:
                            log.warning(f"No {key} found in {data_product.path}")
                            value = None
                        values[key].append(value)
                    else:
                        values[key] = function

    # Special cases of arrays
    for key in ("RV_COMPONENTS", ):
        try:
            values[key] = np.array(values[key])
        except:
            None

    columns = []
    for key, function in mappings:
        if function is None: continue
        columns.append(
            fits.Column(
                name=key,
                array=values[key],
                unit=None, 
                **fits_column_kwargs(values[key])
            )
        )

    hdu = fits.BinTableHDU.from_columns(columns, header=header)
    for dtype_name, category_header in table_category_headers:
        index = 1 + hdu.data.dtype.names.index(dtype_name)
        key = f"TTYPE{index}"
        hdu.header.insert(key, BLANK_CARD)
        hdu.header.insert(key, ("", category_header))
        
    # Add explanatory things from glossary
    for key in hdu.header.keys():
        hdu.header.comments[key] = GLOSSARY.get(key, None)
    for i, key in enumerate(hdu.data.dtype.names, start=1):
        hdu.header.comments[f"TTYPE{i}"] = GLOSSARY.get(key, None)
    if FILLER_CARD_KEY is not None:
        try:
            del hdu.header[FILLER_CARD_KEY]
        except:
            None

    return (hdu, resampled)


def get_catalog_identifier(source: Union[Source, int]):
    """
    Return a catalog identifer given either a source, or catalog identifier (as string or int).
    
    :param source:
        The astronomical source, or the SDSS-V catalog identifier.
    """
    return source.catalogid if isinstance(source, Source) else int(source)


def get_cartons_and_programs(source: Union[Source, int]):
    """
    Return the name of cartons and programs that this source is matched to.

    :param source:
        The astronomical source, or the SDSS-V catalog identifier.

    :returns:
        A two-length tuple containing a list of carton names (e.g., `mwm_snc_250pc`)
        and a list of program names (e.g., `mwm_snc`).
    """
    
    catalogid = get_catalog_identifier(source)
    from peewee import fn
    from sdssdb.peewee.sdss5db import database as sdss5_database
    sdss5_database.set_profile("operations") # TODO: HOW CAN WE SET THIS AS DEFAULT!?!

    from sdssdb.peewee.sdss5db.targetdb import (Target, CartonToTarget, Carton)

    sq = (
        Carton.select(
            Target.catalogid, 
            Carton.carton,
            Carton.program
        )
        .distinct()
        .join(CartonToTarget)
        .join(Target)
        .where(Target.catalogid == catalogid)
        .alias("distinct_cartons")
    )

    q_cartons = (
        Target.select(
            Target.catalogid, 
            fn.STRING_AGG(sq.c.carton, ",").alias("cartons"),
            fn.STRING_AGG(sq.c.program, ",").alias("programs"),
        )
        .join(sq, on=(sq.c.catalogid == Target.catalogid))
        .group_by(Target.catalogid)
        .tuples()
    )
    _, cartons, programs = q_cartons.first()
    return (cartons.split(","), programs.split(","))


def get_auxiliary_source_data(source: Union[Source, int]):
    """
    Return auxiliary data (e.g., photometry) for a given SDSS-V source.

    :param source:
        The astronomical source, or the SDSS-V catalog identifier.
    """
    from peewee import Alias, JOIN
    from sdssdb.peewee.sdss5db import database as sdss5_database
    sdss5_database.set_profile("operations") # TODO: HOW CAN WE SET THIS AS DEFAULT!?!

    from sdssdb.peewee.sdss5db.catalogdb import (Catalog, CatalogToTIC_v8, TIC_v8 as TIC, TwoMassPSC)

    try:
        from sdssdb.peewee.sdss5db.catalogdb import Gaia_DR3 as Gaia
    except ImportError:
        from sdssdb.peewee.sdss5db.catalogdb import Gaia_DR2 as Gaia
        log.warning(f"Gaia DR3 not yet available in sdssdb.peewee.sdss5db.catalogdb. Using Gaia DR2.")

    catalogid = get_catalog_identifier(source)
    tic_dr = TIC.__name__.split("_")[-1]
    gaia_dr = Gaia.__name__.split("_")[-1]

    ignore = lambda c: c is None or isinstance(c, str)

    # Define the columns and associated comments.    
    field_descriptors = [
        BLANK_CARD,
        ("",            "IDENTIFIERS",                  None),
        ("SDSS_ID",     Catalog.catalogid,              f"SDSS-V catalog identifier"),
        ("TIC_ID",      TIC.id.alias("tic_id"),         f"TESS Input Catalog ({tic_dr}) identifier"),
        ("GAIA_ID",     Gaia.source_id,                 f"Gaia {gaia_dr} source identifier"),
        BLANK_CARD,
        ("",            "ASTROMETRY",                   None),
        ("RA",          Catalog.ra,                     "SDSS-V catalog right ascension (J2000) [deg]"),
        ("DEC",         Catalog.dec,                    "SDSS-V catalog declination (J2000) [deg]"),
        ("GAIA_RA",     Gaia.ra,                        f"Gaia {gaia_dr} right ascension [deg]"),
        ("GAIA_DEC",    Gaia.dec,                       f"Gaia {gaia_dr} declination [deg]"),        
        ("PLX",         Gaia.parallax,                  f"Gaia {gaia_dr} parallax [mas]"),
        ("E_PLX",       Gaia.parallax_error,            f"Gaia {gaia_dr} parallax error [mas]"),
        ("PMRA",        Gaia.pmra,                      f"Gaia {gaia_dr} proper motion in RA [mas/yr]"),
        ("E_PMRA",      Gaia.pmra_error,                f"Gaia {gaia_dr} proper motion in RA error [mas/yr]"),
        ("PMDE",        Gaia.pmdec,                     f"Gaia {gaia_dr} proper motion in DEC [mas/yr]"),
        ("E_PMDE",      Gaia.pmdec_error,               f"Gaia {gaia_dr} proper motion in DEC error [mas/yr]"),
        ("VRAD",        Gaia.radial_velocity,           f"Gaia {gaia_dr} radial velocity [km/s]"),
        ("E_VRAD",      Gaia.radial_velocity_error,     f"Gaia {gaia_dr} radial velocity error [km/s]"),
        BLANK_CARD,
        ("",            "PHOTOMETRY",                   None),
        ("G_MAG",       Gaia.phot_g_mean_mag,           f"Gaia {gaia_dr} mean apparent G magnitude [mag]"),
        ("BP_MAG",      Gaia.phot_bp_mean_mag,          f"Gaia {gaia_dr} mean apparent BP magnitude [mag]"),
        ("RP_MAG",      Gaia.phot_rp_mean_mag,          f"Gaia {gaia_dr} mean apparent RP magnitude [mag]"),
        ("J_MAG",       TwoMassPSC.j_m,                 f"2MASS mean apparent J magnitude [mag]"),
        ("E_J_MAG",     TwoMassPSC.j_cmsig,             f"2MASS mean apparent J magnitude error [mag]"),
        ("H_MAG",       TwoMassPSC.h_m,                 f"2MASS mean apparent H magnitude [mag]"),        
        ("E_H_MAG",     TwoMassPSC.h_cmsig,             f"2MASS mean apparent H magnitude error [mag]"),
        ("K_MAG",       TwoMassPSC.k_m,                 f"2MASS mean apparent K magnitude [mag]"),
        ("E_K_MAG",     TwoMassPSC.k_cmsig,             f"2MASS mean apparent K magnitude error [mag]"),
    ]

    q = (
        Catalog.select(*[c for k, c, comment in field_descriptors if not ignore(c)])
               .distinct(Catalog.catalogid)
               .join(CatalogToTIC_v8, JOIN.LEFT_OUTER)
               .join(TIC)
               .join(Gaia, JOIN.LEFT_OUTER)
               .switch(TIC)
               .join(TwoMassPSC, JOIN.LEFT_OUTER)
               .where(Catalog.catalogid == catalogid)
               .dicts()
    )
    row = q.first()

    # Return as a list of entries suitable for a FITS header card.
    data = []
    for key, field, comment in field_descriptors:
        if ignore(field):
            data.append((key, field, comment))
        else:
            data.append((
                key,
                row[field._alias if isinstance(field, Alias) else field.name],
                comment
            ))

    # Add carton and target information
    cartons, programs = get_cartons_and_programs(source)
    data.extend([
        BLANK_CARD,
        ("",            "TARGETING",        None),
        ("CARTONS",     ",".join(cartons), f"Comma-separated SDSS-V program names"),
        ("PROGRAMS",    ",".join(programs), f"Comma-separated SDSS-V carton names"),
        ("MAPPERS",     ",".join([p.split("_")[0] for p in programs]), f"Comma-separated SDSS-V Mappers")
    ])
    return data



def create_primary_hdu(
    source: Union[Source, int],
    hdu_descriptions: Optional[List[str]] = None
) -> fits.PrimaryHDU:
    """
    Create primary HDU (headers only) for a Milky Way Mapper data product, given some source.
    
    :param source:
        The astronomical source, or the SDSS-V catalog identifier.

    :param hdu_descriptions: [optional]
        A list of strings describing all HDUs.
    """
    catalogid = get_catalog_identifier(source)
    
    # Sky position.
    ra, dec = get_sky_position(catalogid)
    nside = 128
    healpix = ang2pix(nside, ra, dec, lonlat=True)

    # I would like to use .isoformat(), but it is too long and makes headers look disorganised.
    # Even %Y-%m-%d %H:%M:%S is one character too long! ARGH!
    datetime_fmt = "%y-%m-%d %H:%M:%S"
    created = datetime.datetime.utcnow().strftime(datetime_fmt)
    
    cards = [
        BLANK_CARD,
        ("",        "METADATA",     None),
        ("ASTRA",   astra_version,  f"Astra version"),
        ("CREATED", created,        f"File creation time (UTC {datetime_fmt})"),
        ("HEALPIX", healpix,        f"Healpix location ({nside} sides)")
    ]
    # Get photometry and other auxiliary data.
    cards.extend(get_auxiliary_source_data(source))

    if hdu_descriptions is not None:
        cards.extend([
            BLANK_CARD,
            ("",        "HDU DESCRIPTIONS",     None),
            *[(f"COMMENT", f"HDU {i}: {desc}", None) for i, desc in enumerate(hdu_descriptions)]
        ])
        
    return fits.PrimaryHDU(header=fits.Header(cards))





def resample_visit_spectra(
    resampled_wavelength,
    num_pixels_per_resolution_element,
    radial_velocity,
    wavelength,
    flux,
    flux_error=None,
    scale_by_pseudo_continuum=False,
    use_smooth_filtered_spectrum_for_bad_pixels=False,
    bad_pixel_mask=None,
    median_filter_size=501,
    median_filter_mode="reflect",
    gaussian_filter_size=100,
):
    """
    Resample visit spectra onto a common wavelength array.

    :param scale_by_pseudo_continuum: [optional]
        Optionally scale each visit spectrum by its pseudo-continuum (a gaussian median filter) when
        stacking to keep them on the same relative scale (default: False).

    :param use_smooth_filtered_spectrum_for_bad_pixels: [optional]
        For any bad pixels (defined by the `bad_pixel_mask`), use a smooth filtered spectrum (a median
        filtered spectrum with a gaussian convolution) to fill in bad pixel values (default:False).

    :param median_filter_size: [optional]
        The filter width (in pixels) to use for any median filters (default: 501).
    
    :param median_filter_mode: [optional]
        The mode to use for any median filters (default: reflect).

    :param gaussian_filter_size: [optional]
        The filter size (in pixels) to use for any gaussian filter applied.    
    """

    try:
        n_chips = len(num_pixels_per_resolution_element)
    except:
        n_chips = None
    finally:
        num_pixels_per_resolution_element = flatten(num_pixels_per_resolution_element)
    
    n_visits, n_pixels = shape = (len(wavelength), len(resampled_wavelength))

    resampled_flux = np.zeros(shape)
    resampled_flux_error = np.zeros(shape)
    resampled_pseudo_cont = np.ones(shape)

    visit_and_chip = lambda f, i, j: None if f is None else (f[i][j] if n_chips is not None else f[i])
    smooth_filter = lambda f: gaussian_filter(median_filter(f, [median_filter_size], mode=median_filter_mode), gaussian_filter_size)

    if radial_velocity is None:
        radial_velocity = np.zeros(n_visits)

    if len(radial_velocity) != n_visits:
        raise ValueError(f"Unexpected number of radial velocities ({len(radial_velocity)} != {n_visits})")

    for i, v_rad in enumerate(radial_velocity):
        for j, n_res in enumerate(num_pixels_per_resolution_element):
            
            chip_wavelength = visit_and_chip(wavelength, i, j)
            chip_flux = visit_and_chip(flux, i, j)
            chip_flux_error = visit_and_chip(flux_error, i, j)
            
            if bad_pixel_mask is not None and use_smooth_filtered_spectrum_for_bad_pixels:
                chip_bad_pixel_mask = visit_and_chip(bad_pixel_mask, i, j)
                if any(chip_bad_pixel_mask):
                    chip_flux[chip_bad_pixel_mask] = smooth_filter(chip_flux)[chip_bad_pixel_mask]
                    if chip_flux_error is not None:
                        chip_flux_error[chip_bad_pixel_mask] = smooth_filter(chip_flux_error)[chip_bad_pixel_mask]

            pixel = wave_to_pixel(resampled_wavelength * (1 + v_rad/C_KM_S), chip_wavelength)
            finite, = np.where(np.isfinite(pixel))

            (resampled_chip_flux, resampled_chip_flux_error), = sincint(
                pixel[finite], 
                n_res,
                [
                    [chip_flux, chip_flux_error]
                ]
            )

            resampled_flux[i, finite] = resampled_chip_flux
            resampled_flux_error[i, finite] = resampled_chip_flux_error

            # Scale by continuum?
            if scale_by_pseudo_continuum:
                # TODO: If there are gaps in `finite` then this will cause issues because median filter and gaussian filter 
                #       don't receive the x array
                # TODO: Take a closer look at this process.
                resampled_pseudo_cont[i, finite] = smooth_filter(resampled_chip_flux)

                resampled_flux[i, finite] /= resampled_pseudo_cont[i, finite]
                resampled_flux_error[i, finite] /= resampled_pseudo_cont[i, finite]

    # TODO: return flux ivar instead?
    
    return (
        resampled_flux,
        resampled_flux_error,
        resampled_pseudo_cont,
    )


def get_boss_radial_velocity(
    image: fits.hdu.hdulist.HDUList, 
    visit: Union[DataProduct, str]
):
    """
    Return the (current) best-estimate of the radial velocity (in km/s) of this 
    visit spectrum from the image headers. 
    
    This defaults to using the `XCSAO_RV` radial velocity.

    :param image:
        The FITS image of the BOSS SpecLite data product.
    
    :param visit:
        The supplied visit.
    """    
    return image[2].data["XCSAO_RV"][0]


def resample_boss_visit_spectra(
    visits: List[Union[DataProduct, str]],
    crval: float = 3.5523,
    cdelt: float = 1e-4,
    num_pixels: int = 4_648,
    num_pixels_per_resolution_element: int = 5,
    radial_velocities: Optional[Union[Callable, List[float]]] = None,
    scale_by_pseudo_continuum: bool = True,
    median_filter_size: int = 501,
    median_filter_mode: str = "reflect",
    gaussian_filter_size: float = 100,
    **kwargs
):
    """
    Resample BOSS visit spectra onto a common wavelength array.
    
    :param visits:
        A list of specLite data products, or their paths.
    
    :param crval: [optional]
        The log10(lambda) of the wavelength of the first pixel to resample to.
    
    :param cdelt: [optional]
        The log (base 10) of the wavelength spacing to use when resampling.
    
    :param num_pixels: [optional]
        The number of pixels to use for the resampled array.

    :param num_pixels_per_resolution_element: [optional]
        The number of pixels per resolution element assumed when performing sinc interpolation.
    
    :param radial_velocities: [optional]
        Either a list of radial velocities (one per visit), or a callable that takes two arguments
        (the FITS image of the data product, and the input visit) and returns a radial velocity
        in units of km/s.

        If `None` is given then we use `get_boss_radial_velocity`.
    
    :param scale_by_pseudo_continuum: [optional]
        Optionally scale each visit spectrum by its pseudo-continuum (a gaussian median filter) when
        stacking to keep them on the same relative scale (default: True).

    :param median_filter_size: [optional]
        The filter width (in pixels) to use for any median filters (default: 501).
    
    :param median_filter_mode: [optional]
        The mode to use for any median filters (default: reflect).

    :param gaussian_filter_size: [optional]
        The filter size (in pixels) to use for any gaussian filter applied.    
    
    """
    if radial_velocities is None:
        radial_velocities = get_boss_radial_velocity

    wavelength, v_shift, flux, flux_error, sky_flux = ([], [], [], [], [])
    for i, visit in enumerate(visits):
        path = visit.path if isinstance(visit, DataProduct) else visit

        with fits.open(path) as image:
            if callable(radial_velocities):
                v = radial_velocities(image, visit)
            else:
                v = radial_velocities[i]
            
            v_shift.append(v)
            wavelength.append(10**image[1].data["LOGLAM"])
            flux.append(image[1].data["FLUX"])
            flux_error.append(image[1].data["IVAR"]**-0.5)
            sky_flux.append(image[1].data["SKY"])
        

    resampled_wavelength = log_lambda_dispersion(crval, cdelt, num_pixels)
    args = (resampled_wavelength, num_pixels_per_resolution_element, v_shift, wavelength)
    kwds = dict(
        median_filter_size=median_filter_size,
        median_filter_mode=median_filter_mode,
        gaussian_filter_size=gaussian_filter_size,
        scale_by_pseudo_continuum=scale_by_pseudo_continuum
    )
    kwds.update(kwargs)

    resampled_flux, resampled_flux_error, resampled_pseudo_cont = resample_visit_spectra(
        *args,
        flux,
        flux_error,
        **kwds
    )

    # BOSS DRP gives no per-pixel bitmask array AFAIK
    # Which is why `use_smooth_filtered_spectrum_for_bad_pixels` is not an option here 
    # because we have no bad pixel mask anyways. The user can supply these through kwargs.
    resampled_bitmask = np.zeros(resampled_flux.shape, dtype=int) 

    # TODO: have resample_visit_spectra return this so we dont repeat ourselves
    meta = dict(
        crval=crval,
        cdelt=cdelt,
        wavelength=wavelength,
        v_shift=v_shift,
        num_pixels=num_pixels,
        num_pixels_per_resolution_element=num_pixels_per_resolution_element,
        median_filter_size=median_filter_size,
        median_filter_mode=median_filter_mode,
        gaussian_filter_size=gaussian_filter_size,
        scale_by_pseudo_continuum=scale_by_pseudo_continuum,
    )

    return (
        resampled_flux, 
        resampled_flux_error, 
        resampled_pseudo_cont, 
        resampled_bitmask,
        meta
    )


def get_apogee_visit_metadata(
    image: fits.hdu.hdulist.HDUList,
):

    # Things that are only relevant to APOGEE, but not for a particular visit:

    # Headers:
    # OBJ       Object name
    # HEALPIX   Healpix location
    # TELESCOP  Telescope name
    # APRED     APOGEE reduction pipeline version
    # APSTAR 

    # Radial velocity:
    # V_BARY
    # V_SCATTER
    # N_COMP

    # Stellar parameters from Doppler
    # TEFF
    # LOGG
    # FEH

    # Line spread function:
    # FIB_MEAN  S/N weighted mean fiber number
    # N_RES     # number of pixels per resolution element
    

    # FLUXFLAM 
    # HPLUS
    # HMINUS
    
    
    # Things that are relevant to individual visits:
    # FIELD
    # MJD
    # APRED
    # FIBER
    # TELESCOPE
    # PLATE


    # DATE-OBS
    # EXPTIME
    # NCOMBINE
    # FRAME1
    # FRAME2
    # NPAIRS


    #  RADIAL VELOCITY
    # JD
    # V_BC
    # V_RAD
    # V_BARY
    # SNR
    # CHI_SQ
    # STARFLAG




    fields = [
        ("LOCID", "LOC_ID"),
        ("PLATE", "PLATE"),
        ("TELESCOP", "TELESCOP"),
        ("MJD5", "MJD"),
        ("FIBERID", "FIBER_ID"),
        ("DATE-OBS", None),
        ("EXPTIME", "EXPTIME"),
        ("JD-MID", None),
        ("UT-MID", None),
        ("NCOMBINE", "NCOMBINE"),
        ("FRAME1", "FRAME1")
    ]


    fields = [
        (0, "TELESCOP", "TELESCOP", None),
        (0, "PLATE", "PLATE", None),
        # TODO: How are location ID and plate ID different?
        (0, "LOCID", "LOC_ID", None), 
    ]

    # visit data for th etable:

    #MJD5, "MJD",
    

    return get_visit_metadata(image, fields)

def get_visit_metadata(
    image: fits.hdu.hdulist.HDUList,
    fields: List
):
    """
    Return metadata fields for a given FITS image.
    This will return a list of fields (key, value, comment) suitable for
    a new FITS header.

    :param image:
        The FITS image of the ApVisit data product.
    
    :param fields:
        A list of 4-length tuples that each contain:
            - 0-indexed HDU where to access the header in the ApVisit file
            - Existing header key in the ApVisit file
            - Header key to use for the new data product
            - Comment to use in the new data product
    """    
    metadata = []
    for (hdu, key, new_key, comment) in fields:
        if hdu is None:
            metadata.append((new_key, comment, None))
        else:
            value = image[hdu].header[key]
            metadata.append((new_key, value, comment))
    return metadata




def get_apogee_visit_radial_velocity(data_product: DataProduct):
    """
    Return the (current) best-estimate of the radial velocity (in km/s) of this 
    visit spectrum that is stored in the APOGEE DRP database.

    :param image:
        The FITS image of the ApVisit data product.
    
    :param visit:
        The supplied visit.
    """
    from astra.database.apogee_drpdb import (RvVisit, Visit)
    if data_product.filetype.lower() != "apvisit":
        raise ValueError(f"Data product must have filetype 'apVisit'")
    
    # Note: The rv_visit table contains *NEARLY* everything to uniquely identify
    #       a visit. It contains mjd, apred, fiber (fibreid), telescope, plate,
    #       but does not contain FIELD. So we must cross-match with the visit table.
    q = (
        RvVisit.select()
               .join(Visit, on=(RvVisit.visit_pk == Visit.pk))
               .where(
                    (Visit.telescope == data_product.kwargs["telescope"])
                &   (Visit.fiberid == data_product.kwargs["fiber"])
                &   (Visit.mjd == data_product.kwargs["mjd"])
                &   (Visit.apred == data_product.kwargs["apred"])
                &   (Visit.field == data_product.kwargs["field"])
                &   (Visit.plate == data_product.kwargs["plate"])
               )
               .order_by(RvVisit.created.desc())
    )
    result = q.first()
    # Sanity check
    if (data_product.sources[0].catalogid != result.catalogid):
        raise ValueError(
            f"Data product {data_product} catalogid does not match record in APOGEE DRP "
            f"table ({data_product.sources[0].catalogid} != {result.catalogid}) "
            f"on APOGEE rv_visit.pk={result.pk} and visit.pk={result.visit_pk}"
        )
    # Return the named metadata we need, using keys from common glossary.
    return {
        "V_BC": result.bc,
        "V_REL": result.vrel,
        "E_V_REL": result.vrelerr,
        "V_TYPE": result.vtype, # 1=chisq, 2=xcorr
        "JD": result.jd,
        "DATE-OBS": result.dateobs,
        "TEFF_DOPPLER": result.rv_teff,
        "E_TEFF_DOPPLER": result.rv_tefferr,
        "LOGG_DOPPLER": result.rv_logg,
        "E_LOGG_DOPPLER": result.rv_loggerr,
        "FEH_DOPPLER": result.rv_feh,
        "E_FEH_DOPPLER": result.rv_feherr,
        "VISIT_PK": result.visit_pk,
        "RV_VISIT_PK": result.pk,
        "V_BARY": result.vheliobary,
        "RCHISQ": result.chisq,
        "N_RV_COMPONENTS": result.n_components,
        "V_REL_XCORR": result.xcorr_vrel,
        "E_V_REL_XCORR": result.xcorr_vrelerr,
        "V_BARY_XCORR": result.xcorr_vheliobary,
        "RV_COMPONENTS": result.rv_components,
    }  
    

def resample_apogee_visit_spectra(
    visits: List[Union[DataProduct, str]],
    crval: float = 4.179,
    cdelt: float = 6e-6,
    num_pixels: int = 8_575,
    num_pixels_per_resolution_element=(5, 4.25, 3.5),
    radial_velocities: Optional[Union[Callable, List[float]]] = None,
    use_smooth_filtered_spectrum_for_bad_pixels: bool = True,
    scale_by_pseudo_continuum: bool = True,
    median_filter_size: int = 501,
    median_filter_mode: str = "reflect",
    gaussian_filter_size: float = 100,
    **kwargs
):
    """
    Resample APOGEE visit spectra onto a common wavelength array.
    
    :param visits:
        A list of ApVisit data products, or their paths.
    
    :param crval: [optional]
        The log10(lambda) of the wavelength of the first pixel to resample to.
    
    :param cdelt: [optional]
        The log (base 10) of the wavelength spacing to use when resampling.
    
    :param num_pixels: [optional]
        The number of pixels to use for the resampled array.

    :param num_pixels_per_resolution_element: [optional]
        The number of pixels per resolution element assumed when performing sinc interpolation.
        If a tuple is given, then it is assumed the input visits are multi-dimensional (e.g., multiple
        chips) and a different number of pixels per resolution element should be used per chip.
    
    :param radial_velocities: [optional]
        Either a list of radial velocities (one per visit), or a callable that takes two arguments
        (the FITS image of the data product, and the input visit) and returns a radial velocity
        in units of km/s.

        If `None` is given then we take the most recent radial velocity measurement from the APOGEE DRP
        database.

    :param use_smooth_filtered_spectrum_for_bad_pixels: [optional]
        For any bad pixels (defined by the bad pixel mask) use a smooth filtered spectrum (a median
        filtered spectrum with a gaussian convolution) to fill in bad pixel values (default: True).

    :param scale_by_pseudo_continuum: [optional]
        Optionally scale each visit spectrum by its pseudo-continuum (a gaussian median filter) when
        stacking to keep them on the same relative scale (default: True).

    :param median_filter_size: [optional]
        The filter width (in pixels) to use for any median filters (default: 501).
    
    :param median_filter_mode: [optional]
        The mode to use for any median filters (default: reflect).

    :param gaussian_filter_size: [optional]
        The filter size (in pixels) to use for any gaussian filter applied.            

    :returns:
        A 7-length tuple containing:
            - a list of visits that were included (e.g., no problems finding the file)
            - radial velocities [km/s] used for stacking
            - array of shape (P, ) of resampled wavelengths
            - array of shape (V, P) containing the resampled flux values, where V is the number of visits
            - array of shape (V, P) containing the resampled flux error values
            - array of shape (V, P) containing the pseudo-continuum values
            - array of shape (V, P) containing the resampled bitmask values
    """

    pixel_mask = PixelBitMask()

    if radial_velocities is None:
        radial_velocities = get_apogee_visit_radial_velocity

    include_visits, wavelength, v_shift = ([], [], [])
    flux, flux_error = ([], [])
    bitmasks, bad_pixel_mask = ([], [])

    for i, visit in enumerate(visits):
        path = visit.path if isinstance(visit, DataProduct) else visit
        if not os.path.exists(path):
            log.warning(f"Missing file: {path} from {visit}")
            continue

        with fits.open(path) as image:
            if callable(radial_velocities):
                v = radial_velocities(image, visit)
            else:
                v = radial_velocities[i]

            hdu_header, hdu_flux, hdu_flux_error, hdu_bitmask, hdu_wl, *_ = range(11)
        
            v_shift.append(v)
            wavelength.append(image[hdu_wl].data)
            flux.append(image[hdu_flux].data)
            flux_error.append(image[hdu_flux_error].data)
            # We resample the bitmasks, and we provide a bad pixel mask.
            bitmasks.append(image[hdu_bitmask].data)
            bad_pixel_mask.append((bitmasks[-1] & pixel_mask.badval()) > 0)

        include_visits.append(visit)

    resampled_wavelength = log_lambda_dispersion(crval, cdelt, num_pixels)
    args = (resampled_wavelength, num_pixels_per_resolution_element, v_shift, wavelength)

    kwds = dict(
        scale_by_pseudo_continuum=scale_by_pseudo_continuum,
        use_smooth_filtered_spectrum_for_bad_pixels=use_smooth_filtered_spectrum_for_bad_pixels,
        bad_pixel_mask=bad_pixel_mask,
        median_filter_size=median_filter_size,
        median_filter_mode=median_filter_mode,
        gaussian_filter_size=gaussian_filter_size
    )
    kwds.update(kwargs)

    resampled_flux, resampled_flux_error, resampled_pseudo_cont = resample_visit_spectra(
        *args,
        flux,
        flux_error,
        **kwds
    )
    # TODO: have resample_visit_spectra return this so we dont repeat ourselves
    meta = dict(
        crval=crval,
        cdelt=cdelt,
        num_pixels=num_pixels,
        wavelength=resampled_wavelength,
        v_shift=v_shift,
        num_pixels_per_resolution_element=num_pixels_per_resolution_element,
        median_filter_size=median_filter_size,
        median_filter_mode=median_filter_mode,
        gaussian_filter_size=gaussian_filter_size,
        scale_by_pseudo_continuum=scale_by_pseudo_continuum,
    )

    resampled_bitmask, *_ = resample_visit_spectra(*args, bitmasks)
    return (
        resampled_flux, 
        resampled_flux_error, 
        resampled_pseudo_cont, 
        resampled_bitmask,
        meta
    )


def combine_spectra(
    flux,
    flux_error,
    pseudo_continuum=None,
    bitmask=None,
):
    """
    Combine resampled spectra into a single spectrum, weighted by the inverse variance of each pixel.

    :param flux:
        An (N, P) array of fluxes of N visit spectra, each with P pixels.
    
    :param flux_error:
        An (N, P) array of flux errors of N visit spectra.
    
    :param pseudo_continuum: [optional]
        Pseudo continuum array used when resampling spectra.        

    :param bitmask: [optional]
        An optional bitmask array (of shape (N, P)) to combine (by 'and').
    """
    ivar = flux_error**-2
    ivar[ivar == 0] = 0

    stacked_ivar = np.sum(ivar, axis=0)
    stacked_flux = np.sum(flux * ivar, axis=0) / stacked_ivar
    stacked_flux_error = np.sqrt(1/stacked_ivar)

    if pseudo_continuum is not None:
        cont = np.median(pseudo_continuum, axis=0) # TODO: check
        stacked_flux *= cont
        stacked_flux_error *= cont
    else:
        cont = 1

    if bitmask is not None:
        stacked_bitmask = np.bitwise_and.reduce(bitmask, 0)
    else:
        stacked_bitmask = np.zeros(stacked_flux.shape, dtype=int)
    
    return (stacked_flux, stacked_flux_error, cont, stacked_bitmask)


def fits_column_kwargs(values):
    if all(isinstance(v, str) for v in values):
        max_len = max(map(len, values))
        return dict(format=f"{max_len}A")

    """
    FITS format code         Description                     8-bit bytes

    L                        logical (Boolean)               1
    X                        bit                             *
    B                        Unsigned byte                   1
    I                        16-bit integer                  2
    J                        32-bit integer                  4
    K                        64-bit integer                  8
    A                        character                       1
    E                        single precision float (32-bit) 4
    D                        double precision float (64-bit) 8
    C                        single precision complex        8
    M                        double precision complex        16
    P                        array descriptor                8
    Q                        array descriptor                16
    """

    mappings = [
        ("E", lambda v: isinstance(v[0], (float, np.floating))), # all 32-bit
        ("J", lambda v: isinstance(v[0], (int, np.integer)) and (int(max(v) >> 32) == 0)), # 32-bit integers
        ("K", lambda v: isinstance(v[0], (int, np.integer)) and (int(max(v) >> 32) > 0)), # 64-bit integers
        ("L", lambda v: isinstance(v[0], (bool, np.bool_))), # bools
    ]
    flat_values = np.array(values).flatten()
    for format_code, check in mappings:
        if check(flat_values):
            break
    else:
        return {}

    kwds = {}
    if isinstance(values, np.ndarray):
        #S = values.size
        V, P = np.atleast_2d(values).shape
        if values.ndim == 2:
            kwds["format"] = f"{P:.0f}{format_code}"
            kwds["dim"] = f"({P}, )"
        else:
            kwds["format"] = f"{format_code}"

    else:
        kwds["format"] = format_code
    return kwds



# TODO: Refactor this to something that can be used by astra/operators/sdss and here.
def get_boss_visits(catalogid):
    from astropy.table import Table
    from astra.utils import expand_path
    data = Table.read(expand_path("$BOSS_SPECTRO_REDUX/master/spAll-master.fits"))
    matches = np.where(data["CATALOGID"] == catalogid)[0]

    kwds = []
    for row in data[matches]:
        kwds.append(dict(
            # TODO: remove this when the path is fixed in sdss_access
            fieldid=f"{row['FIELD']:0>6.0f}",
            mjd=int(row["MJD"]),
            catalogid=int(catalogid),
            run2d=row["RUN2D"],
            isplate=""
        ))
    return kwds

from sdss_access import SDSSPath
paths = [SDSSPath("sdss5").full("specLite", **kwds) for kwds in get_boss_visits(27021597917837494)]

# Utilities below.


def log_lambda_dispersion(crval, cdelt, num_pixels):
    return 10**(crval + cdelt * np.arange(num_pixels))


    

def wave_to_pixel(wave,wave0) :
    """ convert wavelength to pixel given wavelength array
    Args :
       wave(s) : wavelength(s) (\AA) to get pixel of
       wave0 : array with wavelength as a function of pixel number 
    Returns :
       pixel(s) in the chip
    """
    pix0= np.arange(len(wave0))
    # Need to sort into ascending order
    sindx= np.argsort(wave0)
    wave0= wave0[sindx]
    pix0= pix0[sindx]
    # Start from a linear baseline
    baseline= np.polynomial.Polynomial.fit(wave0,pix0,1)
    ip= interpolate.InterpolatedUnivariateSpline(wave0,pix0/baseline(wave0),k=3)
    out= baseline(wave)*ip(wave)
    # NaN for out of bounds
    out[wave > wave0[-1]]= np.nan
    out[wave < wave0[0]]= np.nan
    return out


def sincint(x, nres, speclist) :
    """ Use sinc interpolation to get resampled values
        x : desired positions
        nres : number of pixels per resolution element (2=Nyquist)
        speclist : list of [quantity, variance] pairs (variance can be None)
    """

    dampfac = 3.25*nres/2.
    ksize = int(21*nres/2.)
    if ksize%2 == 0 : ksize +=1
    nhalf = ksize//2 

    #number of output and input pixels
    nx = len(x)
    nf = len(speclist[0][0])

    # integer and fractional pixel location of each output pixel
    ix = x.astype(int)
    fx = x-ix

    # outputs
    outlist=[]
    for spec in speclist :
        if spec[1] is None :
            outlist.append([np.full_like(x,0),None])
        else :
            outlist.append([np.full_like(x,0),np.full_like(x,0)])

    for i in range(len(x)) :
        xkernel = np.arange(ksize)-nhalf - fx[i]
        # in units of Nyquist
        xkernel /= (nres/2.)
        u1 = xkernel/dampfac
        u2 = np.pi*xkernel
        sinc = np.exp(-(u1**2)) * np.sin(u2) / u2
        sinc /= (nres/2.)

        lobe = np.arange(ksize) - nhalf + ix[i]
        vals = np.zeros(ksize)
        vars = np.zeros(ksize)
        gd = np.where( (lobe>=0) & (lobe<nf) )[0]

        for spec,out in zip(speclist,outlist) :
            vals = spec[0][lobe[gd]]
            out[0][i] = (sinc[gd]*vals).sum()
            if spec[1] is not None : 
                var = spec[1][lobe[gd]]
                out[1][i] = (sinc[gd]**2*var).sum()

    for out in outlist :
       if out[1] is not None : out[1] = np.sqrt(out[1])
    
    return outlist
