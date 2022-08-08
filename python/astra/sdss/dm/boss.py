import numpy as np
from astropy.io import fits
from typing import Union, List, Callable, Optional, Dict, Tuple
from functools import partial

from astra.database.astradb import Source, DataProduct

from astra.utils import log, list_to_dict
from astra.sdss.dm import base
from .util import (calculate_snr, log_lambda_dispersion)
from .combine import resample_visit_spectra, pixel_weighted_spectrum

def create_boss_hdus(
    data_products: List[DataProduct], 
    crval: float = 3.5523,
    cdelt: float = 1e-4,
    num_pixels: int = 4_648,
    num_pixels_per_resolution_element: int = 5,
    **kwargs
) -> Tuple[fits.BinTableHDU]:
    """
    Create a HDU for resampled BOSS visits, and a stacked BOSS spectrum,
    given the input BOSS data products.

    :param data_products:
        A list of specLite data products.

    :param crval: [optional]
        The log10(lambda) of the wavelength of the first pixel to resample to.
    
    :param cdelt: [optional]
        The log (base 10) of the wavelength spacing to use when resampling.
    
    :param num_pixels: [optional]
        The number of pixels to use for the resampled array.

    :param num_pixels_per_resolution_element: [optional]
        The number of pixels per resolution element assumed when performing sinc interpolation.
    """

    telescope, instrument = ("apo25m", "BOSS")
    common_headers = (
        ("BOSS DATA REDUCTION PIPELINE", None),
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

    if len(data_products) == 0:
        empty_hdu = base.create_empty_hdu(telescope, instrument),            
        return (empty_hdu, empty_hdu)

    # Data reduction pipeline keywords
    drp_cards = base.headers_as_cards(data_products[0], common_headers)

    flux, flux_error, continuum, bitmask, meta = resample_boss_visit_spectra(
        data_products,
        crval=crval,
        cdelt=cdelt,
        num_pixels=num_pixels,   
        num_pixels_per_resolution_element=num_pixels_per_resolution_element,
        **kwargs
    )
    snr_visit = calculate_snr(flux, flux_error, axis=1)
    
    # Let's define some quality criteria of what to include in a stack.
    use_in_stack = (
        (np.isfinite(snr_visit) & (snr_visit > 3))
    &   (np.array(meta["v_meta"]["RXC_XCSAO"]) > 6)
    )

    combined_flux, combined_flux_error, combined_bitmask = \
        pixel_weighted_spectrum(flux, flux_error, continuum, bitmask)

    snr_star = calculate_snr(combined_flux, combined_flux_error, axis=None)
    raise a

    DATA_HEADER_CARD = ("SPECTRAL DATA", None)
    star_mappings = [
        DATA_HEADER_CARD,
        ("FLUX", combined_flux),
        ("E_FLUX", combined_flux_error),
        ("BITMASK", combined_bitmask),
        ("SNR", np.array([snr_star]))
    ]    
    visit_mappings = [
        DATA_HEADER_CARD,
        ("FLUX", flux),
        ("E_FLUX", flux_error),
        ("BITMASK", bitmask),
        ("SNR", snr_visit), 

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
        ("V_HELIO_XCSAO", lambda dp, image: image[2].data["XCSAO_RV"][0]),
        ("E_V_HELIO_XCSAO", lambda dp, image: image[2].data["XCSAO_ERV"][0]),
        ("RXC_XCSAO", lambda dp, image: image[2].data["XCSAO_RXC"][0]),
        ("V_HC", lambda dp, image: image[0].header["HELIO_RV"]),
        ("TEFF_XCSAO", lambda dp, image: image[2].data["XCSAO_TEFF"][0]),
        ("E_TEFF_XCSAO", lambda dp, image: image[2].data["XCSAO_ETEFF"][0]),
        ("LOGG_XCSAO", lambda dp, image: image[2].data["XCSAO_LOGG"][0]),
        ("E_LOGG_XCSAO", lambda dp, image: image[2].data["XCSAO_ELOGG"][0]),
        ("FEH_XCSAO", lambda dp, image: image[2].data["XCSAO_FEH"][0]),
        ("E_FEH_XCSAO", lambda dp, image: image[2].data["XCSAO_EFEH"][0]),

        ("SPECTRUM SAMPLING AND STACKING", None),
        ("V_SHIFT", meta["v_shift"]),
        ("IN_STACK", use_in_stack)
    ]

    spectrum_sampling_cards = base.spectrum_sampling_cards(**meta)
    wavelength_cards = base.wavelength_cards(**meta)

    # These cards will be common to visit and star data products.
    header = fits.Header([
        *base.metadata_cards(telescope, instrument),
        *drp_cards,
        *spectrum_sampling_cards,
        *wavelength_cards,
        base.FILLER_CARD,
    ])

    hdu_star = base.hdu_from_data_mappings(data_products, star_mappings, header)
    hdu_visit = base.hdu_from_data_mappings(data_products, visit_mappings, header)
    
    return (hdu_visit, hdu_star)


def get_boss_relative_velocity(
    image: fits.hdu.hdulist.HDUList, 
    visit: Union[DataProduct, str]
) -> float:
    """
    Return the (current) best-estimate of the relative velocity (in km/s) of this 
    visit spectrum from the image headers. 
    
    This defaults to returning `V_REL`:

        V_REL = XCSAO_RV - HELIO_RV
    
    where XCSAO_RV is the measured heliocentric velocity (HDU 2) and
          HELIO_RV is the heliocentric correction (HDU 0).
    
    :param image:
        The FITS image of the BOSS SpecLite data product.
    
    :param visit:
        The supplied visit.
    """    
    v = image[2].data["XCSAO_RV"][0] - image[0].header["HELIO_RV"]
    meta = {
        "RXC_XCSAO": image[2].data["XCSAO_RXC"][0],
        "V_HELIO_XCSAO": image[2].data["XCSAO_RV"][0],
        "E_V_HELIO_XCSAO": image[2].data["XCSAO_ERV"][0]
    }
    return (v, meta)


def resample_boss_visit_spectra(
    visits: List[DataProduct],
    crval: float,
    cdelt: float,
    num_pixels: int,
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
    
    :param crval:
        The log10(lambda) of the wavelength of the first pixel to resample to.
    
    :param cdelt: 
        The log (base 10) of the wavelength spacing to use when resampling.
    
    :param num_pixels: 
        The number of pixels to use for the resampled array.

    :param num_pixels_per_resolution_element: [optional]
        The number of pixels per resolution element assumed when performing sinc interpolation.
    
    :param radial_velocities: [optional]
        Either a list of radial velocities (one per visit), or a callable that takes two arguments
        (the FITS image of the data product, and the input visit) and returns a radial velocity
        in units of km/s.

        If `None` is given then we use `get_boss_relative_velocity`.
    
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
        radial_velocities = get_boss_relative_velocity

    additional_meta = []
    wavelength, v_shift, flux, flux_error, sky_flux = ([], [], [], [], [])
    for i, visit in enumerate(visits):
        path = visit.path if isinstance(visit, DataProduct) else visit

        with fits.open(path) as image:
            if callable(radial_velocities):
                try:
                    v, v_meta = radial_velocities(image, visit)
                except:
                    v = radial_velocities(image, visit)
                else:
                    additional_meta.append(v_meta)
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
    if additional_meta:
        meta["v_meta"] = list_to_dict(additional_meta)

    return (
        resampled_flux, 
        resampled_flux_error, 
        resampled_pseudo_cont, 
        resampled_bitmask,
        meta
    )