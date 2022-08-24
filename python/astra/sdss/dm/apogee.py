import numpy as np
from astropy.io import fits
from functools import partial
from typing import Union, List, Callable, Optional, Dict, Tuple

from astra.database.astradb import DataProduct
from astra.utils import log, list_to_dict
from astra.sdss.dm import (base, combine, util)

from astra.sdss.apogee_bitmask import PixelBitMask # TODO: put elsewhere

def create_apogee_hdus(
    data_products: List[DataProduct],
    crval: float = 4.179,
    cdelt: float = 6e-6,
    num_pixels: int = 8_575,
    num_pixels_per_resolution_element=(5, 4.25, 3.5),
    observatory: Optional[str] = None,
    **kwargs
) -> Tuple[fits.BinTableHDU]:
    """
    Create a HDU for resampled APOGEE visits, and a HDU for a stacked APOGEE spectrum, given the input APOGEE data products.

    :param data_products:
        A list of ApVisit data products.

    :param crval: [optional]
        The log10(lambda) of the wavelength of the first pixel to resample to.
    
    :param cdelt: [optional]
        The log (base 10) of the wavelength spacing to use when resampling.
    
    :param num_pixels: [optional]
        The number of pixels to use for the resampled array.

    :param num_pixels_per_resolution_element: [optional]
        The number of pixels per resolution element assumed when performing sinc interpolation. 
        
        If a list-like is given, this is assumed to be a value per chip.
    
    :param observatory: [optional]
        Short name for the observatory where the data products originated from. If `None` is given, this will
        be inferred from the data model keywords.
    """

    instrument = "APOGEE"
    if observatory is None:
        try:
            observatory = ",".join(list(set(
                [dp.kwargs["telescope"][:3].upper() for dp in data_products]
            )))
        except:
            log.warning(f"Cannot infer observatory from data products.")
            observatory = None

    common_headers = (
        (f"{instrument} DATA REDUCTION PIPELINE", None),
        "V_APRED", 
    )
    if len(data_products) == 0:
        empty_visits_hdu = base.create_empty_hdu(observatory, instrument)
        empty_star_hdu = base.create_empty_hdu(observatory, instrument)
        return (empty_visits_hdu, empty_star_hdu)

    # Data reduction pipeline keywords
    drp_cards = base.headers_as_cards(data_products[0], common_headers)

    # First get the velocity information from the APOGEE data reduction pipeline database.
    velocity_meta = list_to_dict(tuple(map(get_apogee_visit_radial_velocity, data_products)))

    flux, flux_error, continuum, bitmask, meta = resample_apogee_visit_spectra(
        data_products,
        crval=crval,
        cdelt=cdelt,
        num_pixels=num_pixels,   
        num_pixels_per_resolution_element=num_pixels_per_resolution_element,
        radial_velocities=velocity_meta["V_REL"],
        **kwargs
    )

    # Increase the flux uncertainties at the pixel level due to persistence and significant skylines,
    # in the same way that is done for apStar data products.
    increase_flux_uncertainties_due_to_persistence(flux_error, bitmask)
    increase_flux_uncertainties_due_to_skylines(flux_error, bitmask)

    # Now calculate the S/N.
    snr_visit = util.calculate_snr(flux, flux_error, axis=1)

    # Deal with apVisits that have different number of pixels, and include this as the DITHERED 
    # or not 
    # Let's define some quality criteria of what to include in a stack.
    # TODO: Check in again with APOGEE DRP,.. still not clear which visits get included or why
    use_in_stack = (
            (np.isfinite(snr_visit) & (snr_visit > 3))
        &   (np.isfinite(velocity_meta["RCHISQ"]))
    )

    combined_flux, combined_flux_error, combined_bitmask = combine.pixel_weighted_spectrum(
        flux[use_in_stack], 
        flux_error[use_in_stack], 
        continuum[use_in_stack], 
        bitmask[use_in_stack]
    )
    snr_star = util.calculate_snr(combined_flux, combined_flux_error, axis=None)

    DATA_HEADER_CARD = ("SPECTRAL DATA", None)
    star_mappings = [
        DATA_HEADER_CARD,
        ("FLUX", combined_flux),
        ("E_FLUX", combined_flux_error),
        ("BITMASK", combined_bitmask),
    ]

    visit_mappings = [
        DATA_HEADER_CARD,
        ("SNR", snr_visit), 
        ("FLUX", flux),
        ("E_FLUX", flux_error),
        ("BITMASK", bitmask),

        ("INPUT DATA MODEL KEYWORDS", None),   
        ("RELEASE", lambda dp, image: dp.release),
        ("FILETYPE", lambda dp, image: dp.filetype),
        # https://stackoverflow.com/questions/6076270/lambda-function-in-list-comprehensions
        *[(k.upper(), partial(lambda dp, image, _k: dp.kwargs[_k], _k=k)) for k in data_products[0].kwargs.keys()],

        ("OBSERVING CONDITIONS", None),
        #TODO: DATE_OBS? OBS_DATE? remove entirely since we have MJD?
        # DATE-OBS looks to be start of observation, since it is different from UT-MID
        ("DATE-OBS", lambda dp, image: image[0].header["DATE-OBS"]), 
        ("EXPTIME", lambda dp, image: image[0].header["EXPTIME"]),
        ("FLUXFLAM", lambda dp, image: image[0].header["FLUXFLAM"]),
        ("NPAIRS", lambda dp, image: image[0].header["NPAIRS"]),
        #("NCOMBINE", lambda dp, image: image[0].header["NCOMBINE"]),
        ("DITHERED", lambda dp, image: 1.0 if image[1].data.size == (3 * 4096) else 0.0),

        ("RADIAL VELOCITIES (DOPPLER)", None),
        *((k, velocity_meta[k]) for k in (
            "JD",
            "V_RAD", "E_V_RAD", "V_REL", "V_BC",  
            "RCHISQ",
        )),
        
        ("RADIAL VELOCITIES (CROSS-CORRELATION)", None),
        *((k, velocity_meta[k]) for k in (
            "V_RAD_XCORR", "V_REL_XCORR", "E_V_RAD_XCORR",
            "N_RV_COMPONENTS", #"RV_COMPONENTS",
        )),

        ("SPECTRUM SAMPLING AND STACKING", None),
        ("V_SHIFT", meta["v_shift"]),
        ("IN_STACK", use_in_stack),

        ("DATABASE PRIMARY KEYS", None),
        ("VISIT_PK", velocity_meta["VISIT_PK"]),
        ("RV_VISIT_PK", velocity_meta["RV_VISIT_PK"]),
    ]

    spectrum_sampling_cards = base.spectrum_sampling_cards(**meta)
    wavelength_cards = base.wavelength_cards(**meta)

    doppler_cards = [
        # Since DOPPLER uses the same Cannon model for the final fit of all individual visits,
        # we include it here instead of repeating the information many times in the data table.
        base.BLANK_CARD,
        (" ", "DOPPLER STELLAR PARAMETERS", None),
        *[(f"{k}_D", f"{velocity_meta[f'{k}_DOPPLER'][0]:.0f}") for k in ("TEFF", "E_TEFF")],
        *[(f"{k}_D", f"{velocity_meta[f'{k}_DOPPLER'][0]:.3f}") for k in ("LOGG", "E_LOGG")],
        *[(f"{k}_D", f"{velocity_meta[f'{k}_DOPPLER'][0]:.3f}") for k in ("FEH", "E_FEH")],
    ]

    # These cards will be common to visit and star data products.
    visit_header = fits.Header([
        *base.metadata_cards(observatory, instrument),
        *drp_cards,
        *spectrum_sampling_cards,
        *doppler_cards,
        *wavelength_cards,
        base.FILLER_CARD,
    ])    
    star_header = fits.Header([
        *base.metadata_cards(observatory, instrument),
        *drp_cards,
        *spectrum_sampling_cards,
        *doppler_cards,
        *wavelength_cards,
        base.FILLER_CARD,
    ])


    hdu_star = base.hdu_from_data_mappings(data_products, star_mappings, star_header)
    hdu_visit = base.hdu_from_data_mappings(data_products, visit_mappings, visit_header)

    # Add S/N for the stacked spectrum.
    hdu_star.header.insert("TTYPE1", "SNR")
    hdu_star.header["SNR"] = f"{snr_star:.1f}"
    hdu_star.header.comments["SNR"] = base.GLOSSARY.get("SNR", None)

    return (hdu_visit, hdu_star)


def get_apogee_visit_radial_velocity(data_product: DataProduct) -> dict:
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
    if result is None:
        # No RV measurement for this visit.
        return {
                "V_BC": np.nan,
                "V_REL": 0,
                "V_RAD": np.nan,
                "E_V_REL": np.nan,
                "V_TYPE": -1,
                "JD": -1,
                "DATE-OBS": "",
                "TEFF_DOPPLER": np.nan,
                "E_TEFF_DOPPLER": np.nan,
                "LOGG_DOPPLER": np.nan,
                "E_LOGG_DOPPLER": np.nan,
                "FEH_DOPPLER": np.nan,
                "E_FEH_DOPPLER": np.nan,
                "VISIT_PK": -1,
                "RV_VISIT_PK": -1,
                "RCHISQ": np.nan,
                "N_RV_COMPONENTS": 0, 
                "V_REL_XCORR": np.nan,
                "E_V_RAD_XCORR": np.nan,
                "V_RAD_XCORR": np.nan,
                "RV_COMPONENTS": np.array([np.nan, np.nan, np.nan]),
            }  
        
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
        "V_RAD": result.vheliobary,
        "E_V_REL": result.vrelerr,
        "E_V_RAD": result.vrelerr,
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
        "RCHISQ": result.chisq,
        "N_RV_COMPONENTS": result.n_components,
        "V_REL_XCORR": result.xcorr_vrel,
        "E_V_RAD_XCORR": result.xcorr_vrelerr,
        "V_RAD_XCORR": result.xcorr_vheliobary,
        "RV_COMPONENTS": result.rv_components,
    }  


def resample_apogee_visit_spectra(
    visits: List[DataProduct],
    crval: float,
    cdelt: float,
    num_pixels: int,
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
    bitmask, bad_pixel_mask = ([], [])

    for i, visit in enumerate(visits):
        path = visit.path if isinstance(visit, DataProduct) else visit
        
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
            # We resample the bitmask, and we provide a bad pixel mask.
            bitmask.append(image[hdu_bitmask].data)
            bad_pixel_mask.append((bitmask[-1] & pixel_mask.badval()) > 0)

        include_visits.append(visit)

    resampled_wavelength = util.log_lambda_dispersion(crval, cdelt, num_pixels)
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

    resampled_flux, resampled_flux_error, resampled_pseudo_cont, resampled_bitmask = combine.resample_visit_spectra(
        *args,
        flux,
        flux_error,
        bitmask,
        **kwds
    )
    # TODO: have combine.resample_visit_spectra return this so we dont repeat ourselves
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

    return (
        resampled_flux, 
        resampled_flux_error, 
        resampled_pseudo_cont, 
        resampled_bitmask,
        meta
    )


def increase_flux_uncertainties_due_to_persistence(resampled_flux_error, resampled_bitmask) -> None:
    """
    Increase the pixel-level resampled flux uncertainties (in-place, no array copying) due to persistence flags in the resampled bitmask.
    
    This logic follows directly from what is performed when constructing the apStar files. See:
        https://github.com/sdss/apogee_drp/blob/73cfd3f7a7fbb15963ddd2190e24a15261fb07b1/python/apogee_drp/apred/rv.py#L780-L791
    """

    V, P = resampled_bitmask.shape
    pixel_mask = PixelBitMask()
    is_high = ((resampled_bitmask & pixel_mask.getval("PERSIST_HIGH")) > 0)
    is_medium = ((resampled_bitmask & pixel_mask.getval("PERSIST_MED")) > 0)
    is_low = ((resampled_bitmask & pixel_mask.getval("PERSIST_LOW")) > 0)

    resampled_flux_error[is_high] *= np.sqrt(5)
    resampled_flux_error[is_medium & ~is_high] *= np.sqrt(4)
    resampled_flux_error[is_low & ~is_medium & ~is_low] *= np.sqrt(3)
    return None


def increase_flux_uncertainties_due_to_skylines(resampled_flux_error, resampled_bitmask) -> None:
    """
    Increase the pixel-level resampled flux uncertainties (in-place; no array copying) due to significant skylines.

    This logic follows directly from what is performed when constructing the apStar files. See:
        https://github.com/sdss/apogee_drp/blob/73cfd3f7a7fbb15963ddd2190e24a15261fb07b1/python/apogee_drp/apred/rv.py#L780-L791
    """

    is_significant_skyline = ((resampled_bitmask & PixelBitMask.getval("SIG_SKYLINE")) > 0)
    resampled_flux_error[is_significant_skyline] *= np.sqrt(100)
    return None