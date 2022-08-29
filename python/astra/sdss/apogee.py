import numpy as np
from astropy.io import fits
from typing import Union, List, Callable, Optional, Dict, Tuple
from functools import partial

from astra.database.astradb import Source, DataProduct

from astra.utils import log, list_to_dict
from astra.sdss.dm import base
from .util import (calculate_snr, log_lambda_dispersion)
from .combine import resample_visit_spectra, pixel_weighted_spectrum


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
    visits: List[DataProduct, str],
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
    bitmasks, bad_pixel_mask = ([], [])

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
