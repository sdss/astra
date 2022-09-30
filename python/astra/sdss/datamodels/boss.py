"""Create HDUs in mwmVisit/mwmStar products with BOSS spectra."""

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from functools import partial
from typing import Union, List, Callable, Optional, Dict, Tuple

from astra import log
from astra.database.astradb import DataProduct
from astra.utils import list_to_dict

from astra.sdss.datamodels import base, util, combine


def _safe_read_header_value(image, hdu, key, default_value):
    # TODO: put this somewhere common?
    try:
        return image[hdu].header[key]
    except:
        return default_value


def create_boss_hdus(
    data_products: List[DataProduct],
    crval: float = 3.5523,
    cdelt: float = 1e-4,
    num_pixels: int = 4_648,
    num_pixels_per_resolution_element: int = 5,
    observatory: Optional[str] = "APO",
    **kwargs,
) -> Tuple[fits.BinTableHDU]:
    """
    Create a HDU for resampled BOSS visits, and a HDU for a stacked BOSS spectrum, given the input BOSS data products.

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

    :param observatory: [optional]
        Short name for the observatory where the data products originated from, since this information
        is not currently part of the BOSS specLite data model path (default: APO).
    """

    instrument = "BOSS"
    common_headers = (
        (f"{instrument} DATA REDUCTION PIPELINE", None),
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
        "DIDFLUSH",
        "CARTID",
        "PSFSKY",
        "PREJECT",
        "LOWREJ",
        "HIGHREJ",
        "SCATPOLY",
        "PROFTYPE",
        "NFITPOLY",
        "SKYCHI2",
        "SCHI2MIN",
        "SCHI2MAX",
        ("HELIO_RV", "V_HELIO"),
        "RDNOISE0",
    )
    if len(data_products) == 0:
        empty_visits_hdu = base.create_empty_hdu(observatory, instrument)
        empty_star_hdu = base.create_empty_hdu(observatory, instrument)
        return (empty_visits_hdu, empty_star_hdu)

    # Data reduction pipeline keywords
    drp_cards = base.headers_as_cards(data_products[0], common_headers)

    flux, flux_error, bitmask, meta = resample_boss_visit_spectra(
        data_products,
        crval=crval,
        cdelt=cdelt,
        num_pixels=num_pixels,
        num_pixels_per_resolution_element=num_pixels_per_resolution_element,
        **kwargs,
    )
    snr_visit = util.calculate_snr(flux, flux_error, axis=1)

    # There's some metadata that we need before we can create the mappings.
    # We need ZWARNING before everything else.
    zwarnings = np.zeros(len(data_products), dtype=int)
    for i, data_product in enumerate(data_products):
        with fits.open(data_product.path) as image:
            try:
                zwarning = image[2].data["ZWARNING"][0]
            except:
                zwarning = -1
            finally:
                zwarnings[i] = zwarning

    missing_zwarnings = np.sum(zwarnings == -1)
    if missing_zwarnings > 0:
        log.warning(f"Missing ZWARNING from {missing_zwarnings} data products:")
        for dp, zwarning in zip(data_products, zwarnings):
            if zwarning == -1:
                log.warning(f"\t{dp}: {dp.path}")
        log.warning(
            "We will assume nothing is wrong with them (e.g., as if ZWARNING = 0)!"
        )

    # Let's define some quality criteria of what to include in a stack.
    # Check if it's in a WD carton.
    cartons, programs = base.get_cartons_and_programs(data_products[0].sources[0])
    in_wd_carton = ("mwm_wd_core" in cartons)

    use_in_stack = (
        (np.isfinite(snr_visit) & (snr_visit > 3))
        & 
        (
            (np.array(meta["v_meta"]["RXC_XCSAO"]) > 6) | in_wd_carton
        ) 
        & (zwarnings <= 0)
    )

    (
        combined_flux,
        combined_flux_error,
        combined_bitmask,
        continuum,
        meta_combine,
    ) = combine.pixel_weighted_spectrum(
        flux[use_in_stack], 
        flux_error[use_in_stack], 
        bitmask[use_in_stack], 
        **kwargs
    )
    meta.update(meta_combine)
    wavelength = util.log_lambda_dispersion(crval, cdelt, num_pixels)

    # Disallow zero fluxes.
    bad_flux, bad_combined_flux = ((flux_error == 0), (combined_flux_error == 0))
    flux[bad_flux] = np.nan
    flux_error[bad_flux] = np.inf
    combined_flux[bad_combined_flux] = np.nan
    combined_flux_error[bad_combined_flux] = np.inf

    DATA_HEADER_CARD = ("SPECTRAL DATA", None)

    nanify = lambda x: np.nan if x == "NaN" else x
    log.info(
        f"Using default 'sdss5' for data product release if not specified. TODO: Andy, you can remove this in next database burn."
    )        
    visit_mappings = [
        DATA_HEADER_CARD,
        ("SNR", snr_visit),
        # ("LAMBDA", wavelength), --> too big
        ("FLUX", flux),
        ("E_FLUX", flux_error),
        ("BITMASK", bitmask),
        ("WRESL", meta["resampled_wresl"]),
        ("INPUT DATA MODEL KEYWORDS", None),
        ("RELEASE", lambda dp, image: dp.release or "sdss5"),
        ("FILETYPE", lambda dp, image: dp.filetype),
        # https://stackoverflow.com/questions/6076270/lambda-function-in-list-comprehensions
        *[
            (k.upper(), partial(lambda dp, image, _k: dp.kwargs[_k], _k=k))
            for k in data_products[0].kwargs.keys()
        ],
        ("OBSERVING CONDITIONS", None),
        ("ALT", lambda dp, image: nanify(_safe_read_header_value(image, 0, "ALT", np.nan))),
        ("AZ", lambda dp, image: nanify(_safe_read_header_value(image, 0, "AZ", np.nan))),
        ("SEEING", lambda dp, image: image[2].data["SEEING50"][0]),
        ("AIRMASS", lambda dp, image: image[2].data["AIRMASS"][0]),
        (
            "AIRTEMP",
            lambda dp, image: nanify(
                _safe_read_header_value(image, 0, "AIRTEMP", np.nan)
            ),
        ),
        (
            "DEWPOINT",
            lambda dp, image: nanify(
                _safe_read_header_value(image, 0, "DEWPOINT", np.nan)
            ),
        ),
        (
            "HUMIDITY",
            lambda dp, image: nanify(
                _safe_read_header_value(image, 0, "HUMIDITY", np.nan)
            ),
        ),
        (
            "PRESSURE",
            lambda dp, image: nanify(
                _safe_read_header_value(image, 0, "PRESSURE", np.nan)
            ),
        ),
        ("GUSTD", lambda dp, image: nanify(_safe_read_header_value(image, 0, "GUSTD", np.nan))),
        ("GUSTS", lambda dp, image: nanify(_safe_read_header_value(image, 0, "GUSTS", np.nan))),
        ("WINDD", lambda dp, image: nanify(_safe_read_header_value(image, 0, "WINDD", np.nan))),
        ("WINDS", lambda dp, image: nanify(_safe_read_header_value(image, 0, "WINDS", np.nan))),
        (
            "MOON_DIST_MEAN",
            lambda dp, image: np.mean(
                list(map(float, image[2].data["MOON_DIST"][0][0].split(" ")))
            ),
        ),
        (
            "MOON_PHASE_MEAN",
            lambda dp, image: np.mean(
                list(map(float, image[2].data["MOON_PHASE"][0][0].split(" ")))
            ),
        ),
        ("EXPTIME", lambda dp, image: image[2].data["EXPTIME"][0]),
        ("NEXP", lambda dp, image: image[2].data["NEXP"][0]),
        ("NGUIDE", lambda dp, image: image[0].header["NGUIDE"]),
        ("TAI-BEG", lambda dp, image: image[0].header["TAI-BEG"]),
        ("TAI-END", lambda dp, image: image[0].header["TAI-END"]),
        ("FIBER_OFFSET", lambda dp, image: bool(image[2].data["FIBER_OFFSET"][0])),
        ("DELTA_RA", lambda dp, image: image[2].data["DELTA_RA"][0]),
        ("DELTA_DEC", lambda dp, image: image[2].data["DELTA_DEC"][0]),
        ("RADIAL VELOCITIES (XCSAO)", None),
        ("V_RAD", get_radial_velocity),
        ("E_V_RAD", lambda dp, image: image[2].data["XCSAO_ERV"][0]),
        ("RXC_XCSAO", lambda dp, image: image[2].data["XCSAO_RXC"][0]),
        # We're using radial velocities in the Solar system barycentric rest frame, not heliocentric rest frame.
        # ("V_HC", lambda dp, image: image[0].header["HELIO_RV"]),
        # Casey (22-09-17): I calculated the barycentric correction for
        #   /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/bhm/boss/spectro/redux/v6_0_9/spectra/full/019092/59630/spec-019092-59630-27021597919936514.fits
        # and confirmed it matched the HELIO_RV:
        # from astropy: -15.22828104 km/s (using only MJD, not precise)
        #   header:    15.9146695709
        # so there is a sign difference.
        # v_true = v_measured + v_barycentric + (v_barycentric * v_measured)/c
        # See https://docs.astropy.org/en/stable/coordinates/velocities.html
        ("V_BC", get_or_calculate_barycentric_velocity_correction),
        ("TEFF_XCSAO", lambda dp, image: image[2].data["XCSAO_TEFF"][0]),
        ("E_TEFF_XCSAO", lambda dp, image: image[2].data["XCSAO_ETEFF"][0]),
        ("LOGG_XCSAO", lambda dp, image: image[2].data["XCSAO_LOGG"][0]),
        ("E_LOGG_XCSAO", lambda dp, image: image[2].data["XCSAO_ELOGG"][0]),
        ("FEH_XCSAO", lambda dp, image: image[2].data["XCSAO_FEH"][0]),
        ("E_FEH_XCSAO", lambda dp, image: image[2].data["XCSAO_EFEH"][0]),
        ("SPECTRUM SAMPLING AND STACKING", None),
        ("V_SHIFT", meta["v_shift"]),
        ("IN_STACK", use_in_stack),
        ("ZWARNING", zwarnings),
    ]

    spectrum_sampling_cards = base.spectrum_sampling_cards(**meta)
    wavelength_cards = base.wavelength_cards(**meta)

    # These cards will be common to visit and star data products.
    header = fits.Header(
        [
            *base.metadata_cards(observatory, instrument),
            *drp_cards,
            *spectrum_sampling_cards,
            *wavelength_cards,
            base.FILLER_CARD,
        ]
    )

    hdu_visit = base.hdu_from_data_mappings(data_products, visit_mappings, header)
    if any(use_in_stack):
        snr_star = util.calculate_snr(combined_flux, combined_flux_error, axis=None)
        star_data_shape = ((1, -1))
        star_mappings = [
            DATA_HEADER_CARD,
            ("SNR", np.array([snr_star]).reshape(star_data_shape)),
            ("LAMBDA", wavelength.reshape(star_data_shape)),
            ("FLUX", combined_flux.reshape(star_data_shape)),
            ("E_FLUX", combined_flux_error.reshape(star_data_shape)),
            ("BITMASK", combined_bitmask.reshape(star_data_shape)),
            ("WRESL", np.mean(meta["resampled_wresl"][use_in_stack], axis=0)),
        ]
        hdu_star = base.hdu_from_data_mappings(data_products, star_mappings, header)
    else:
        hdu_star = base.create_empty_hdu(observatory, instrument)
    return (hdu_visit, hdu_star)


def get_radial_velocity(dp, image):
    """
    Return the radial velocity of the source.
    
    Often this is measured from the Solar system barycentric rest frame, so we can
    just report the value ``XCSAO_RV`` from the HDU index 2. 
    
    But when there was an error in the calculation of the Solar system barycentric
    velocity correction, ``HELIO_RV`` in HDU0 will be NaN, and no shift was applied.
    In this case, we need to calculate the correction ourselves and add it to the
    measured radial velocity.
    """

    # Try get HELIO_RV
    try:
        v_bc = image[0].header["HELIO_RV"]
    except:
        v_bc = np.nan
    else:
        if isinstance(v_bc, str): v_bc = np.nan
    
    v_measured = image[2].data["XCSAO_RV"][0]
    if np.isfinite(v_bc):
        return v_measured
    else:
        # No barycentric correction applied yet
        # TODO: Silly that we are doing this twice, but I am time-poor.
        v_bc = get_or_calculate_barycentric_velocity_correction(dp, image)
        # TODO: Be consistent with V_BC sign
        return v_measured - v_bc


def get_or_calculate_barycentric_velocity_correction(dp, image):
    value = _safe_read_header_value(image, 0, "HELIO_RV", np.nan)
    value = np.nan if value == "NaN" else value
    
    if not np.isfinite(value):
        log.warning("HELIO_RV is not finite, calculating barycentric velocity")
        apo = EarthLocation.of_site("APO")

        coord = SkyCoord(
            ra=image[0].header["RA"] * u.deg,
            dec=image[0].header["DEC"] * u.deg
        )
        beg_time = image[0].header["TAI-BEG"]/86400.0
        end_time = image[0].header["TAI-END"]/86400.0
        mid_time = Time((beg_time + end_time) / 2, format="mjd")
        value = -coord.radial_velocity_correction(obstime=mid_time, location=apo).to(u.km/u.s).value

    return value


def get_boss_relative_velocity(
    image: fits.hdu.hdulist.HDUList, visit: Union[DataProduct, str]
) -> Tuple[float, dict]:
    """
    Return the (current) best-estimate of the relative velocity (in km/s) of this
    visit spectrum from the image headers.

    The true radial velocity is (approximately) related to the measured radial velocity by:

        v_true = v_measured + v_bc + (v_bc * v_measured)/c

    Where v_true is the true radial velocity, v_measured is the measured radial
    velocity, and v_bc is the barycentric velocity correction. The keywords for each
    quantity are:

        v_true: XCSAO_RV
        v_measured: V_REL
        v_bc: HELIO_RV

    This function returns -v_measured. Imagine an absorption line at rest:

                    At rest
                    x
          --------------------
          01234567890123456789

    The barycentric motion places it +2
                    x-|
    The radial motion places it at -6
              |-----x
    Together, the barycentric and radial motion place it at -4
                |---x
    Then the BOSS data reduction pipeline shifts the wavelength solution to account
    for barycentric motion (-2), placing the line at -6:
              |-----x
    We want the source to be at rest frame, so we need to shift it by +6

    :param image:
        The FITS image of the BOSS SpecLite data product.

    :param visit:
        The supplied visit.
    """

    try:
        v_correction = image[0].header["HELIO_RV"]
    except:
        log.warning(f"No 'VHELIO_RV' key found in HDU 0 of visit {visit}")
        # No shift was applied!
        v_correction = 0

    v_measured = image[2].data["XCSAO_RV"][0] - v_correction
    meta = {
        "RXC_XCSAO": image[2].data["XCSAO_RXC"][0],
        "V_HELIO_XCSAO": image[2].data["XCSAO_RV"][0],
        "E_V_HELIO_XCSAO": image[2].data["XCSAO_ERV"][0],
    }
    return (-v_measured, meta)


def resample_boss_visit_spectra(
    visits: List[DataProduct],
    crval: float,
    cdelt: float,
    num_pixels: int,
    num_pixels_per_resolution_element: int,
    radial_velocities: Optional[Union[Callable, List[float]]] = None,
    median_filter_size: int = 501,
    median_filter_mode: str = "reflect",
    gaussian_filter_size: float = 100,
    **kwargs,
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

    :param num_pixels_per_resolution_element:
        The number of pixels per resolution element assumed when performing sinc interpolation.

    :param radial_velocities: [optional]
        Either a list of radial velocities (one per visit), or a callable that takes two arguments
        (the FITS image of the data product, and the input visit) and returns a radial velocity
        in units of km/s.

        If `None` is given then we use `get_boss_relative_velocity`.

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

    wavelength, v_shift, flux, flux_error, sky_flux, bitmask, bad_pixel_mask, wresl = ([], [], [], [], [], [], [], [])
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
            wavelength.append(10 ** image[1].data["LOGLAM"])
            flux.append(image[1].data["FLUX"])
            flux_error.append(image[1].data["IVAR"] ** -0.5)
            sky_flux.append(image[1].data["SKY"])
            bitmask.append(image[1].data["OR_MASK"])
            bad_pixel_mask.append((image[1].data["IVAR"] == 0))
            wresl.append(image[1].data["WRESL"])

    resampled_wavelength = util.log_lambda_dispersion(crval, cdelt, num_pixels)
    args = (
        resampled_wavelength,
        num_pixels_per_resolution_element,
        v_shift,
        wavelength,
    )
    kwds = dict(
        median_filter_size=median_filter_size,
        median_filter_mode=median_filter_mode,
        gaussian_filter_size=gaussian_filter_size,
        bad_pixel_mask=bad_pixel_mask,
        use_smooth_filtered_spectrum_for_bad_pixels=True
    )
    kwds.update(kwargs)

    (
        resampled_flux,
        resampled_flux_error,
        resampled_bitmask,
    ) = combine.resample_visit_spectra(*args, flux, flux_error, **kwds)

    # Do something dumb(er) for WRESL.
    # TODO: This is not a good thing to do, but the WRESL looks well behaved but for weird
    # spikes, and it is well behaved in real life. And i'm time-limited, and only one 
    # pipeline claims to use wresl
    resampled_wresl = np.vstack(wresl)
    V, P = resampled_wresl.shape
    for i in range(V):
        bad_wresl = (resampled_wresl[i] == 0) + ~np.isfinite(resampled_wresl[i])
        resampled_wresl[i, bad_wresl] = np.interp(
            np.arange(P)[bad_wresl],
            np.arange(P)[~bad_wresl], 
            resampled_wresl[i][~bad_wresl]
        )
        
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
        resampled_wresl=resampled_wresl,
    )
    if additional_meta:
        meta["v_meta"] = list_to_dict(additional_meta)

    return (
        resampled_flux,
        resampled_flux_error,
        resampled_bitmask,
        meta,
    )
