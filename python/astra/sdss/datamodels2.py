import datetime
import os
import numpy as np
from typing import Union, List, Callable, Optional
from astropy.io import fits
from astropy.constants import c
from astropy import units as u
from healpy import ang2pix

from astra import (log, __version__ as astra_version)
from astra.utils import flatten
from astra.database.astradb import Source, DataProduct
from astra.base import ExecutableTask, Parameter

from scipy import interpolate
from scipy.ndimage.filters import median_filter, gaussian_filter

from astra.sdss.apogee_bitmask import PixelBitMask # TODO: put elsewhere


from .catalog import get_sky_position

C_KM_S = c.to(u.km / u.s).value


class CreateMilkyWayMapperDataProducts(ExecutableTask):

    def execute(self):
        

        for task, input_data_products, _ in self.iterable():

            source = input_data_products[0].source

            primary_hdu = create_primary_hdu(source)
            #boss_visits_hdu, boss_resampled = create_boss_visits_hdu(fdp(source.data)_pinput_data_products, "specLite"))
            #apo25m_visits_hdu, apo25m_resampled = create_apogee_visits_hdu(fdp(input_data_products, "apVisit", "apo25m"))
            #lco25m_visits_hdu, lco25m_resampled = create_apogee_visits_hdu(fdp(input_data_products, "apVisit", "lco25m"))



# Resample all spectra.

# Stack spectra.

# Create header card
'''
return (
    include_visits, 
    v_rad, 
    resampled_wavelength, 
    resampled_flux, 
    resampled_flux_error, 
    resampled_pseudo_cont, 
    resampled_bitmask
)
'''

_filter_data_products = lambda idp, filetype, telescope=None: filter(idp, lambda dp: (dp.filetype == filetype) and ((telescope is None) or (telescope == dp.kwargs["telescope"])))


def create_mwm_data_product(source):
    
    primary_hdu = create_primary_hdu(source)
    boss_visits_hdu, boss_resampled = create_boss_visits_hdu(_filter_data_products(source.data_products, "specLite"))
    apo25m_hdu, apo25m_resampled = create_apogee_visits_hdu(
        _filter_data_products(source.data_products, "apVisit", "apo25m"),
        description_suffix="-North"
    )
    lco25m_hdu, lco25m_resampled = create_apogee_visits_hdu(
        _filter_data_products(source.data_products, "apVisit", "lco25m"),
        description_suffix="-South"
    )

    raise a


def create_boss_visits_hdu(data_products: List[Union[DataProduct, str]], **kwargs):
    """
    Create a binary table HDU containing all BOSS visit spectra.

    :param data_products:
        A list of SpecLite data products.
    """

    data_products_used, vrad, wavelength, flux, flux_error, cont, bitmask, meta = resampled = resample_boss_visit_spectra(
        data_products,
        **kwargs
    )
    header = fits.Header([
        ("DESCR", "BOSS visits resampled to rest-frame vacuum wavelengths"),
        ("CRVAL1", meta["crval"]),
        ("CDELT1", meta["cdelt"]),
        ("CRPIX1", 1),
        ("CTYPE1", "LOG-LINEAR"),
        ("NAXIS1", meta["num_pixels"]),
        ("DC-FLAG", 1)
    ])
    
    #visit_basenames = [os.path.basename(dp.path if isinstance(dp, DataProduct) else dp) for dp in data_products_used]
    
    # https://docs.astropy.org/en/stable/io/fits/usage/table.html#column-creation
    
    S = flux.size
    V, P = flux.shape
    snr = np.nanmean(flux / flux_error, axis=1)
    flux_unit = "10^-17 erg/s/cm^2/Ang"
    columns = [
        fits.Column(name="flux", array=flux, format=f"{S:.0f}E", dim=f"({P}, )", unit=flux_unit),
        fits.Column(name="flux_error", array=flux_error, format=f"{S:.0f}E", dim=f"({P}, )", unit=flux_unit),
        fits.Column(name="bitmask", array=bitmask, format=f"{S:.0f}K", dim=f"({P}, )"),
        fits.Column(name="snr", array=snr, format="E"),
        fits.Column(name="v_rad", array=vrad, format="E", unit="km/s"),
        # TODO: Add other metadata that we retrieved from the visit files.
    ]

    try:
        for key in data_products_used[0].kwargs.keys():
            values = [v.kwargs[key] for v in data_products_used]
            columns.append(
                fits.Column(name=key, array=values, format=fits_format_string(values))
            )
    except:
        log.exception(f"Unable to add visit meta columns for BOSS HDU")

    hdu = fits.BinTableHDU.from_columns(columns, header=header)
    return (hdu, resampled)


'''
from astra.sdss.datamodels2 import create_boss_visits_hdu
from astra.database.astradb import Source

source = Source.get(catalogid=27021597917837494)
hdu, resampled = create_boss_visits_hdu(source.data_products)
'''

    

def create_apogee_visits_hdu(
    data_products: List[Union[DataProduct, str]], 
    description_suffix: Optional[str] = None,
    **kwargs
):
    """
    Create a binary table HDU containing all given APOGEE visit spectra.

    :param data_products:
        A list of ApVisit data products.
    """

    data_products_used, vrad, wavelength, flux, flux_error, cont, bitmask, meta = resampled = resample_apogee_visit_spectra(
        data_products,
        **kwargs
    )
    header = fits.Header([
        ("DESCR", f"APOGEE{description_suffix or ''} visits resampled to rest-frame vacuum wavelengths"),
        ("CRVAL1", meta["crval"]),
        ("CDELT1", meta["cdelt"]),
        ("CRPIX1", 1),
        ("CTYPE1", "LOG-LINEAR"),
        ("NAXIS1", meta["num_pixels"]),
        ("DC-FLAG", 1)
    ])

    S = flux.size
    V, P = flux.shape
    snr = np.nanmean(flux / flux_error, axis=1)
    flux_unit = "10^-17 erg/s/cm^2/Ang"
    columns = [
        fits.Column(name="flux", array=flux, format=f"{S:.0f}E", dim=f"({P}, )", unit=flux_unit),
        fits.Column(name="flux_error", array=flux_error, format=f"{S:.0f}E", dim=f"({P}, )", unit=flux_unit),
        fits.Column(name="bitmask", array=bitmask, format=f"{S:.0f}K", dim=f"({P}, )"),
        fits.Column(name="snr", array=snr, format="E"),
        fits.Column(name="v_rad", array=vrad, format="E", unit="km/s"),
        # TODO: Add other metadata that we retrieved from the visit files.
    ]

    # TODO: DRY
    try:
        for key in data_products_used[0].kwargs.keys():
            values = [v.kwargs[key] for v in data_products_used]
            columns.append(
                fits.Column(name=key, array=values, format=fits_format_string(values))
            )
    except:
        log.exception(f"Unable to add visit meta columns for APOGEE visits")

    hdu = fits.BinTableHDU.from_columns(columns, header=header)
    return (hdu, resampled)


def create_primary_hdu(source: Union[Source, int]) -> fits.PrimaryHDU:
    """
    Create primary HDU (headers only) for a Milky Way Mapper data product, given some source.
    
    :param source:
        The astronomical source, or the SDSS-V catalog identifier.
    """
    catalogid = source.catalogid if isinstance(source, Source) else int(source)

    # Sky position.
    ra, dec = get_sky_position(catalogid)
    healpix = ang2pix(128, ra, dec, lonlat=True)

    return fits.PrimaryHDU(
        header=fits.Header(
            [
                (
                    "DATE", 
                    datetime.datetime.utcnow().strftime("%Y-%m-%d"),
                    "File creation date (UTC)"
                ),
                (   
                    "ASTRAVER", 
                    astra_version,
                    "Software version of Astra"
                ),
                (
                    "CATID", 
                    catalogid,
                    "SDSS-V catalog identifier"
                ),
                (
                    "RA",
                    ra,
                    "RA (J2000)"
                ),
                (
                    "DEC",
                    dec,
                    "DEC (J2000)" 
                ),
                (
                    "HEALPIX",
                    healpix,
                    "HEALPix location"
                ),
            ]
        )
    )



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

    include_visits, wavelength, v_rad, flux, flux_error, sky_flux = ([], [], [], [], [], [])
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
            
            v_rad.append(v)
            wavelength.append(10**image[1].data["LOGLAM"])
            flux.append(image[1].data["FLUX"])
            flux_error.append(image[1].data["IVAR"]**-0.5)
            sky_flux.append(image[1].data["SKY"])
        
        include_visits.append(visit)

    resampled_wavelength = log_lambda_dispersion(crval, cdelt, num_pixels)
    args = (resampled_wavelength, num_pixels_per_resolution_element, v_rad, wavelength)
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
    meta = dict(
        crval=crval,
        cdelt=cdelt,
        num_pixels=num_pixels
    )

    return (
        include_visits, 
        v_rad, 
        resampled_wavelength, 
        resampled_flux, 
        resampled_flux_error, 
        resampled_pseudo_cont, 
        resampled_bitmask,
        meta
    )


def get_apogee_radial_velocity(
    image: fits.hdu.hdulist.HDUList, 
    visit: Union[DataProduct, str]
):
    """
    Return the (current) best-estimate of the radial velocity (in km/s) of this 
    visit spectrum that is stored in the APOGEE DRP database.

    :param image:
        The FITS image of the ApVisit data product.
    
    :param visit:
        The supplied visit.
    """
    from astra.database.apogee_drpdb import RvVisit
    q = (
        RvVisit.select()
               .where(RvVisit.catalogid == image[0].header["CATID"])
               .order_by(RvVisit.created.desc())
    )
    return q.first().vrel


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
        radial_velocities = get_apogee_radial_velocity

    include_visits, wavelength, v_rad = ([], [], [])
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
        
            v_rad.append(v)
            wavelength.append(image[hdu_wl].data)
            flux.append(image[hdu_flux].data)
            flux_error.append(image[hdu_flux_error].data)
            # We resample the bitmasks, and we provide a bad pixel mask.
            bitmasks.append(image[hdu_bitmask].data)
            bad_pixel_mask.append(
                np.where(bitmasks[-1] & pixel_mask.badval())[0]
            )

        include_visits.append(visit)

    resampled_wavelength = log_lambda_dispersion(crval, cdelt, num_pixels)
    args = (resampled_wavelength, num_pixels_per_resolution_element, v_rad, wavelength)

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
    resampled_bitmask, *_ = resample_visit_spectra(*args, bitmasks)
    return (
        include_visits, 
        v_rad, 
        resampled_wavelength, 
        resampled_flux, 
        resampled_flux_error, 
        resampled_pseudo_cont, 
        resampled_bitmask
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


def fits_format_string(values):
    if all(isinstance(v, str) for v in values):
        max_len = max(map(len, values))
        return f"{max_len}A"
    
    first_type = type(values[0])
    return {
        int: "J", # 32 bit
        float: "E", # 32 bit
        bool: "L"
    }.get(first_type)



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
