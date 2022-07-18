

# combine_boss_visits just needs the data products because the redshift is stored in the header

# combine_apogee_visits needs the best measured relative velocity for each visit, which is stored in the apogee_drp database
#   -> could just take visit data products and look up the radial velocity

import os
import numpy as np
from typing import Union, List
from astropy.io import fits
from astra import log
from astra.utils import flatten
from astra.database.astradb import DataProduct

from scipy import interpolate
from scipy.ndimage.filters import median_filter, gaussian_filter

# we need:
#various create_header_cards

#resample_boss_visit_spectra() # -> I/O and redshift etc
#resample_apogee_visit_spectra() # -> I/O and redshfit etc
#resample_visit_spectra() # --> does the actual work

#stack_spectra() # -> stacking given some resampled visits, sky, bitmasks, etc


# create_mwm_visits
    # -> resamples all visits
    # -> stores them


def resample_visit_spectra(
    resampled_wavelength,
    num_pixels_per_resolution_element,
    redshift,
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

    if redshift is None:
        redshift = np.zeros(n_visits)

    if len(redshift) != n_visits:
        raise ValueError(f"Unexpected number of redshifts ({len(redshift)} != {n_visits})")

    for i, z in enumerate(redshift):
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

            pixel = wave_to_pixel(resampled_wavelength * (1 + z), chip_wavelength)
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

    return (
        resampled_flux,
        resampled_flux_error,
        resampled_pseudo_cont,
    )



def log_lambda_dispersion(crval, cdelt, num_pixels):
    return 10**(crval + cdelt * np.arange(num_pixels))

def resample_boss_visit_spectra(
    visits: List[Union[DataProduct, str]],
    crval: float = 3.5523,
    cdelt: float = 1e-4,
    num_pixels: int = 4_648,
    num_pixels_per_resolution_element: int = 5,
):
    """
    Resample BOSS visit spectra onto a common wavelength array.
    
    :param visits:
        A list of specLite data products, or their paths.
    
    :param num_pixels_per_resolution_element: [optional]
        The number of pixels per resolution element assumed when performing sinc interpolation.
    """

    include_visits, wavelength, redshift, flux, flux_error, sky_flux = ([], [], [], [], [], [])
    for visit in visits:
        path = visit.path if isinstance(visit, DataProduct) else visit
        if not os.path.exists(path):
            log.warning(f"Missing file: {path} from {visit}")
            continue

        with fits.open(path) as image:
            z, = image[2].data["Z"]

            redshift.append(z)
            wavelength.append(10**image[1].data["LOGLAM"])
            flux.append(image[1].data["FLUX"])
            flux_error.append(image[1].data["IVAR"]**-0.5)
            sky_flux.append(image[1].data["SKY"])
        
        include_visits.append(visit)

    resampled_wavelength = log_lambda_dispersion(crval, cdelt, num_pixels)
    args = (resampled_wavelength, num_pixels_per_resolution_element, redshift, wavelength)

    resampled_flux, resampled_flux_error, resampled_pseudo_cont = resample_visit_spectra(
        *args,
        flux,
        flux_error,
        scale_by_pseudo_continuum=True,    
    )
    resampled_sky_flux, *_ = resample_visit_spectra(
        *args,
        sky_flux,
    )


    raise a


def resample_apogee_visit_spectra(
    visits: List[Union[DataProduct, str]],
    crval: float = 4.179,
    cdelt: float = 6e-6,
    num_pixels: int = 8_575,
    num_pixels_per_resolution_element=(5, 4.25, 3.5)
):
    """
    Resample APOGEE visit spectra onto a common wavelength array.
    
    :param visits:
        A list of ApVisit data products, or their paths.
    
    :param num_pixels_per_resolution_element: [optional]
        The number of pixels per resolution element assumed when performing sinc interpolation.
        This is given per chip.
    """

    include_visits, wavelength, redshift, flux, flux_error, sky_flux = ([], [], [], [], [], [])
    for visit in visits:
        path = visit.path if isinstance(visit, DataProduct) else visit
        if not os.path.exists(path):
            log.warning(f"Missing file: {path} from {visit}")
            continue

        # Get the redshift from the APOGEE_DRP database.

        with fits.open(path) as image:
            hdu_header, hdu_flux, hdu_flux_error, hdu_bitmask, hdu_wl, \
                hdu_sky, hdu_sky_error, \
                hdu_telluric, hdu_telluric_error, \
                hdu_wl_coeff, hdu_lsf_coeff = range(11)
        
            wavelength.append(image[hdu_wl].data)
            flux.append(image[hdu_flux].data)
            flux_error.append(image[hdu_flux_error].data)
            


        include_visits.append(visit)

    resampled_wavelength = log_lambda_dispersion(4.179, 6e-6, 8575)
    

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
