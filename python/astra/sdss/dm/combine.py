import numpy as np
import numpy as np
from collections import OrderedDict
from scipy import interpolate
from scipy.ndimage.filters import median_filter, gaussian_filter
from astropy.constants import c
from astropy import units as u

from typing import Union, List, Callable, Optional, Dict

from astra.utils import flatten

C_KM_S = c.to(u.km / u.s).value

def pixel_weighted_spectrum(flux, flux_error, continuum, bitmask):

    # Pixel-by-pixel weighted average
    resampled_ivar = 1.0 / flux_error**2
    resampled_ivar[flux_error == 0] = 0

    cont = np.median(continuum, axis=0) # TODO: is this right?
    stacked_ivar = np.sum(resampled_ivar, axis=0)
    stacked_flux = np.sum(flux * resampled_ivar, axis=0) / stacked_ivar * cont
    stacked_flux_error = np.sqrt(1.0/stacked_ivar) * cont

    stacked_bitmask = np.bitwise_or.reduce(bitmask, 0)

    return (stacked_flux, stacked_flux_error, stacked_bitmask)


def separate_bitmasks(bitmasks):
    """
    Separate a bitmask array into arrays of bitmasks for each bit. Assumes base-2.

    :param bitmasks:
        An list of bitmask arrays.
    """

    q_max = max([int(np.log2(np.max(bitmask))) for bitmask in bitmasks])
    separated = OrderedDict()
    for q in range(q_max):
        separated[q] = []
        for bitmask in bitmasks:                
            is_set = (bitmask & np.uint64(2**q)) > 0
            separated[q].append(np.clip(is_set, 0, 1).astype(float))
    return separated

"""
from astra.database.astradb import Source
source = Source.get(catalogid=4551536934)
from astra.sdss.dm.mwm import create_mwm_data_products

foo = create_mwm_data_products(source)
"""

def resample_visit_spectra(
    resampled_wavelength,
    num_pixels_per_resolution_element,
    radial_velocity,
    wavelength,
    flux,
    flux_error=None,
    bitmask=None,
    scale_by_pseudo_continuum=False,
    use_smooth_filtered_spectrum_for_bad_pixels=False,
    bad_pixel_mask=None,
    median_filter_size=501,
    median_filter_mode="reflect",
    gaussian_filter_size=100,
):
    """
    Resample visit spectra onto a common wavelength array.

    :param resampled_wavelength:
        An array of wavelength values to resample the flux onto.
    
    :param num_pixels_per_resolution_element:
        The number of pixels per resolution element to assume when performing sinc interpolation.
    
    :param radial_velocity:
        The radial velocity of each flux. This should be an array of length N, where the `flux` argument
        has shape (N, P).
    
    :param flux:
        A flux array of shape (N, P) where N is the number of visits and P is the number of pixels.
    
    :param flux_error: [optional]
        A flux error array of shape (N, P) where N is the number of visits and P is the number of pixels.
    
    :param bitmask: [optional]
        A bitmask array the same shape as `flux`.

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

    if bitmask is not None:
        # Separate all bitmask values.
        separated_bitmasks = separate_bitmasks(bitmask)
        n_flags = len(separated_bitmasks)
        resampled_bitmasks = np.zeros((n_visits, n_pixels, n_flags))
        #num_flagged_pixels = { flag: np.sum(a > 0, axis=-1).astype(int) for flag, a in separated_bitmasks.items() }

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

            if bitmask is not None:
                # Do the sinc interpolation on each bitmask value.
                output = sincint(
                    pixel[finite],
                    n_res,
                    [
                        [visit_and_chip(flag_bitmask, i, j), None] for flag_bitmask in separated_bitmasks.values()
                    ]
                )

                for k, (flag, (resampled_bitmask_flag, _)) in enumerate(zip(separated_bitmasks.keys(), output)):
                    #if num_flagged_pixels[flag][i, j] == 0: continue

                    # The resampling will produce a continuous (fraction) of bitmask values everywhere
                    # with an exponential sinc function pattern. In SDSS-IV they decided just to take
                    # any pixel with a fraction > 0.1 (in most cases) and assign pixels like that with
                    # the bitmask.

                    # If you have a *single* pixel that is flagged, and zero radial velocity (so no shift)
                    # then this >0.1 metric would end up flagging the neighbouring pixels as well, even
                    # though there was no change to the flux. 

                    # Instead, here I will take metric to be whatever is needed to keep the same *number*
                    # of pixels originally flagged.
                    # and we take the absolute so that we don't imprint a fringe pattern on the bitmask
                    #metric = np.sort(np.abs(resampled_bitmask_flag))[-num_flagged_pixels[flag][i, j]]
                    #print(f"Took metric={metric:.1f} for bitmask {flag} on visit {i} chip {j}")
                    
                    # Turns out that this was not a good idea. Let's be more conservative.
                    metric = 0.1
                    resampled_bitmasks[i, finite, k] = (resampled_bitmask_flag >= metric).astype(int)
                

            # Scale by continuum?
            if scale_by_pseudo_continuum:
                # TODO: If there are gaps in `finite` then this will cause issues because median filter and gaussian filter 
                #       don't receive the x array
                # TODO: Take a closer look at this process.
                resampled_pseudo_cont[i, finite] = smooth_filter(resampled_chip_flux)

                resampled_flux[i, finite] /= resampled_pseudo_cont[i, finite]
                resampled_flux_error[i, finite] /= resampled_pseudo_cont[i, finite]

    # TODO: return flux ivar instead?
    resampled_bitmask = np.zeros(resampled_flux.shape, dtype=np.uint64)
    if bitmask is not None:
        # Sum them together.
        for k, flag in enumerate(separated_bitmasks.keys()):
            resampled_bitmask += (resampled_bitmasks[:, :, k] * (2**flag)).astype(np.uint64)


    return (
        resampled_flux,
        resampled_flux_error,
        resampled_pseudo_cont,
        resampled_bitmask
    )


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

        # the sinc function value at x = 0 is defined by the limit, -> 1
        sinc[u2 == 0] = 1

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
