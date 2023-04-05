import numpy as np
from scipy import interpolate


def resample(spectrum, wavelength, n_res):        
    pixel = wave_to_pixel(spectrum.wavelength, wavelength)

    (finite,) = np.where(np.isfinite(pixel))

    ((finite_flux, finite_e_flux), ) = sincint(
        pixel[finite], n_res, [
            [spectrum.flux[finite], 1/spectrum.ivar[finite]]
        ]
    )

    flux = np.zeros_like(spectrum.wavelength)
    e_flux = np.zeros_like(spectrum.wavelength)
    flux[finite] = finite_flux
    e_flux[finite] = finite_e_flux
    

    spectrum.flux = flux
    spectrum.ivar = e_flux**-2
    return None



def resample_apmadgics(spectrum, n_res):       
    x = np.arange(spectrum.wavelength.size) 

    pixel = wave_to_pixel(x + spectrum.v_rad_pixel, x)
    (finite, ) = np.where(np.isfinite(pixel))

    ((finite_flux, finite_e_flux), ) = sincint(
        pixel[finite], n_res, [
            [spectrum.flux, 1/spectrum.ivar]
        ]
    )

    flux = np.nan * np.ones(spectrum.wavelength.size)
    e_flux = np.nan * np.ones(spectrum.wavelength.size)
    flux[finite] = finite_flux
    e_flux[finite] = finite_e_flux
    
    spectrum.flux = flux
    spectrum.ivar = e_flux**-2
    return None


def wave_to_pixel(wave, wave0):
    """convert wavelength to pixel given wavelength array
    Args :
       wave(s) : wavelength(s) (\AA) to get pixel of
       wave0 : array with wavelength as a function of pixel number
    Returns :
       pixel(s) in the chip
    """
    pix0 = np.arange(len(wave0))
    # Need to sort into ascending order
    sindx = np.argsort(wave0)
    wave0 = wave0[sindx]
    pix0 = pix0[sindx]
    # Start from a linear baseline
    baseline = np.polynomial.Polynomial.fit(wave0, pix0, 1)
    ip = interpolate.InterpolatedUnivariateSpline(wave0, pix0 / baseline(wave0), k=3)
    out = baseline(wave) * ip(wave)
    # NaN for out of bounds
    out[wave > wave0[-1]] = np.nan
    out[wave < wave0[0]] = np.nan
    return out


def sincint(x, nres, speclist):
    """Use sinc interpolation to get resampled values
    x : desired positions
    nres : number of pixels per resolution element (2=Nyquist)
    speclist : list of [quantity, variance] pairs (variance can be None)

    NOTE: This takes in variance, but returns ERROR.
    """

    dampfac = 3.25 * nres / 2.0
    ksize = int(21 * nres / 2.0)
    if ksize % 2 == 0:
        ksize += 1
    nhalf = ksize // 2

    # number of output and input pixels
    nx = len(x)
    nf = len(speclist[0][0])

    # integer and fractional pixel location of each output pixel
    ix = x.astype(int)
    fx = x - ix

    # outputs
    outlist = []
    for spec in speclist:
        if spec[1] is None:
            outlist.append([np.full_like(x, 0), None])
        else:
            outlist.append([np.full_like(x, 0), np.full_like(x, 0)])

    for i in range(len(x)):
        xkernel = np.arange(ksize) - nhalf - fx[i]
        # in units of Nyquist
        xkernel /= nres / 2.0
        u1 = xkernel / dampfac
        u2 = np.pi * xkernel
        sinc = np.exp(-(u1**2)) * np.sin(u2) / u2
        sinc /= nres / 2.0

        # the sinc function value at x = 0 is defined by the limit, -> 1
        sinc[u2 == 0] = 1

        lobe = np.arange(ksize) - nhalf + ix[i]
        vals = np.zeros(ksize)
        gd = np.where((lobe >= 0) & (lobe < nf))[0]

        for spec, out in zip(speclist, outlist):
            vals = spec[0][lobe[gd]]
            out[0][i] = (sinc[gd] * vals).sum()
            if spec[1] is not None:
                var = spec[1][lobe[gd]]
                out[1][i] = (sinc[gd] ** 2 * var).sum()

    for out in outlist:
        if out[1] is not None:
            out[1] = np.sqrt(out[1])

    return outlist
