import numpy as np
from scipy import interpolate
from astropy.constants import c
from astropy import units as u
from collections import OrderedDict
from scipy.ndimage.filters import median_filter, gaussian_filter


C_KM_S = c.to(u.km / u.s).value



'''
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
'''
def separate_bitmasks(bitmasks):
    """
    Separate a bitmask array into arrays of bitmasks for each bit. Assumes base-2.

    :param bitmasks:
        An list of bitmask arrays.
    """

    q_max = max([int(np.log2(np.max(bitmask))) for bitmask in bitmasks])
    #q_max = int(np.ceil(np.log2(np.hstack([bitmasks]).flatten().max())))
    separated = OrderedDict()
    for q in range(1 + q_max):
        separated[q] = []
        for bitmask in bitmasks:
            is_set = (bitmask & np.int64(2**q)) > 0
            separated[q].append(np.clip(is_set, 0, 1).astype(float))
    return separated


smooth_filter = lambda f, median_filter_size=501, gaussian_filter_size=100, median_filter_mode='reflect': gaussian_filter(
    median_filter(f, [median_filter_size], mode=median_filter_mode),
    gaussian_filter_size,
)

def resample(old_wavelength, new_wavelength, flux, ivar, n_res, pixel_flags=None, fill_flux=0, fill_ivar=0, min_bitmask_value=0.1):
    # TODO: Check inputs

    new_flux = fill_flux * np.ones(new_wavelength.size)
    new_ivar = fill_ivar * np.ones(new_wavelength.size)

    if pixel_flags is not None:
        pixel_flags = np.atleast_2d(pixel_flags)
        separate_pixel_flags = separate_bitmasks(pixel_flags)
        n_flags = len(separate_pixel_flags)
        resampled_flags = np.zeros((n_flags, new_wavelength.size), dtype=pixel_flags.dtype)
    else:
        new_pixel_flags = np.zeros(new_wavelength.size, dtype=int)        

    old_wavelength = np.atleast_2d(old_wavelength)
    flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
        
    n_res = np.atleast_1d(n_res)
    for i, chip_wavelength in enumerate(old_wavelength):
        pixel = wave_to_pixel(new_wavelength, chip_wavelength)
        (finite, ) = np.where(np.isfinite(pixel))

        # do a smoothing of bad pixels
        flux_smooth = smooth_filter(flux[i])
        with np.errstate(divide='ignore', invalid='ignore'):
            var_smooth = smooth_filter(1/ivar[i])
        
        bad = (ivar[i] == 0) | (flux[i] < 0) | ~np.isfinite(flux[i]) | ~np.isfinite(ivar[i])
        
        sinc_flux = flux[i].copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc_var = 1/ivar[i]
        
        sinc_flux[bad] = flux_smooth[bad]
        sinc_var[bad] = var_smooth[bad]
        
        ((finite_flux, finite_e_flux), ) = sincint(
            pixel[finite], n_res[i], [
                [sinc_flux, sinc_var]
            ]
        )
        new_flux[finite] = finite_flux
        with np.errstate(divide='ignore', invalid='ignore'):
            new_ivar[finite] = finite_e_flux**(-2)

        if pixel_flags is not None:
        
            # Do the sinc interpolation on each bitmask value.
            output = sincint(
                pixel[finite],
                n_res[i],
                [
                    [flag_this_pixel[i], None]
                    for flag_this_pixel in separate_pixel_flags.values()
                ],
            )

            for k, (flag, (resampled_bitmask_flag, _)) in enumerate(
                zip(separate_pixel_flags.keys(), output)
            ):
                # if num_flagged_pixels[flag][i, j] == 0: continue

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
                # metric = np.sort(np.abs(resampled_bitmask_flag))[-num_flagged_pixels[flag][i, j]]
                # print(f"Took metric={metric:.1f} for bitmask {flag} on visit {i} chip {j}")

                # Turns out that this was not a good idea. Let's be more conservative.
                resampled_flags[k, finite] = (
                    np.abs(resampled_bitmask_flag) > min_bitmask_value
                ).astype(resampled_flags.dtype)            

    if pixel_flags is not None:
        new_pixel_flags = np.zeros(new_wavelength.size, dtype=pixel_flags.dtype)
        # Sum them together.
        for k, flag in enumerate(separate_pixel_flags.keys()):
            new_pixel_flags += (resampled_flags[k] * (2**flag)).astype(
                pixel_flags.dtype
            )

    return (new_flux, new_ivar, new_pixel_flags)


def pixel_weighted_spectrum(
    flux: np.ndarray,
    ivar: np.ndarray,
    bitmask: np.ndarray,
    scale_by_pseudo_continuum=False,
    median_filter_size=501,
    median_filter_mode="reflect",
    gaussian_filter_size=100,
    **kwargs,
):
    """
    Combine input spectra to produce a single pixel-weighted spectrum.

    :param flux:
        An array of flux values, resampled to the same wavelength values.

    :param ivar:
        inverse variance
        

    :param bitmask:
        A pixel bitmask, same shape as `flux`.

    :param scale_by_pseudo_continuum: [optional]
        Optionally scale each visit spectrum by its pseudo-continuum (a gaussian median filter) when
        stacking to keep them on the same relative scale (default: False).

    :param median_filter_size: [optional]
        The filter width (in pixels) to use for any median filters (default: 501).

    :param median_filter_mode: [optional]
        The mode to use for any median filters (default: reflect).

    :param gaussian_filter_size: [optional]
        The filter size (in pixels) to use for any gaussian filter applied.
    """
    meta = dict(
        scale_by_pseudo_continuum=scale_by_pseudo_continuum,
        median_filter_mode=median_filter_mode,
        gaussian_filter_size=gaussian_filter_size,
    )

    if flux.size == 0:
        return (None, None, None, None, meta)

    V, P = flux.shape
    continuum = np.ones((V, P), dtype=float)
    if scale_by_pseudo_continuum:
        smooth_filter = lambda f: gaussian_filter(
            median_filter(f, [median_filter_size], mode=median_filter_mode),
            gaussian_filter_size,
        )
        # TODO: If there are gaps in `finite` then this will cause issues because median filter and gaussian filter
        #       don't receive the x array
        # TODO: Take a closer look at this process.
        for v in range(V):
            finite = np.isfinite(flux)
            continuum[v, finite] = smooth_filter(flux[v, finite])


    #cont = np.median(continuum, axis=0)  # TODO: is this right?
    stacked_ivar = np.sum(ivar, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stacked_flux = np.sum(flux * ivar, axis=0) / stacked_ivar #* cont

    stacked_bitmask = np.bitwise_or.reduce(bitmask.astype(int), 0)
    return (stacked_flux, stacked_ivar, stacked_bitmask, continuum, meta)


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
    bad = ~np.isfinite(spectrum.ivar)
    spectrum.ivar[bad] = 0
    
    return None


def wave_to_pixel(wave, wave0):
    r"""convert wavelength to pixel given wavelength array
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



# Hogg start smashing here

def design_matrix(xs, P=None, L=None):
    """
    Take in a set of $x$ positions `xs` and return the Fourier design matrix.

    :param xs:
        An array of $x$ positions.
    
    :param P:
        The number of Fourier modes to include.
    
    :param L: [optional]
        The length scale. If `None` is given it defaults to `max(xs) - min(xs)`.

    .. notes:
        - The code looks different from the paper because Python zero-indexes.
        - This could be replaced with something that makes use of finufft.
    """
    P = P or xs.size
    L = L or (np.max(xs) - np.min(xs))
    scale = np.pi * xs / L

    # TODO: for square design matrices pre-calculate using sin(ab) = sin(a)cos(b) + cos(a)sin(b)
    return np.vstack(
        [
            np.ones_like(xs).reshape((1, -1)),
            np.array([
                (np.cos(scale * j) if j % 2 == 0 else np.sin(scale * (j + 1))) for j in range(1, P)
            ]),
        ]
    ).T


def old_design_matrix(xs, P=None, L=None):
    P = P or xs.size
    L = L or (np.max(xs) - np.min(xs))
    X = np.ones_like(xs).reshape(len(xs), 1)
    for j in range(1, P):
        if j % 2 == 0:
            X = np.concatenate((X, np.cos(np.pi * j * xs / L)[:, None]), axis=1)
        else:
            X = np.concatenate((X, np.sin(np.pi * (j + 1) * xs / L)[:, None]), axis=1)
    return X

def _design_matrix(self, dispersion: np.array) -> np.array:
    scale = 2 * (np.pi / self.L)
    return np.vstack(
        [
            np.ones_like(dispersion).reshape((1, -1)),
            np.array(
                [
                    [np.cos(o * scale * dispersion), np.sin(o * scale * dispersion)]
                    for o in range(1, self.deg + 1)
                ]
            ).reshape((2 * self.deg, dispersion.size)),
        ]
    )


def pack_matrices(xs, ys, ivar, bs, Delta_xs, P):
    """
    Rearrange data into big matrices for `lstsq()`.

    ## Bugs:
    - Needs comment header.
    """
    x_min, x_max = (np.min(xs), np.max(xs))
    L = x_max - x_min

    XX = np.array([])
    YY = np.array([])
    II = np.array([])
    for bb, yy, ii, Dx in zip(bs, ys, ivar, Delta_xs):
        x_rest = (xs - Dx)[bb > 0.5]
        I = np.logical_and(x_rest > x_min, x_rest < x_max)
        YY = np.append(YY, yy[bb > 0.5][I])
        XX = np.append(XX, x_rest[I])
        II = np.append(II, ii[I])
    return (design_matrix(XX, P, L), YY, II)

'''
def forward_model_tm(xs, ys, bs, Delta_xs, P, xstar):
    """
    Sample the forward model at the given `x_star` positions.

    ## Bugs:
    - Doesn't take the inverse variance on the data!
    - Doesn't return the inverse variance tensor for the output.
    """
    X, Y = pack_matrices(xs, ys, bs, Delta_xs, P)
    Xstar = design_matrix(xstar, P)
    thetahat, foo, bar, whatevs = np.linalg.lstsq(X, Y, rcond=None)
    return Xstar @ thetahat
'''

def sample_forward_model(wavelength, flux, ivar, pixel_flags, delta_wavelength, P, new_wavelength):
    """
    Sample the forward model at the required wavelengths.
    """
    A, Y, Cinv = pack_matrices(wavelength, flux, ivar, pixel_flags, delta_wavelength, P)
    Xstar = design_matrix(new_wavelength, P)
    theta_hat, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)

    raise a
    return Xstar @ theta_hat


if __name__ == "__main__":

    from astra.models.apogee import ApogeeVisitSpectrum

    spectrum = ApogeeVisitSpectrum.get(1178032)

    new_wavelength = 10**(4.179 + 6e-6 * np.arange(8575))
    wavelength = spectrum.wavelength.flatten()
    flux = np.atleast_2d(spectrum.flux.flatten())
    ivar = np.atleast_2d(spectrum.ivar.flatten())
    pixel_flags = np.atleast_2d(np.ones_like(flux))


    sample_forward_model(
        wavelength,
        flux,
        ivar,
        pixel_flags,
        [0],
        8575,
        new_wavelength,
    )



