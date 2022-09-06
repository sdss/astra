"""
Continuum-normalization using sines and cosines.
"""

import numpy as np
import os

from astra.utils import log


def normalize(
    spectrum,
    continuum_regions=None,
    L=1400,
    order=3,
    regions=([3000, 10000], [15090, 15822], [15823, 16451], [16452, 16971]),
    fill_value=1.0,
    **kwargs,
):
    """
    Pseudo-continuum-normalize the flux using a defined set of continuum pixels and a sum of sine
    and cosine functions.

    :param spectrum:
        The `Spectrum1D` object to normalize.

    :param continuum_regions: [optional]
        A list of two-length tuples that describe the start and end of regions
        that should be treated as continuum.

    :param L: [optional]
        The length scale for the sines and cosines.

    :param order: [optional]
        The number of sine/cosine functions to use in the fit.

    :param regions: [optional]
        Specify sections of the spectra that should be fitted separately in each
        star. This may be due to gaps between CCDs, or some other physically-
        motivated reason. These values should be specified in the same units as
        the `dispersion`, and should be given as a list of `[(start, end), ...]`
        values. For example, APOGEE spectra have gaps near the following
        wavelengths which could be used as `regions`:

        >> regions = ([15090, 15822], [15823, 16451], [16452, 16971])

    :param fill_value: [optional]
        The continuum value to use for when no continuum was calculated for that
        particular pixel (e.g., the pixel is outside of the `regions`).

    :param full_output: [optional]
        If set as True, then a metadata dictionary will also be returned.

    :returns:
        The continuum values for all pixels, and a dictionary that contains
        metadata about the fit.
    """

    if continuum_regions is None:
        default_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../etc/continuum-regions.list",
        )
        continuum_regions = np.loadtxt(default_path)

    elif isinstance(continuum_regions, str):
        continuum_regions = np.loadtxt(continuum_regions)

    dispersion = spectrum.wavelength.value
    flux, ivar = (spectrum.flux.value, spectrum.uncertainty.array)

    # Work out the continuum pixels.
    mask = np.zeros(dispersion.size, dtype=bool)
    if continuum_regions is not None:
        for start, end in dispersion.searchsorted(continuum_regions):
            mask[start:end] = True
    else:
        # Use all pixels as 'continuum'.
        mask[:] = True

    continuum_pixels = np.arange(dispersion.size)[mask]

    continuum, metadata = sines_and_cosines(
        dispersion,
        flux,
        ivar,
        continuum_pixels,
        L=L,
        order=order,
        regions=regions,
        fill_value=fill_value,
        **kwargs,
    )

    spectrum._data /= continuum
    spectrum._uncertainty.array *= continuum * continuum

    no_information = spectrum._uncertainty.array == 0
    spectrum._data[no_information] = 1.0

    non_finite_pixels = ~np.isfinite(spectrum._data)
    spectrum._data[non_finite_pixels] = 1.0
    spectrum._uncertainty.array[non_finite_pixels] = 0.0

    if "slice_args" in kwargs or "shape" in kwargs:
        return slice_and_shape(spectrum, **kwargs)
    return spectrum


def slice_and_shape(spectrum, slice_args, shape, repeat=None, **kwargs):

    if repeat is not None:
        spectrum._data = np.repeat(spectrum._data, repeat)
        spectrum._uncertainty.array = np.repeat(spectrum._uncertainty.array, repeat)

    if slice_args is not None:
        slices = tuple([slice(*each) for each in slice_args])

        spectrum._data = spectrum._data[slices]
        spectrum._uncertainty.array = spectrum._uncertainty.array[slices]

        try:
            spectrum.meta["snr"] = spectrum.meta["snr"][slices[0]]
        except:
            log.warning(f"Unable to slice 'snr' metadata with {slice_args}")

    spectrum._data = spectrum._data.reshape(shape)
    spectrum._uncertainty.array = spectrum._uncertainty.array.reshape(shape)

    return spectrum


def sines_and_cosines(
    dispersion,
    flux,
    ivar,
    continuum_pixels,
    L=1400,
    order=3,
    regions=None,
    fill_value=1.0,
    **kwargs,
):
    """
    Fit the flux values of pre-defined continuum pixels using a sum of sine and
    cosine functions.

    :param dispersion:
        The dispersion values.

    :param flux:
        The flux values for all pixels, as they correspond to the `dispersion`
        array.

    :param ivar:
        The inverse variances for all pixels, as they correspond to the
        `dispersion` array.

    :param continuum_pixels:
        A mask that selects pixels that should be considered as 'continuum'.

    :param L: [optional]
        The length scale for the sines and cosines.

    :param order: [optional]
        The number of sine/cosine functions to use in the fit.

    :param regions: [optional]
        Specify sections of the spectra that should be fitted separately in each
        star. This may be due to gaps between CCDs, or some other physically-
        motivated reason. These values should be specified in the same units as
        the `dispersion`, and should be given as a list of `[(start, end), ...]`
        values. For example, APOGEE spectra have gaps near the following
        wavelengths which could be used as `regions`:

        >> regions = ([15090, 15822], [15823, 16451], [16452, 16971])

    :param fill_value: [optional]
        The continuum value to use for when no continuum was calculated for that
        particular pixel (e.g., the pixel is outside of the `regions`).

    :param full_output: [optional]
        If set as True, then a metadata dictionary will also be returned.

    :returns:
        The continuum values for all pixels, and a dictionary that contains
        metadata about the fit.
    """

    scalar = kwargs.pop("__magic_scalar", 1e-6)  # MAGIC
    flux, ivar = np.atleast_2d(flux), np.atleast_2d(ivar)

    bad = ~np.isfinite(ivar) + ~np.isfinite(flux) + (ivar == 0)
    ivar[bad] = 0
    flux[bad] = 1

    if regions is None:
        regions = [(dispersion[0], dispersion[-1])]

    region_masks = []
    region_matrices = []
    continuum_masks = []
    continuum_matrices = []
    pixel_included_in_regions = np.zeros_like(flux).astype(int)
    for i, (start, end) in enumerate(regions):
        # Build the masks for this region.
        si, ei = np.searchsorted(dispersion, (start, end))

        if si == ei:
            # No pixels. Not a valid region.
            continue

        region_mask = (end >= dispersion) * (dispersion >= start)
        region_masks.append(region_mask)
        pixel_included_in_regions[:, region_mask] += 1

        continuum_masks.append(
            continuum_pixels[(ei >= continuum_pixels) * (continuum_pixels >= si)]
        )

        # Build the design matrices for this region.
        region_matrices.append(
            _continuum_design_matrix(dispersion[region_masks[-1]], L, order)
        )
        continuum_matrices.append(
            _continuum_design_matrix(dispersion[continuum_masks[-1]], L, order)
        )

        # TODO: ISSUE: Check for overlapping regions and raise an warning.

    # Check for non-zero pixels (e.g. ivar > 0) that are not included in a
    # region. We should warn about this very loudly!
    warn_on_pixels = (pixel_included_in_regions == 0) * (ivar > 0)

    metadata = []
    continuum = np.ones_like(flux) * fill_value
    for i in range(flux.shape[0]):

        warn_indices = np.where(warn_on_pixels[i])[0]
        if any(warn_indices):
            # Split by deltas so that we give useful warning messages.
            segment_indices = np.where(np.diff(warn_indices) > 1)[0]
            segment_indices = np.sort(
                np.hstack([0, segment_indices, segment_indices + 1, len(warn_indices)])
            )
            segment_indices = segment_indices.reshape(-1, 2)

            segments = ", ".join(
                [
                    "{:.1f} to {:.1f}".format(dispersion[s], dispersion[e], e - s)
                    for s, e in segment_indices
                ]
            )

            log.warning(
                f"Some pixels in have measured flux values (e.g., ivar > 0) but "
                f"are not included in any specified region ({segments})."
            )

        # Get the flux and inverse variance for this object.
        object_metadata = []
        object_flux, object_ivar = (flux[i], ivar[i])

        # Normalize each region.
        for region_mask, region_matrix, continuum_mask, continuum_matrix in zip(
            region_masks, region_matrices, continuum_masks, continuum_matrices
        ):
            if continuum_mask.size == 0:
                # Skipping..
                object_metadata.append([order, L, fill_value, scalar, [], None])
                continue

            # We will fit to continuum pixels only.
            continuum_disp = dispersion[continuum_mask]
            continuum_flux, continuum_ivar = (
                object_flux[continuum_mask],
                object_ivar[continuum_mask],
            )

            # Solve for the amplitudes.
            M = continuum_matrix
            MTM = np.dot(M, continuum_ivar[:, None] * M.T)
            MTy = np.dot(M, (continuum_ivar * continuum_flux).T)

            eigenvalues = np.linalg.eigvalsh(MTM)
            MTM[np.diag_indices(len(MTM))] += scalar * np.max(eigenvalues)
            eigenvalues = np.linalg.eigvalsh(MTM)
            condition_number = max(eigenvalues) / min(eigenvalues)

            amplitudes = np.linalg.solve(MTM, MTy)
            continuum[i, region_mask] = np.dot(region_matrix.T, amplitudes)
            object_metadata.append(
                (order, L, fill_value, scalar, amplitudes, condition_number)
            )

        metadata.append(object_metadata)

    return (continuum, metadata)


def _continuum_design_matrix(dispersion, L, order):
    """
    Build a design matrix for the continuum determination, using sines and
    cosines.

    :param dispersion:
        An array of dispersion points.

    :param L:
        The length-scale for the sine and cosine functions.

    :param order:
        The number of sines and cosines to use in the fit.
    """

    L, dispersion = float(L), np.array(dispersion)
    scale = 2 * (np.pi / L)
    return np.vstack(
        [
            np.ones_like(dispersion).reshape((1, -1)),
            np.array(
                [
                    [np.cos(o * scale * dispersion), np.sin(o * scale * dispersion)]
                    for o in range(1, order + 1)
                ]
            ).reshape((2 * order, dispersion.size)),
        ]
    )


if __name__ == "__main__":

    # path = '/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/stars/apo25m/97/97480/apStar-daily-apo25m-2M21241108+0023308.fits'
    path = (
        path
    ) = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/stars/apo25m/56/56822/apStar-daily-apo25m-2M11322307+2500207-59314.fits"
    from astra.tools.spectrum import Spectrum1D

    spec = Spectrum1D.read(path)

    normalize(spec)
