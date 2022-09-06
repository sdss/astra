import numpy as np


def log_lambda_dispersion(crval, cdelt, num_pixels):
    return 10 ** (crval + cdelt * np.arange(num_pixels))


def calculate_snr(flux, flux_error, axis=None):
    snr_pixel = flux / flux_error
    bad_pixels = ~np.isfinite(snr_pixel) | (flux < 0) | (flux_error < 0)
    snr_pixel[bad_pixels] = np.nan
    return np.nanmean(snr_pixel, axis=axis)
