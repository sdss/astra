"""General utilities for creating data products."""

import numpy as np

def log_lambda_dispersion(crval, cdelt, num_pixels):
    return 10 ** (crval + cdelt * np.arange(num_pixels))

def calculate_snr(flux, flux_error, axis=None):
    snr_pixel = np.clip(flux, 0, np.inf) / np.clip(flux_error, 0, np.inf)
    bad_pixels = ~np.isfinite(snr_pixel) | (flux_error == 0)
    snr_pixel[bad_pixels] = 0
    return np.mean(snr_pixel, axis=axis)
