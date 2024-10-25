from astropy.io import fits
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from juliacall import Main as jl
# TODO dev it instead
jl.include("../src/Grok.jl")
Grok = jl.Grok

def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))

def read_apstar(path, inflate_errors=True, use_ferre_mask=True):
    with fits.open(expand_path(path)) as image:
        flux = image[1].data[0]
        e_flux = image[2].data[0]
        pixel_flags = image[3].data[0]
        
    if inflate_errors:
        flux, e_flux = inflate_errors_at_bad_pixels(
            flux, 
            e_flux,
            pixel_flags,
        )
    
    if use_ferre_mask:
        ferre_mask = np.loadtxt(expand_path("ferre_mask.dat"))
        use_pixel = (ferre_mask == 1)        
        e_flux[~use_pixel] = np.inf

    wl = 10**(4.179 + 6e-6 * np.arange(8575))
    
    return (wl, flux, e_flux, pixel_flags)

def inflate_errors_at_bad_pixels(
    flux,
    e_flux,
    bitfield,
    skyline_sigma_multiplier=100,
    bad_pixel_flux_value=1e-4,
    bad_pixel_error_value=1e10,
    spike_threshold_to_inflate_uncertainty=3,
    min_sigma_value=0.05,
):
    # Inflate errors around skylines,
    skyline_mask = (bitfield & 4096) > 0 # significant skyline
    e_flux[skyline_mask] *= skyline_sigma_multiplier

    # Sometimes FERRE will run forever.
    if spike_threshold_to_inflate_uncertainty > 0:

        flux_median = np.nanmedian(flux)
        flux_stddev = np.nanstd(flux)
        e_flux_median = np.median(e_flux)

        delta = (flux - flux_median) / flux_stddev
        is_spike = (delta > spike_threshold_to_inflate_uncertainty)
        #* (
        #    sigma_ < (parameters["spike_threshold_to_inflate_uncertainty"] * e_flux_median)
        #)
        #if np.any(is_spike):
        #    sum_spike = np.sum(is_spike)
            #fraction = sum_spike / is_spike.size
            #log.warning(
            #    f"Inflating uncertainties for {sum_spike} pixels ({100 * fraction:.2f}%) that were identified as spikes."
            #)
            #for pi in range(is_spike.shape[0]):
            #    n = np.sum(is_spike[pi])
            #    if n > 0:
            #        log.debug(f"  {n} pixels on spectrum index {pi}")
        e_flux[is_spike] = bad_pixel_error_value

    # Set bad pixels to have no useful data.
    if bad_pixel_flux_value is not None or bad_pixel_error_value is not None:                            
        bad = (
            ~np.isfinite(flux)
            | ~np.isfinite(e_flux)
            | (flux < 0)
            | (e_flux < 0)
            | ((bitfield & 16639) > 0) # any bad value (level = 1)
        )

        flux[bad] = bad_pixel_flux_value
        e_flux[bad] = bad_pixel_error_value        

    if min_sigma_value is not None:
        e_flux = np.clip(e_flux, min_sigma_value, np.inf)

    return (flux, e_flux)

def read_tayar_files():
    from astropy.table import Table
    t = Table.read("tayar_2015/apj514696t1_mrt_xm_aspcap.fits")    
    is_measurement = (t["f_vsini"] != "<")
    paths = []
    for twomass_id in t[is_measurement]["2MASS"]:
        apogee_id = twomass_id.lstrip("J")
        paths.append(f"tayar_2015/spectra/apStar-dr17-2M{apogee_id}.fits")
    
    fluxes = []
    ivars = []
    for path in tqdm(paths[1:10], "load spectra"):
        wl, flux, e_flux, pixel_flags = read_apstar(path)
        fluxes.append(flux)
        ivars.append(e_flux ** (-1/2))

    return fluxes, ivars, t

if __name__ == '__main__':
    fluxes, ivars, t = read_tayar_files()

    best_fit_nodes = Grok.get_best_nodes(fluxes, ivars, "../../grok_old/korg_grid_old.h5")
    print(best_fit_nodes)

    #names=("apogee_id", "grok_teff", "grok_logg", "grok_m_h", "grok_v_micro", "grok_v_sini", "chi2")
    #output_path = "/Users/andycasey/research/Grok.jl/sandbox/tayar_2015/20240209_grok_results.fits"
    #Table(rows=results, names=names).write(output_path, overwrite=True)

