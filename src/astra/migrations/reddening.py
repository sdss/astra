import numpy as np
import os
import warnings
import concurrent.futures
from astra.utils import log, expand_path, silenced
from astra.models.source import Source
from peewee import chunked
from astropy.coordinates import SkyCoord
from astropy import units as u
from functools import cache
from astra.migrations.utils import ProgressContext

# dust-maps data directory: $MWM_ASTRA/aux/dust-maps/
from dustmaps.sfd import SFDQuery
from dustmaps.edenhofer2023 import Edenhofer2023Query
from dustmaps.bayestar import BayestarQuery

# TODO: This python file is a lot of spaghetti code. sorry about that. refactor this!

von = lambda v: v or np.nan

def _update_reddening_on_source(source, sfd, edenhofer2023, bayestar2019, raise_exceptions=True):
    """
    Compute reddening and reddening uncertainties for a source using various methods.

    :param source:
        An astronomical source.
    """

    try:
        coord = SkyCoord(ra=source.ra * u.deg, dec=source.dec * u.deg)

        # Zhang et al. 2023
        source.ebv_zhang_2023 = 0.829 * von(source.zgr_e)
        source.e_ebv_zhang_2023 = 0.829 * von(source.zgr_e_e)

        # RJCE_GLIMPSE
        ebv_ehw2 = 2.61
        source.ebv_rjce_glimpse = ebv_ehw2 * (von(source.h_mag) - von(source.mag4_5) - 0.08)
        source.e_ebv_rjce_glimpse = ebv_ehw2 * np.sqrt(von(source.e_h_mag)**2 + von(source.d4_5m)**2)

        # RJCE_ALLWISE
        # We store unWISE (not ALLWISE) and we have only w2 fluxes, not w2 magnitudes.
        # See https://catalog.unwise.me/catalogs.html (Flux Scale) for justification of 32 mmag offset
        w2_mag_vega = -2.5 * np.log10(von(source.w2_flux)) + 22.5 - 32 * 1e-3 # Vega
        e_w2_mag_vega = (2.5 / np.log(10)) * von(source.w2_dflux) / von(source.w2_flux)
        source.ebv_rjce_allwise = ebv_ehw2 * (von(source.h_mag) - w2_mag_vega - 0.08)
        source.e_ebv_rjce_allwise = ebv_ehw2 * np.sqrt(von(source.e_h_mag)**2 + e_w2_mag_vega**2)

        # SFD
        e_sfd = sfd(coord)
        source.ebv_sfd = 0.884 * e_sfd
        source.e_ebv_sfd = np.sqrt(0.01**2 + (0.1 * e_sfd)**2)

        d = von(source.r_med_geo) # [pc]
        d_err = 0.5 * (von(source.r_hi_geo) - von(source.r_lo_geo))

        d_samples = np.clip(np.random.normal(d, d_err, 20), 1, np.inf) #

        coord_samples = SkyCoord(
            ra=source.ra * u.deg,
            dec=source.dec * u.deg,
            distance=d_samples * u.pc
        )

        # Edenhofer
        if d is not None:
            if d < 69:
                coord_integrated = SkyCoord(ra=source.ra * u.deg, dec=source.dec * u.deg, distance=69 * u.pc)
                ed = edenhofer2023(coord_integrated)
                source.ebv_edenhofer_2023 = 0.829 * ed
                # TODO: document says 'reddening uncertainty = the reddening value' -> the scaled 0.829 value?
                source.e_ebv_edenhofer_2023 = 0.829 * ed
                source.flag_ebv_from_edenhofer_2023 = True

            else:
                ed = edenhofer2023(coord_samples, mode="samples")
                with np.errstate(divide='ignore', invalid='ignore'):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        # Take the nanmedian and nanstd as the samples are often NaNs
                        source.ebv_edenhofer_2023 = 0.829 * np.nanmedian(ed)
                        source.e_ebv_edenhofer_2023 = 0.829 * np.nanstd(ed)

        #assert np.isfinite(source.ebv_edenhofer_2023)

        # Bayestar 2019
        bs_samples = bayestar2019(coord_samples, mode="samples").flatten()
        source.ebv_bayestar_2019 = 0.88 * np.median(bs_samples)
        source.e_ebv_bayestar_2019 = 0.88 * np.std(bs_samples)

        # Logic to decide preferred reddening value

        if source.zgr_e is not None and source.zgr_quality_flags < 8: # target is in Zhang
            # Zhang et al. (2023)
            source.flag_ebv_from_zhang_2023 = True
            source.ebv = source.ebv_zhang_2023
            source.e_ebv = source.e_ebv_zhang_2023

        elif d is not None and (69 < d < 1_250):
            # Edenhofer et al. (2023)
            source.flag_ebv_from_edenhofer_2023 = True
            source.ebv = source.ebv_edenhofer_2023
            source.e_ebv = source.e_ebv_edenhofer_2023

        elif d is not None and d < 69:
            # Edenhofer et al. (2023) using inner integrated 69 pc
            source.flag_ebv_from_edenhofer_2023 = True
            source.flag_ebv_upper_limit = True
            source.ebv = source.ebv_edenhofer_2023
            source.e_ebv = source.e_ebv_edenhofer_2023

        elif np.abs(coord.galactic.b.value) > 30:
            # SFD
            source.flag_ebv_from_sfd = True
            source.ebv = source.ebv_sfd
            source.e_ebv = source.e_ebv_sfd

        elif source.h_mag is not None and source.mag4_5 is not None and source.mag4_5 < 99:
            # RJCE_GLIMPSE
            source.flag_ebv_from_rjce_glimpse = True
            source.ebv = source.ebv_rjce_glimpse
            source.e_ebv = source.e_ebv_rjce_glimpse

        elif source.h_mag is not None and source.w2_flux is not None and source.w2_flux > 0:
            # RJCE_ALLWISE
            source.flag_ebv_from_rjce_allwise = True
            source.ebv = source.ebv_rjce_allwise
            source.e_ebv = source.e_ebv_rjce_allwise

        else:
            # SFD Upper limit
            source.flag_ebv_from_sfd = True
            source.flag_ebv_upper_limit = True
            source.ebv = source.ebv_sfd
            source.e_ebv = source.e_ebv_sfd
    except:
        log.exception(f"Exception when computing reddening for source {source}")
        if raise_exceptions:
            raise
        return None
    else:
        return source


_cached_maps = None

def load_maps(queue=None):
    """Load dust maps, using cache if available."""
    global _cached_maps
    if _cached_maps is not None:
        return _cached_maps

    if queue is None:
        queue = ProgressContext()

    with silenced():
        sfd = SFDQuery()
        queue.put(dict(advance=1))
        edenhofer2023 = Edenhofer2023Query(load_samples=True, integrated=True)
        queue.put(dict(advance=1))
        bayestar2019 = BayestarQuery()
        queue.put(dict(advance=1))

    _cached_maps = (sfd, edenhofer2023, bayestar2019)
    return _cached_maps


def preload_dust_maps(queue=None):
    """
    Preload dust maps early in the migration pipeline.

    This allows the expensive I/O operation of loading dust maps to happen
    in parallel with other migration tasks (like photometry ingestion).
    The maps are cached globally so update_reddening() will get instant access.
    """
    if queue is None:
        queue = ProgressContext()

    # Check if there are sources that will need reddening computed
    count = (
        Source
        .select()
        .where(
            Source.ebv.is_null()
            & Source.ra.is_null(False)
            & Source.dec.is_null(False)
        )
        .count()
    )

    if count == 0:
        queue.put(dict(total=1, completed=1, description="No sources need reddening"))
        queue.put(Ellipsis)
        return 0

    queue.put(dict(total=3, completed=0, description="Preloading dust maps"))
    load_maps(queue)
    queue.put(Ellipsis)
    return count




def _compute_reddening_batch(pks, ra, dec, zgr_e, zgr_e_e, zgr_quality_flags,
                             h_mag, e_h_mag, mag4_5, d4_5m, w2_flux, w2_dflux,
                             r_med_geo, r_hi_geo, r_lo_geo,
                             sfd, edenhofer2023, bayestar2019, n_samples=20):
    """
    Compute reddening for a batch of sources using vectorized operations.
    Returns list of dicts with pk and computed values.
    """
    n = len(pks)
    if n == 0:
        return []

    # Pre-allocate result arrays
    ebv_zhang = np.full(n, np.nan)
    e_ebv_zhang = np.full(n, np.nan)
    ebv_rjce_glimpse = np.full(n, np.nan)
    e_ebv_rjce_glimpse = np.full(n, np.nan)
    ebv_rjce_allwise = np.full(n, np.nan)
    e_ebv_rjce_allwise = np.full(n, np.nan)
    ebv_sfd_arr = np.full(n, np.nan)
    e_ebv_sfd_arr = np.full(n, np.nan)
    ebv_bayestar = np.full(n, np.nan)
    e_ebv_bayestar = np.full(n, np.nan)
    ebv_edenhofer = np.full(n, np.nan)
    e_ebv_edenhofer = np.full(n, np.nan)
    ebv_final = np.full(n, np.nan)
    e_ebv_final = np.full(n, np.nan)
    ebv_flags = np.zeros(n, dtype=np.int32)

    # Flag bit positions (adjust based on actual model definition)
    FLAG_ZHANG = 1
    FLAG_EDENHOFER = 2
    FLAG_SFD = 4
    FLAG_RJCE_GLIMPSE = 8
    FLAG_RJCE_ALLWISE = 16
    FLAG_UPPER_LIMIT = 32

    # Convert to numpy arrays
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    zgr_e = np.asarray(zgr_e, dtype=np.float64)
    zgr_e_e = np.asarray(zgr_e_e, dtype=np.float64)
    zgr_quality_flags = np.asarray(zgr_quality_flags, dtype=np.float64)
    h_mag = np.asarray(h_mag, dtype=np.float64)
    e_h_mag = np.asarray(e_h_mag, dtype=np.float64)
    mag4_5 = np.asarray(mag4_5, dtype=np.float64)
    d4_5m = np.asarray(d4_5m, dtype=np.float64)
    w2_flux = np.asarray(w2_flux, dtype=np.float64)
    w2_dflux = np.asarray(w2_dflux, dtype=np.float64)
    r_med_geo = np.asarray(r_med_geo, dtype=np.float64)
    r_hi_geo = np.asarray(r_hi_geo, dtype=np.float64)
    r_lo_geo = np.asarray(r_lo_geo, dtype=np.float64)

    # Zhang et al. 2023 - fully vectorized
    ebv_zhang = 0.829 * zgr_e
    e_ebv_zhang = 0.829 * zgr_e_e

    # RJCE_GLIMPSE - fully vectorized
    ebv_ehw2 = 2.61
    ebv_rjce_glimpse = ebv_ehw2 * (h_mag - mag4_5 - 0.08)
    e_ebv_rjce_glimpse = ebv_ehw2 * np.sqrt(e_h_mag**2 + d4_5m**2)

    # RJCE_ALLWISE - fully vectorized
    with np.errstate(divide='ignore', invalid='ignore'):
        w2_mag_vega = -2.5 * np.log10(w2_flux) + 22.5 - 32e-3
        e_w2_mag_vega = (2.5 / np.log(10)) * w2_dflux / w2_flux
    ebv_rjce_allwise = ebv_ehw2 * (h_mag - w2_mag_vega - 0.08)
    e_ebv_rjce_allwise = ebv_ehw2 * np.sqrt(e_h_mag**2 + e_w2_mag_vega**2)

    # Create batched SkyCoord for SFD query
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    # SFD - batched query
    e_sfd = sfd(coord)
    ebv_sfd_arr = 0.884 * e_sfd
    e_ebv_sfd_arr = np.sqrt(0.01**2 + (0.1 * e_sfd)**2)

    # Compute galactic latitude for SFD flag logic
    gal_b = coord.galactic.b.value

    # Distance and uncertainty
    d = r_med_geo
    d_err = 0.5 * (r_hi_geo - r_lo_geo)

    # Generate distance samples for all sources at once (n x n_samples)
    # Use a fixed seed per batch for reproducibility
    rng = np.random.default_rng()
    d_samples_all = rng.normal(d[:, np.newaxis], np.abs(d_err[:, np.newaxis]), (n, n_samples))
    d_samples_all = np.clip(d_samples_all, 1, np.inf)

    # Edenhofer 2023 - process in sub-batches by distance category
    # Category 1: d < 69 pc - use integrated at 69 pc
    mask_near = np.isfinite(d) & (d < 69)
    if np.any(mask_near):
        idx_near = np.where(mask_near)[0]
        coord_near = SkyCoord(ra=ra[mask_near] * u.deg, dec=dec[mask_near] * u.deg, distance=69 * u.pc)
        ed_near = edenhofer2023(coord_near)
        ebv_edenhofer[idx_near] = 0.829 * ed_near
        e_ebv_edenhofer[idx_near] = 0.829 * ed_near  # uncertainty = value for near sources

    # Category 2: d >= 69 pc - use samples
    mask_far = np.isfinite(d) & (d >= 69)
    if np.any(mask_far):
        idx_far = np.where(mask_far)[0]
        # Process in smaller sub-batches to avoid memory issues
        sub_batch_size = 1
        for i in range(0, len(idx_far), sub_batch_size):
            sub_idx = idx_far[i:i+sub_batch_size]
            sub_d_samples = d_samples_all[sub_idx]  # shape: (sub_n, n_samples)
            sub_n = len(sub_idx)

            # Create flattened coordinates for batch query
            ra_rep = np.repeat(ra[sub_idx], n_samples)
            dec_rep = np.repeat(dec[sub_idx], n_samples)
            d_flat = sub_d_samples.flatten()

            coord_samples = SkyCoord(ra=ra_rep * u.deg, dec=dec_rep * u.deg, distance=d_flat * u.pc)
            ed_flat = edenhofer2023(coord_samples, mode="samples")

            # Reshape and compute median/std
            #ed_reshaped = ed_flat.reshape(sub_n, n_samples)
            ed_reshaped = ed_flat.flatten()
            with np.errstate(divide='ignore', invalid='ignore'):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ebv_edenhofer[sub_idx] = 0.829 * np.nanmedian(ed_reshaped)
                    e_ebv_edenhofer[sub_idx] = 0.829 * np.nanstd(ed_reshaped)

    # Bayestar 2019 - batched query with samples
    # Process all sources with valid distances
    mask_has_dist = np.isfinite(d)
    if np.any(mask_has_dist):
        idx_dist = np.where(mask_has_dist)[0]
        sub_batch_size = 1
        for i in range(0, len(idx_dist), sub_batch_size):
            sub_idx = idx_dist[i:i+sub_batch_size]
            sub_d_samples = d_samples_all[sub_idx]
            sub_n = len(sub_idx)

            ra_rep = np.repeat(ra[sub_idx], n_samples)
            dec_rep = np.repeat(dec[sub_idx], n_samples)
            d_flat = sub_d_samples.flatten()

            coord_samples = SkyCoord(ra=ra_rep * u.deg, dec=dec_rep * u.deg, distance=d_flat * u.pc)
            bs_flat = bayestar2019(coord_samples, mode="samples").flatten()

            #bs_reshaped = bs_flat.reshape(sub_n, n_samples)
            with np.errstate(divide='ignore', invalid='ignore'):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ebv_bayestar[sub_idx] = 0.88 * np.nanmedian(bs_flat)
                    e_ebv_bayestar[sub_idx] = 0.88 * np.nanstd(bs_flat)

    # Logic to decide preferred reddening value (vectorized)
    # Priority: Zhang -> Edenhofer (69-1250pc) -> Edenhofer (<69pc) -> SFD (|b|>30) -> RJCE_GLIMPSE -> RJCE_ALLWISE -> SFD upper limit

    # Initialize all as unset
    chosen = np.zeros(n, dtype=bool)

    # 1. Zhang et al. (2023)
    mask_zhang = ~chosen & np.isfinite(zgr_e) & (zgr_quality_flags < 8)
    ebv_final[mask_zhang] = ebv_zhang[mask_zhang]
    e_ebv_final[mask_zhang] = e_ebv_zhang[mask_zhang]
    ebv_flags[mask_zhang] |= FLAG_ZHANG
    chosen |= mask_zhang

    # 2. Edenhofer (69 < d < 1250)
    mask_eden_mid = ~chosen & np.isfinite(d) & (d > 69) & (d < 1250)
    ebv_final[mask_eden_mid] = ebv_edenhofer[mask_eden_mid]
    e_ebv_final[mask_eden_mid] = e_ebv_edenhofer[mask_eden_mid]
    ebv_flags[mask_eden_mid] |= FLAG_EDENHOFER
    chosen |= mask_eden_mid

    # 3. Edenhofer (d < 69) - upper limit
    mask_eden_near = ~chosen & np.isfinite(d) & (d < 69)
    ebv_final[mask_eden_near] = ebv_edenhofer[mask_eden_near]
    e_ebv_final[mask_eden_near] = e_ebv_edenhofer[mask_eden_near]
    ebv_flags[mask_eden_near] |= FLAG_EDENHOFER | FLAG_UPPER_LIMIT
    chosen |= mask_eden_near

    # 4. SFD (|b| > 30)
    mask_sfd_highb = ~chosen & (np.abs(gal_b) > 30)
    ebv_final[mask_sfd_highb] = ebv_sfd_arr[mask_sfd_highb]
    e_ebv_final[mask_sfd_highb] = e_ebv_sfd_arr[mask_sfd_highb]
    ebv_flags[mask_sfd_highb] |= FLAG_SFD
    chosen |= mask_sfd_highb

    # 5. RJCE_GLIMPSE
    mask_rjce_glimpse = ~chosen & np.isfinite(h_mag) & np.isfinite(mag4_5)
    ebv_final[mask_rjce_glimpse] = ebv_rjce_glimpse[mask_rjce_glimpse]
    e_ebv_final[mask_rjce_glimpse] = e_ebv_rjce_glimpse[mask_rjce_glimpse]
    ebv_flags[mask_rjce_glimpse] |= FLAG_RJCE_GLIMPSE
    chosen |= mask_rjce_glimpse

    # 6. RJCE_ALLWISE
    mask_rjce_allwise = ~chosen & np.isfinite(h_mag) & np.isfinite(w2_flux) & (w2_flux > 0)
    ebv_final[mask_rjce_allwise] = ebv_rjce_allwise[mask_rjce_allwise]
    e_ebv_final[mask_rjce_allwise] = e_ebv_rjce_allwise[mask_rjce_allwise]
    ebv_flags[mask_rjce_allwise] |= FLAG_RJCE_ALLWISE
    chosen |= mask_rjce_allwise

    # 7. SFD upper limit (fallback)
    mask_sfd_fallback = ~chosen
    ebv_final[mask_sfd_fallback] = ebv_sfd_arr[mask_sfd_fallback]
    e_ebv_final[mask_sfd_fallback] = e_ebv_sfd_arr[mask_sfd_fallback]
    ebv_flags[mask_sfd_fallback] |= FLAG_SFD | FLAG_UPPER_LIMIT

    # Build result list
    results = []
    for i in range(n):
        results.append({
            'pk': pks[i],
            'ebv_zhang_2023': None if ~np.isfinite(ebv_zhang[i]) else float(ebv_zhang[i]),
            'e_ebv_zhang_2023': None if ~np.isfinite(e_ebv_zhang[i]) else float(e_ebv_zhang[i]),
            'ebv_rjce_glimpse': None if ~np.isfinite(ebv_rjce_glimpse[i]) else float(ebv_rjce_glimpse[i]),
            'e_ebv_rjce_glimpse': None if ~np.isfinite(e_ebv_rjce_glimpse[i]) else float(e_ebv_rjce_glimpse[i]),
            'ebv_rjce_allwise': None if ~np.isfinite(ebv_rjce_allwise[i]) else float(ebv_rjce_allwise[i]),
            'e_ebv_rjce_allwise': None if ~np.isfinite(e_ebv_rjce_allwise[i]) else float(e_ebv_rjce_allwise[i]),
            'ebv_sfd': None if ~np.isfinite(ebv_sfd_arr[i]) else float(ebv_sfd_arr[i]),
            'e_ebv_sfd': None if ~np.isfinite(e_ebv_sfd_arr[i]) else float(e_ebv_sfd_arr[i]),
            'ebv_bayestar_2019': None if ~np.isfinite(ebv_bayestar[i]) else float(ebv_bayestar[i]),
            'e_ebv_bayestar_2019': None if ~np.isfinite(e_ebv_bayestar[i]) else float(e_ebv_bayestar[i]),
            'ebv_edenhofer_2023': None if ~np.isfinite(ebv_edenhofer[i]) else float(ebv_edenhofer[i]),
            'e_ebv_edenhofer_2023': None if ~np.isfinite(e_ebv_edenhofer[i]) else float(e_ebv_edenhofer[i]),
            'ebv': None if ~np.isfinite(ebv_final[i]) else float(ebv_final[i]),
            'e_ebv': None if ~np.isfinite(e_ebv_final[i]) else float(e_ebv_final[i]),
            'ebv_flags': int(ebv_flags[i]),
        })

    return results


def _update_reddening_worker(args):
    """
    Worker function for parallel reddening computation.
    Must be at module level for multiprocessing.

    Each worker:
    1. Loads its own dust maps (cannot be pickled across processes)
    2. Computes reddening for the batch
    3. Writes results directly to the database
    4. Returns the count of updated rows
    """
    chunk_data, table = args

    from astra.models.base import database

    # Close any inherited connection from parent process and open a fresh one
    if not database.is_closed():
        database.close()
    database.connect()

    try:
        # Load dust maps in this worker process
        with silenced():
            sfd = SFDQuery()
            edenhofer2023 = Edenhofer2023Query(load_samples=True, integrated=True)
            bayestar2019 = BayestarQuery()

        # Unpack the chunk data
        (pks, ra, dec, zgr_e, zgr_e_e, zgr_quality_flags,
         h_mag, e_h_mag, mag4_5, d4_5m, w2_flux, w2_dflux,
         r_med_geo, r_hi_geo, r_lo_geo) = chunk_data

        # Filter valid coordinates
        valid_mask = [
            (r is not None and d is not None and np.isfinite(r) and np.isfinite(d))
            for r, d in zip(ra, dec)
        ]

        valid_pks = [pks[i] for i in range(len(pks)) if valid_mask[i]]
        if not valid_pks:
            return 0

        valid_ra = [ra[i] for i in range(len(ra)) if valid_mask[i]]
        valid_dec = [dec[i] for i in range(len(dec)) if valid_mask[i]]
        valid_zgr_e = [zgr_e[i] for i in range(len(zgr_e)) if valid_mask[i]]
        valid_zgr_e_e = [zgr_e_e[i] for i in range(len(zgr_e_e)) if valid_mask[i]]
        valid_zgr_qf = [zgr_quality_flags[i] for i in range(len(zgr_quality_flags)) if valid_mask[i]]
        valid_h_mag = [h_mag[i] for i in range(len(h_mag)) if valid_mask[i]]
        valid_e_h_mag = [e_h_mag[i] for i in range(len(e_h_mag)) if valid_mask[i]]
        valid_mag4_5 = [mag4_5[i] for i in range(len(mag4_5)) if valid_mask[i]]
        valid_d4_5m = [d4_5m[i] for i in range(len(d4_5m)) if valid_mask[i]]
        valid_w2_flux = [w2_flux[i] for i in range(len(w2_flux)) if valid_mask[i]]
        valid_w2_dflux = [w2_dflux[i] for i in range(len(w2_dflux)) if valid_mask[i]]
        valid_r_med = [r_med_geo[i] for i in range(len(r_med_geo)) if valid_mask[i]]
        valid_r_hi = [r_hi_geo[i] for i in range(len(r_hi_geo)) if valid_mask[i]]
        valid_r_lo = [r_lo_geo[i] for i in range(len(r_lo_geo)) if valid_mask[i]]

        results = _compute_reddening_batch(
            valid_pks, valid_ra, valid_dec,
            valid_zgr_e, valid_zgr_e_e, valid_zgr_qf,
            valid_h_mag, valid_e_h_mag, valid_mag4_5, valid_d4_5m,
            valid_w2_flux, valid_w2_dflux,
            valid_r_med, valid_r_hi, valid_r_lo,
            sfd, edenhofer2023, bayestar2019
        )

        if not results:
            return 0

        # Bulk update using single SQL with VALUES clause
        columns = [
            'ebv_zhang_2023', 'e_ebv_zhang_2023',
            'ebv_rjce_glimpse', 'e_ebv_rjce_glimpse',
            'ebv_rjce_allwise', 'e_ebv_rjce_allwise',
            'ebv_sfd', 'e_ebv_sfd',
            'ebv_bayestar_2019', 'e_ebv_bayestar_2019',
            'ebv_edenhofer_2023', 'e_ebv_edenhofer_2023',
            'ebv', 'e_ebv', 'ebv_flags'
        ]

        def fmt_val(v, is_int=False):
            if v is None:
                return 'NULL::double precision' if not is_int else 'NULL::integer'
            elif isinstance(v, int) or is_int:
                return str(int(v))
            else:
                return str(float(v))

        values_rows = []
        for row in results:
            vals = [str(row['pk'])]
            for col in columns:
                is_int = (col == 'ebv_flags')
                vals.append(fmt_val(row[col], is_int=is_int))
            values_rows.append(f"({', '.join(vals)})")

        values_sql = ', '.join(values_rows)
        set_clauses = ', '.join([f"{col} = v.{col}" for col in columns])
        col_list = ', '.join(['pk'] + columns)

        sql = f"""
            UPDATE {table}
            SET {set_clauses}, modified = NOW()
            FROM (VALUES {values_sql}) AS v({col_list})
            WHERE {table}.pk = v.pk
        """

        with database.atomic():
            database.execute_sql(sql)

        return len(results)

    except Exception as e:
        log.exception(f"Worker error: {e}")
        return 0

    finally:
        # Close connection when done to avoid connection leaks
        if not database.is_closed():
            database.close()


def update_reddening(
    where=(
        Source.ebv.is_null()
    &   Source.ra.is_null(False)
    &   Source.dec.is_null(False)
    ),
    batch_size=5000,
    max_workers=6,
    queue=None
):
    """
    Update reddening estimates for sources using parallel batch processing.

    :param where:
        Peewee expression to filter sources.
    :param batch_size:
        Number of sources per batch (each batch is processed by one worker).
    :param max_workers:
        Maximum number of parallel worker processes.
    :param queue:
        Progress queue for status updates.
    """
    from astra.models.base import database

    if queue is None:
        queue = ProgressContext()

    # check if dustmaps configured
    if not _dustmaps_are_configured():
        queue.put(dict(total=None, description="Dust maps not found — running setup"))
        setup_dustmaps()

    # Select only needed columns for efficiency
    q = (
        Source
        .select(
            Source.pk,
            Source.ra,
            Source.dec,
            Source.zgr_e,
            Source.zgr_e_e,
            Source.zgr_quality_flags,
            Source.h_mag,
            Source.e_h_mag,
            Source.mag4_5,
            Source.d4_5m,
            Source.w2_flux,
            Source.w2_dflux,
            Source.r_med_geo,
            Source.r_hi_geo,
            Source.r_lo_geo,
        )
    )
    if where:
        q = q.where(where)

    # Check if there are any sources to process
    if not q.exists():
        queue.put(Ellipsis)
        return None

    table = f"{Source._meta.schema}.{Source._meta.table_name}"
    total = q.count()

    # Prepare batches
    queue.put(dict(total=None, description="Preparing batches for reddening"))
    batches = []
    for chunk in chunked(q.tuples(), batch_size):
        chunk_list = list(chunk)
        if not chunk_list:
            continue

        # Pack data for worker
        chunk_data = (
            [row[0] for row in chunk_list],   # pks
            [row[1] for row in chunk_list],   # ra
            [row[2] for row in chunk_list],   # dec
            [row[3] for row in chunk_list],   # zgr_e
            [row[4] for row in chunk_list],   # zgr_e_e
            [row[5] if row[5] is not None else 999 for row in chunk_list],  # zgr_quality_flags
            [row[6] for row in chunk_list],   # h_mag
            [row[7] for row in chunk_list],   # e_h_mag
            [row[8] for row in chunk_list],   # mag4_5
            [row[9] for row in chunk_list],   # d4_5m
            [row[10] for row in chunk_list],  # w2_flux
            [row[11] for row in chunk_list],  # w2_dflux
            [row[12] for row in chunk_list],  # r_med_geo
            [row[13] for row in chunk_list],  # r_hi_geo
            [row[14] for row in chunk_list],  # r_lo_geo
        )
        batches.append((chunk_data, table))

    if not batches:
        queue.put(Ellipsis)
        return 0

    # Close parent connection before forking workers
    if not database.is_closed():
        database.close()

    # Process batches in parallel
    total_updated = 0
    with queue.subtask("Computing extinction", total=total) as compute_step:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_update_reddening_worker, batch): batch for batch in batches}

            for future in concurrent.futures.as_completed(futures):
                try:
                    n_updated = future.result()
                    total_updated += n_updated
                except Exception as e:
                    log.exception(f"Batch failed: {e}")
                    raise

                # Update progress (estimate based on batch size)
                compute_step.update(advance=batch_size)

    # Reconnect in parent process
    if database.is_closed():
        database.connect()

    queue.put(Ellipsis)
    return total_updated


def _dustmaps_are_configured(data_dir="$MWM_ASTRA/aux/dust-maps"):
    """Return True if the required dust map files are present."""
    data_dir = expand_path(data_dir)
    req_files = [
        os.path.join(data_dir, "sfd", "SFD_dust_4096_ngp.fits"),
        os.path.join(data_dir, "sfd", "SFD_dust_4096_sgp.fits"),
        os.path.join(data_dir, "bayestar", "bayestar2019.h5"),
        os.path.join(data_dir, "edenhofer_2023", "mean_and_std_healpix.fits"),
        os.path.join(data_dir, "edenhofer_2023", "samples_healpix.fits"),
    ]
    return all(os.path.exists(f) for f in req_files)


def setup_dustmaps(data_dir="$MWM_ASTRA/aux/dust-maps"):
    """
    Set up dustmaps package.
    """

    from dustmaps.config import config
    from dustmaps import sfd, bayestar, edenhofer2023

    with silenced():
        config.reset()

        config["data_dir"] = expand_path(data_dir)

        os.makedirs(expand_path(data_dir), exist_ok=True)

        sfd.fetch()

        bayestar.fetch()

        edenhofer2023.fetch(fetch_samples=True)
