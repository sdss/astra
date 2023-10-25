import numpy as np
import os
from astra.utils import log, expand_path
from astra.models.source import Source
from peewee import chunked
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm
from functools import cache


# dust-maps data directory: $MWM_ASTRA/aux/dust-maps/
from dustmaps.sfd import SFDQuery
from dustmaps.edenhofer2023 import Edenhofer2023Query
from dustmaps.bayestar import BayestarQuery

# TODO: This python file is a lot of spaghetti code. sorry about that. refactor this!

von = lambda v: v or np.nan
    
def _fix_w2_flux():
    (
        Source
        .update(w2_flux=None, w2_dflux=None)
        .where(Source.w2_flux == 0)
        .execute()
    )
    
def _fix_rjce_glimpse(sfd, edenhofer2023, bayestar2019):
    sources = list(
        Source
        .select()
        .where(Source.mag4_5 > 99)
    )
    for s in sources:
        s.mag4_5 = None
        s.d4_5m = None
        s.rms_f4_5 = None
        
        s = _update_reddening_on_source(s, sfd, edenhofer2023, bayestar2019)
        s.save()    
            
    
def _update_reddening_on_source(source, sfd, edenhofer2023, bayestar2019, raise_exceptions=False):
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
            
        elif source.h_mag is not None and source.mag4_5 is not None:
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


@cache
def load_maps():
    sfd = SFDQuery()
    edenhofer2023 = Edenhofer2023Query(load_samples=True, integrated=True)
    bayestar2019 = BayestarQuery()
    return (sfd, edenhofer2023, bayestar2019)    

def _reddening_worker(sources):

    maps = load_maps()

    updated = []
    for source in sources:
        s = _update_reddening_on_source(source, *maps)
        if s is not None:
            updated.append(s)
    return updated
        

def update_reddening(where=Source.ebv.is_null(), batch_size=1000, max_workers: int = 16):
    """
    Update reddening estimates for sources.
    """
    
    maps = load_maps()
    
    _fix_w2_flux()
    # Fix any problem children first:
    # TODO: mag4_5 uses 99.999 as a BAD value. set to NaNs.
    _fix_rjce_glimpse(*maps)
    
    
    q = (
        Source
        .select()
    )
    if where:
        q = q.where(where)

    fields = [
        Source.ebv_zhang_2023,
        Source.e_ebv_zhang_2023,
        Source.ebv_rjce_glimpse,
        Source.e_ebv_rjce_glimpse,
        Source.ebv_rjce_allwise,
        Source.e_ebv_rjce_allwise,
        Source.ebv_sfd,
        Source.e_ebv_sfd,
        Source.ebv_bayestar_2019,
        Source.e_ebv_bayestar_2019,
        Source.ebv_edenhofer_2023,
        Source.e_ebv_edenhofer_2023,
        Source.ebv,
        Source.e_ebv,
        Source.ebv_flags,
    ]
    

    with tqdm(total=len(q)) as pb:
            
        for chunk in chunked(q, batch_size):
            updated = []
            for source in chunk:
                s = _update_reddening_on_source(source, *maps)
                if s is not None:
                    updated.append(s)    
            
            if len(updated) > 0:
                Source.bulk_update(updated, fields)
            
            pb.update(batch_size)
            
    

    """
    executor = concurrent.futures.ProcessPoolExecutor(max_workers)
    
    '''
    updated = []
    for source in tqdm(q, desc="Computing reddening"):
        update_reddening_on_source(source)
        updated.append(source)

    n_updated = 0
    with tqdm(total=len(updated), desc="Updating") as pb:
        for chunk in chunked(updated, batch_size):
            n_updated += (
                Source
                .bulk_update(
                    chunk,
                    fields=fields
                )
            )
            pb.update(batch_size)
    '''
    
    futures, chunk_size = ([], int(np.ceil(len(q) / max_workers)))
    for chunk in tqdm(chunked(q, chunk_size), total=max_workers, desc="Chunking"):
        futures.append(executor.submit(_reddening_worker, chunk))
    
    updated = []
    for future in tqdm(concurrent.futures.as_completed(futures), desc="Computing reddening"):
        r = future.result()
        if r is not None:
            updated.append(r)

    n_updated = 0
    with tqdm(total=len(updated), desc="Updating") as pb:
        for chunk in chunked(updated, batch_size):
            n_updated += (
                Source
                .bulk_update(
                    chunk,
                    fields=fields
                )
            )
            pb.update(batch_size)        
        
    return n_updated
    """


def setup_dustmaps(data_dir="$MWM_ASTRA/aux/dust-maps"):
    """
    Set up dustmaps package.
    """

    from dustmaps.config import config
    from dustmaps import sfd, bayestar, edenhofer2023

    config.reset()

    config["data_dir"] = expand_path(data_dir)

    os.makedirs(expand_path(data_dir), exist_ok=True)

    log.info(f"Set dust map directory to {config['data_dir']}")

    log.info(f"Downloading SFD")
    sfd.fetch()

    log.info(f"Downloading Bayestar")
    bayestar.fetch()

    log.info(f"Downloading Edenhofer et al. (2023)")
    edenhofer2023.fetch(fetch_samples=True)
