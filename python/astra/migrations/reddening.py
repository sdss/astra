import numpy as np
from astra.utils import log, expand_path
from astra.models.source import Source
from peewee import chunked
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm

# dust-maps data directory: $MWM_ASTRA/aux/dust-maps/
from dustmaps.sfd import SFDQuery
from dustmaps.edenhofer2023 import Edenhofer2023Query

sfd = SFDQuery()
edenhofer2023 = Edenhofer2023Query()


def update_reddening(where=Source.ebv.is_null(), batch_size=1000):
    """
    Update reddening estimates for sources.
    """
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
        Source.ebv_edenhofer2023,
        Source.e_ebv_edenhofer2023,
        Source.flag_ebv_edenhofer_2023_upper_limit,
        Source.ebv,
        Source.e_ebv,
        Source.flag_ebv_upper_limit,
        Source.ebv_method_flags,
    ]

    updated = []
    for source in tqdm(q, total=1, desc="Computing reddening"):
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
    
    return n_updated


def update_reddening_on_source(source):
    """
    Compute reddening and reddening uncertainties for a source using various methods.
    """

    coord = SkyCoord(ra=source.ra * u.deg, dec=source.dec * u.deg)

    # Zhang et al. 2023
    source.ebv_zhang_2023 = 0.829 * source.zgr_e
    source.e_ebv_zhang_2023 = 0.829 * source.zgr_e_e

    # RJCE_GLIMPSE
    ebv_ehw2 = 2.61
    source.ebv_rjce_glimpse = ebv_ehw2 * (source.h_mag - source.mag4_5 - 0.08)
    source.e_ebv_rjce_glimpse = ebv_ehw2 * np.sqrt(source.e_h_mag**2 + source.e_mag4_5**2)

    # RJCE_ALLWISE
    # We store unWISE (not ALLWISE) and we have only w2 fluxes, not w2 magnitudes.
    # See https://catalog.unwise.me/catalogs.html (Flux Scale) for justification of 32 mmag offset
    w2_mag_vega = -2.5 * np.log10(source.w2_flux) + 22.5 - 32 * 1e-3 # Vega
    e_w2_mag_vega = (2.5 / np.log(10)) * source.w2_dflux / source.w2_flux
    source.ebv_rjce_allwise = ebv_ehw2 * (source.h_mag - w2_mag_vega - 0.08)
    source.e_ebv_rjce_allwise = ebv_ehw2 * np.sqrt(source.e_h_mag**2 + e_w2_mag_vega**2)

    # SFD
    e_sfd = sfd(coord)
    source.ebv_sfd = 0.884 * e_sfd
    source.e_ebv_sfd = np.sqrt(0.01**2 + (0.1 * e_sfd)**2)


    d = source.r_med_geo # [pc]

    # Logic to decide preferred reddening value

    if source.zgr_e is not None: # target is in Zhang
        # Zhang et al. (2023)
        source.flag_ebv_from_zhang_2023 = True
        source.ebv = source.ebv_zhang_2023
        source.e_ebv = source.e_ebv_zhang_2023

    elif d is not None and (69 < d < 1_250):
        # Edenhofer et al. (2023)
        source.flag_ebv_from_edenhofer_2023 = True

        n_draws = 20

        mean, sigma = (source.r_med_geo, 0.5 * (source.r_hi_geo - source.r_lo_geo))
        
        #for draw in np.random.normal(mean, sigma, n_draws):
        #edenhofer2023(coord,)
        raise a



        raise NotImplementedError("awaiting advice on how to sample distance PDFs")

    elif d is not None and d < 69:
        # Edenhofer et al. (2023) using inner integrated 69 pc
        source.flag_ebv_from_edenhofer_2023 = True

        source.flag_ebv_upper_limit = True
        raise NotImplementedError("awaiting dust map download")
    
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

    elif source.h_mag is not None and source.w2_flux is not None: #TODO: unWISE w2 or ALLWISE?
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
        


def setup_dustmaps():
    """
    Set up dustmaps package.
    """

    from dustmaps.config import config
    from dustmaps import sfd, bayestar, edenhofer2023

    config.reset()

    config["data_dir"] = expand_path("$MWM_ASTRA/aux/dust-maps")

    log.info(f"Set dust map directory to {config['data_dir']}")

    log.info(f"Downloading SFD")
    sfd.fetch()

    log.info(f"Downloading Bayestar")
    bayestar.fetch()

    log.info(f"Downloading Edenhofer et al. (2023)")
    edenhofer2023.fetch(fetch_samples=True)
