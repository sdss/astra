import os
import numpy as np
from peewee import chunked, fn, JOIN
from typing import Optional
from astra.models.source import Source, SDSSCatalog
from astra.utils import flatten, log
from tqdm import tqdm


def migrate_tic_v8_identifier(batch_size: Optional[int] = 500, limit: Optional[int] = None):
    from astra.migrations.sdss5db.catalogdb import CatalogToTIC_v8

    log.info(f"Migrating TIC v8 identifiers")

    q = (
        Source
        .select(
            Source.id,
            Source.sdss5_catalogid_v1
        )
        .where(
                Source.tic_v8_id.is_null()
            &   Source.sdss5_catalogid_v1.is_null(False)
        )
        .order_by(
            Source.sdss5_catalogid_v1.asc()
        )
        .limit(limit)
        .iterator()
    )    

    updated = 0
    with tqdm(total=limit) as pb:
        for batch in chunked(q, batch_size):
            q_tic_v8 = (
                CatalogToTIC_v8
                .select(
                    CatalogToTIC_v8.catalogid.alias("catalogid"),
                    CatalogToTIC_v8.target.alias("tic_v8_id"),
                )
                .where(
                    CatalogToTIC_v8.catalogid.in_([r.sdss5_catalogid_v1 for r in batch])
                )
                .order_by(CatalogToTIC_v8.catalogid.asc())
                .tuples()
                .iterator()
            )
            
            sources = { s.sdss5_catalogid_v1: s for s in batch}
            update = []
            for sdss5_catalogid_v1, tic_v8_id in q_tic_v8:
                source = sources[sdss5_catalogid_v1]
                source.tic_v8_id = tic_v8_id
                update.append(source)

            if update:             
                updated += (
                    Source
                    .bulk_update(
                        update,
                        fields=[Source.tic_v8_id]
                    )
                )

            pb.update(batch_size)

    log.info(f"Updated {updated} records")
    return updated    


def migrate_twomass_photometry(
    where=(
        (
            Source.j_mag.is_null()
        |   Source.h_mag.is_null()
        |   Source.k_mag.is_null()
        )
        &   Source.sdss5_catalogid_v1.is_null(False)
    ),
    limit: Optional[int] = None,
    batch_size: Optional[int] = 500, 
):

    from astra.migrations.sdss5db.catalogdb import TwoMassPSC, CatalogToTwoMassPSC

    log.info(f"Migrating 2MASS photometry")
    q = (
        Source
        .select(Source.sdss5_catalogid_v1)
        .distinct()
        .where(where)
        .order_by(
            Source.sdss5_catalogid_v1.asc()
        )
        .tuples()
        .limit(limit)
    )    

    limit = limit or q.count()

    twomass_data = {}

    with tqdm(total=limit, desc="Retrieving photometry") as pb:
        for batch in chunked(q, batch_size):
            q_twomass = (
                TwoMassPSC
                .select(
                    CatalogToTwoMassPSC.catalogid.alias("catalogid"),
                    TwoMassPSC.j_m.alias("j_mag"),
                    TwoMassPSC.j_cmsig.alias("e_j_mag"),
                    TwoMassPSC.h_m.alias("h_mag"),
                    TwoMassPSC.h_cmsig.alias("e_h_mag"),
                    TwoMassPSC.k_m.alias("k_mag"),
                    TwoMassPSC.k_cmsig.alias("e_k_mag"),
                    TwoMassPSC.ph_qual,
                    TwoMassPSC.bl_flg,
                    TwoMassPSC.cc_flg,
                )
                .join(CatalogToTwoMassPSC)
                .where(
                    CatalogToTwoMassPSC.catalogid.in_(flatten(batch))
                )
                .order_by(CatalogToTwoMassPSC.catalogid.asc())
                .dicts()
            )
            for r in q_twomass:
                twomass_data[r.pop("catalogid")] = r
            pb.update(min(batch_size, len(batch)))

    q = (
        Source
        .select(
            Source.id,
            Source.sdss5_catalogid_v1
        )
        .where(where)
        .order_by(
            Source.sdss5_catalogid_v1.asc()
        )
        .limit(limit)
    )    

    updated_sources = []
    for source in q:
        try:
            d = twomass_data[source.sdss5_catalogid_v1]
        except KeyError:
            continue

        for key, value in d.items():
            setattr(source, key, value or np.nan)
        updated_sources.append(source)

    updated = 0
    with tqdm(total=len(updated_sources), desc="Updating sources") as pb:
        for chunk in chunked(updated_sources, batch_size):
            updated += (
                Source
                .bulk_update(
                    chunk,
                    fields=[
                        Source.j_mag,
                        Source.e_j_mag,
                        Source.h_mag,
                        Source.e_h_mag,
                        Source.k_mag,
                        Source.e_k_mag,
                        Source.ph_qual,
                        Source.bl_flg,
                        Source.cc_flg,
                    ]
                )
            )
            pb.update(min(batch_size, len(chunk)))

    log.info(f"Updated {updated} records")
    return updated



def migrate_unwise_photometry(batch_size: Optional[int] = 500, limit: Optional[int] = None):

    from astra.migrations.sdss5db.catalogdb import unWISE, CatalogTounWISE

    log.info(f"Migrating UNWISE photometry")
    q = (
        Source
        .select(
            Source.id,
            Source.sdss5_catalogid_v1
        )
        .where(
            (
                Source.w1_flux.is_null()
            |   Source.w2_flux.is_null()
            )
            &   Source.sdss5_catalogid_v1.is_null(False)
        )
        .order_by(
            Source.sdss5_catalogid_v1.asc()
        )
        .limit(limit)
        .iterator()
    )    

    updated = 0
    with tqdm(total=limit) as pb:
        for batch in chunked(q, batch_size):
            q_phot = (
                unWISE
                .select(
                    CatalogTounWISE.catalogid.alias("catalogid"),
                    unWISE.flux_w1.alias("w1_flux"),
                    unWISE.dflux_w1.alias("w1_dflux"),
                    unWISE.flux_w2.alias("w2_flux"),
                    unWISE.dflux_w2.alias("w2_dflux"),
                    unWISE.fracflux_w1.alias("w1_frac"),
                    unWISE.fracflux_w2.alias("w2_frac"),
                    unWISE.flags_unwise_w1.alias("w1uflags"),
                    unWISE.flags_unwise_w2.alias("w2uflags"),
                    unWISE.flags_info_w1.alias("w1aflags"),
                    unWISE.flags_info_w2.alias("w2aflags")
                )
                .join(CatalogTounWISE)
                .where(
                    CatalogTounWISE.catalogid.in_([r.sdss5_catalogid_v1 for r in batch])
                )
                .order_by(CatalogTounWISE.catalogid.asc())
                .dicts()
                .iterator()
            )

            update = []
            sources = { s.sdss5_catalogid_v1: s for s in batch }
            for r in q_phot:
                source = sources[r["catalogid"]]
                for key, value in r.items():
                    setattr(source, key, value)
                update.append(source)
                
            updated += (
                Source
                .bulk_update(
                    update,
                    fields=[
                        Source.w1_flux,
                        Source.w1_dflux,
                        Source.w2_flux,
                        Source.w2_dflux,
                        Source.w1_frac,
                        Source.w2_frac,
                        Source.w1uflags,
                        Source.w2uflags,
                        Source.w1aflags,
                        Source.w2aflags
                    ]
                )
            )

            pb.update(batch_size)

    log.info(f"Updated {updated} records")
    return updated




def migrate_gaia_dr3_astrometry_and_photometry(limit: Optional[int] = None, batch_size: Optional[int] = 500):
    """
    Migrate Gaia DR3 astrometry and photometry from the SDSS-V database for any sources (`astra.models.Source`)
    that have a Gaia DR3 source identifier (`astra.models.Source.gaia_dr3_source_id`) but are missing Gaia
    photometry.

    :param batch_size: [optional]
        The batch size to use for updates.
    
    :param limit: [optional]
        Limit the update to `limit` records. Useful for testing.
    """

    from astra.migrations.sdss5db.catalogdb import Gaia_DR3

    log.info(f"Updating Gaia astrometry and photometry")

    # Retrieve sources which have gaia identifiers but not astrometry
    q = (
        Source
        .select(Source.gaia_dr3_source_id)
        .distinct()
        .where(
            (
                Source.g_mag.is_null()
            |   Source.bp_mag.is_null()
            |   Source.rp_mag.is_null()
            )
            &   Source.gaia_dr3_source_id.is_null(False)
        )
        .order_by(
            Source.gaia_dr3_source_id.asc()
        )
        .limit(limit)
        .tuples()
    )

    total = limit or q.count()

    gaia_data = {}

    with tqdm(total=total) as pb:
        for batch in chunked(q.iterator(), batch_size):
            q_gaia = (
                Gaia_DR3
                .select(
                    Gaia_DR3.source_id.alias("gaia_dr3_source_id"),
                    Gaia_DR3.phot_g_mean_mag.alias("g_mag"),
                    Gaia_DR3.phot_bp_mean_mag.alias("bp_mag"),
                    Gaia_DR3.phot_rp_mean_mag.alias("rp_mag"),
                    Gaia_DR3.parallax.alias("plx"),
                    Gaia_DR3.parallax_error.alias("e_plx"),
                    Gaia_DR3.pmra,
                    Gaia_DR3.pmra_error.alias("e_pmra"),
                    Gaia_DR3.pmdec.alias("pmde"),
                    Gaia_DR3.pmdec_error.alias("e_pmde"),
                    Gaia_DR3.radial_velocity.alias("gaia_v_rad"),
                    Gaia_DR3.radial_velocity_error.alias("gaia_e_v_rad"),
                )
                .where(
                    Gaia_DR3.source_id.in_(flatten(batch))
                )
                .dicts()
                .iterator()
            )

            for source in q_gaia:
                gaia_data[source["gaia_dr3_source_id"]] = source
            pb.update(min(batch_size, len(batch)))
    
    q = (
        Source
        .select(
            Source.id,
            Source.gaia_dr3_source_id
        )
        .where(
            (
                Source.g_mag.is_null()
            |   Source.bp_mag.is_null()
            |   Source.rp_mag.is_null()
            )
            &   Source.gaia_dr3_source_id.is_null(False)
        )        
    )

    updated_sources = []
    for source in q:
        for k, v in gaia_data[source.gaia_dr3_source_id].items():
            setattr(source, k, v or np.nan)
        updated_sources.append(source)

    updated = 0
    for chunk in chunked(updated_sources, batch_size):
        updated += (
            Source
            .bulk_update(
                chunk,
                fields=[
                    Source.g_mag,
                    Source.bp_mag,
                    Source.rp_mag,
                    Source.plx,
                    Source.e_plx,
                    Source.pmra,
                    Source.e_pmra,
                    Source.pmde,
                    Source.e_pmde,
                    Source.gaia_v_rad,
                    Source.gaia_e_v_rad
                ]        
            )
        )

    log.info(f"Updated {updated} records ({len(gaia_data)} gaia sources)")
    return updated
    '''
                source = sources[]
                for key, value in gaia_source.items():
                    setattr(source, key, value or np.nan)
                update.append(source)
            
            if update:
                updated += (
                    Source
                    .bulk_update(
                        update,
                        fields=[
                            Source.g_mag,
                            Source.bp_mag,
                            Source.rp_mag,
                            Source.plx,
                            Source.e_plx,
                            Source.pmra,
                            Source.e_pmra,
                            Source.pmde,
                            Source.e_pmde,
                            Source.gaia_v_rad,
                            Source.gaia_e_v_rad
                        ]
                    )
                )

            pb.update(batch_size)

    log.info(f"Updated {updated} records")
    return updated
    '''

