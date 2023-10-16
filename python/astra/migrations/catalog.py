
from typing import Optional
from tqdm import tqdm
from peewee import chunked, IntegerField,  fn, JOIN
from astra.models.source import Source
from astra.migrations.sdss5db.utils import get_approximate_rows
from astra.utils import log, flatten
import numpy as np



def migrate_healpix(
    where=(
            Source.healpix.is_null()
        &   Source.ra.is_null(False)
        &   Source.dec.is_null(False)
    ),
    batch_size: Optional[int] = 500,
    limit: Optional[int] = None,
    nside: Optional[int] = 128,
    lonlat: Optional[bool] = True
):
    """
    Migrate HEALPix values for any sources that have positions, but no HEALPix assignment.

    :param batch_size: [optional]
        The batch size to use when upserting data.    
    
    :param limit: [optional]
        Limit the initial catalog queries for testing purposes.
    
    :param nside: [optional]
        The number of sides to use for the HEALPix map.
    
    :param lonlat: [optional]
        The HEALPix map is oriented in longitude and latitude coordinates.
    """
    from healpy import ang2pix
    
    log.info(f"Migrating healpix")
    q = (
        Source
        .select(
            Source.pk,
            Source.ra,
            Source.dec,
        )
        .where(where)
        .limit(limit)
    )    
    
    updated, total = (0, limit or q.count())
    with tqdm(total=total) as pb:
        for batch in chunked(q.iterator(), batch_size):
            for record in batch:
                record.healpix = ang2pix(nside, record.ra, record.dec, lonlat=lonlat)
            updated += (
                Source
                .bulk_update(
                    batch,
                    fields=[Source.healpix],
                )
            )
            pb.update(len(batch))

    log.info(f"Updated {updated} records")
    return updated


def migrate_bailer_jones_distances(
    where=(Source.r_med_geo.is_null() & Source.gaia_dr3_source_id.is_null(False) & (Source.gaia_dr3_source_id > 0)), 
    batch_size=500, 
    limit=None
):

    from astra.migrations.sdss5db.catalogdb import BailerJonesEDR3

    q = (
        Source
        .select()
    )
    if where:
        q = q.where(where)
        
    q = (
        q
        .order_by(Source.gaia_dr3_source_id.asc())
        .limit(limit)
    )

    n_updated = 0
    with tqdm(total=1, desc="Upserting") as pb:
        for chunk in chunked(q.iterator(), batch_size):

            q_bj = (
                BailerJonesEDR3
                .select(
                    BailerJonesEDR3.source_id,
                    BailerJonesEDR3.r_med_geo,
                    BailerJonesEDR3.r_lo_geo,
                    BailerJonesEDR3.r_hi_geo,
                    BailerJonesEDR3.r_med_photogeo,
                    BailerJonesEDR3.r_lo_photogeo,
                    BailerJonesEDR3.r_hi_photogeo,
                    BailerJonesEDR3.flag.alias("bailer_jones_flags"),
                )
                .where(BailerJonesEDR3.source_id.in_([s.gaia_dr3_source_id for s in chunk]))
                .dicts()
            )

            sources = { s.gaia_dr3_source_id: s for s in chunk }
            update = []
            for record in q_bj:
                source_id = record.pop("source_id")
                source = sources[source_id]

                for key, value in record.items():
                    setattr(source, key, value)
                
                update.append(source)
            
            n_updated += (
                Source
                .bulk_update(
                    update,
                    fields=[
                        Source.r_med_geo,
                        Source.r_lo_geo,
                        Source.r_hi_geo,
                        Source.r_med_photogeo,
                        Source.r_lo_photogeo,
                        Source.r_hi_photogeo,
                        Source.bailer_jones_flags,
                    ]
                )
            )

            pb.update(batch_size)
    
    return n_updated


def migrate_gaia_synthetic_photometry(
    where=(Source.gaia_dr3_source_id.is_null(False)), 
    batch_size=500, 
    limit=None
):
    from astra.migrations.sdss5db.catalogdb import Gaia_dr3_synthetic_photometry_gspc

    q = (
        Source
        .select()
    )
    if where:
        q = q.where(where)
        
    q = (
        q
        .order_by(Source.gaia_dr3_source_id.asc())
        .limit(limit)
    )

    n_updated = 0
    with tqdm(total=1, desc="Upserting") as pb:
        for chunk in chunked(q.iterator(), batch_size):

            q_bj = (
                Gaia_dr3_synthetic_photometry_gspc
                .select(
                    Gaia_dr3_synthetic_photometry_gspc.source_id,
                    Gaia_dr3_synthetic_photometry_gspc.c_star,
                    Gaia_dr3_synthetic_photometry_gspc.u_jkc_mag,
                    Gaia_dr3_synthetic_photometry_gspc.u_jkc_flag.alias("u_jkc_mag_flag"),                    
                    Gaia_dr3_synthetic_photometry_gspc.b_jkc_mag,
                    Gaia_dr3_synthetic_photometry_gspc.b_jkc_flag.alias("b_jkc_mag_flag"),                    
                    Gaia_dr3_synthetic_photometry_gspc.v_jkc_mag,
                    Gaia_dr3_synthetic_photometry_gspc.v_jkc_flag.alias("v_jkc_mag_flag"),                                        
                    Gaia_dr3_synthetic_photometry_gspc.r_jkc_mag,
                    Gaia_dr3_synthetic_photometry_gspc.r_jkc_flag.alias("r_jkc_mag_flag"),                                        
                    Gaia_dr3_synthetic_photometry_gspc.i_jkc_mag,
                    Gaia_dr3_synthetic_photometry_gspc.i_jkc_flag.alias("i_jkc_mag_flag"),                                                        
                    Gaia_dr3_synthetic_photometry_gspc.u_sdss_mag,
                    Gaia_dr3_synthetic_photometry_gspc.u_sdss_flag.alias("u_sdss_mag_flag"),
                    Gaia_dr3_synthetic_photometry_gspc.g_sdss_mag,
                    Gaia_dr3_synthetic_photometry_gspc.g_sdss_flag.alias("g_sdss_mag_flag"),
                    Gaia_dr3_synthetic_photometry_gspc.r_sdss_mag,
                    Gaia_dr3_synthetic_photometry_gspc.r_sdss_flag.alias("r_sdss_mag_flag"),
                    Gaia_dr3_synthetic_photometry_gspc.i_sdss_mag,
                    Gaia_dr3_synthetic_photometry_gspc.i_sdss_flag.alias("i_sdss_mag_flag"),
                    Gaia_dr3_synthetic_photometry_gspc.z_sdss_mag,
                    Gaia_dr3_synthetic_photometry_gspc.z_sdss_flag.alias("z_sdss_mag_flag"),
                    Gaia_dr3_synthetic_photometry_gspc.y_ps1_mag,
                    Gaia_dr3_synthetic_photometry_gspc.y_ps1_flag.alias("y_ps1_flag_mag"),                    
                    
                )
                .where(Gaia_dr3_synthetic_photometry_gspc.source_id.in_([s.gaia_dr3_source_id for s in chunk]))
                .dicts()
            )

            sources = { s.gaia_dr3_source_id: s for s in chunk }
            update = []
            for record in q_bj:
                source_id = record.pop("source_id")
                source = sources[source_id]

                for key, value in record.items():
                    setattr(source, key, value)
                
                update.append(source)
            
            n_updated += (
                Source
                .bulk_update(
                    update,
                    fields=[
                        Source.c_star,
                        Source.u_jkc_mag,
                        Source.u_jkc_mag_flag,                    
                        Source.b_jkc_mag,
                        Source.b_jkc_mag_flag,                    
                        Source.v_jkc_mag,
                        Source.v_jkc_mag_flag,                                        
                        Source.r_jkc_mag,
                        Source.r_jkc_mag_flag,                                        
                        Source.i_jkc_mag,
                        Source.i_jkc_mag_flag,                                                        
                        Source.u_sdss_mag,
                        Source.u_sdss_mag_flag,
                        Source.g_sdss_mag,
                        Source.g_sdss_mag_flag,
                        Source.r_sdss_mag,
                        Source.r_sdss_mag_flag,
                        Source.i_sdss_mag,
                        Source.i_sdss_mag_flag,
                        Source.z_sdss_mag,
                        Source.z_sdss_mag_flag,
                        Source.y_ps1_mag,
                        Source.y_ps1_mag_flag,   
                    ]
                )
            )

            pb.update(batch_size)
    
    return n_updated


def migrate_zhang_stellar_parameters(where=None, batch_size: Optional[int] = 500, limit: Optional[int] = None):
    """
    Migrate stellar parameters derived using Gaia XP spectra from Zhang, Green & Rix (2023) using the cross-match with `catalogid31` (v1).
    """

    from astra.migrations.sdss5db.catalogdb import CatalogdbModel, Gaia_DR3, BigIntegerField, ForeignKeyField

    class Gaia_Stellar_Parameters(CatalogdbModel):

        gdr3_source_id = BigIntegerField(primary_key=True)

        gaia = ForeignKeyField(Gaia_DR3,
                            field='source_id',
                            column_name='gdr3_source_id',
                            object_id_name='gdr3_source_id',
                            backref='stellar_parameters')

        class Meta:
            table_name = 'gaia_stellar_parameters'


    log.info(f"Migrating Zhang et al. stellar parameters")
    q = (
        Source
        .select(
            Source.pk,
            Source.gaia_dr3_source_id
        )
    )
    if where:
        q = q.where(where)
    q = (
        q
        .where(
            (Source.zgr_teff.is_null() & Source.gaia_dr3_source_id.is_null(False))
        )
        .limit(limit)
        .iterator()
    )    

    updated = 0
    with tqdm(total=limit) as pb:
        for batch in chunked(q, batch_size):
            q_phot = (
                Gaia_Stellar_Parameters
                .select(
                    Gaia_Stellar_Parameters.gdr3_source_id.alias("gaia_dr3_source_id"),
                    Gaia_Stellar_Parameters.stellar_params_est_teff.alias("zgr_teff"),
                    Gaia_Stellar_Parameters.stellar_params_est_logg.alias("zgr_logg"),
                    Gaia_Stellar_Parameters.stellar_params_est_fe_h.alias("zgr_fe_h"),
                    Gaia_Stellar_Parameters.stellar_params_est_e.alias("zgr_e"),
                    Gaia_Stellar_Parameters.stellar_params_est_parallax.alias("zgr_plx"),
                    Gaia_Stellar_Parameters.stellar_params_err_teff.alias("zgr_e_teff"),
                    Gaia_Stellar_Parameters.stellar_params_err_logg.alias("zgr_e_logg"),
                    Gaia_Stellar_Parameters.stellar_params_err_fe_h.alias("zgr_e_fe_h"),
                    Gaia_Stellar_Parameters.stellar_params_err_e.alias("zgr_e_e"),
                    Gaia_Stellar_Parameters.stellar_params_err_parallax.alias("zgr_e_plx"),
                    Gaia_Stellar_Parameters.teff_confidence.alias("zgr_teff_confidence"),
                    Gaia_Stellar_Parameters.logg_confidence.alias("zgr_logg_confidence"),
                    Gaia_Stellar_Parameters.feh_confidence.alias("zgr_fe_h_confidence"),
                    Gaia_Stellar_Parameters.ln_prior.alias("zgr_ln_prior"),
                    Gaia_Stellar_Parameters.chi2_opt.alias("zgr_chi2"),
                    Gaia_Stellar_Parameters.quality_flags.alias("zgr_quality_flags")
                )
                .where(Gaia_Stellar_Parameters.gdr3_source_id.in_([s.gaia_dr3_source_id for s in batch]))
                .dicts()
                .iterator()
            )

            update = []
            sources = { s.gaia_dr3_source_id: s for s in batch }
            for r in q_phot:
                source = sources[r["gaia_dr3_source_id"]]
                for key, value in r.items():
                    if key in ("zgr_teff", "zgr_e_teff"):
                        # The ZGR catalog stores these in 'kiloKelvin'...
                        transformed_value = 1000 * value
                    else:
                        transformed_value = value

                    setattr(source, key, transformed_value)
                update.append(source)
            
            if update:                    
                updated += (
                    Source
                    .bulk_update(
                        update,
                        fields=[
                            Source.zgr_teff,
                            Source.zgr_logg,
                            Source.zgr_fe_h,
                            Source.zgr_e_teff,
                            Source.zgr_e_logg,
                            Source.zgr_e_fe_h,
                            Source.zgr_e,
                            Source.zgr_plx,
                            Source.zgr_e_e,
                            Source.zgr_e_plx,
                            Source.zgr_teff_confidence,
                            Source.zgr_logg_confidence,
                            Source.zgr_fe_h_confidence,
                            Source.zgr_quality_flags,
                            Source.zgr_ln_prior,
                            Source.zgr_chi2
                        ]
                    )
                )

            pb.update(batch_size)

    log.info(f"Updated {updated} records")
    return updated




def migrate_tic_v8_identifier(catalogid_field_name="catalogid21", batch_size: Optional[int] = 500, limit: Optional[int] = None):
    from astra.migrations.sdss5db.catalogdb import CatalogToTIC_v8

    catalogid_field = getattr(Source, catalogid_field_name)

    log.info(f"Migrating TIC v8 identifiers")

    q = (
        Source
        .select(
            Source.pk,
            catalogid_field
        )
        .where(
                Source.tic_v8_id.is_null()
            &   catalogid_field.is_null(False)
        )
        .order_by(
            catalogid_field.asc()
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
                    CatalogToTIC_v8.catalogid.in_([getattr(r, catalogid_field_name) for r in batch])
                )
                .order_by(CatalogToTIC_v8.catalogid.asc())
                .tuples()
                .iterator()
            )
            
            sources = { getattr(s, catalogid_field_name): s for s in batch}
            update = []
            for catalogid, tic_v8_id in q_tic_v8:
                source = sources[catalogid]
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
        &   Source.catalogid31.is_null(False)
    ),
    limit: Optional[int] = None,
    batch_size: Optional[int] = 500, 
):
    """
    Migrate 2MASS photometry from the database, using the cross-match with `catalogid31` (v1).
    """

    from astra.migrations.sdss5db.catalogdb import TwoMassPSC, CatalogToTwoMassPSC

    log.info(f"Migrating 2MASS photometry")
    q = (
        Source
        .select(Source.catalogid31)
        .distinct()
        .where(where)
        .order_by(
            Source.catalogid31.asc()
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
            Source.pk,
            Source.catalogid31
        )
        .where(where)
        .order_by(
            Source.catalogid31.asc()
        )
        .limit(limit)
    )    

    updated_sources = []
    for source in q:
        try:
            d = twomass_data[source.catalogid31]
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



def migrate_unwise_photometry(
    where=(
        (
            Source.w1_flux.is_null()
        |   Source.w2_flux.is_null()
        )
        &   Source.catalogid21.is_null(False)
    ),
    catalogid_field_name="catalogid21", 
    batch_size: Optional[int] = 500, 
    limit: Optional[int] = None
):
    """
    Migrate 2MASS photometry from the database, using the cross-match with `catalogid21` (v0).

    As of 2023-09-14, the cross-match does not yield anything with `catalog31`.
    """

    from astra.migrations.sdss5db.catalogdb import unWISE, CatalogTounWISE

    catalogid_field = getattr(Source, catalogid_field_name)

    log.info(f"Migrating UNWISE photometry")
    q = (
        Source
        .select(
            Source.pk,
            catalogid_field
        )
        .where(where)
        .order_by(catalogid_field.asc())
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
                    CatalogTounWISE.catalogid.in_([getattr(r, catalogid_field_name) for r in batch])
                )
                .order_by(CatalogTounWISE.catalogid.asc())
                .dicts()
                .iterator()
            )

            update = []
            sources = { getattr(s, catalogid_field_name): s for s in batch }
            for r in q_phot:
                source = sources[r["catalogid"]]
                for key, value in r.items():
                    setattr(source, key, value)
                update.append(source)
            
            if update:                    
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




def migrate_glimpse_photometry(catalogid_field_name="catalogid31", batch_size: Optional[int] = 500, limit: Optional[int] = None):
    """
    Migrate Glimpse photometry from the database, using the cross-match with `catalogid31` (v1).
    """

    from astra.migrations.sdss5db.catalogdb import GLIMPSE, CatalogToGLIMPSE

    catalogid_field = getattr(Source, catalogid_field_name)

    log.info(f"Migrating GLIMPSE photometry")

    q = (
        Source
        .select(
            Source.pk,
            catalogid_field
        )
        .where(
            (
                Source.mag4_5.is_null()
            )
            &   catalogid_field.is_null(False)
        )
        .limit(limit)
    )    

    updated = 0
    with tqdm(total=limit) as pb:
        for batch in chunked(q, batch_size):
            catalogids = [getattr(r, catalogid_field_name) for r in batch]
            q_phot = list(
                GLIMPSE
                .select(
                    CatalogToGLIMPSE.catalogid.alias("catalogid"),
                    GLIMPSE.mag4_5,
                    GLIMPSE.d4_5m,
                    GLIMPSE.rms_f4_5,
                    GLIMPSE.sqf_4_5,
                    GLIMPSE.mf4_5,
                    GLIMPSE.csf,
                )
                .join(CatalogToGLIMPSE, on=(CatalogToGLIMPSE.target_id == GLIMPSE.pk))
                .where(
                    CatalogToGLIMPSE.catalogid.in_(catalogids)
                )
                .dicts()
            )

            update = []
            sources = { getattr(s, catalogid_field_name): s for s in batch }
            for r in q_phot:
                source = sources[r["catalogid"]]
                for key, value in r.items():
                    setattr(source, key, value)
                update.append(source)
            
            if update:                    
                updated += (
                    Source
                    .bulk_update(
                        update,
                        fields=[
                            Source.mag4_5,
                            Source.d4_5m,
                            Source.rms_f4_5,
                            Source.sqf_4_5,
                            Source.mf4_5,
                            Source.csf,
                        ]
                    )
                )

            pb.update(batch_size)

    log.info(f"Updated {updated} records")
    return updated


def migrate_gaia_source_ids(
    where=(
        (Source.gaia_dr3_source_id.is_null())
    |   (Source.gaia_dr3_source_id == 0)
    |   (Source.gaia_dr2_source_id.is_null())
    |   (Source.gaia_dr2_source_id == 0)
    ),
    limit: Optional[int] = None,
    batch_size: Optional[int] = 500
):
    """
    Migrate Gaia source IDs for anything that we might have missed.
    """
    
    from astra.migrations.sdss5db.catalogdb import CatalogToGaia_DR3, CatalogToGaia_DR2

    q = (
        Source
        .select()
        .where(where)
        .limit(limit)
    )

    updated = []
    with tqdm(total=1) as pb:
        for chunk in chunked(q, batch_size):

            source_by_catalogid = {}
            for source in chunk:
                for key in ("catalogid", "catalogid21", "catalogid25", "catalogid31"):
                    if getattr(source, key) is not None:
                        source_by_catalogid[getattr(source, key)] = source

            q = (
                CatalogToGaia_DR3
                .select(
                    CatalogToGaia_DR3.catalogid,
                    CatalogToGaia_DR3.target_id.alias("gaia_dr3_source_id")
                )
                .where(CatalogToGaia_DR3.catalogid.in_(list(source_by_catalogid.keys())))
                .tuples()
            )
            for catalogid, gaia_dr3_source_id in q:
                source = source_by_catalogid[catalogid]
                source.gaia_dr3_source_id = gaia_dr3_source_id
                updated.append(source)

            q = (
                CatalogToGaia_DR2
                .select(
                    CatalogToGaia_DR2.catalogid,
                    CatalogToGaia_DR2.target_id.alias("gaia_dr2_source_id")
                )
                .where(CatalogToGaia_DR2.catalogid.in_(list(source_by_catalogid.keys())))
                .tuples()
            )
            for catalogid, gaia_dr2_source_id in q:
                source = source_by_catalogid[catalogid]
                source.gaia_dr2_source_id = gaia_dr2_source_id
                updated.append(source)            
    
        pb.update(batch_size)
    
    n_updated, updated = (0, list(set(updated)))
    for chunk in chunked(updated, batch_size):
        n_updated += (
            Source
            .bulk_update(
                chunk,
                fields=[
                    Source.gaia_dr2_source_id,
                    Source.gaia_dr3_source_id
                ]
            )
        )
        
    return n_updated
        
        


def migrate_gaia_dr3_astrometry_and_photometry(where = None, limit: Optional[int] = None, batch_size: Optional[int] = 500):
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
            &   (
                Source.gaia_dr3_source_id.is_null(False)
            &   (Source.gaia_dr3_source_id > 0)
            )
        )
        .order_by(
            Source.gaia_dr3_source_id.asc()
        )
    )
    if where is not None:
        q = q.where(where)
    
    q = (
        q
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
            Source.pk,
            Source.gaia_dr3_source_id
        )
        .where(
            (
                Source.g_mag.is_null()
            |   Source.bp_mag.is_null()
            |   Source.rp_mag.is_null()
            )
            &   (
                Source.gaia_dr3_source_id.is_null(False)
            &   (Source.gaia_dr3_source_id > 0)
            )
        )        
    )
    if where:
        q = q.where(where)

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


def migrate_sources_from_sdss5_catalogdb(batch_size: Optional[int] = 500, limit: Optional[int] = None):
    """
    Migrate all catalog sources stored in the SDSS-V database.

    This creates a unique identifier per astronomical source (akin to a `sdss_id`) and links all possible
    catalog identifiers (`catalogdb.catalog.catalogid`) to those unique sources.
            
    :param batch_size: [optional]
        The batch size to use when upserting data.    
    
    :param limit: [optional]
        Limit the initial catalog queries for testing purposes.
    
    :returns:
        A tuple of new `sdss_id` identifiers created.
    """
    raise ProgrammingError
    
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel
    
    class Catalog_ver25_to_ver31_full_unique(CatalogdbModel):

        id = IntegerField(primary_key=True)

        class Meta:
            table_name = 'catalog_ver25_to_ver31_full_unique'

    log.info(f"Querying catalogdb.catalog_ver25_to_ver31_unique")

    q = (
        Catalog_ver25_to_ver31_full_unique
        .select(
            Catalog_ver25_to_ver31_full_unique.id,
            Catalog_ver25_to_ver31_full_unique.lowest_catalogid,
            Catalog_ver25_to_ver31_full_unique.highest_catalogid,
        )
        .limit(limit)
        .tuples()
        .iterator()
    )

    # Sometimes the highest_catalogid appears twice. There's a good reason for this.
    # I just don't know what it is. But we need unique-ness, and we need to link to
    # the lower catalog identifiers.
    next_sdss_id = 1
    source_data, lookup_sdss_id_from_catalog_id, lookup_catalog_id_from_sdss_id = ({}, {}, {})
    for sdss_id, lowest, highest in tqdm(q, total=limit or get_approximate_rows(Catalog_ver25_to_ver31_full_unique)):

        # Do we already have an sdss_id assigned to this highest catalog identifier?
        sdss_id_1 = lookup_sdss_id_from_catalog_id.get(highest, None)
        sdss_id_2 = lookup_sdss_id_from_catalog_id.get(lowest, None)

        if sdss_id_1 is not None and sdss_id_2 is not None and sdss_id_1 != sdss_id_2:
            # We need to amalgamate these two.
            affected = []
            affected.extend(lookup_catalog_id_from_sdss_id[sdss_id_1])
            affected.extend(lookup_catalog_id_from_sdss_id[sdss_id_2])
            
            # merge both into sdss_id_1
            source_data[sdss_id_1] = dict(
                sdss_id=sdss_id_1,
                sdss5_catalogid_v1=max(affected)
            )
            for catalogid in affected:
                lookup_sdss_id_from_catalog_id[catalogid] = sdss_id_1

            lookup_catalog_id_from_sdss_id[sdss_id_1] = affected
            
            del source_data[sdss_id_2]
            del lookup_catalog_id_from_sdss_id[sdss_id_2]
        
        else:
            sdss_id = sdss_id_1 or sdss_id_2
            if sdss_id is None:
                sdss_id = 0 + next_sdss_id
                next_sdss_id += 1
        
            lookup_catalog_id_from_sdss_id.setdefault(sdss_id, [])
            lookup_catalog_id_from_sdss_id[sdss_id].extend((lowest, highest))

            lookup_sdss_id_from_catalog_id[lowest] = sdss_id
            lookup_sdss_id_from_catalog_id[highest] = sdss_id
            source_data[sdss_id] = dict(
                sdss_id=sdss_id,
                sdss5_catalogid_v1=highest
            )

    log.info(f"There are {len(source_data)} unique `sdss_id` entries so far")

    class Catalog_ver25_to_ver31_full_all(CatalogdbModel):

        id = IntegerField(primary_key=True)

        class Meta:
            table_name = 'catalog_ver25_to_ver31_full_all'
    
    log.info(f"Querying catalogdb.catalog_ver25_to_ver31_full_all")

    q = (
        Catalog_ver25_to_ver31_full_all
        .select(
            Catalog_ver25_to_ver31_full_all.lowest_catalogid,
            Catalog_ver25_to_ver31_full_all.highest_catalogid
        )
        .limit(limit)
        .tuples()
        .iterator()
    )
    
    for lowest, highest in tqdm(q, total=limit or get_approximate_rows(Catalog_ver25_to_ver31_full_all)):

        sdss_id_1 = lookup_sdss_id_from_catalog_id.get(highest, None)
        sdss_id_2 = lookup_sdss_id_from_catalog_id.get(lowest, None)

        if sdss_id_1 is not None and sdss_id_2 is not None and sdss_id_1 != sdss_id_2:
            # We need to amalgamate these two.
            affected = []
            affected.extend(lookup_catalog_id_from_sdss_id[sdss_id_1])
            affected.extend(lookup_catalog_id_from_sdss_id[sdss_id_2])
            
            # merge both into sdss_id_1
            source_data[sdss_id_1] = dict(
                sdss_id=sdss_id_1,
                sdss5_catalogid_v1=max(affected)
            )
            for catalogid in affected:
                lookup_sdss_id_from_catalog_id[catalogid] = sdss_id_1

            lookup_catalog_id_from_sdss_id[sdss_id_1] = affected
            
            del source_data[sdss_id_2]
            del lookup_catalog_id_from_sdss_id[sdss_id_2]
        
        else:
            sdss_id = sdss_id_1 or sdss_id_2
            if sdss_id is None:
                sdss_id = 0 + next_sdss_id
                next_sdss_id += 1
        
            lookup_catalog_id_from_sdss_id.setdefault(sdss_id, [])
            lookup_catalog_id_from_sdss_id[sdss_id].extend((lowest, highest))

            lookup_sdss_id_from_catalog_id[lowest] = sdss_id
            lookup_sdss_id_from_catalog_id[highest] = sdss_id
            source_data[sdss_id] = dict(
                sdss_id=sdss_id,
                sdss5_catalogid_v1=highest
            )
            
    log.info(f"There are now {len(source_data)} unique `sdss_id` entries so far")

    # Create the Source
    new_source_ids = []
    with database.atomic():
        # Need to chunk this to avoid SQLite limits.
        with tqdm(desc="Upserting", unit="sources", total=len(source_data)) as pb:
            for chunk in chunked(source_data.values(), batch_size):
                new_source_ids.extend(
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .returning(Source.sdss_id)
                    .tuples()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    log.info(f"Inserted {len(new_source_ids)} new sources")

    log.info("Linking catalog identifiers to SDSS identifiers")
    
    data_generator = (
        dict(catalogid=catalogid, sdss_id=sdss_id) 
        for catalogid, sdss_id in lookup_sdss_id_from_catalog_id.items()
    )
    
    with database.atomic():
        with tqdm(desc="Linking catalog identifiers to unique sources", total=len(lookup_sdss_id_from_catalog_id)) as pb:
            for chunk in chunked(data_generator, batch_size):
                (
                    SDSSCatalog
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .returning(SDSSCatalog.catalogid)
                    .tuples()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()
    
    return tuple(new_source_ids)





def migrate_catalog_sky_positions(batch_size=1000):

    raise RuntimeError

    from astra.migrations.sdss5db.catalogdb import Catalog

    log.info(f"Querying for sources without basic metadata")
    where_source = (
            Source.ra.is_null()
        &   (Source.sdss5_catalogid_v1.is_null(False))        
    )
    q = (
        Source
        .select(
            Source.sdss5_catalogid_v1
        )
        .where(where_source)
        .tuples()
        .iterator()
    )

    log.info(f"Querying catalogdb.catalog for basic catalog metadata")
    catalog_data = {}
    n_q = 0
    for batch in tqdm(chunked(q, batch_size), total=1):        
        q_catalog = (
            Catalog
            .select(
                Catalog.catalogid.alias("sdss5_catalogid_v1"),
                Catalog.ra,
                Catalog.dec,
                Catalog.lead,
                Catalog.version_id
            )
            .where(Catalog.catalogid.in_(flatten(batch)))
            .tuples()
            .iterator()
        )

        for sdss5_catalogid_v1, ra, dec, lead, version_id in q_catalog:
            catalog_data[sdss5_catalogid_v1] = dict(
                ra=ra,
                dec=dec,
                version_id=version_id,
                lead=lead
            )
        n_q += len(batch)

    q = (
        Source
        .select()
        .where(where_source)
    )

    updated = 0
    with tqdm(total=n_q) as pb:
        for batch in chunked(q.iterator(), batch_size):
            for source in batch:
                try:
                    for key, value in catalog_data[source.sdss5_catalogid_v1].items():
                        setattr(source, key, value)
                except:
                    continue
            updated += (
                Source
                .bulk_update(
                    batch,
                    fields=[
                        Source.ra,
                        Source.dec,
                        Source.version_id,
                        Source.lead
                    ],
                )
            )
            pb.update(len(batch))            
            

    # You might think that we should only do subsequent photometry/astrometry queries for things that were 
    # actually targeted, but you'd be wrong. We need to include everything from SDSS-IV / DR17 too, which
    # was not assigned to any SDSS-V carton.

    '''    
    # Only do the carton/target cleverness if we are using a postgresql database.
    if isinstance(database, PostgresqlDatabase):
        log.info(f"Querying cartons and target assignments")    

        from astra.migrations.sdss5db.targetdb import Target, CartonToTarget

        q = (
            CartonToTarget
            .select(
                Target.catalogid,
                CartonToTarget.carton_pk
            )
            .join(Target, on=(CartonToTarget.target_pk == Target.pk))
            .tuples()
            .iterator()
        )
        lookup_cartons_by_catalogid = {}
        for catalogid, carton_pk in tqdm(q, total=get_approximate_rows(CartonToTarget)):
            lookup_cartons_by_catalogid.setdefault(catalogid, [])
            lookup_cartons_by_catalogid[catalogid] = carton_pk
        
        # We will actually assign these to sdss_id entries later on, when it's time to create
        # the sources.
        raise NotImplementedError
    
    else:
        log.warning(f"Not including carton and target assignments right now")
    '''
    log.info(f"Inserting sources")

        


def migrate_cata():

    log.info("Preparing data for catalog queries..")

    log.info(f"Getting Gaia DR3 information")

    # Add Gaia DR3 identifiers
    
    q = (
        Gaia_DR3
        .select(
            CatalogToGaia_DR3.catalogid,
            Gaia_DR3.source_id.alias("gaia_dr3_source_id"),
            Gaia_DR3.parallax.alias("plx"),
            Gaia_DR3.parallax_error.alias("e_plx"),
            Gaia_DR3.pmra,
            Gaia_DR3.pmra_error.alias("e_pmra"),
            Gaia_DR3.pmdec.alias("pmde"),
            Gaia_DR3.pmdec_error.alias("e_pmde"),
            Gaia_DR3.phot_g_mean_mag.alias("g_mag"),
            Gaia_DR3.phot_bp_mean_mag.alias("bp_mag"),
            Gaia_DR3.phot_rp_mean_mag.alias("rp_mag"),
        )
        .join(CatalogToGaia_DR3, on=(CatalogToGaia_DR3.target_id == CatalogToGaia_DR3.source_id))
        .where(
            CatalogToGaia_DR3.catalogid.in_(reference_catalogids)
        )
        .dicts()
    )
    for row in tqdm(q):
        sdss_id = lookup_sdss_id_from_catalog_id[row["catalogid"]]
        source_data[sdss_id].update(**row)

    log.info(f"Querying TIC v8")
    
    q = (
        CatalogToTIC_v8
        .select(
            CatalogToTIC_v8.catalogid,
            CatalogToTIC_v8.target_id
        )
        .where(
            CatalogToTIC_v8.catalogid.in_(reference_catalogids)
        )
        .dicts()
    )
    for catalogid, tic_v8_id in q:
        sdss_id = lookup_sdss_id_from_catalog_id[catalogid]
        source_data[sdss_id].update(tic_v8_id=tic_v8_id)
    
    # Now do SDSSCatalog
    
    raise a

    '''
    log.info(f"Upserting sources..")

    new_sources = []
    with database.atomic():
        # Need to chunk this to avoid SQLite limits.
        with tqdm(desc="Upserting", unit="sources", total=len(sources)) as pb:
            for chunk in chunked(sources, batch_size):
                new_sources.extend(
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .returning(Source.sdss_id, Source.sdss5_catalogid_v1)
                    .tuples()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    raise a
    sdss_ids = dict(new_sources)
    sources_to_catalogids = [{ "sdss_id": sdss_id, "catalogid": catalogid } for sdss_id, catalogid in sources_to_catalogids]

    log.info("Upserting links")

    with database.atomic():
        with tqdm(desc="Linking catalog identifiers to unique sources", total=len(sources_to_catalogids)) as pb:
            for chunk in chunked(sources_to_catalogids, batch_size):
                (
                    SDSSCatalog
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .tuples()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()
    
    log.info(f"Inserting basic catalog information")
    new_catalogids = [c for s, c in new_sources]
    '''


    # 
    cids = {}
    for s, c in new_sources:
        cids.setdefault(c, [])
        cids[c].append(s)

    data = []
    for sdss5_catalogid_v1, ra, dec, lead, version_id in q:
        for sdss_id in cids[sdss5_catalogid_v1]:
            data.append(
                {
                    "ra": ra,
                    "dec": dec,
                    "lead": lead,
                    "version_id": version_id,
                    "sdss_id": sdss_id
                }
            )

    with database.atomic():
        with tqdm(desc="Upserting basic catalog information", total=len(data)) as pb:
            for chunk in chunked(data, batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_replace()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()        

    # Add Gaia DR3 identifiers
    log.info(f"Querying Gaia DR3")
    
    q = (
        CatalogToGaia_DR3
        .select(
            CatalogToGaia_DR3.catalog,
            CatalogToGaia_DR3.target
        )
        .where(
            CatalogToGaia_DR3.catalog.in_(new_catalogids)
        )
        .tuples()
    )
    data = []
    for catalogid, targetid in q:
        for sdss_id in cids[catalogid]:
            data.append({
                "sdss_id": sdss_id,
                "gaia_dr3_source_id": targetid
            })

    with database.atomic():
        with tqdm(desc="Linking Gaia DR3", total=len(data)) as pb:
            for chunk in chunked(data, batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_replace()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()               

    # Add Gaia DR2 identifiers

    # Add TIC v8 identifier
    # Add Gaia DR3 identifiers
    log.info(f"Querying TIC v8")
    
    q = (
        CatalogToTIC_v8
        .select(
            CatalogToTIC_v8.catalogid,
            CatalogToTIC_v8.target_id
        )
        .where(
            CatalogToTIC_v8.catalogid.in_(new_catalogids)
        )
        .tuples()
    )
    data = []
    for catalogid, targetid in q:
        for sdss_id in cids[catalogid]:
            data.append({
                "sdss_id": sdss_id,
                "tic_v8_id": targetid
            })

    with database.atomic():
        with tqdm(desc="Linking TIC v8", total=len(data)) as pb:
            for chunk in chunked(data, batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_replace()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()               

    raise a
    raise a



if __name__ == "__main__":
    from astra.models.source import Source
    models = [Spectrum, ApogeeVisitSpectrum, Source, SDSSCatalog]
    #database.drop_tables(models)
    if models[0].table_exists():
        database.drop_tables(models)
    database.create_tables(models)

    #migrate_apvisit_from_sdss5_apogee_drpdb()

    migrate_sources_from_sdss5_catalogdb()

