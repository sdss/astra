
from typing import Optional
from tqdm import tqdm
from peewee import chunked, IntegerField,  fn, JOIN, IntegrityError
#from astra.migrations.sdss5db.utils import get_approximate_rows
from astra.migrations.utils import NoQueue
from astra.utils import log, flatten
import numpy as np


def migrate_healpix(
    batch_size: Optional[int] = 500,
    limit: Optional[int] = None,
    nside: Optional[int] = 128,
    lonlat: Optional[bool] = True,
    queue=None,
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
    from astra.models.base import database
    from astra.models.source import Source

    from healpy import ang2pix
    if queue is None:
        queue = NoQueue()
    
    q = (
        Source
        .select(
            Source.pk,
            Source.ra,
            Source.dec,
        )
        .where(
            Source.healpix.is_null()
        &   Source.ra.is_null(False)
        &   Source.dec.is_null(False)
        )
        .limit(limit)
    )    
    
    updated, total = (0, limit or q.count())
    queue.put(dict(description="Migrating HEALPix values", total=total, completed=0))
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
        queue.put(dict(advance=len(batch)))

    queue.put(Ellipsis)
    return updated


def migrate_bailer_jones_distances(
    batch_size=500, 
    limit=None,
    queue=None
):
    from astra.models.base import database
    from astra.models.source import Source

    if queue is None:
        queue = NoQueue()

    from astra.migrations.sdss5db.catalogdb import BailerJonesEDR3

    q = (
        Source
        .select()
        .where(
            (Source.r_med_geo.is_null() & Source.gaia_dr3_source_id.is_null(False) & (Source.gaia_dr3_source_id > 0))
        )
    )
        
    q = (
        q
        .order_by(Source.gaia_dr3_source_id.asc())
        .limit(limit)
    )

    n_updated = 0
    queue.put(dict(total=limit or q.count(), completed=0))
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
                if value is not None:
                    setattr(source, key, value)
            
            update.append(source)
        
        if len(update) > 0:
            fields = []
            consider_keys = ("r_med_geo", "r_lo_geo", "r_hi_geo", "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo")
            for key in consider_keys:
                for l in update:
                    if getattr(l, key) is not None:
                        fields.append(getattr(Source, key))
                        break
                
            n_updated += (
                Source
                .bulk_update(
                    update,
                    fields=fields + [Source.bailer_jones_flags]
                )
            )

        queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)
    return n_updated


def migrate_gaia_synthetic_photometry(
    batch_size=500, 
    limit=None,
    queue=None
):
    from astra.models.base import database
    from astra.models.source import Source

    from astra.migrations.sdss5db.catalogdb import Gaia_dr3_synthetic_photometry_gspc
    if queue is None:
        queue = NoQueue()

    q = (
        Source
        .select()
        .where((Source.gaia_dr3_source_id.is_null(False) & Source.g_sdss_mag.is_null()))
    )
        
    q = (
        q
        .order_by(Source.gaia_dr3_source_id.asc())
        .limit(limit)
    )

    n_updated = 0
    queue.put(dict(total=q.count()))
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
                Gaia_dr3_synthetic_photometry_gspc.y_ps1_flag.alias("y_ps1_mag_flag"),                    
                
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
                if value is None:
                    if key.endswith("_mag"):
                        value = np.nan
                    elif key.endswith("_flag"):
                        value = 0

                setattr(source, key, value)
            
            update.append(source)
        
        if len(update) > 0:
            with database.atomic():                    
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

        queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)
    return n_updated


def migrate_zhang_stellar_parameters(where=None, batch_size: Optional[int] = 500, limit: Optional[int] = None, queue=None):
    """
    Migrate stellar parameters derived using Gaia XP spectra from Zhang, Green & Rix (2023) using the cross-match with `catalogid31` (v1).
    """
    from astra.models.base import database
    from astra.models.source import Source

    from astra.migrations.sdss5db.catalogdb import CatalogdbModel, Gaia_DR3, BigIntegerField, ForeignKeyField

    # Sigh, this catalog is on operations, but not pipelines.
    if queue is None:
        queue = NoQueue()
    from sdssdb.peewee.sdss5db import SDSS5dbDatabaseConnection

    class Gaia_Stellar_Parameters(CatalogdbModel):

        gdr3_source_id = BigIntegerField(primary_key=True)

        gaia = ForeignKeyField(Gaia_DR3,
                            field='source_id',
                            column_name='gdr3_source_id',
                            object_id_name='gdr3_source_id',
                            backref='stellar_parameters')

        class Meta:
            table_name = 'gaia_stellar_parameters'
            database = SDSS5dbDatabaseConnection(profile="operations")

    #log.info(f"Migrating Zhang et al. stellar parameters")
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
    )    

    updated = 0
    queue.put(dict(total=limit or q.count()))
    #with tqdm(total=limit) as pb:
    for batch in chunked(q.iterator(), batch_size):
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

        queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)
    #log.info(f"Updated {updated} records")
    return updated




def migrate_tic_v8_identifier(catalogid_field_name="catalogid21", batch_size: Optional[int] = 500, limit: Optional[int] = None, queue=None):
    if queue is None:
        queue = NoQueue()
    from astra.models.base import database
    from astra.models.source import Source

    from astra.migrations.sdss5db.catalogdb import CatalogToTIC_v8

    catalogid_field = getattr(Source, catalogid_field_name)

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
    )    

    updated = 0
    queue.put(dict(total=q.count()))
    if q:            
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

            queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)
    return updated    


def migrate_twomass_photometry(
    limit: Optional[int] = None,
    batch_size: Optional[int] = 500, 
    queue = None
):
    """
    Migrate 2MASS photometry from the database, using the cross-match with `catalogid31` (v1).
    """

    from astra.models.base import database
    from astra.models.source import Source

    if queue is None:
        queue = NoQueue()
    from astra.migrations.sdss5db.catalogdb import TwoMassPSC, CatalogToTwoMassPSC

    where = (
        (
            Source.j_mag.is_null()
        |   Source.h_mag.is_null()
        |   Source.k_mag.is_null()
        )
        &   Source.catalogid31.is_null(False)                
    )

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

    queue.put(dict(total=limit or q.count()))

    twomass_data = {}
    if q:
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
            queue.put(dict(advance=min(batch_size, len(batch))))

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

    queue.put(dict(description="Assigning 2MASS photometry", total=limit, completed=0))

    updated_sources = []
    for source in q:
        try:
            d = twomass_data[source.catalogid31]
        except KeyError:
            None
        else:
            for key, value in d.items():
                setattr(source, key, value or np.nan)
            updated_sources.append(source)
        finally:
            queue.put(dict(advance=1))


    queue.put(dict(description="Updating sources with 2MASS photometry", total=len(updated_sources), completed=0))
    
    updated = 0
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
        queue.put(dict(advance=min(batch_size, len(chunk))))

    queue.put(Ellipsis)
    return updated



def migrate_unwise_photometry(
    catalogid_field_name="catalogid21", 
    batch_size: Optional[int] = 500, 
    limit: Optional[int] = None,
    queue = None,
):
    """
    Migrate 2MASS photometry from the database, using the cross-match with `catalogid21` (v0).

    As of 2023-09-14, the cross-match does not yield anything with `catalog31`.
    """

    from astra.models.base import database
    from astra.models.source import Source

    if queue is None:
        queue = NoQueue()



    from astra.migrations.sdss5db.catalogdb import unWISE, CatalogTounWISE

    catalogid_field = getattr(Source, catalogid_field_name)

    q = (
        Source
        .select(
            Source.pk,
            Source.sdss_id,
            catalogid_field
        )
        .where(
            (
                Source.w1_flux.is_null()
            |   Source.w2_flux.is_null()
            )
            &   Source.catalogid21.is_null(False)            
        )
        .order_by(catalogid_field.asc())
        .limit(limit)
    )    
    
    updated = 0
    queue.put(dict(total=limit or q.count()))
    if q:
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

            queue.put(dict(advance=batch_size))
    queue.put(Ellipsis)
    return updated




def migrate_glimpse_photometry(catalogid_field_name="catalogid31", batch_size: Optional[int] = 500, limit: Optional[int] = None, queue=None):
    """
    Migrate Glimpse photometry from the database, using the cross-match with `catalogid31` (v1).
    """

    from astra.models.base import database
    from astra.models.source import Source

    if queue is None:
        queue = NoQueue()

    from astra.migrations.sdss5db.catalogdb import GLIMPSE, CatalogToGLIMPSE

    catalogid_field = getattr(Source, catalogid_field_name)
    
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
    queue.put(dict(total=limit or q.count()))

    updated = 0
    if q:
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

            queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)
    return updated


def migrate_gaia_source_ids(
    limit: Optional[int] = None,
    batch_size: Optional[int] = 1000,
    queue=None
):
    """
    Migrate Gaia source IDs for anything that we might have missed.
    """
    from astra.models.base import database
    from astra.models.source import Source

    if queue is None:
        queue = NoQueue()

    queue.put(Ellipsis)
    return None
    
    from astra.migrations.sdss5db.catalogdb import CatalogToGaia_DR3, CatalogToGaia_DR2

    q = (
        Source
        .select()
        .where(
            (Source.gaia_dr3_source_id.is_null())
        |   (Source.gaia_dr3_source_id == 0)
        |   (Source.gaia_dr2_source_id.is_null())
        |   (Source.gaia_dr2_source_id == 0)            
        )
        .limit(limit)
    )

    updated = []
    queue.put(dict(total=limit or q.count(), description="Querying Gaia source IDs"))

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
            source.gaia_dr3_source_id = gaia_dr3_source_id or -1
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
            source.gaia_dr2_source_id = gaia_dr2_source_id or -1
            updated.append(source)            

        queue.put(dict(advance=batch_size))
    
    fields = list(fields)
    n_updated, updated = (0, list(set(updated)))
    queue.put(dict(total=len(updated), completed=0, description="Ingesting Gaia DR3 source IDs"))
    integrity_errors = []
    for chunk in chunked(updated, batch_size):
        try:
            n_updated += (
                Source
                .bulk_update(
                    chunk,
                    fields=[
                        Source.gaia_dr3_source_id,
                        Source.gaia_dr2_source_id
                    ]
                )
            )
        except IntegrityError:
            integrity_errors.append(chunk)
            raise a
                
        queue.put(dict(advance=batch_size))
    #if integrity_errors:
    #    log.warning(f"Integrity errors encountered for {len(integrity_errors)} chunks")
    queue.put(Ellipsis)
    return n_updated
        
        


def migrate_gaia_dr3_astrometry_and_photometry(
    limit: Optional[int] = None, 
    batch_size: Optional[int] = 500,
    queue=None
):
    """
    Migrate Gaia DR3 astrometry and photometry from the SDSS-V database for any sources (`astra.models.Source`)
    that have a Gaia DR3 source identifier (`astra.models.Source.gaia_dr3_source_id`) but are missing Gaia
    photometry.

    :param batch_size: [optional]
        The batch size to use for updates.
    
    :param limit: [optional]
        Limit the update to `limit` records. Useful for testing.
    """
    from astra.models.base import database
    from astra.models.source import Source

    if queue is None:
        queue = NoQueue()

    from astra.migrations.sdss5db.catalogdb import Gaia_DR3 as _Gaia_DR3

    # Just as I said would happen, the commonly used tables have not been copied over to the pipelines database.
    # The pipelines database has introduced more difficulties and edge cases than what it has fixed.
    from sdssdb.peewee.sdss5db import SDSS5dbDatabaseConnection

    class Gaia_DR3(_Gaia_DR3):

        class Meta:
            table_name = "gaia_dr3_source"
            database = SDSS5dbDatabaseConnection(profile="operations")

    
    #log.info(f"Updating Gaia astrometry and photometry")

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
    
    q = (
        q
        .limit(limit)
        .tuples()
    )

    total = limit or q.count()

    gaia_data = {}
    queue.put(dict(total=total))
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
        queue.put(dict(advance=len(batch)))

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

    updated_sources = []
    for source in q:
        try:
            z = gaia_data[source.gaia_dr3_source_id]
        except KeyError:
            log.warning(f"Source pk={source} seems to have an incorrect Gaia DR3 source identifier: {source.gaia_dr3_source_id} (sdss_id={source.sdss_id})")
            continue
        else:
            for k, v in z.items():
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

    #log.info(f"Updated {updated} records ({len(gaia_data)} gaia sources)")
    queue.put(Ellipsis)
    return updated


