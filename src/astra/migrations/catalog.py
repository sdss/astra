
from typing import Optional
from tqdm import tqdm
from peewee import chunked, IntegerField,  fn, JOIN, IntegrityError
#from astra.migrations.sdss5db.utils import get_approximate_rows
from astra.migrations.utils import ProgressContext
from astra.utils import log, flatten
import numpy as np
import re



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
        queue = ProgressContext()

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

    updated = 0
    queue.put(dict(description="Migrating HEALPix values", total=limit, completed=0))
    for batch in chunked(q.iterator(), batch_size):
        for record in batch:
            try:
                record.healpix = ang2pix(nside, record.ra, record.dec, lonlat=lonlat)
            except ValueError:
                record.healpix = -1
        with database.atomic():
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


def migrate_bailer_jones_distances(queue=None, **kwargs):
    """
    Migrate Bailer-Jones distances from catalogdb for any sources that have a
    Gaia DR3 source identifier but are missing distance estimates.
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.migrations.sdss5db.catalogdb import BailerJonesEDR3

    if queue is None:
        queue = ProgressContext()

    with database.atomic():
        n = (
            Source
            .update(
                r_med_geo=BailerJonesEDR3.r_med_geo,
                r_lo_geo=BailerJonesEDR3.r_lo_geo,
                r_hi_geo=BailerJonesEDR3.r_hi_geo,
                r_med_photogeo=BailerJonesEDR3.r_med_photogeo,
                r_lo_photogeo=BailerJonesEDR3.r_lo_photogeo,
                r_hi_photogeo=BailerJonesEDR3.r_hi_photogeo,
                bailer_jones_flags=BailerJonesEDR3.flag,
            )
            .from_(BailerJonesEDR3)
            .where(
                (Source.gaia_dr3_source_id == BailerJonesEDR3.source_id)
            &   Source.r_med_geo.is_null()
            &   Source.gaia_dr3_source_id.is_null(False)
            )
            .execute()
        )
    queue.put(Ellipsis)
    return n


def migrate_gaia_synthetic_photometry(queue=None, **kwargs):
    """
    Migrate Gaia DR3 synthetic photometry from catalogdb for any sources that have a
    Gaia DR3 source identifier but are missing synthetic photometry.
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.migrations.sdss5db.catalogdb import Gaia_dr3_synthetic_photometry_gspc
    from peewee import fn

    if queue is None:
        queue = ProgressContext()

    G = Gaia_dr3_synthetic_photometry_gspc

    with database.atomic():
        n = (
            Source
            .update(
                c_star=G.c_star,
                u_jkc_mag=fn.COALESCE(G.u_jkc_mag, float('nan')),
                u_jkc_mag_flag=fn.COALESCE(G.u_jkc_flag, 0),
                b_jkc_mag=fn.COALESCE(G.b_jkc_mag, float('nan')),
                b_jkc_mag_flag=fn.COALESCE(G.b_jkc_flag, 0),
                v_jkc_mag=fn.COALESCE(G.v_jkc_mag, float('nan')),
                v_jkc_mag_flag=fn.COALESCE(G.v_jkc_flag, 0),
                r_jkc_mag=fn.COALESCE(G.r_jkc_mag, float('nan')),
                r_jkc_mag_flag=fn.COALESCE(G.r_jkc_flag, 0),
                i_jkc_mag=fn.COALESCE(G.i_jkc_mag, float('nan')),
                i_jkc_mag_flag=fn.COALESCE(G.i_jkc_flag, 0),
                u_sdss_mag=fn.COALESCE(G.u_sdss_mag, float('nan')),
                u_sdss_mag_flag=fn.COALESCE(G.u_sdss_flag, 0),
                g_sdss_mag=fn.COALESCE(G.g_sdss_mag, float('nan')),
                g_sdss_mag_flag=fn.COALESCE(G.g_sdss_flag, 0),
                r_sdss_mag=fn.COALESCE(G.r_sdss_mag, float('nan')),
                r_sdss_mag_flag=fn.COALESCE(G.r_sdss_flag, 0),
                i_sdss_mag=fn.COALESCE(G.i_sdss_mag, float('nan')),
                i_sdss_mag_flag=fn.COALESCE(G.i_sdss_flag, 0),
                z_sdss_mag=fn.COALESCE(G.z_sdss_mag, float('nan')),
                z_sdss_mag_flag=fn.COALESCE(G.z_sdss_flag, 0),
                y_ps1_mag=fn.COALESCE(G.y_ps1_mag, float('nan')),
                y_ps1_mag_flag=fn.COALESCE(G.y_ps1_flag, 0),
            )
            .from_(G)
            .where(
                (Source.gaia_dr3_source_id == G.source_id)
            &   Source.gaia_dr3_source_id.is_null(False)
            &   Source.g_sdss_mag.is_null()
            )
            .execute()
        )
    queue.put(Ellipsis)
    return n


def migrate_zhang_stellar_parameters(where=None, batch_size: Optional[int] = 500, limit: Optional[int] = None, queue=None):
    """
    Migrate stellar parameters derived using Gaia XP spectra from Zhang, Green & Rix (2023) using the cross-match with `catalogid31` (v1).
    """
    from astra.models.base import database
    from astra.models.source import Source

    from astra.migrations.sdss5db.catalogdb import CatalogdbModel, Gaia_DR3, BigIntegerField, ForeignKeyField, Gaia_Stellar_Parameters

    # Sigh, this catalog is on operations, but not pipelines.
    if queue is None:
        queue = ProgressContext()
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
    queue.put(dict(total=limit, description="Migrating Zhang stellar parameters"))
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
            with database.atomic():
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




def migrate_tic_v8_identifier(queue=None, **kwargs):
    """
    Migrate TIC v8 identifiers from catalogdb for sources that have an sdss_id
    but are missing tic_v8_id.
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.migrations.sdss5db.catalogdb import CatalogToTIC_v8, CatalogdbModel

    if queue is None:
        queue = ProgressContext()

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    with database.atomic():
        n = (
            Source
            .update(tic_v8_id=CatalogToTIC_v8.target)
            .from_(SDSS_ID_Flat, CatalogToTIC_v8)
            .where(
                (Source.sdss_id == SDSS_ID_Flat.sdss_id)
            &   (SDSS_ID_Flat.catalogid == CatalogToTIC_v8.catalogid)
            &   (SDSS_ID_Flat.rank == 1)
            &   Source.tic_v8_id.is_null()
            &   Source.sdss_id.is_null(False)
            )
            .execute()
        )
    queue.put(Ellipsis)
    return n


def migrate_twomass_photometry(queue=None, **kwargs):
    """
    Migrate 2MASS photometry from catalogdb for sources that have an sdss_id
    but are missing 2MASS photometry.
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.migrations.sdss5db.catalogdb import TwoMassPSC, CatalogToTwoMassPSC, CatalogdbModel

    if queue is None:
        queue = ProgressContext()

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    with database.atomic():
        n = (
            Source
            .update(
                j_mag=TwoMassPSC.j_m,
                e_j_mag=TwoMassPSC.j_cmsig,
                h_mag=TwoMassPSC.h_m,
                e_h_mag=TwoMassPSC.h_cmsig,
                k_mag=TwoMassPSC.k_m,
                e_k_mag=TwoMassPSC.k_cmsig,
                ph_qual=TwoMassPSC.ph_qual,
                bl_flg=TwoMassPSC.bl_flg,
                cc_flg=TwoMassPSC.cc_flg,
            )
            .from_(SDSS_ID_Flat, CatalogToTwoMassPSC, TwoMassPSC)
            .where(
                (Source.sdss_id == SDSS_ID_Flat.sdss_id)
            &   (SDSS_ID_Flat.catalogid == CatalogToTwoMassPSC.catalogid)
            &   (CatalogToTwoMassPSC.target == TwoMassPSC.pts_key)
            &   (SDSS_ID_Flat.rank == 1)
            &   (
                    Source.j_mag.is_null()
                |   Source.h_mag.is_null()
                |   Source.k_mag.is_null()
                |   Source.ph_qual.is_null()
                )
            &   Source.sdss_id.is_null(False)
            )
            .execute()
        )
    queue.put(Ellipsis)
    return n



def migrate_unwise_photometry(
    batch_size: Optional[int] = 10_000,
    limit: Optional[int] = None,
    queue = None,
):
    """
    Migrate unWISE photometry from the database using sdss_id to look up catalogids.
    """

    from astra.models.base import database
    from astra.models.source import Source

    if queue is None:
        queue = ProgressContext()

    from astra.migrations.sdss5db.catalogdb import unWISE, CatalogTounWISE, CatalogdbModel

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    where = (
        (
            Source.w1_flux.is_null()
        |   Source.w2_flux.is_null()
        )
        &   Source.sdss_id.is_null(False)
    )

    # Single query to get pk and sdss_id
    q = (
        Source
        .select(Source.pk, Source.sdss_id)
        .where(where)
        .order_by(Source.sdss_id.asc())
        .limit(limit)
    )

    queue.put(dict(total=None, description="Fetching sources for unWISE photometry"))

    # Fetch all at once to avoid slow count
    rows = list(q.tuples())
    total = len(rows)

    if total == 0:
        queue.put(Ellipsis)
        return 0

    queue.put(dict(total=total, completed=total))

    # Build lookup of sdss_id -> pk
    source_pks = {sdss_id: pk for pk, sdss_id in rows}

    sdss_ids = list(source_pks.keys())
    table = f"{Source._meta.schema}.{Source._meta.table_name}"
    columns = ["w1_flux", "w1_dflux", "w2_flux", "w2_dflux", "w1_frac", "w2_frac",
               "w1uflags", "w2uflags", "w1aflags", "w2aflags"]

    updated = 0
    queue.put(dict(total=len(sdss_ids), completed=0, description="Updating unWISE photometry"))

    for batch in chunked(sdss_ids, batch_size):
        # Get catalogids for these sdss_ids from sdss_id_flat
        catalogid_map = {}  # catalogid -> sdss_id
        for row in (
            SDSS_ID_Flat
            .select(SDSS_ID_Flat.sdss_id, SDSS_ID_Flat.catalogid)
            .where(
                (SDSS_ID_Flat.sdss_id.in_(batch))
            &   (SDSS_ID_Flat.rank == 1)
            )
            .tuples()
        ):
            catalogid_map[row[1]] = row[0]

        if not catalogid_map:
            queue.put(dict(advance=len(batch)))
            continue

        # Fetch unWISE data for this batch
        q_phot = (
            unWISE
            .select(
                CatalogTounWISE.catalogid,
                unWISE.flux_w1,
                unWISE.dflux_w1,
                unWISE.flux_w2,
                unWISE.dflux_w2,
                unWISE.fracflux_w1,
                unWISE.fracflux_w2,
                unWISE.flags_unwise_w1,
                unWISE.flags_unwise_w2,
                unWISE.flags_info_w1,
                unWISE.flags_info_w2
            )
            .join(CatalogTounWISE)
            .where(CatalogTounWISE.catalogid.in_(list(catalogid_map.keys())))
            .tuples()
        )

        # Build VALUES rows for bulk update
        values_rows = []
        for row in q_phot:
            catalogid = row[0]
            sdss_id = catalogid_map.get(catalogid)
            if sdss_id is None:
                continue
            pk = source_pks.get(sdss_id)
            if pk is None:
                continue

            # Format values: pk, then all photometry columns
            # Use explicit casts to avoid PostgreSQL type inference issues
            # Order: w1_flux, w1_dflux, w2_flux, w2_dflux, w1_frac, w2_frac (real),
            #        w1uflags, w2uflags, w1aflags, w2aflags (bigint)
            vals = [f"{pk}::bigint"]
            for i, v in enumerate(row[1:], start=1):
                if i <= 6:  # float fields (real)
                    if v is None:
                        vals.append("NULL::real")
                    else:
                        vals.append(f"{float(v)}::real")
                else:  # integer fields (bigint) - flags
                    if v is None:
                        vals.append("NULL::bigint")
                    else:
                        vals.append(f"{int(v)}::bigint")
            values_rows.append(f"({', '.join(vals)})")

        if values_rows:
            values_sql = ", ".join(values_rows)
            set_clauses = ", ".join([f"{col} = v.{col}" for col in columns])
            col_list = "pk, " + ", ".join(columns)

            sql = f"""
                UPDATE {table}
                SET {set_clauses}
                FROM (VALUES {values_sql}) AS v({col_list})
                WHERE {table}.pk = v.pk
            """
            with database.atomic():
                database.execute_sql(sql)
            updated += len(values_rows)

        queue.put(dict(advance=len(batch)))

    queue.put(Ellipsis)
    return updated




def migrate_glimpse_photometry(queue=None, **kwargs):
    """
    Migrate GLIMPSE photometry from the database using sdss_id to look up catalogids.

    Uses a single UPDATE...FROM query joining Source -> SDSS_ID_Flat -> CatalogToGLIMPSE -> GLIMPSE.
    """

    from astra.models.base import database
    from astra.models.source import Source

    if queue is None:
        queue = ProgressContext()

    from astra.migrations.sdss5db.catalogdb import GLIMPSE, CatalogToGLIMPSE, CatalogdbModel

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    queue.put(dict(total=None, description="Migrating GLIMPSE photometry"))

    with database.atomic():
        n = (
            Source
            .update(
                mag4_5=GLIMPSE.mag4_5,
                d4_5m=GLIMPSE.d4_5m,
                rms_f4_5=GLIMPSE.rms_f4_5,
                sqf_4_5=GLIMPSE.sqf_4_5,
                mf4_5=GLIMPSE.mf4_5,
                csf=GLIMPSE.csf,
            )
            .from_(SDSS_ID_Flat, CatalogToGLIMPSE, GLIMPSE)
            .where(
                (Source.sdss_id == SDSS_ID_Flat.sdss_id)
                & (SDSS_ID_Flat.catalogid == CatalogToGLIMPSE.catalogid)
                & (CatalogToGLIMPSE.target_id == GLIMPSE.pk)
                & (SDSS_ID_Flat.rank == 1)
                & Source.mag4_5.is_null()
                & Source.sdss_id.is_null(False)
            )
            .execute()
        )

    queue.put(Ellipsis)
    return n


def migrate_gaia_source_ids(
    limit: Optional[int] = None,
    batch_size: Optional[int] = 10_000,
    queue=None
):
    """
    Migrate Gaia source IDs for anything that we might have missed, using sdss_id to look up catalogids.

    This function uses crossmatch_flags to track whether cross-matches have been attempted:
    - flag_gaia_dr3_crossmatch_attempted = False: We haven't tried to cross-match yet
    - flag_gaia_dr3_crossmatch_attempted = True: We've attempted cross-match (may or may not have found a match)

    After processing:
    - gaia_dr3_source_id = NULL: No match found in Gaia
    - gaia_dr3_source_id = <value>: Found a Gaia source

    On subsequent runs, only sources where flag_gaia_dr*_crossmatch_attempted is False are processed.

    When conflicts are detected (same Gaia source ID already assigned to another source),
    the sources are merged using merge_sources().
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.migrations.source import merge_sources

    if queue is None:
        queue = ProgressContext()

    from astra.migrations.sdss5db.catalogdb import CatalogToGaia_DR3, CatalogToGaia_DR2, CatalogdbModel

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    # First, bulk update crossmatch flags for sources that already have Gaia IDs
    # This avoids expensive remote queries for sources we don't need to look up
    # BitField flags: DR3 = 2**0 = 1, DR2 = 2**1 = 2
    DR3_FLAG = 1
    DR2_FLAG = 2
    queue.update(total=3)
    with queue.subtask("Updating flags for sources with existing Gaia IDs", total=2) as flag_step:
        # Update DR3 flags for sources that have gaia_dr3_source_id but flag not set
        with database.atomic():
            n_dr3_flagged = (
                Source
                .update(crossmatch_flags=Source.crossmatch_flags.bin_or(DR3_FLAG))
                .where(
                    (Source.crossmatch_flags.bin_and(DR3_FLAG) == 0)
                    & Source.gaia_dr3_source_id.is_null(False)
                )
                .execute()
            )
        flag_step.update(advance=1)

        # Update DR2 flags for sources that have gaia_dr2_source_id but flag not set
        with database.atomic():
            n_dr2_flagged = (
                Source
                .update(crossmatch_flags=Source.crossmatch_flags.bin_or(DR2_FLAG))
                .where(
                    (Source.crossmatch_flags.bin_and(DR2_FLAG) == 0)
                    & Source.gaia_dr2_source_id.is_null(False)
                )
                .execute()
            )
        flag_step.update(advance=1)
    queue.update(advance=1)

    with queue.subtask("Migrating Gaia source IDs", total=2) as migrate_step:

        # let's assume no conflicts
        with database.atomic():
            (
                Source
                .update(
                    gaia_dr3_source_id=CatalogToGaia_DR3.target,
                    crossmatch_flags=Source.crossmatch_flags.bin_or(DR3_FLAG)
                )
                .from_(SDSS_ID_Flat, CatalogToGaia_DR3)
                .where(
                    (Source.sdss_id == SDSS_ID_Flat.sdss_id)
                &   (SDSS_ID_Flat.catalogid == CatalogToGaia_DR3.catalogid)
                &   Source.sdss_id.is_null(False)
                &   (Source.flag_gaia_dr3_crossmatch_attempted == False)
                &   (Source.gaia_dr3_source_id.is_null())
                &   (SDSS_ID_Flat.rank == 1)
                )
                .execute()
            )
        migrate_step.update(advance=1)
        queue.update(advance=1)

        # conflict for Gaia DR2 2195644907296409088 which we will ignore for now
        excluded_ids = set()
        while True:
            try:
                q = (
                    Source
                    .update(
                        gaia_dr2_source_id=CatalogToGaia_DR2.target,
                        crossmatch_flags=Source.crossmatch_flags.bin_or(DR2_FLAG)
                    )
                    .from_(SDSS_ID_Flat, CatalogToGaia_DR2)
                    .where(
                        (Source.sdss_id == SDSS_ID_Flat.sdss_id)
                    &   (SDSS_ID_Flat.catalogid == CatalogToGaia_DR2.catalogid)
                    &   Source.sdss_id.is_null(False)
                    &   (Source.flag_gaia_dr2_crossmatch_attempted == False)
                    &   (Source.gaia_dr2_source_id.is_null())
                    &   (SDSS_ID_Flat.rank == 1)
                    )
                )
                if excluded_ids:
                    q = q.where(~Source.gaia_dr2_source_id.in_(excluded_ids))

                with database.atomic():
                    q.execute()
                break

            except IntegrityError as e:
                match = re.search(r'gaia_dr2_source_id\)=\((\d+)\)', str(e))
                if match:
                    conflicting_id = int(match.group(1))
                    excluded_ids.add(conflicting_id)
                else:
                    raise  # different error, re-raise

        migrate_step.update(advance=1)
        queue.update(advance=1)

    queue.put(Ellipsis)
    return None



def migrate_gaia_dr3_astrometry_and_photometry(queue=None, **kwargs):
    """
    Migrate Gaia DR3 astrometry and photometry from catalogdb for any sources (`astra.models.Source`)
    that have a Gaia DR3 source identifier (`astra.models.Source.gaia_dr3_source_id`) but are missing Gaia
    photometry.
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.migrations.sdss5db.catalogdb import Gaia_DR3

    if queue is None:
        queue = ProgressContext()

    with database.atomic():
        n = (
            Source
            .update(
                g_mag=Gaia_DR3.phot_g_mean_mag,
                bp_mag=Gaia_DR3.phot_bp_mean_mag,
                rp_mag=Gaia_DR3.phot_rp_mean_mag,
                plx=Gaia_DR3.parallax,
                e_plx=Gaia_DR3.parallax_error,
                pmra=Gaia_DR3.pmra,
                e_pmra=Gaia_DR3.pmra_error,
                pmde=Gaia_DR3.pmdec,
                e_pmde=Gaia_DR3.pmdec_error,
                gaia_v_rad=Gaia_DR3.radial_velocity,
                gaia_e_v_rad=Gaia_DR3.radial_velocity_error,
            )
            .from_(Gaia_DR3)
            .where(
                (Source.gaia_dr3_source_id == Gaia_DR3.source_id)
            &   (
                    Source.g_mag.is_null()
                |   Source.bp_mag.is_null()
                |   Source.rp_mag.is_null()
                |   Source.pmra.is_null()
                |   Source.pmde.is_null()
                )
            &   Source.gaia_dr3_source_id.is_null(False)
            )
            .execute()
        )
    queue.put(Ellipsis)
    return None


def migrate_sdss4_apogee_id(
    limit: Optional[int] = None,
    batch_size: Optional[int] = 10_000,
    queue=None
):
    """
    Migrate sdss4_apogee_id from catalogdb.allstar_dr17_synspec_rev1 for sources
    that have a match in catalog_to_allstar_dr17_synspec_rev1.

    This function starts from the allstar catalog (~500k entries) rather than
    iterating through all sources (~6M), making it much faster.
    """
    from astra.models.base import database
    from astra.models.source import Source

    if queue is None:
        queue = ProgressContext()

    from peewee import BigIntegerField, TextField
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel

    class CatalogToAllstarDR17SynspecRev1(CatalogdbModel):
        catalogid = BigIntegerField()
        target_id = TextField()  # This is the apstar_id (e.g., "apogee.apo1m.stars.Bestars.2M00431825+6154402")

        class Meta:
            table_name = "catalog_to_allstar_dr17_synspec_rev1"
            primary_key = False

    class AllstarDR17SynspecRev1(CatalogdbModel):
        apogee_id = TextField()  # The actual APOGEE ID (e.g., "2M00431825+6154402")
        apstar_id = TextField()  # Matches target_id in catalog_to table

        class Meta:
            table_name = "allstar_dr17_synspec_rev1"
            primary_key = False

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    table = f"{Source._meta.schema}.{Source._meta.table_name}"

    # Phase 1: Get all apogee_id -> catalogid mappings from the allstar catalog
    # This is the small dataset (~500k) we should iterate through
    with queue.subtask("Fetching apogee_ids from allstar catalog", total=None) as fetch_progress:
        q_allstar = (
            CatalogToAllstarDR17SynspecRev1
            .select(
                CatalogToAllstarDR17SynspecRev1.catalogid,
                AllstarDR17SynspecRev1.apogee_id
            )
            .join(
                AllstarDR17SynspecRev1,
                on=(CatalogToAllstarDR17SynspecRev1.target_id == AllstarDR17SynspecRev1.apstar_id)
            )
            .tuples()
        )
        # catalogid -> apogee_id
        allstar_data = list(q_allstar)
        fetch_progress.update(total=len(allstar_data), completed=len(allstar_data))

    if not allstar_data:
        return 0

    # Phase 2: Get catalogid -> sdss_id mapping for all catalogids in allstar
    catalogids = [row[0] for row in allstar_data]
    apogee_id_by_catalogid = {row[0]: row[1] for row in allstar_data}

    with queue.subtask("Fetching sdss_ids for catalogids", total=len(catalogids)) as fetch_progress:
        catalogid_to_sdss_id = {}
        for batch in chunked(catalogids, batch_size):
            for row in (
                SDSS_ID_Flat
                .select(SDSS_ID_Flat.catalogid, SDSS_ID_Flat.sdss_id)
                .where(
                    SDSS_ID_Flat.catalogid.in_(batch)
                &   (SDSS_ID_Flat.rank == 1)
                )
                .tuples()
            ):
                catalogid_to_sdss_id[row[0]] = row[1]
            fetch_progress.update(advance=len(batch))

    if not catalogid_to_sdss_id:
        return 0

    # Phase 3: Get sdss_id -> pk mapping for sources that need updating
    sdss_ids = list(set(catalogid_to_sdss_id.values()))

    with queue.subtask("Fetching source PKs", total=len(sdss_ids)) as fetch_progress:
        source_pks = {}  # sdss_id -> pk
        for batch in chunked(sdss_ids, batch_size):
            for row in (
                Source
                .select(Source.pk, Source.sdss_id)
                .where(
                    Source.sdss_id.in_(batch)
                    & Source.sdss4_apogee_id.is_null()  # Only sources that need updating
                )
                .tuples()
            ):
                source_pks[row[1]] = row[0]
            fetch_progress.update(advance=len(batch))

    if not source_pks:
        return 0

    # Track apogee_ids we've already assigned (unique constraint)
    # Pre-load existing apogee_ids from the database
    with queue.subtask("Loading existing apogee_ids", total=None) as fetch_progress:
        existing = Source.select(Source.pk, Source.sdss4_apogee_id).where(Source.sdss4_apogee_id.is_null(False)).tuples()
        assigned_apogee_ids = {row[1]: row[0] for row in existing}  # apogee_id -> pk
        fetch_progress.update(total=len(assigned_apogee_ids), completed=len(assigned_apogee_ids))

    # Phase 4: Build the update list
    conflicts = []
    updates = []  # (pk, apogee_id)

    for catalogid, apogee_id in allstar_data:
        if apogee_id is None:
            continue
        sdss_id = catalogid_to_sdss_id.get(catalogid)
        if sdss_id is None:
            continue
        pk = source_pks.get(sdss_id)
        if pk is None:
            continue

        # Check for unique constraint conflict
        if apogee_id in assigned_apogee_ids:
            existing_pk = assigned_apogee_ids[apogee_id]
            # Same source reached via multiple catalogids is not a conflict
            if existing_pk != pk:
                conflicts.append((pk, existing_pk))
            continue

        assigned_apogee_ids[apogee_id] = pk
        updates.append((pk, apogee_id))

    # Phase 5: Apply updates
    updated = 0
    with queue.subtask("Updating sdss4_apogee_id", total=len(updates)) as update_progress:
        for batch in chunked(updates, batch_size):
            values_rows = [f"({pk}::bigint, '{apogee_id}'::text)" for pk, apogee_id in batch]
            if values_rows:
                values_sql = ", ".join(values_rows)
                sql = f"""
                    UPDATE {table}
                    SET sdss4_apogee_id = v.sdss4_apogee_id
                    FROM (VALUES {values_sql}) AS v(pk, sdss4_apogee_id)
                    WHERE {table}.pk = v.pk
                """
                with database.atomic():
                    database.execute_sql(sql)
                updated += len(batch)
            update_progress.update(advance=len(batch))

    # Phase 6: Merge conflicting sources (if any)
    if conflicts:
        from astra.migrations.source import merge_sources

        with queue.subtask(f"Merging {len(conflicts)} duplicate sources", total=len(conflicts)) as merge_progress:
            for old_pk, new_pk in conflicts:
                with database.atomic():
                    merge_sources(keep_pk=new_pk, remove_pk=old_pk, database=database)
                merge_progress.update(advance=1)

    return updated
