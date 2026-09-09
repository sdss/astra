import numpy as np
import datetime
from astropy.time import Time
import astropy.coordinates as coord
import astropy.units as u
from scipy.signal import argrelmin
from peewee import chunked, fn, JOIN, EXCLUDED
import concurrent.futures

import pickle
from astra.utils import flatten, expand_path, log

from astra.migrations.utils import ProgressContext
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import Tuple, Optional

from astra import __version__


von = lambda v: v or np.nan

def update_sdss_id_related_fields(queue=None, batch_size=10_000):
    """
    Update sdss_id and related fields (version_id, n_associated, catalogid21/25/31)
    on Source records that have catalogids but are missing these fields.

    Optimized to avoid slow cross-database JOINs against 300M+ row tables.
    """
    from astra.migrations.utils import ProgressContext
    if queue is None:
        queue = ProgressContext()

    from astra.models.source import Source
    from astra.models.base import database
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel, Catalog

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"

    table = f"{Source._meta.schema}.{Source._meta.table_name}"
    n_updated_sdss_id = 0
    n_updated_misc = 0
    n_update_catalogids = 0
    n_updated_leads = 0
    n_merged = 0

    # =========================================================================
    # Step 1: Update missing sdss_id values
    # =========================================================================
    with queue.subtask("Finding sources needing sdss_id", total=None) as step1:
        sources_needing_sdss_id = list(
            Source
            .select(Source.pk, Source.catalogid, Source.catalogid21, Source.catalogid25, Source.catalogid31)
            .where(
                Source.sdss_id.is_null()
                & (
                    Source.catalogid21.is_null(False)
                    | Source.catalogid25.is_null(False)
                    | Source.catalogid31.is_null(False)
                    | Source.catalogid.is_null(False)
                )
            )
            .tuples()
        )
        step1.update(total=len(sources_needing_sdss_id), completed=len(sources_needing_sdss_id))

    if sources_needing_sdss_id:
        # Collect all unique catalogids we need to look up
        all_catalogids = set()
        for row in sources_needing_sdss_id:
            for cid in row[1:]:  # catalogid, catalogid21, catalogid25, catalogid31
                if cid is not None:
                    all_catalogids.add(cid)

        # Fetch sdss_id mappings from SDSS_ID_Flat for these catalogids (in batches)
        with queue.subtask("Fetching sdss_id mappings", total=len(all_catalogids)) as fetch_progress:
            catalogid_to_sdss_id = {}
            for batch in chunked(list(all_catalogids), batch_size):
                for row in SDSS_ID_Flat.select(SDSS_ID_Flat.catalogid, SDSS_ID_Flat.sdss_id).where(
                    (SDSS_ID_Flat.catalogid.in_(batch)) & (SDSS_ID_Flat.rank == 1)
                ).tuples():
                    catalogid_to_sdss_id[row[0]] = row[1]
                fetch_progress.update(advance=len(batch))

        # Build updates
        updates = []  # (pk, sdss_id)
        for pk, catalogid, catalogid21, catalogid25, catalogid31 in sources_needing_sdss_id:
            # Try each catalogid in order of preference
            for cid in (catalogid, catalogid31, catalogid25, catalogid21):
                if cid is not None and cid in catalogid_to_sdss_id:
                    updates.append((pk, catalogid_to_sdss_id[cid]))
                    break

        # Bulk update using raw SQL
        with queue.subtask(f"Updating sdss_id for {len(updates)} sources", total=len(updates)) as update_progress:
            for batch in chunked(updates, batch_size):
                values_rows = [f"({pk}, {sdss_id})" for pk, sdss_id in batch]
                values_sql = ", ".join(values_rows)
                sql = f"""
                    UPDATE {table}
                    SET sdss_id = v.sdss_id, modified = NOW()
                    FROM (VALUES {values_sql}) AS v(pk, sdss_id)
                    WHERE {table}.pk = v.pk
                """
                with database.atomic():
                    database.execute_sql(sql)
                n_updated_sdss_id += len(batch)
                update_progress.update(advance=len(batch))

    # =========================================================================
    # Step 2: Update version_id, n_associated, catalogid21/25/31
    # =========================================================================
    with queue.subtask("Finding sources needing version_id/catalogids", total=None) as step2:
        sources_needing_misc = list(
            Source
            .select(Source.pk, Source.sdss_id)
            .where(
                Source.sdss_id.is_null(False)
                & (
                    Source.version_id.is_null()
                    | Source.catalogid21.is_null()
                    | Source.catalogid25.is_null()
                    | Source.catalogid31.is_null()
                    | (Source.n_associated == -1)
                )
            )
            .tuples()
        )
        step2.update(total=len(sources_needing_misc), completed=len(sources_needing_misc))

    if sources_needing_misc:
        sdss_ids = [row[1] for row in sources_needing_misc]
        pk_by_sdss_id = {row[1]: row[0] for row in sources_needing_misc}

        # Fetch from SDSS_ID_Flat (version_id, n_associated, catalogid)
        with queue.subtask(f"Fetching flat metadata for {len(sdss_ids)} sources", total=len(sdss_ids)) as flat_progress:
            flat_data = {}  # sdss_id -> {version_id, n_associated, catalogid}
            for batch in chunked(sdss_ids, batch_size):
                q = (
                    SDSS_ID_Flat
                    .select(
                        SDSS_ID_Flat.sdss_id,
                        SDSS_ID_Flat.version_id,
                        SDSS_ID_Flat.n_associated,
                        SDSS_ID_Flat.catalogid,
                    )
                    .where((SDSS_ID_Flat.sdss_id.in_(batch)) & (SDSS_ID_Flat.rank == 1))
                    .order_by(SDSS_ID_Flat.sdss_id, SDSS_ID_Flat.version_id.desc())
                    .tuples()
                )
                for sdss_id, version_id, n_associated, catalogid in q:
                    if sdss_id not in flat_data:  # Keep first (highest version_id due to ORDER BY)
                        flat_data[sdss_id] = {
                            'version_id': version_id,
                            'n_associated': n_associated,
                            'catalogid': catalogid,
                        }
                flat_progress.update(advance=len(batch))

        # Fetch from SDSS_ID_Stacked (catalogid21, catalogid25, catalogid31)
        with queue.subtask(f"Fetching stacked catalogids for {len(sdss_ids)} sources", total=len(sdss_ids)) as stacked_progress:
            stacked_data = {}  # sdss_id -> {catalogid21, catalogid25, catalogid31}
            for batch in chunked(sdss_ids, batch_size):
                q = (
                    SDSS_ID_Stacked
                    .select(
                        SDSS_ID_Stacked.sdss_id,
                        SDSS_ID_Stacked.catalogid21,
                        SDSS_ID_Stacked.catalogid25,
                        SDSS_ID_Stacked.catalogid31,
                    )
                    .where(SDSS_ID_Stacked.sdss_id.in_(batch))
                    .tuples()
                )
                for sdss_id, catalogid21, catalogid25, catalogid31 in q:
                    stacked_data[sdss_id] = {
                        'catalogid21': catalogid21,
                        'catalogid25': catalogid25,
                        'catalogid31': catalogid31,
                    }
                stacked_progress.update(advance=len(batch))

        # Build and execute updates
        updates = []
        for sdss_id, pk in pk_by_sdss_id.items():
            flat = flat_data.get(sdss_id, {})
            stacked = stacked_data.get(sdss_id, {})
            if flat or stacked:
                updates.append((
                    pk,
                    flat.get('version_id'),
                    flat.get('n_associated'),
                    flat.get('catalogid'),
                    stacked.get('catalogid21'),
                    stacked.get('catalogid25'),
                    stacked.get('catalogid31'),
                ))

        with queue.subtask(f"Updating version_id/catalogids for {len(updates)} sources", total=len(updates)) as update_progress:
            for batch in chunked(updates, batch_size):
                def fmt(v):
                    return 'NULL' if v is None else str(v)

                values_rows = [
                    f"({pk}, {fmt(version_id)}, {fmt(n_associated)}, {fmt(catalogid)}::bigint, {fmt(c21)}::bigint, {fmt(c25)}::bigint, {fmt(c31)}::bigint)"
                    for pk, version_id, n_associated, catalogid, c21, c25, c31 in batch
                ]
                values_sql = ", ".join(values_rows)
                sql = f"""
                    UPDATE {table}
                    SET version_id = v.version_id,
                        n_associated = v.n_associated,
                        catalogid = COALESCE(v.catalogid, {table}.catalogid),
                        catalogid21 = COALESCE(v.catalogid21, {table}.catalogid21),
                        catalogid25 = COALESCE(v.catalogid25, {table}.catalogid25),
                        catalogid31 = COALESCE(v.catalogid31, {table}.catalogid31),
                        modified = NOW()
                    FROM (VALUES {values_sql}) AS v(pk, version_id, n_associated, catalogid, catalogid21, catalogid25, catalogid31)
                    WHERE {table}.pk = v.pk
                """
                with database.atomic():
                    database.execute_sql(sql)
                n_updated_misc += len(batch)
                update_progress.update(advance=len(batch))

    # =========================================================================
    # Step 3: Update missing catalogid from catalogid31/25/21
    # =========================================================================
    with queue.subtask("Updating missing catalogid from catalogid31/25/21", total=None) as step3:
        with database.atomic():
            n_update_catalogids = (
                Source
                .update(
                    catalogid=fn.COALESCE(
                        Source.catalogid31,
                        Source.catalogid25,
                        Source.catalogid21
                    ),
                    modified=fn.NOW(),
                )
                .where(
                    Source.catalogid.is_null()
                    & Source.sdss_id.is_null(False)
                    & Source.catalogid31.is_null(False)
                )
                .execute()
            )
        step3.update(total=n_update_catalogids, completed=n_update_catalogids)

    # =========================================================================
    # Step 4: Update catalog lead names
    # =========================================================================
    with queue.subtask("Finding sources needing lead update", total=None) as step4:
        sources_needing_lead = list(
            Source
            .select(Source.pk, Source.catalogid, Source.version_id)
            .where(
                Source.lead.is_null()
                & Source.catalogid.is_null(False)
                & Source.version_id.is_null(False)
            )
            .tuples()
        )
        step4.update(total=len(sources_needing_lead), completed=len(sources_needing_lead))

    if sources_needing_lead:
        # Get unique (catalogid, version_id) pairs
        cv_pairs = list(set((row[1], row[2]) for row in sources_needing_lead))
        pk_by_cv = {}
        for pk, catalogid, version_id in sources_needing_lead:
            pk_by_cv[(catalogid, version_id)] = pk

        # Fetch leads from Catalog
        with queue.subtask(f"Fetching leads for {len(cv_pairs)} catalogids", total=len(cv_pairs)) as fetch_progress:
            lead_data = {}
            for batch in chunked(cv_pairs, batch_size):
                catalogids = [cv[0] for cv in batch]
                q = (
                    Catalog
                    .select(Catalog.catalogid, Catalog.version_id, Catalog.lead)
                    .where(Catalog.catalogid.in_(catalogids))
                    .tuples()
                )
                for catalogid, version_id, lead in q:
                    lead_data[(catalogid, version_id)] = lead
                fetch_progress.update(advance=len(batch))

        # Build updates
        updates = []
        for (catalogid, version_id), pk in pk_by_cv.items():
            lead = lead_data.get((catalogid, version_id))
            if lead is not None:
                updates.append((pk, lead))

        with queue.subtask(f"Updating lead for {len(updates)} sources", total=len(updates)) as update_progress:
            for batch in chunked(updates, batch_size):
                values_rows = [f"({pk}, '{lead}')" for pk, lead in batch]
                values_sql = ", ".join(values_rows)
                sql = f"""
                    UPDATE {table}
                    SET lead = v.lead, modified = NOW()
                    FROM (VALUES {values_sql}) AS v(pk, lead)
                    WHERE {table}.pk = v.pk
                """
                with database.atomic():
                    database.execute_sql(sql)
                n_updated_leads += len(batch)
                update_progress.update(advance=len(batch))

    # =========================================================================
    # Step 5: Find and merge duplicate sources via sdss_id_stacked
    # =========================================================================
    # Sources with different sdss_ids but whose catalogids map to the same
    # row in sdss_id_stacked are actually the same physical source and should
    # be merged. We keep the source with the lower sdss_id.
    from astra.migrations.source import merge_sources

    stacked_table = f"{SDSS_ID_Stacked._meta.schema}.{SDSS_ID_Stacked._meta.table_name}"

    with queue.subtask("Finding duplicate sources via sdss_id_stacked", total=None) as step5:
        duplicate_query = f"""
            WITH source_to_stacked AS (
                -- Map each source to the sdss_id from sdss_id_stacked via any matching catalogid
                SELECT DISTINCT
                    s.pk,
                    s.sdss_id,
                    ss.sdss_id as stacked_sdss_id
                FROM {table} s
                JOIN {stacked_table} ss ON (
                    (s.catalogid21 IS NOT NULL AND s.catalogid21 = ss.catalogid21)
                    OR (s.catalogid25 IS NOT NULL AND s.catalogid25 = ss.catalogid25)
                    OR (s.catalogid31 IS NOT NULL AND s.catalogid31 = ss.catalogid31)
                )
                WHERE s.sdss_id IS NOT NULL
            )
            -- Find pairs that share the same stacked_sdss_id but have different sdss_ids
            SELECT
                a.pk as keep_pk,
                a.sdss_id as keep_sdss_id,
                b.pk as remove_pk,
                b.sdss_id as remove_sdss_id
            FROM source_to_stacked a
            JOIN source_to_stacked b ON a.stacked_sdss_id = b.stacked_sdss_id
            WHERE a.sdss_id < b.sdss_id  -- keep lower sdss_id, also avoids duplicates/self-joins
        """
        duplicate_pairs = list(database.execute_sql(duplicate_query).fetchall())
        step5.update(total=len(duplicate_pairs), completed=len(duplicate_pairs))

    if duplicate_pairs:
        with queue.subtask(f"Merging {len(duplicate_pairs)} duplicate source pairs", total=len(duplicate_pairs)) as merge_progress:
            for keep_pk, keep_sdss_id, remove_pk, remove_sdss_id in duplicate_pairs:
                try:
                    merge_sources(keep_pk, remove_pk, database=database)
                    n_merged += 1
                except Exception as e:
                    from astra.utils import log
                    log.warning(f"Failed to merge sources {keep_pk} (sdss_id={keep_sdss_id}) <- {remove_pk} (sdss_id={remove_sdss_id}): {e}")
                merge_progress.update(advance=1)

    queue.put(Ellipsis)
    return (n_updated_sdss_id, n_update_catalogids, n_updated_misc, n_updated_leads, n_merged)


def compute_w1mag_and_w2mag(
    limit=None,
    batch_size=1000,
    queue=None
):
    from astra.models.base import database
    from astra.models.source import Source
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.boss import BossVisitSpectrum
    if queue is None:
        queue = ProgressContext()

    q = (
        Source
        .select(
            Source.pk,
            Source.w1_flux,
            Source.w1_dflux,
            Source.w2_flux,
            Source.w2_dflux,
        )
        .where(
            (Source.w1_flux.is_null(False) & Source.w1_mag.is_null(True))
        |   (Source.w2_flux.is_null(False) & Source.w2_mag.is_null(True))
        )
        .limit(limit)
    )
    n_updated = 0
    queue.put(dict(total=None, description="Computing W1/W2 magnitudes"))
    now = datetime.datetime.now()

    for batch in chunked(q.iterator(), batch_size):

        for source in batch:
            # See https://catalog.unwise.me/catalogs.html (Flux Scale) for justification of 32 mmag offset in W2, and 4 mmag offset in W1
            source.w1_mag = -2.5 * np.log10(von(source.w1_flux)) + 22.5 - 4 * 1e-3 # Vega
            source.e_w1_mag = (2.5 / np.log(10)) * von(source.w1_dflux) / von(source.w1_flux)
            source.w2_mag = -2.5 * np.log10(von(source.w2_flux)) + 22.5 - 32 * 1e-3 # Vega
            source.e_w2_mag = (2.5 / np.log(10)) * von(source.w2_dflux) / von(source.w2_flux)
            source.modified = now

        with database.atomic():
            n_updated += Source.bulk_update(
                batch,
                fields=[
                    Source.w1_mag,
                    Source.e_w1_mag,
                    Source.w2_mag,
                    Source.e_w2_mag,
                    Source.modified,
                ]
            )
        queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)

    return n_updated



def update_galactic_coordinates(queue=None, **kwargs):
    """
    Compute galactic coordinates (l, b) from equatorial coordinates (ra, dec).

    Uses pure SQL for the ICRS to Galactic transformation, which is much faster
    than the previous Python/astropy approach.

    The transformation uses the IAU definition of the Galactic coordinate system:
    - North Galactic Pole (J2000): RA = 192.85948°, Dec = 27.12825°
    - Galactic longitude of ascending node: 32.93192°
    """
    from astra.models.base import database
    from astra.models.source import Source

    if queue is None:
        queue = ProgressContext()

    table = f"{Source._meta.schema}.{Source._meta.table_name}"

    # IAU Galactic coordinate system parameters (J2000)
    # North Galactic Pole: RA = 192.85948°, Dec = 27.12825°
    # l_NCP (galactic longitude of north celestial pole) = 122.93192°
    sql = f"""
        UPDATE {table}
        SET
            l = 122.93192 - DEGREES(
                ATAN2(
                    SIN(RADIANS(ra - 192.85948)),
                    TAN(RADIANS(dec)) * COS(RADIANS(27.12825))
                    - SIN(RADIANS(27.12825)) * COS(RADIANS(ra - 192.85948))
                )
            ),
            b = DEGREES(
                ASIN(
                    SIN(RADIANS(dec)) * SIN(RADIANS(27.12825))
                    + COS(RADIANS(dec)) * COS(RADIANS(27.12825)) * COS(RADIANS(ra - 192.85948))
                )
            ),
            modified = NOW()
        WHERE ra IS NOT NULL AND l IS NULL
    """

    with database.atomic():
        cursor = database.execute_sql(sql)
        n_updated = cursor.rowcount

        # Normalize l to [0, 360) range
        database.execute_sql(f"""
            UPDATE {table}
            SET l = l + 360
            WHERE l < 0
        """)
        database.execute_sql(f"""
            UPDATE {table}
            SET l = l - 360
            WHERE l >= 360
        """)

    queue.put(Ellipsis)
    return n_updated


def fix_unsigned_apogee_flags(queue):

    from astra.models.base import database
    from astra.models.source import Source
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.boss import BossVisitSpectrum

    if queue is None:
        queue = ProgressContext()

    delta = 2**32 - 2**31
    field_names = [
        "sdss4_apogee_target1_flags",
        "sdss4_apogee_target2_flags",
        "sdss4_apogee2_target1_flags",
        "sdss4_apogee2_target2_flags",
        "sdss4_apogee2_target3_flags",
        "sdss4_apogee_member_flags",
        "sdss4_apogee_extra_target_flags"
    ]
    updated = {}
    queue.put(dict(total=len(field_names)))
    for field_name in field_names:
        field = getattr(Source, field_name)
        kwds = { field_name: field + delta, 'modified': fn.NOW() }
        with database.atomic():
            updated[field_name] = (
                Source
                .update(**kwds)
                .where(field < 0)
                .execute()
            )
        queue.put(dict(advance=1))
    queue.put(Ellipsis)
    return updated


def compute_gonzalez_hernandez_irfm_effective_temperatures_from_vmk(
    model,
    logg_field,
    fe_h_field,
    dwarf_giant_logg_split=3.8,
    batch_size=10_000
):
    '''
    # These are from Table 2 of https://arxiv.org/pdf/0901.3034.pdf
    A_dwarf = np.array([2.3522, -1.8817, 0.6229, -0.0745, 0.0371, -0.0990, -0.0052])
    A_giant = np.array([2.1304, -1.5438, 0.4562, -0.0483, 0.0132, 0.0456, -0.0026])

    dwarf_colour_range = [1, 3]
    dwarf_fe_h_range = [-3.5, 0.3]

    giant_colour_range = [0.7, 3.8]
    giant_fe_h_range = [-4.0, 0.1]
    '''
    from astra.models.base import database
    from astra.models.source import Source
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.boss import BossVisitSpectrum

    B_dwarf = np.array([0.5201, 0.2511, -0.0118, -0.0186, 0.0408, 0.0033])
    B_giant = np.array([0.5293, 0.2489, -0.0119, -0.0042, 0.0135, 0.0010])

    #dwarf_colour_range = [0.1, 0.8] ### WRONG
    dwarf_colour_range = [0.7, 3.0]
    dwarf_fe_h_range = [-3.5, 0.5]

    giant_colour_range = [1.1, 3.4]
    giant_fe_h_range = [-4, 0.2]

    q = (
        model
        .select(
            model,
            Source,
        )
        .join(Source, on=(model.source_pk == Source.pk), attr="_source")
        .where(
            (model.v_astra == __version__)
        &   logg_field.is_null(False)
        &   fe_h_field.is_null(False)
        &   Source.v_jkc_mag.is_null(False)
        &   Source.k_mag.is_null(False)
        )
    )

    n_updated, batch = (0, [])
    for row in tqdm(q.iterator()):
        X = (row._source.v_jkc_mag or np.nan) - (row._source.k_mag or np.nan)
        fe_h = getattr(row, fe_h_field.name) or np.nan
        logg = getattr(row, logg_field.name) or np.nan

        if logg >= dwarf_giant_logg_split:
            # dwarf
            B = B_dwarf
            valid_v_k = dwarf_colour_range
            valid_fe_h = dwarf_fe_h_range
            row.flag_as_dwarf_for_irfm_teff = True
        else:
            # giant
            B = B_giant
            valid_v_k = giant_colour_range
            valid_fe_h = giant_fe_h_range
            row.flag_as_giant_for_irfm_teff = True

        theta = np.sum(B * np.array([1, X, X**2, X*fe_h, fe_h, fe_h**2]))

        row.irfm_teff = 5040/theta
        row.flag_out_of_v_k_bounds = not (valid_v_k[0] <= X <= valid_v_k[1])
        row.flag_out_of_fe_h_bounds = not (valid_fe_h[0] <= fe_h <= valid_fe_h[1])
        row.flag_extrapolated_v_mag = (row._source.v_jkc_mag_flag == 0)
        row.flag_poor_quality_k_mag = (
            (row._source.ph_qual is None)
        or  (row._source.ph_qual[-1] != "A")
        or  (row._source.e_k_mag > 0.1)
        )
        row.flag_ebv_used_is_upper_limit = row._source.flag_ebv_upper_limit
        batch.append(row)

        if len(batch) >= batch_size:
            with database.atomic():
                model.bulk_update(
                    batch,
                    fields=[
                        model.irfm_teff,
                        model.irfm_teff_flags,
                    ]
                )
            n_updated += batch_size
            batch = []

    if len(batch) > 0:
        with database.atomic():
            model.bulk_update(
                batch,
                fields=[
                    model.irfm_teff,
                    model.irfm_teff_flags,
                ]
            )
        n_updated += len(batch)

    return n_updated



def compute_casagrande_irfm_effective_temperatures(
    model,
    fe_h_field,
    batch_size=10_000
):
    """
    Compute IRFM effective temperatures using the V-Ks colour and the Casagrande et al. (2010) scale.
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.boss import BossVisitSpectrum

    valid_v_k = [0.78, 3.15]

    #https://www.aanda.org/articles/aa/full_html/2010/04/aa13204-09/aa13204-09.html
    a0, a1, a2, a3, a4, a5 = (
        +0.5057,
        +0.2600,
        -0.0146,
        -0.0131,
        +0.0288,
        +0.0016
    )

    q = (
        model
        .select(
            model,
            Source,
        )
        .join(Source, on=(model.source_pk == Source.pk), attr="_source")
        .where(
            Source.v_jkc_mag.is_null(False)
        &   Source.k_mag.is_null(False)
        )
    )

    n_updated, batch = (0, [])
    for row in tqdm(q.iterator()):

        X = (row._source.v_jkc_mag or np.nan) - (row._source.k_mag or np.nan)
        fe_h = getattr(row, fe_h_field.name) or np.nan
        theta = a0 + a1 * X + a2*X**2 + a3*X*fe_h + a4*fe_h + a5*fe_h**2

        row.irfm_teff = 5040/theta
        row.e_irfm_teff = np.nan

        #_source.e_irfm_teff = 5040 * np.sqrt(
        #    (a1 + 2*a2*X + a3*fe_h) ** 2 * _source.e_v_jkc_mag**2
        #+   (a1 + 2*a2*X + a3*fe_h) ** 2 * _source.e_k_mag**2
        #+   (a3*X + a4 + 2*a5*fe_h) ** 2 * _source.e_fe_h**2
        #)


        row.flag_out_of_v_k_bounds = not (valid_v_k[0] <= X <= valid_v_k[1])
        row.flag_extrapolated_v_mag = (row._source.v_jkc_mag_flag == 0)
        row.flag_poor_quality_k_mag = (row._source.ph_qual is None) or (row._source.ph_qual[-1] != "A")
        row.flag_ebv_used_is_upper_limit = row._source.flag_ebv_upper_limit
        batch.append(row)

        if len(batch) >= batch_size:
            with database.atomic():
                model.bulk_update(
                    batch,
                    fields=[
                        model.irfm_teff,
                        model.e_irfm_teff,
                        model.irfm_teff_flags,
                    ]
                )
            n_updated += batch_size
            batch = []

    if len(batch) > 0:
        with database.atomic():
            model.bulk_update(
                batch,
                fields=[
                    model.irfm_teff,
                    model.e_irfm_teff,
                    model.irfm_teff_flags,
                ]
            )
        n_updated += len(batch)

    return n_updated



def update_visit_spectra_counts(
    apogee_where=None,
    boss_where=None,
    batch_size=10_000,
    queue=None,
    k=1000,
    incremental=False
):
    """
    Update source visit counts using batched SQL UPDATE statements.

    Only processes sources where n_apogee_visits or n_boss_visits is NULL.
    Batches both the computation and update phases for progress visibility.
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.boss import BossVisitSpectrum

    if queue is None:
        queue = ProgressContext()

    schema = Source._meta.schema

    # Update APOGEE counts - only for sources with NULL n_apogee_visits
    # First get the source_pks that need updating
    if ApogeeVisitSpectrum.select().count() > 0:

        queue.put(dict(total=None, completed=0, description="Finding sources needing APOGEE counts"))

        source_pks_needing_apogee = list(
            Source
            .select(Source.pk)
            .where(Source.n_apogee_visits.is_null())
            .tuples()
        )
        source_pks_needing_apogee = [row[0] for row in source_pks_needing_apogee]

        if source_pks_needing_apogee:
            total = len(source_pks_needing_apogee)
            queue.put(dict(total=total, completed=0, description=f"Updating APOGEE counts for {total} sources"))

            for chunk in chunked(source_pks_needing_apogee, batch_size):
                pks_str = ",".join(str(pk) for pk in chunk)

                sql = f"""
                    WITH counts AS (
                        SELECT
                            source_pk,
                            COUNT(*) as n_apogee_visits,
                            MIN(mjd) as apogee_min_mjd,
                            MAX(mjd) as apogee_max_mjd
                        FROM {schema}.apogee_visit_spectrum
                        WHERE source_pk IN ({pks_str})
                        GROUP BY source_pk
                    )
                    UPDATE {schema}.source
                    SET
                        n_apogee_visits = counts.n_apogee_visits,
                        apogee_min_mjd = counts.apogee_min_mjd,
                        apogee_max_mjd = counts.apogee_max_mjd,
                        modified = NOW()
                    FROM counts
                    WHERE {schema}.source.pk = counts.source_pk
                """
                with database.atomic():
                    database.execute_sql(sql)

                # Also set n_apogee_visits=0 for sources with no spectra
                pks_with_spectra_result = database.execute_sql(f"""
                    SELECT DISTINCT source_pk FROM {schema}.apogee_visit_spectrum
                    WHERE source_pk IN ({pks_str})
                """).fetchall()
                pks_with_spectra = {row[0] for row in pks_with_spectra_result}
                pks_without_spectra = [pk for pk in chunk if pk not in pks_with_spectra]

                if pks_without_spectra:
                    pks_no_spectra_str = ",".join(str(pk) for pk in pks_without_spectra)
                    with database.atomic():
                        database.execute_sql(f"""
                            UPDATE {schema}.source
                            SET n_apogee_visits = 0, modified = NOW()
                            WHERE pk IN ({pks_no_spectra_str}) AND n_apogee_visits IS NULL
                        """)

                queue.put(dict(advance=len(chunk)))

    if BossVisitSpectrum.select().count() > 0:

        # Update BOSS counts - only for sources with NULL n_boss_visits
        queue.put(dict(total=None, completed=0, description="Finding sources needing BOSS counts"))

        source_pks_needing_boss = list(
            Source
            .select(Source.pk)
            .where(Source.n_boss_visits.is_null())
            .tuples()
        )
        source_pks_needing_boss = [row[0] for row in source_pks_needing_boss]

        if source_pks_needing_boss:
            total = len(source_pks_needing_boss)
            queue.put(dict(total=total, completed=0, description=f"Updating BOSS counts for {total} sources"))

            for chunk in chunked(source_pks_needing_boss, batch_size):
                pks_str = ",".join(str(pk) for pk in chunk)

                sql = f"""
                    WITH counts AS (
                        SELECT
                            source_pk,
                            COUNT(*) as n_boss_visits,
                            MIN(mjd) as boss_min_mjd,
                            MAX(mjd) as boss_max_mjd
                        FROM {schema}.boss_visit_spectrum
                        WHERE source_pk IN ({pks_str})
                        GROUP BY source_pk
                    )
                    UPDATE {schema}.source
                    SET
                        n_boss_visits = counts.n_boss_visits,
                        boss_min_mjd = counts.boss_min_mjd,
                        boss_max_mjd = counts.boss_max_mjd,
                        modified = NOW()
                    FROM counts
                    WHERE {schema}.source.pk = counts.source_pk
                """
                with database.atomic():
                    database.execute_sql(sql)

                # Also set n_boss_visits=0 for sources with no spectra
                pks_with_spectra_result = database.execute_sql(f"""
                    SELECT DISTINCT source_pk FROM {schema}.boss_visit_spectrum
                    WHERE source_pk IN ({pks_str})
                """).fetchall()
                pks_with_spectra = {row[0] for row in pks_with_spectra_result}
                pks_without_spectra = [pk for pk in chunk if pk not in pks_with_spectra]

                if pks_without_spectra:
                    pks_no_spectra_str = ",".join(str(pk) for pk in pks_without_spectra)
                    with database.atomic():
                        database.execute_sql(f"""
                            UPDATE {schema}.source
                            SET n_boss_visits = 0, modified = NOW()
                            WHERE pk IN ({pks_no_spectra_str}) AND n_boss_visits IS NULL
                        """)

                queue.put(dict(advance=len(chunk)))

    queue.put(Ellipsis)
    return None


def _compute_n_neighborhood_batch(args):
    """
    Worker function for parallel n_neighborhood computation.
    Must be at module level for multiprocessing.
    """
    source_ids, table, radius_deg, brightness = args

    from astra.models.base import database

    # Close any inherited connection from parent process and open a fresh one
    if not database.is_closed():
        database.close()
    database.connect()

    # Set search_path to include public schema for q3c functions
    database.execute_sql("SET search_path TO public, catalogdb, airflow")

    sql = f"""
        WITH targets AS (
            SELECT source_id, ra, dec, phot_g_mean_mag
            FROM catalogdb.gaia_dr3_source
            WHERE source_id = ANY(%(source_ids)s)
        ),
        counts AS (
            SELECT
                t.source_id,
                COUNT(*) - 1 as n_neighborhood
            FROM targets t
            JOIN catalogdb.gaia_dr3_source g
                ON q3c_radial_query(g.ra, g.dec, t.ra, t.dec, %(radius)s)
            WHERE g.phot_g_mean_mag > t.phot_g_mean_mag - %(brightness)s
            GROUP BY t.source_id
        )
        UPDATE {table} s
        SET n_neighborhood = c.n_neighborhood, modified = NOW()
        FROM counts c
        WHERE s.gaia_dr3_source_id = c.source_id
    """

    try:
        with database.atomic():
            cursor = database.execute_sql(sql, {
                'source_ids': list(source_ids),
                'radius': radius_deg,
                'brightness': brightness
            })
            return cursor.rowcount
    finally:
        # Close connection when done to avoid connection leaks
        if not database.is_closed():
            database.close()


def compute_n_neighborhood(
    radius=3,  # arcseconds
    brightness=5,  # magnitudes
    batch_size=10_000,
    limit=None,
    max_workers=8,
    queue=None
):
    """
    Compute n_neighborhood: count of Gaia sources within `radius` arcseconds
    that are brighter than (source_mag - brightness).

    Uses raw SQL with q3c_join for efficient spatial queries.
    Supports parallel execution with max_workers processes.

    :param radius: Search radius in arcseconds (default: 3)
    :param brightness: Magnitude difference threshold (default: 5)
    :param batch_size: Number of sources per batch (default: 10,000)
    :param limit: Maximum number of sources to process (default: None = all)
    :param max_workers: Number of parallel workers (default: 8, set to 1 for serial)
    :param queue: Progress queue
    """
    from astra.models.base import database
    from astra.models.source import Source
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if queue is None:
        queue = ProgressContext()

    # Ensure fresh database connection in subprocess
    if not database.is_closed():
        database.close()
    database.connect()

    # Get gaia_dr3_source_ids that need processing
    queue.put(dict(total=None, completed=0, description="Finding sources needing n_neighborhood"))

    # Use iterator to avoid loading all at once, and count separately if no limit
    q = (
        Source
        .select(Source.gaia_dr3_source_id)
        .where(
            (Source.n_neighborhood.is_null() | (Source.n_neighborhood < 0))
        &   Source.gaia_dr3_source_id.is_null(False)
        )
        .limit(limit)
        .tuples()
    )

    source_ids = [row[0] for row in q.iterator()]
    if not source_ids:
        queue.put(Ellipsis)
        return 0

    table = f"{Source._meta.schema}.{Source._meta.table_name}"
    radius_deg = radius / 3600.0

    # Prepare batches
    batches = list(chunked(source_ids, batch_size))
    batch_args = [(batch, table, radius_deg, brightness) for batch in batches]

    n_updated = 0
    queue.put(dict(total=len(source_ids), completed=0, description="Computing n_neighborhood"))

    if max_workers == 1:
        # Serial execution
        for args in batch_args:
            n_updated += _compute_n_neighborhood_batch(args)
            queue.put(dict(advance=len(args[0])))
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_n_neighborhood_batch, args): len(args[0])
                for args in batch_args
            }
            for future in as_completed(futures):
                batch_len = futures[future]
                try:
                    n_updated += future.result()
                except Exception as e:
                    log.warning(f"Batch failed: {e}")
                queue.put(dict(advance=batch_len))

    queue.put(Ellipsis)
    return n_updated


def set_missing_gaia_source_ids_to_null():
    """
    Reset gaia_dr*_source_id values <= 0 to NULL.

    This can be used to clean up any incorrectly set Gaia source IDs (e.g., 0 or negative values).
    Note that with the crossmatch_flags approach in migrate_gaia_source_ids, setting these to NULL
    will NOT cause re-processing unless you also reset the flag_gaia_dr*_crossmatch_attempted flags.
    """
    from astra.models.base import database
    from astra.models.source import Source

    with database.atomic():
        (
            Source
            .update(gaia_dr3_source_id=None, modified=fn.NOW())
            .where(Source.gaia_dr3_source_id <= 0)
            .execute()
        )
        (
            Source
            .update(gaia_dr2_source_id=None, modified=fn.NOW())
            .where(Source.gaia_dr2_source_id <= 0)
            .execute()
        )

def compute_f_night_time_for_boss_visits(limit=None, batch_size=1000, n_time=256, max_workers=64, queue=None):
    """
    Compute `f_night_time`, which is the observation mid-point expressed as a fraction of time between local sunset and sunrise.

    :param where:
        A peewee expression to filter the visits to compute `f_night_time` for.

    :param limit:
        The maximum number of visits to compute `f_night_time` for.

    :param batch_size:
        The number of visits to update at a time.

    :param n_time:
        The number of points to use when computing the sun's position.

    :param max_workers:
        The maximum number of workers to use when computing `f_night_time`.
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.boss import BossVisitSpectrum
    # Only select fields we actually need: pk, telescope, tai_beg, tai_end
    q = (
        BossVisitSpectrum
        .select(
            BossVisitSpectrum.pk,
            BossVisitSpectrum.telescope,
            BossVisitSpectrum.tai_beg,
            BossVisitSpectrum.tai_end
        )
        .where(
            BossVisitSpectrum.f_night_time.is_null()
        &   BossVisitSpectrum.tai_end.is_null(False)
        &   BossVisitSpectrum.tai_beg.is_null(False) # sometimes we don't have tai_beg or tai_end
        )
        .limit(limit)
    )

    get_obs_time = lambda v: Time((v.tai_beg + 0.5 * (v.tai_end - v.tai_beg))/(24*3600), format="mjd").datetime

    return _compute_f_night_time_for_visits(q, BossVisitSpectrum, get_obs_time, batch_size, n_time, max_workers, queue)


def compute_f_night_time_for_apogee_visits(limit=None, batch_size=1000, n_time=256, max_workers=64, queue=None):
    """
    Compute `f_night_time`, which is the observation mid-point expressed as a fraction of time between local sunset and sunrise.

    :param where:
        A peewee expression to filter the visits to compute `f_night_time` for.

    :param limit:
        The maximum number of visits to compute `f_night_time` for.

    :param batch_size:
        The number of visits to update at a time.

    :param n_time:
        The number of points to use (per 24 hour period) when computing the sun's position.

    :param max_workers:
        The maximum number of workers to use when computing `f_night_time`.
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.boss import BossVisitSpectrum
    # Only select fields we actually need: pk, telescope, date_obs
    q = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum.pk,
            ApogeeVisitSpectrum.telescope,
            ApogeeVisitSpectrum.date_obs
        )
        .where(ApogeeVisitSpectrum.f_night_time.is_null())
        .limit(limit)
    )
    return _compute_f_night_time_for_visits(q, ApogeeVisitSpectrum, lambda v: v.date_obs, batch_size, n_time, max_workers, queue)


def _compute_sunset_sunrise(observatory, mjd_int, n_time=256):
    """
    Compute sunset and sunrise times for a given observatory and MJD date.
    Returns (sunset_mjd, sunrise_mjd) as floats.
    """
    # Use noon of the given MJD as reference point
    ref_time = Time(mjd_int + 0.5, format="mjd")
    time_grid = ref_time + np.linspace(-24, 24, 2 * n_time) * u.hour

    sun = coord.get_sun(time_grid)
    altaz_frame = coord.AltAz(location=observatory, obstime=time_grid)
    sun_altaz = sun.transform_to(altaz_frame)

    # Find minima in altitude^2 (horizon crossings)
    alt_sq = sun_altaz.alt.degree ** 2
    min_idx = argrelmin(alt_sq, mode="wrap")[0]

    # Take the two closest to the middle (sunset before, sunrise after)
    center = n_time
    sunset_idx, sunrise_idx = min_idx[min_idx.searchsorted(center) - 1:][:2]

    sunset_mjd = time_grid[sunset_idx].mjd
    sunrise_mjd = time_grid[sunrise_idx].mjd

    return (sunset_mjd, sunrise_mjd)


def _compute_f_night_time_for_visits(q, model, get_obs_time, batch_size, n_time, max_workers, queue, k=100):
    """
    Optimized f_night_time computation.
    Key insight: sunset/sunrise only depends on (observatory, date), not individual visits.
    """
    from astra.models.base import database

    if queue is None:
        queue = ProgressContext()

    # Get observatory locations
    observatories = {
        "APO": coord.EarthLocation.of_site("APO"),
        "LCO": coord.EarthLocation.of_site("LCO"),
    }

    # Fetch all visits first to avoid slow count
    with queue.subtask("Fetching visits", total=None) as fetch_step:
        all_visits = list(q)
        total = len(all_visits)
        fetch_step.update(total=total, completed=total)

    if total == 0:
        queue.put(Ellipsis)
        return 0

    # Group visits by (observatory, mjd_int) - all visits on same night share sunset/sunrise
    visits_by_night = {}  # (obs_name, mjd_int) -> [(pk, obs_mjd), ...]
    with queue.subtask("Grouping visits by night", total=total) as group_step:
        for i, visit in enumerate(all_visits):
            if i % 100 == 0:
                group_step.update(advance=100)
            obs_time = get_obs_time(visit)
            if obs_time is None:
                continue
            obs_mjd = Time(obs_time).mjd
            mjd_int = int(obs_mjd)
            obs_name = visit.telescope[:3].upper()
            key = (obs_name, mjd_int)
            if key not in visits_by_night:
                visits_by_night[key] = []
            visits_by_night[key].append((visit.pk, obs_mjd))
        group_step.update(completed=total)

    # Compute sunset/sunrise for each unique (observatory, night) - this is the expensive part
    # but now we only do it once per night, not once per visit
    sunset_sunrise_cache = {}
    with queue.subtask("Computing sunset/sunrise times", total=len(visits_by_night)) as sun_step:
        for (obs_name, mjd_int) in visits_by_night.keys():
            observatory = observatories[obs_name]
            sunset_sunrise_cache[(obs_name, mjd_int)] = _compute_sunset_sunrise(observatory, mjd_int, n_time)
            sun_step.update(advance=1)

    table = f"{model._meta.schema}.{model._meta.table_name}"
    n_updated = 0

    # Process in batches for the SQL UPDATE
    all_updates = []  # (pk, f_night_time)
    for (obs_name, mjd_int), visits in visits_by_night.items():
        sunset_mjd, sunrise_mjd = sunset_sunrise_cache[(obs_name, mjd_int)]
        night_duration = sunrise_mjd - sunset_mjd

        for pk, obs_mjd in visits:
            f_night_time = (obs_mjd - sunset_mjd) / night_duration
            all_updates.append((pk, f_night_time))

    # Bulk update using raw SQL with VALUES clause
    with queue.subtask("Updating f_night_time", total=len(all_updates)) as update_step:
        for batch in chunked(all_updates, batch_size):
            values_rows = [f"({pk}, {f_night:.10f})" for pk, f_night in batch]
            values_sql = ", ".join(values_rows)

            sql = f"""
                UPDATE {table}
                SET f_night_time = v.f_night_time
                FROM (VALUES {values_sql}) AS v(pk, f_night_time)
                WHERE {table}.pk = v.pk
            """
            with database.atomic():
                database.execute_sql(sql)
            n_updated += len(batch)
            update_step.update(advance=len(batch))

    queue.put(Ellipsis)
    return n_updated
