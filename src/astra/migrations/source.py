"""
Unified source creation and spectrum-source linking for all spectrum types.

This module provides functions to:
1. Create Source entries from spectrum catalogids
2. Link spectra to their corresponding sources
3. Merge duplicate sources
"""
import numpy as np

from typing import Optional, Set, List, Tuple
from peewee import JOIN, chunked, fn, BigIntegerField, TextField

from astra.migrations.utils import ProgressContext
from astra.utils import log

def update_sdss5_dr19_apogee_flag(chunk_size: int = 1000):
    from astropy.io import fits
    from tqdm import tqdm

    with fits.open("/uufs/chpc.utah.edu/common/home/sdss50/dr19/spectro/astra/0.6.0/summary/mwmAllStar-0.6.0.fits.gz") as image:
        sdss_ids = list(set(image[2].data["sdss_id"]).difference({0, -1}))

    from astra.models import Source
    for chunk in tqdm(chunked(sdss_ids, chunk_size)):
        (
            Source
            .update(sdss5_dr19_apogee_flag=True)
            .where(Source.sdss_id.in_(chunk))
            .execute()
        )




def merge_sources(keep_pk: int, remove_pk: int, database=None) -> None:
    """
    Merge two source records by moving all foreign key references from remove_pk to keep_pk,
    then deleting the remove_pk source.

    If updating a foreign key would violate a unique constraint (because keep_pk already
    has a record in that table with the same unique key), both affected rows are deleted.

    :param keep_pk: The source pk to keep (foreign keys will point here)
    :param remove_pk: The source pk to remove (will be deleted after migration)
    :param database: Optional database connection (defaults to astra database)
    """
    if keep_pk == remove_pk:
        return

    if database is None:
        from astra.models.base import database

    from astra.models.source import Source

    schema = Source._meta.schema

    def parse_pg_array(pg_array_str: str) -> List[str]:
        """Parse a PostgreSQL array string like '{a,b,c}' into a Python list."""
        if pg_array_str is None:
            return []
        # Remove braces and split by comma
        return pg_array_str.strip('{}').split(',')

    def get_primary_key_column(schema: str, table_name: str) -> str:
        """Get the primary key column name for a table."""
        result = database.execute_sql(f"""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = '{schema}'
            AND tc.table_name = '{table_name}'
            AND tc.constraint_type = 'PRIMARY KEY'
            LIMIT 1
        """).fetchone()
        return result[0] if result else 'pk'

    # Find all tables with source_pk column (excluding source itself)
    tables_with_source_pk = database.execute_sql(f"""
        SELECT table_name
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
        AND column_name = 'source_pk'
        AND table_name != 'source'
    """).fetchall()
    tables_with_source_pk = [row[0] for row in tables_with_source_pk]

    # For each table, try to update source_pk, handling unique constraint violations
    for tbl in tables_with_source_pk:
        # Get the primary key column name for this table
        pk_col = get_primary_key_column(schema, tbl)

        # Get unique constraints for this table that include source_pk
        unique_constraints = database.execute_sql(f"""
            SELECT
                tc.constraint_name,
                array_agg(kcu.column_name ORDER BY kcu.ordinal_position) as columns
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = '{schema}'
            AND tc.table_name = '{tbl}'
            AND tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
            GROUP BY tc.constraint_name
            HAVING 'source_pk' = ANY(array_agg(kcu.column_name))
        """).fetchall()

        if unique_constraints:
            # There are unique constraints involving source_pk
            # We need to find rows that would conflict and delete them

            for constraint_name, columns_str in unique_constraints:
                # Parse the PostgreSQL array string into a Python list
                columns = parse_pg_array(columns_str)
                # Build the columns list excluding source_pk for comparison
                other_cols = [c for c in columns if c != 'source_pk']

                if other_cols:
                    # Find rows from remove_pk that would conflict with keep_pk rows
                    other_cols_str = ", ".join(other_cols)
                    other_cols_compare = " AND ".join(f"old_rows.{c} = new_rows.{c}" for c in other_cols)

                    # Delete conflicting rows from both old and new source
                    # First get the PKs of conflicting rows
                    conflict_sql = f"""
                        WITH old_rows AS (
                            SELECT {pk_col}, {other_cols_str}
                            FROM {schema}.{tbl}
                            WHERE source_pk = {remove_pk}
                        ),
                        new_rows AS (
                            SELECT {pk_col}, {other_cols_str}
                            FROM {schema}.{tbl}
                            WHERE source_pk = {keep_pk}
                        ),
                        conflicts AS (
                            SELECT old_rows.{pk_col} as old_pk, new_rows.{pk_col} as new_pk
                            FROM old_rows
                            JOIN new_rows ON {other_cols_compare}
                        )
                        DELETE FROM {schema}.{tbl}
                        WHERE {pk_col} IN (SELECT old_pk FROM conflicts)
                           OR {pk_col} IN (SELECT new_pk FROM conflicts)
                    """
                    database.execute_sql(conflict_sql)

        # Now update remaining rows (non-conflicting) to point to keep_pk
        database.execute_sql(f"""
            UPDATE {schema}.{tbl}
            SET source_pk = {keep_pk}
            WHERE source_pk = {remove_pk}
        """)

    # Delete the old source row
    database.execute_sql(f"""
        DELETE FROM {schema}.source
        WHERE pk = {remove_pk}
    """)


def fill_dr17_apogee_catalogids(batch_size: int = 10000, queue=None):
    """
    Fill in catalogid for DR17 APOGEE spectra that are missing it.

    DR17 spectra have an `obj` field containing the apogee_id. We can look this up
    in AllstarDR17SynspecRev1 (via apogee_id) and then get the catalogid from
    CatalogToAllstarDR17SynspecRev1.
    """
    from astra.models.base import database
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel

    queue = queue or ProgressContext()

    class CatalogToAllstarDR17SynspecRev1(CatalogdbModel):
        catalogid = BigIntegerField()
        target_id = TextField()  # This is the apstar_id

        class Meta:
            table_name = "catalog_to_allstar_dr17_synspec_rev1"
            primary_key = False

    class AllstarDR17SynspecRev1(CatalogdbModel):
        apogee_id = TextField()
        apstar_id = TextField()

        class Meta:
            table_name = "allstar_dr17_synspec_rev1"
            primary_key = False

    # Find DR17 spectra missing catalogid
    with queue.subtask("Finding DR17 APOGEE spectra missing catalogid", total=None) as find_step:
        dr17_missing = list(
            ApogeeVisitSpectrum
            .select(ApogeeVisitSpectrum.pk, ApogeeVisitSpectrum.obj)
            .where(
                (ApogeeVisitSpectrum.release == "dr17")
                & (
                    ApogeeVisitSpectrum.catalogid.is_null()
                    | (ApogeeVisitSpectrum.catalogid <= 0)
                )
            )
            .tuples()
        )
        find_step.update(total=len(dr17_missing), completed=len(dr17_missing))

    if not dr17_missing:
        return 0

    total_spectra = len(dr17_missing)

    # Build lookup of obj (apogee_id) -> pk
    with queue.subtask("Building APOGEE ID lookup", total=total_spectra) as lookup_step:
        pk_by_obj = {}
        for pk, obj in dr17_missing:
            if obj:
                pk_by_obj.setdefault(obj, []).append(pk)
        lookup_step.update(completed=total_spectra)

    # Query catalogdb to get apogee_id -> catalogid mapping
    # Join AllstarDR17SynspecRev1 to CatalogToAllstarDR17SynspecRev1 via apstar_id = target_id
    apogee_ids = list(pk_by_obj.keys())
    catalogid_by_apogee_id = {}

    with queue.subtask(f"Querying catalogdb for {len(apogee_ids)} APOGEE IDs", total=len(apogee_ids)) as query_step:
        for batch in chunked(apogee_ids, batch_size):
            q = (
                AllstarDR17SynspecRev1
                .select(
                    AllstarDR17SynspecRev1.apogee_id,
                    CatalogToAllstarDR17SynspecRev1.catalogid
                )
                .join(
                    CatalogToAllstarDR17SynspecRev1,
                    on=(AllstarDR17SynspecRev1.apstar_id == CatalogToAllstarDR17SynspecRev1.target_id)
                )
                .where(AllstarDR17SynspecRev1.apogee_id.in_(batch))
                .tuples()
            )
            for apogee_id, catalogid in q:
                if catalogid:
                    catalogid_by_apogee_id[apogee_id] = catalogid
            query_step.update(advance=len(batch))

    # Build updates
    table = f"{ApogeeVisitSpectrum._meta.schema}.{ApogeeVisitSpectrum._meta.table_name}"
    n_updated = 0

    updates_by_catalogid = {}  # catalogid -> [pk, pk, ...]
    for obj, pks in pk_by_obj.items():
        catalogid = catalogid_by_apogee_id.get(obj)
        if catalogid:
            updates_by_catalogid.setdefault(catalogid, []).extend(pks)

    total_to_update = sum(len(pks) for pks in updates_by_catalogid.values())

    # Bulk update using SQL
    with queue.subtask(f"Updating {total_to_update} DR17 APOGEE spectra", total=total_to_update) as update_step:
        for catalogid, pks in updates_by_catalogid.items():
            for pk_batch in chunked(pks, batch_size):
                pk_list = ", ".join(str(pk) for pk in pk_batch)
                sql = f"""
                    UPDATE {table}
                    SET catalogid = {catalogid}
                    WHERE pk IN ({pk_list})
                """
                with database.atomic():
                    database.execute_sql(sql)
                n_updated += len(pk_batch)
                update_step.update(advance=len(pk_batch))

    return n_updated


def create_sources_from_spectra(batch_size: int = 1000, queue=None):
    """
    Create Source entries for all spectra that have catalogids but no linked source.

    This queries the catalog database to get source information for catalogids
    found in BossVisitSpectrum and ApogeeVisitSpectrum tables.
    """
    from astra.models.base import database
    from astra.models.source import Source
    from astra.models.boss import BossVisitSpectrum
    from astra.models.apogee import ApogeeVisitSpectrum

    from astra.migrations.sdss5db.catalogdb import (
        Catalog,
        CatalogToGaia_DR3,
        CatalogToGaia_DR2,
        CatalogdbModel
    )

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"

    queue = queue or ProgressContext()

    # Collect catalogids from unlinked spectra
    with queue.subtask("Collecting unlinked BOSS catalogids", total=None) as boss_step:
        boss_catalogids = set(
            BossVisitSpectrum
            .select(BossVisitSpectrum.catalogid)
            .where(
                BossVisitSpectrum.source.is_null()
                & (BossVisitSpectrum.catalogid > 0)
            )
            .distinct()
            .tuples()
            .iterator()
        )
        boss_catalogids = {c[0] for c in boss_catalogids}
        boss_step.update(total=len(boss_catalogids), completed=len(boss_catalogids))

    with queue.subtask("Collecting unlinked APOGEE catalogids", total=None) as apogee_step:
        apogee_catalogids = set(
            ApogeeVisitSpectrum
            .select(ApogeeVisitSpectrum.catalogid)
            .where(
                ApogeeVisitSpectrum.source.is_null()
                & (ApogeeVisitSpectrum.catalogid > 0)
            )
            .distinct()
            .tuples()
            .iterator()
        )
        apogee_catalogids = {c[0] for c in apogee_catalogids}
        apogee_step.update(total=len(apogee_catalogids), completed=len(apogee_catalogids))

    all_catalogids = boss_catalogids | apogee_catalogids

    if not all_catalogids:
        return 0

    # Get existing catalogids in Source table to avoid duplicates
    with queue.subtask("Checking existing sources", total=None) as check_step:
        existing_catalogids = set()
        for field in (Source.catalogid, Source.catalogid21, Source.catalogid25, Source.catalogid31):
            q = Source.select(field).where(field.is_null(False)).tuples()
            existing_catalogids.update(c[0] for c in q.iterator())
        check_step.update(total=len(existing_catalogids), completed=len(existing_catalogids))

    new_catalogids = all_catalogids - existing_catalogids

    if not new_catalogids:
        return 0

    # Query catalog database for source information
    source_data = {
        0: {
            "ra": np.nan,
            "dec": np.nan,
            "catalogid": 0,
            "version_id": None,
            "lead": None,
            "gaia_dr3_source_id": None,
            "gaia_dr2_source_id": None,
            "sdss_id": 0,
            "n_associated": 0,
            "catalogid21": None,
            "catalogid25": None,
            "catalogid31": None,
            "sdss4_apogee_id": "VESTA",
        }
    }
    with queue.subtask("Querying catalog for source info", total=len(new_catalogids)) as query_step:
        for chunk_catalogids in chunked(list(new_catalogids), batch_size):
            q = (
                Catalog
                .select(
                    Catalog.ra,
                    Catalog.dec,
                    Catalog.catalogid,
                    Catalog.version_id.alias("version_id"),
                    Catalog.lead,
                    CatalogToGaia_DR3.target.alias("gaia_dr3_source_id"),
                    CatalogToGaia_DR2.target.alias("gaia_dr2_source_id"),
                    SDSS_ID_Flat.sdss_id,
                    SDSS_ID_Flat.n_associated,
                    SDSS_ID_Stacked.catalogid21,
                    SDSS_ID_Stacked.catalogid25,
                    SDSS_ID_Stacked.catalogid31,
                )
                .join(SDSS_ID_Flat, JOIN.LEFT_OUTER, on=(Catalog.catalogid == SDSS_ID_Flat.catalogid))
                .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
                .switch(Catalog)
                .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(Catalog.catalogid == CatalogToGaia_DR3.catalog))
                .switch(Catalog)
                .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(Catalog.catalogid == CatalogToGaia_DR2.catalog))
                .where(
                    Catalog.catalogid.in_(chunk_catalogids)
                &   (SDSS_ID_Flat.rank == 1)
                )
                .dicts()
            )

            for row in q:
                catalogid = row["catalogid"]
                sdss_id = row.get("sdss_id")

                # Use sdss_id as the unique key if available, otherwise catalogid
                key = sdss_id if (sdss_id is not None and sdss_id > 0) else f"cat_{catalogid}"

                if key in source_data:
                    # Merge data, preferring non-null values
                    for k, v in row.items():
                        if source_data[key].get(k) is None and v is not None:
                            source_data[key][k] = v
                else:
                    source_data[key] = row

            query_step.update(advance=len(chunk_catalogids))

    # Insert new sources
    n_new_sources = len(source_data)
    if n_new_sources > 0:
        with queue.subtask("Creating sources", total=n_new_sources) as create_step:
            with database.atomic():
                for chunk in chunked(source_data.values(), batch_size):
                    (
                        Source
                        .insert_many(chunk)
                        .on_conflict_ignore()
                        .execute()
                    )
                    create_step.update(advance=len(chunk))

    sun_pk = Source.get(sdss4_apogee_id="VESTA").pk
    with database.atomic():
        (
            ApogeeVisitSpectrum
            .update(source_pk=sun_pk)
            .where(ApogeeVisitSpectrum.obj == "VESTA")
            .execute()
        )

    return n_new_sources


def link_boss_spectra_to_sources(batch_size: int = 1000, queue=None):
    """
    Link BossVisitSpectrum records to their corresponding Source records.

    This matches spectra to sources using catalogid fields.
    """
    from astra.models.base import database
    from astra.models.boss import BossVisitSpectrum
    from astra.models.source import Source

    queue = queue or ProgressContext()

    # Build mapping of catalogid -> source_pk
    with queue.subtask("Building catalogid to source mapping", total=None) as build_step:
        catalogid_to_source_pk = {}
        q = (
            Source
            .select(
                Source.pk,
                Source.catalogid,
                Source.catalogid21,
                Source.catalogid25,
                Source.catalogid31
            )
            .tuples()
        )
        for pk, *catalogids in q.iterator():
            for catalogid in catalogids:
                if catalogid is not None:
                    catalogid_to_source_pk[catalogid] = pk
        build_step.update(total=len(catalogid_to_source_pk), completed=len(catalogid_to_source_pk))

    # Count unlinked spectra
    unlinked_count = (
        BossVisitSpectrum
        .select()
        .where(
            BossVisitSpectrum.source.is_null()
            & (BossVisitSpectrum.catalogid > 0)
        )
        .count()
    )

    if unlinked_count == 0:
        return 0

    # Update in batches using the catalogid fields
    n_linked = 0
    with queue.subtask("Linking BOSS spectra to sources", total=unlinked_count) as link_step:
        for catalogid_field in (Source.catalogid, Source.catalogid31, Source.catalogid25, Source.catalogid21):
            with database.atomic():
                updated = (
                    BossVisitSpectrum
                    .update(source_pk=Source.pk)
                    .from_(Source)
                    .where(
                        BossVisitSpectrum.source.is_null()
                        & (BossVisitSpectrum.catalogid == catalogid_field)
                    )
                    .execute()
                )
            n_linked += updated
            link_step.update(advance=updated)

    return n_linked


def link_apogee_spectra_to_sources(batch_size: int = 1000, queue=None):
    """
    Link ApogeeVisitSpectrum records to their corresponding Source records.

    This matches spectra to sources using:
    1. sdss_id (most reliable for SDSS5)
    2. sdss4_apogee_id/obj (for DR17 spectra)
    3. catalogid fields (fallback)
    """
    from astra.models.base import database
    from astra.models.apogee import ApogeeVisitSpectrum, ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar
    from astra.models.source import Source

    queue = queue or ProgressContext()

    # Count unlinked visit spectra
    unlinked_count = (
        ApogeeVisitSpectrum
        .select()
        .where(ApogeeVisitSpectrum.source.is_null())
        .count()
    )

    n_linked = 0

    if unlinked_count > 0:
        with queue.subtask("Linking APOGEE visit spectra to sources", total=unlinked_count) as link_step:

            # Try linking DR17 spectra by sdss4_apogee_id (stored in obj field)
            with database.atomic():
                updated = (
                    ApogeeVisitSpectrum
                    .update(source_pk=Source.pk)
                    .from_(Source)
                    .where(
                        ApogeeVisitSpectrum.source.is_null()
                        & (ApogeeVisitSpectrum.release == "dr17")
                        & (ApogeeVisitSpectrum.obj == Source.sdss4_apogee_id)
                    )
                    .execute()
                )
            n_linked += updated
            link_step.update(advance=updated)

            # Then try catalogid fields for any remaining
            for catalogid_field in (Source.catalogid, Source.catalogid31, Source.catalogid25, Source.catalogid21):
                with database.atomic():
                    updated = (
                        ApogeeVisitSpectrum
                        .update(source_pk=Source.pk)
                        .from_(Source)
                        .where(
                            ApogeeVisitSpectrum.source.is_null()
                            & (ApogeeVisitSpectrum.catalogid == catalogid_field)
                            & (ApogeeVisitSpectrum.catalogid > 0)
                        )
                        .execute()
                    )
                n_linked += updated
                link_step.update(advance=updated)

    # Now link ApogeeVisitSpectrumInApStar records. These don't carry their own catalogid,
    # but each one points back to the ApogeeVisitSpectrum it was derived from via
    # drp_spectrum_pk, so once that's linked we can just copy source_pk across. This has
    # to be a real linking step (not just relying on migrate_apogee_visits_in_apStar_files
    # to copy source_pk at insert time) because that function runs concurrently with this
    # one and reads the coadd's source_pk before it has necessarily been set.
    unlinked_in_apstar_count = (
        ApogeeVisitSpectrumInApStar
        .select()
        .where(ApogeeVisitSpectrumInApStar.source.is_null())
        .count()
    )

    if unlinked_in_apstar_count > 0:
        with queue.subtask("Linking APOGEE visit-in-apStar spectra to sources", total=unlinked_in_apstar_count) as in_apstar_step:
            with database.atomic():
                updated = (
                    ApogeeVisitSpectrumInApStar
                    .update(source_pk=ApogeeVisitSpectrum.source_pk)
                    .from_(ApogeeVisitSpectrum)
                    .where(
                        ApogeeVisitSpectrumInApStar.source.is_null()
                        & (ApogeeVisitSpectrumInApStar.drp_spectrum_pk == ApogeeVisitSpectrum.spectrum_pk)
                        & ApogeeVisitSpectrum.source.is_null(False)
                    )
                    .execute()
                )
            n_linked += updated
            in_apstar_step.update(advance=updated)

    # Now link coadded spectra
    unlinked_coadd_count = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .where(ApogeeCoaddedSpectrumInApStar.source.is_null())
        .count()
    )

    if unlinked_coadd_count > 0:
        with queue.subtask("Linking APOGEE coadded spectra to sources", total=unlinked_coadd_count) as coadd_step:
            # Link by catalogid
            for catalogid_field in (Source.catalogid, Source.catalogid31, Source.catalogid25, Source.catalogid21):
                with database.atomic():
                    updated = (
                        ApogeeCoaddedSpectrumInApStar
                        .update(source_pk=Source.pk)
                        .from_(Source)
                        .where(
                            ApogeeCoaddedSpectrumInApStar.source.is_null()
                            & (ApogeeCoaddedSpectrumInApStar.catalogid == catalogid_field)
                            & (ApogeeCoaddedSpectrumInApStar.catalogid > 0)
                        )
                        .execute()
                    )
                n_linked += updated
                coadd_step.update(advance=updated)

    return n_linked


def create_sources_and_link_spectra(batch_size: int = 1000, queue=None):
    """
    Unified function to create sources and link all spectrum types.

    This is the main entry point for the migration task. It:
    1. Fills in catalogid for DR17 APOGEE spectra (from catalogdb)
    2. Creates Source entries for any spectra with catalogids but no source
    3. Links BossVisitSpectrum to sources
    4. Links ApogeeVisitSpectrum to sources
    """
    queue = queue or ProgressContext()

    # Step 0: Fill in catalogid for DR17 APOGEE spectra
    n_dr17_filled = fill_dr17_apogee_catalogids(batch_size=batch_size, queue=queue)

    # Step 1: Create sources
    n_new_sources = create_sources_from_spectra(batch_size=batch_size, queue=queue)

    # Step 2: Link BOSS spectra
    n_boss_linked = link_boss_spectra_to_sources(batch_size=batch_size, queue=queue)

    # Step 3: Link APOGEE spectra
    n_apogee_linked = link_apogee_spectra_to_sources(batch_size=batch_size, queue=queue)

    #log.info(f"Filled {n_dr17_filled} DR17 catalogids, created {n_new_sources} new sources, linked {n_boss_linked} BOSS and {n_apogee_linked} APOGEE spectra")

    queue.put(Ellipsis)
    return (n_new_sources, n_boss_linked, n_apogee_linked)
