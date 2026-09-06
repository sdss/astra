import concurrent.futures
import subprocess
import numpy as np
from datetime import datetime
from peewee import JOIN, chunked, Case, fn, SQL, EXCLUDED, IntegrityError, Tuple
from typing import Optional

from astra.migrations.utils import enumerate_new_spectrum_pks, upsert_many, ProgressContext
from astra.utils import expand_path, flatten, log
from tqdm import tqdm

def _star_has_ingestable_visit(apred: str, max_mjd: int):
    """
    An EXISTS clause matching stars that still have a visit once `max_mjd` is applied.

    `max_mjd` filters the visit query but not the star query, so without this we ingest
    coadds for stars whose visits were all excluded: nothing creates a Source carrying
    their catalogid, and they can never be linked.

    Matching on catalogid, not just (obj, telescope), is what makes this exact. A star's
    catalogid can differ from that of its earliest visits, so "this star has some visit
    before the cutoff" is not sufficient -- it has to have a visit before the cutoff
    carrying the star's own catalogid, since that is what the Source is created from.
    """
    from astra.migrations.sdss5db.apogee_drpdb import Star, Visit

    V = Visit.alias()
    return fn.EXISTS(
        V
        .select(SQL("1"))
        .where(
            (V.apred == apred)
        &   (V.obj == Star.obj)
        &   (V.telescope == Star.telescope)
        &   (V.mjd <= max_mjd)
        &   (V.catalogid == Star.catalogid)
        )
    )


def _limited_star_selection(apred: str, limit: int, incremental: bool = True, max_mjd: Optional[int] = None):
    """
    Build a sub-query selecting the (obj, telescope) pairs that a `limit` applies to.

    `limit` has to mean "this many stars". If it were applied independently to the
    `visit` and `star` queries we would ingest two nearly disjoint samples: the coadds
    would be for stars whose visits were never ingested, so no Source would ever carry
    their catalogid and they could not be linked.

    This is returned as a sub-query rather than a materialised list of pairs because a
    tuple `IN` with tens of thousands of entries exceeds Postgres' `max_stack_depth`.
    The `ORDER BY` matters: `LIMIT` without it is not guaranteed to return the same rows
    each time the sub-query is executed, and it is executed once per calling query.
    """
    from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
    from astra.migrations.sdss5db.apogee_drpdb import Star

    max_star_pk = 0
    if incremental:
        max_star_pk = (
            ApogeeCoaddedSpectrumInApStar
            .select(fn.MAX(ApogeeCoaddedSpectrumInApStar.star_pk))
            .scalar() or 0
        )

    where = (
        (Star.apred_vers == apred)
    &   (Star.pk > max_star_pk)
    )
    if max_mjd is not None:
        # Match the cut the coadd and visit queries apply, so the limit selects stars
        # that will actually be ingested rather than being partly spent on excluded ones.
        where &= _star_has_ingestable_visit(apred, max_mjd)

    return (
        Star
        .select(Star.obj, Star.telescope)
        .where(where)
        .group_by(Star.obj, Star.telescope)
        .order_by(Star.obj, Star.telescope)
        .limit(limit)
    )


def migrate_apogee_spectra_from_sdss5_apogee_drpdb(apred: str, max_mjd: Optional[int] = None, queue=None, limit=None, incremental=True, **kwargs):
    """
    Migrate APOGEE spectrum-level information from the SDSS-V APOGEE DRP database.

    This function only loads spectrum-level data for visits and coadds.
    Source creation and spectrum-to-source linking should be handled separately
    (e.g., via create_sources_and_link_spectra).
    """
    queue = queue or ProgressContext()

    # A `limit` selects stars, and both the visit and coadd queries are then restricted to
    # those same stars, so every ingested coadd has its visits ingested alongside it.
    restrict_to_stars = None
    if limit is not None:
        restrict_to_stars = _limited_star_selection(apred, limit, incremental=incremental, max_mjd=max_mjd)

    # Migrate visits
    v_n_new_spectra, v_n_updated_spectra = migrate_apogee_visits(apred, max_mjd=max_mjd, queue=queue, restrict_to_stars=restrict_to_stars, incremental=incremental, **kwargs)

    # Migrate co-added spectra. `max_mjd` has to be passed here too: it filters visits, so
    # without it we would ingest coadds for stars that have no ingested visit, and therefore
    # no Source carrying their catalogid.
    c_n_new_spectra, c_n_updated_spectra = migrate_apogee_coadds(apred, queue=queue, restrict_to_stars=restrict_to_stars, incremental=incremental, max_mjd=max_mjd, **kwargs)

    queue.put(Ellipsis)
    return None



def migrate_apogee_visits_in_apStar_files(apred: str, max_workers=16, queue=None, limit=None, batch_size=1000):

    from astra.models.base import database
    from astra.models.apogee import ApogeeCoaddedSpectrumInApStar, ApogeeVisitSpectrumInApStar, ApogeeVisitSpectrum

    queue = queue or ProgressContext()

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)
    try:
        q = (
            ApogeeCoaddedSpectrumInApStar
            .select()
            .where(
                (ApogeeCoaddedSpectrumInApStar.apred == apred)
            )
            .limit(limit)
            .iterator()
        )

        apStar_spectra, futures = ({}, [])
        total = 0
        with queue.subtask("Getting apStar metadata", total=None) as get_step:
            for total, spectrum in enumerate(q, start=1):
                futures.append(executor.submit(_get_apstar_metadata, spectrum))
                apStar_spectra[spectrum.spectrum_pk] = spectrum
                get_step.update(advance=1)
            get_step.update(total=total, completed=total)

        visit_spectrum_data = []
        failed_spectrum_pks = []
        with queue.subtask("Collecting apStar metadata", total=total) as collect_step:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                for spectrum_pk, metadata in result.items():
                    if metadata is None:
                        failed_spectrum_pks.append(spectrum_pk)
                        continue

                    spectrum = apStar_spectra[spectrum_pk]

                    mjds = []
                    sfiles = [metadata[f"SFILE{i}"] for i in range(1, int(metadata["NVISITS"]) + 1)]
                    for sfile in sfiles:
                        #if spectrum.telescope == "apo1m":
                        #    #"$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{mjd}/apVisit-{apred}-{mjd}-{reduction}.fits"
                        #    # sometimes it is stored as a float AHGGGGHGGGGHGHGHGH
                        #    mjds.append(int(float(sfile.split("-")[2])))
                        #else:
                        #    mjds.append(int(float(sfile.split("-")[3])))
                        #    # "$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Visit-{apred}-{plate}-{mjd}-{fiber:0>3}.fits"
                        # NOTE: For SDSS5 data this is index 4: 'apVisit-1.2-apo25m-5339-59715-103.fits'
                        mjds.append(int(float(sfile.split("-")[4])))

                    assert len(sfiles) == int(metadata["NVISITS"])

                    spectrum.snr = float(metadata["SNR"])
                    spectrum.mean_fiber = float(metadata["MEANFIB"])
                    spectrum.std_fiber = float(metadata["SIGFIB"])
                    spectrum.n_good_visits = int(metadata["NVISITS"])
                    spectrum.n_good_rvs = int(metadata["NVISITS"])
                    spectrum.v_rad = float(metadata.get("VRAD", metadata.get("VHBARY")))
                    spectrum.e_v_rad = float(metadata["VERR"])
                    spectrum.std_v_rad = float(metadata["VSCATTER"])
                    spectrum.median_e_v_rad = float(metadata.get("VERR_MED", np.nan))
                    spectrum.star_flags = metadata["STARFLAG"]

                    # The MJDS in the apStar file only list the MJDs that were included in the stack.
                    # But there could be other MJDs which were not included in the stack.
                    # TODO: To be consistent elsewhere we should probably not update these based on
                    spectrum.min_mjd = min(mjds)
                    spectrum.max_mjd = max(mjds)

                    star_kwds = dict(
                        source_pk=spectrum.source_pk,
                        release=spectrum.release,
                        filetype=spectrum.filetype,
                        apred=spectrum.apred,
                        apstar=spectrum.apstar,
                        obj=spectrum.obj,
                        telescope=spectrum.telescope,
                        #field=spectrum.field,
                        #prefix=spectrum.prefix,
                        #reduction=spectrum.obj if spectrum.telescope == "apo1m" else None
                    )
                    for i, sfile in enumerate(sfiles, start=1):
                        #if spectrum.telescope != "apo1m":
                        #    plate = sfile.split("-")[2]
                        #else:
                        #    # plate not known..
                        #    plate = metadata["FIELD"].strip()
                        mjd = int(sfile.split("-")[4])
                        plate = sfile.split("-")[3]

                        kwds = star_kwds.copy()
                        kwds.update(
                            mjd=mjd,
                            fiber=int(metadata[f"FIBER{i}"]),
                            plate=plate
                        )
                        visit_spectrum_data.append(kwds)

                collect_step.update(advance=1)
    finally:
        # Always tear down the worker pool; a leaked ProcessPoolExecutor can
        # deadlock interpreter shutdown (especially under the 'fork' start
        # method), which hangs the migration scheduler's process.join().
        executor.shutdown(wait=True)

    with queue.subtask("Updating apStar metadata", total=total) as update_step:
        for chunk in chunked(apStar_spectra.values(), batch_size):
            n_updated = (
                ApogeeCoaddedSpectrumInApStar
                .bulk_update(
                    chunk,
                    fields=[
                        ApogeeCoaddedSpectrumInApStar.snr,
                        ApogeeCoaddedSpectrumInApStar.mean_fiber,
                        ApogeeCoaddedSpectrumInApStar.std_fiber,
                        ApogeeCoaddedSpectrumInApStar.n_good_visits,
                        ApogeeCoaddedSpectrumInApStar.n_good_rvs,
                        ApogeeCoaddedSpectrumInApStar.v_rad,
                        ApogeeCoaddedSpectrumInApStar.e_v_rad,
                        ApogeeCoaddedSpectrumInApStar.std_v_rad,
                        ApogeeCoaddedSpectrumInApStar.median_e_v_rad,
                        ApogeeCoaddedSpectrumInApStar.star_flags,
                        ApogeeCoaddedSpectrumInApStar.min_mjd,
                        ApogeeCoaddedSpectrumInApStar.max_mjd
                    ]
                )
            )
            update_step.update(advance=n_updated)

    q = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum.obj, # using this instead of source_pk because some apogee_ids have two different sources
            ApogeeVisitSpectrum.spectrum_pk,
            ApogeeVisitSpectrum.telescope,
            ApogeeVisitSpectrum.plate,
            ApogeeVisitSpectrum.mjd,
            ApogeeVisitSpectrum.fiber
        )
        .where(ApogeeVisitSpectrum.apred == apred)
    )
    with queue.subtask("Matching to ApogeeVisitSpectrum", total=None) as match_step:
        # Fetch all at once to avoid slow count + row-by-row iteration
        drp_rows = list(q.tuples())
        match_step.update(total=len(drp_rows), completed=len(drp_rows))

        drp_spectrum_data = {}
        for obj, spectrum_pk, telescope, plate, mjd, fiber in drp_rows:
            drp_spectrum_data.setdefault(obj, {})
            key = "_".join(map(str, (telescope, plate, mjd, fiber)))
            drp_spectrum_data[obj][key] = spectrum_pk

    with queue.subtask("Linking to ApogeeVisitSpectrum", total=len(visit_spectrum_data)) as link_step:
        only_ingest_visits = []
        failed_to_match_to_drp_spectrum_pk = []
        for spectrum_pk, visit in enumerate_new_spectrum_pks(visit_spectrum_data):
            key = "_".join(map(str, [visit[k] for k in ("telescope", "plate", "mjd", "fiber")]))
            try:
                drp_spectrum_pk = drp_spectrum_data[visit["obj"]][key]
            except:
                failed_to_match_to_drp_spectrum_pk.append((spectrum_pk, visit))
            else:
                visit.update(
                    spectrum_pk=spectrum_pk,
                    drp_spectrum_pk=drp_spectrum_pk
                )
                only_ingest_visits.append(visit)
        link_step.update(completed=len(visit_spectrum_data))

    if len(failed_to_match_to_drp_spectrum_pk) > 0:
        log.warning(f"There were {len(failed_to_match_to_drp_spectrum_pk)} spectra that we could not match to DRP spectra")
        log.warning(f"Example: {failed_to_match_to_drp_spectrum_pk[0]}")

    n_apogee_visit_in_apstar_inserted = 0
    with queue.subtask("Upserting ApogeeVisitSpectrumInApStar spectra", total=len(only_ingest_visits)) as upsert_step:
        with database.atomic():
            for chunk in chunked(only_ingest_visits, batch_size):
                n_apogee_visit_in_apstar_inserted += len(
                    ApogeeVisitSpectrumInApStar
                    .insert_many(chunk)
                    .on_conflict(
                        conflict_target=[
                            ApogeeVisitSpectrumInApStar.release,
                            ApogeeVisitSpectrumInApStar.apred,
                            ApogeeVisitSpectrumInApStar.apstar,
                            ApogeeVisitSpectrumInApStar.obj,
                            ApogeeVisitSpectrumInApStar.telescope,
                            ApogeeVisitSpectrumInApStar.healpix,
                            ApogeeVisitSpectrumInApStar.field,
                            ApogeeVisitSpectrumInApStar.prefix,
                            ApogeeVisitSpectrumInApStar.plate,
                            ApogeeVisitSpectrumInApStar.mjd,
                            ApogeeVisitSpectrumInApStar.fiber,
                        ],
                        preserve=(
                            ApogeeVisitSpectrumInApStar.drp_spectrum_pk,
                        )
                    )
                    .on_conflict(
                        conflict_target=[ApogeeVisitSpectrumInApStar.drp_spectrum_pk],
                        #action="update"
                        action="ignore"
                    )
                    .returning(ApogeeVisitSpectrumInApStar.pk)
                    .execute()
                )
                upsert_step.update(advance=len(chunk))

    queue.put(Ellipsis)

    return (n_apogee_visit_in_apstar_inserted, failed_to_match_to_drp_spectrum_pk)


def _get_apstar_metadata(
    apstar,
    keys=(
        "SIMPLE",
        "FIELD",
        "MEANFIB",
        "SNR",
        "SIGFIB",
        "VSCATTER",
        "STARFLAG",
        "NVISITS",
        "VHELIO",
        "VRAD",
        "VHBARY",
        "VERR",
        "VERR_MED",
        "SFILE?",
        "FIBER?"
    ),
):

    K = len(keys)
    keys_str = "|".join([f"({k})" for k in keys])

    # 80 chars per line, 150 lines -> 12000
    # (12 lines/visit * 100 visits + 100 lines typical header) * 80 -> 104,000
    command_template = " | ".join([
        'hexdump -n 100000 -e \'80/1 "%_p" "\\n"\' {path} 2>/dev/null', # 2>/dev/null suppresses error messages but keeps what we need
        f'egrep "{keys_str}"',
    ])
    commands = f"{command_template.format(path=apstar.absolute_path)}\n"

    try:
        outputs = subprocess.check_output(
            commands,
            shell=True,
            text=True,
            stderr=subprocess.STDOUT
        )
    except:
        return { apstar.spectrum_pk: None }

    outputs = outputs.strip().split("\n")

    metadata = {}
    for line in outputs:
        try:
            key, value = line.split("=")
            key, value = (key.strip(), value.split()[0].strip(" '"))
        except (IndexError, ValueError): # binary data, probably
            continue

        if key in metadata:
            log.warning(f"Multiple key `{key}` found in {apstar}: {expand_path(apstar.path)}")
        metadata[key] = value

    return { apstar.spectrum_pk: metadata }

def _migrate_dithered_metadata(pk, absolute_path):
    """
    Read FITS header directly to check NAXIS1 for dithered status.
    Much faster than spawning hexdump subprocess for each file.
    """
    try:
        with open(absolute_path, 'rb') as f:
            header_bytes = f.read(28800)

        # Parse the header - each card is 80 bytes
        header_text = header_bytes.decode('ascii', errors='ignore')
        for i in range(0, len(header_text), 80):
            card = header_text[i:i+80]
            if card.startswith('NAXIS1  ='):
                # Extract value after '='
                value_part = card[10:30].strip()  # Value is in columns 11-30
                naxis1 = int(value_part.split()[0])
                # @Nidever: "if there's 2048 then it hasn't been dithered, if it's 4096 then it's dithered."
                # Explicitly return True/False to ensure proper boolean type for DB
                return (pk, True) if naxis1 == 4096 else (pk, False)

        # NAXIS1 not found in header block
        return (pk, None)
    except:
        return (pk, None)

def migrate_dithered_metadata(
    max_workers=128,
    batch_size=1000,
    queue=None,
    limit=None
):
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.base import database
    queue = queue or ProgressContext()

    # Reconnect database to avoid stale connections
    if not database.is_closed():
        database.close()
    database.connect()

    queue.put(dict(description="Finding APOGEE spectra missing dithered info", total=None))

    # Set dithered = True for DR17
    with database.atomic():
        (
            ApogeeVisitSpectrum
            .update(dithered=True)
            .where(
                (ApogeeVisitSpectrum.release == "dr17")
            &   (ApogeeVisitSpectrum.dithered.is_null())
            )
            .execute()
        )

    # Only fetch the fields we need (pk, release, apred, prefix, reduction, telescope, field, plate, mjd, fiber, obj)
    # to compute absolute_path without loading full model instances
    q = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum.pk,
            ApogeeVisitSpectrum.release,
            ApogeeVisitSpectrum.apred,
            ApogeeVisitSpectrum.prefix,
            ApogeeVisitSpectrum.reduction,
            ApogeeVisitSpectrum.telescope,
            ApogeeVisitSpectrum.field,
            ApogeeVisitSpectrum.plate,
            ApogeeVisitSpectrum.mjd,
            ApogeeVisitSpectrum.fiber,
            ApogeeVisitSpectrum.obj
        )
        .where(
            ApogeeVisitSpectrum.dithered.is_null()
        &   ~ApogeeVisitSpectrum.flag_missing_or_corrupted_file
        )
        .limit(limit)
    )

    # Fetch all at once (needed for absolute_path computation which requires model instance)
    work_items = [(s.pk, s.absolute_path) for s in q]

    if not work_items:
        queue.put(Ellipsis)
        return 0

    results = {}  # pk -> dithered value
    queue.put(dict(description="Scraping APOGEE visit spectra headers", total=len(work_items), completed=0))

    # Use ThreadPoolExecutor - much faster for I/O-bound file reading
    completed = 0
    update_interval = max(len(work_items) // 100, 1)  # Update progress ~100 times
    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        futures = {executor.submit(_migrate_dithered_metadata, pk, path): pk for pk, path in work_items}

        for future in concurrent.futures.as_completed(futures):
            pk, dithered = future.result()
            results[pk] = dithered
            completed += 1
            if completed % update_interval == 0:
                queue.put(dict(completed=completed))

    queue.put(dict(completed=len(results)))

    n = 0
    if results:
        # Group by dithered value and use direct UPDATE statements
        # This avoids peewee's CASE statement which has boolean type issues
        true_pks = [pk for pk, dithered in results.items() if dithered is True]
        false_pks = [pk for pk, dithered in results.items() if dithered is False]
        null_pks = [pk for pk, dithered in results.items() if dithered is None]

        with database.atomic():
            if true_pks:
                for batch in chunked(true_pks, batch_size):
                    n += ApogeeVisitSpectrum.update(dithered=True).where(ApogeeVisitSpectrum.pk.in_(batch)).execute()

            if false_pks:
                for batch in chunked(false_pks, batch_size):
                    n += ApogeeVisitSpectrum.update(dithered=False).where(ApogeeVisitSpectrum.pk.in_(batch)).execute()

            if null_pks:
                # flag_missing_or_corrupted_file is bit 26 (2**26)
                missing_file_flag = 2**26
                for batch in chunked(null_pks, batch_size):
                    n += ApogeeVisitSpectrum.update(
                        dithered=None,
                        visit_flags=ApogeeVisitSpectrum.visit_flags.bin_or(missing_file_flag)
                    ).where(ApogeeVisitSpectrum.pk.in_(batch)).execute()

    queue.put(Ellipsis)
    return n


def migrate_apogee_coadds(apred: str, queue=None, batch_size: int = 1000, limit=None, incremental=True, restrict_to_stars=None, max_mjd=None):
    """
    Migrate APOGEE coadd spectrum-level information from the SDSS-V APOGEE DRP database.

    This function only loads spectrum-level data. Source creation and spectrum-to-source
    linking should be handled separately (e.g., via create_sources_and_link_spectra).

    :param restrict_to_stars: [optional]
        A sub-query selecting the (obj, telescope) pairs to ingest, as built by
        `_limited_star_selection`. Prefer this over `limit`: it lets the same star
        selection be shared with `migrate_apogee_visits`.

    :param max_mjd: [optional]
        Only ingest stars that still have a matching visit at or before this MJD (see
        `_star_has_ingestable_visit`). This has to match the cut `migrate_apogee_visits`
        applies, otherwise we ingest coadds for stars whose visits were all excluded, and
        those coadds have no Source to link to.
    """

    from astra.models.apogee import ApogeeVisitSpectrum, ApogeeCoaddedSpectrumInApStar
    from astra.models.base import database
    from astra.migrations.sdss5db.apogee_drpdb import Star, Visit, RvVisit
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"

    queue = queue or ProgressContext()

    max_star_pk = 0
    if incremental:
        max_star_pk = (
            ApogeeCoaddedSpectrumInApStar
            .select(fn.MAX(ApogeeCoaddedSpectrumInApStar.star_pk))
            .scalar() or 0
        )

    restrict_clause = (
        Tuple(Star.obj, Star.telescope).in_(restrict_to_stars)
        if restrict_to_stars is not None
        else None
    )

    # In continuous operations mode, the APOGEE DRP does not update the `star` table to have unique `star_pk`,
    # so we have to sub-query to get the most recent co-add.
    sq_where = (
        (Star.apred_vers == apred)
    &   (Star.pk > max_star_pk)
    )
    if restrict_clause is not None:
        # Safe to narrow here as well as in the outer query: restricting which
        # (obj, telescope) groups are considered does not change MAX(starver) within
        # any surviving group, and it keeps this from scanning the whole `star` table.
        sq_where &= restrict_clause

    # Deliberately NOT pushed into `sq`: catalogid can vary between starver rows of the same
    # star, so filtering there could make a superseded starver win the MAX and be ingested
    # in place of the current one. Applied only to the outer query, where it just drops
    # the star.
    outer_clause = restrict_clause
    if max_mjd is not None:
        mjd_clause = _star_has_ingestable_visit(apred, max_mjd)
        outer_clause = mjd_clause if outer_clause is None else (outer_clause & mjd_clause)

    sq = (
        Star
        .select(
            Star.apogee_id,
            Star.telescope,
            fn.MAX(Star.starver).alias("max")
        )
        .where(sq_where)
        .group_by(Star.apogee_id, Star.telescope)
    )

    q_base = (
        Star
        .select(
            Star.pk.alias("star_pk"),
            Star.apred_vers.alias("apred"),

            SQL("'sdss5'").alias("release"),
            SQL("'apStar'").alias("filetype"),
            SQL("'stars'").alias("apstar"),

            Star.obj,
            Star.telescope,
            Star.healpix,
            fn.Substr(Star.file, 1, 2).alias("prefix"),

            Star.mjdbeg.alias("min_mjd"),
            Star.mjdend.alias("max_mjd"),
            Star.starver,
            Star.nvisits.alias("n_visits"),
            Star.ngoodvisits.alias("n_good_visits"),
            Star.ngoodrvs.alias("n_good_rvs"),
            Star.snr,
            Star.starflag.alias("star_flags"),
            Star.meanfib.alias("mean_fiber"),
            Star.sigfib.alias("std_fiber"),
            Star.vrad.alias("v_rad"),
            Star.verr.alias("e_v_rad"),
            Star.vscatter.alias("std_v_rad"),
            Star.vmederr.alias("median_e_v_rad"),
            Star.rv_teff.alias("doppler_teff"),
            Star.rv_tefferr.alias("doppler_e_teff"),
            Star.rv_logg.alias("doppler_logg"),
            Star.rv_loggerr.alias("doppler_e_logg"),
            Star.rv_feh.alias("doppler_fe_h"),
            Star.rv_feherr.alias("doppler_e_fe_h"),
            Star.chisq.alias("doppler_rchi2"),
            Star.n_components,
            Star.rv_ccpfwhm.alias("ccfwhm"),
            Star.rv_autofwhm.alias("autofwhm"),
            Star.catalogid,
            Star.gaia_sourceid,
            Star.gaia_release,
            Star.sdss_id,
            Star.jmag.alias("j_mag"),
            Star.jerr.alias("e_j_mag"),
            Star.hmag.alias("h_mag"),
            Star.herr.alias("e_h_mag"),
            Star.kmag.alias("k_mag"),
            Star.kerr.alias("e_k_mag"),
            Star.sdss5_target_catalogids,
            SDSS_ID_Stacked.ra_sdss_id.alias("ra"),
            SDSS_ID_Stacked.dec_sdss_id.alias("dec"),
            SDSS_ID_Stacked.catalogid21,
            SDSS_ID_Stacked.catalogid25,
            SDSS_ID_Stacked.catalogid31,
            SDSS_ID_Flat.n_associated,
            SDSS_ID_Flat.version_id,
        )
        .distinct(Star.obj, Star.telescope)
        .join(
            sq,
            on=(
                (Star.apogee_id == sq.c.apogee_id)
            &   (Star.telescope == sq.c.telescope)
            &   (Star.starver == sq.c.max)
            )
        )
        .switch(Star)
        .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(Star.sdss_id == SDSS_ID_Stacked.sdss_id))
        .join(SDSS_ID_Flat, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
        .where(
            (Star.apred_vers == apred)
        &   (Star.pk > max_star_pk)
        &   (SDSS_ID_Flat.rank == 1)
        &   (outer_clause if outer_clause is not None else SQL("TRUE"))
        )
        .limit(None if restrict_to_stars is not None else limit)
    )
    # Count before converting to dicts (peewee .count() doesn't work well with .dicts())
    q_total = q_base.count()
    q = q_base.dicts()

    source_keys = (
        "catalogid",
        "catalogid21",
        "catalogid25",
        "catalogid31",
        "sdss_id",
        "gaia_sourceid",
        "n_associated",
        "version_id",
        "gaia_release",
        "sdss5_target_catalogids",
        "ra",
        "dec",
        "healpix",
        "j_mag",
        "e_j_mag",
        "h_mag",
        "e_h_mag",
        "k_mag",
        "e_k_mag"
    )
    spectrum_data = parse_apogee_coadd_spectrum_data(q, source_keys, queue, f"Parsing APOGEE {apred} coadd spectra", total=q_total)

    preserve = list(
        set(ApogeeCoaddedSpectrumInApStar._meta.fields.values())
    -   {
        ApogeeCoaddedSpectrumInApStar.pk,
        ApogeeCoaddedSpectrumInApStar.created,
        ApogeeCoaddedSpectrumInApStar.spectrum_pk,
        ApogeeCoaddedSpectrumInApStar.source_pk,
    }
    )
    n_updated_coadd_spectra = 0
    if spectrum_data:
        queue.put(dict(description=f"Upserting APOGEE {apred} coadded spectra", total=len(spectrum_data), completed=0))
        with database.atomic():
            for chunk in chunked(spectrum_data, batch_size):
                q = (
                    ApogeeCoaddedSpectrumInApStar
                    .insert_many(chunk)
                    .returning(ApogeeCoaddedSpectrumInApStar.pk)
                    .on_conflict(
                        conflict_target=[
                            ApogeeCoaddedSpectrumInApStar.release,
                            ApogeeCoaddedSpectrumInApStar.apred,
                            ApogeeCoaddedSpectrumInApStar.apstar,
                            ApogeeCoaddedSpectrumInApStar.obj,
                            ApogeeCoaddedSpectrumInApStar.telescope,
                            ApogeeCoaddedSpectrumInApStar.field,
                            ApogeeCoaddedSpectrumInApStar.prefix,
                        ],
                        preserve=preserve,
                        # These `where` conditions are the only scenarios where we would consider the spectrum as `modified`.
                        where=(
                            (EXCLUDED.starver >= ApogeeCoaddedSpectrumInApStar.starver)
                        )
                    )
                    .tuples()
                    .execute()
                )
                for pk in q:
                    n_updated_coadd_spectra += 1
                queue.put(dict(advance=batch_size))

    n_new_coadd_spectra = assign_spectrum_pks(ApogeeCoaddedSpectrumInApStar, batch_size, queue)

    return (n_new_coadd_spectra, n_updated_coadd_spectra)


def migrate_apogee_visits(
    apred: str,
    max_mjd: Optional[int] = None,
    queue=None,
    batch_size: int = 1000,
    limit=None,
    incremental=True,
    where_modified=None,
    restrict_to_stars=None,
):
    """
    Migrate APOGEE visit spectrum-level information from the SDSS-V APOGEE DRP database.

    This function only loads spectrum-level data. Source creation and spectrum-to-source
    linking should be handled separately (e.g., via create_sources_and_link_spectra).

    :param restrict_to_stars: [optional]
        A sub-query selecting the (obj, telescope) pairs to ingest, as built by
        `_limited_star_selection`. Prefer this over `limit`: limiting visits directly
        gives an arbitrary set of visits that does not correspond to the stars whose
        coadds are ingested.

    :param where_modified: [optional]
        A clause to specify when spectra should be considered as 'modified'. Spectra are already
        considered modified if:
        - there is a new RV measurement (`rv_visit_pk`) when there was none before
        - there is an old RV measurement (`rv_visit_pk`) and now it is set to null (old RV measurement was bad)
        - there is an updated RV measurement (`rv_visit_pk` is greater than before)

        One `where_modified` we have used in IPL-4 was:

            ```
            where_modified = (ApogeeVisitSpectrum.spectrum_flags != EXCLUDED.spectrum_flags)
            ```
    """

    from astra.models.apogee import ApogeeVisitSpectrum, ApogeeCoaddedSpectrumInApStar
    from astra.models.base import database
    from astra.migrations.sdss5db.apogee_drpdb import Star, Visit, RvVisit
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel

    _where_modified = (
            (ApogeeVisitSpectrum.rv_visit_pk.is_null() & EXCLUDED.rv_visit_pk.is_null(False))   # New RV measurement; none before.
        |   (ApogeeVisitSpectrum.rv_visit_pk.is_null(False) & EXCLUDED.rv_visit_pk.is_null())   # Old RV measurement was bad.
        |   (EXCLUDED.rv_visit_pk > ApogeeVisitSpectrum.rv_visit_pk)                            # Updated RV measurement.
    )
    if where_modified is not None:
        where_modified |= _where_modified



    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"
    queue = queue or ProgressContext()

    max_rv_visit_pk, max_visit_pk = (0, 0)
    if incremental:
        max_rv_visit_pk += ApogeeVisitSpectrum.select(fn.MAX(ApogeeVisitSpectrum.rv_visit_pk)).scalar() or 0
        max_visit_pk += ApogeeVisitSpectrum.select(fn.MAX(ApogeeVisitSpectrum.visit_pk)).scalar() or 0

    if max_mjd is None:
        max_mjd = 1_000_000


    # Ingest most recent RV measurements for each star.
    # TODO: Should this really be by `starver`, or should we do it by `created`?
    ssq = (
        RvVisit
        .select(
            RvVisit.visit_pk,
            fn.MAX(RvVisit.starver).alias("max")
        )
        .where(
            (RvVisit.apred_vers == apred)
        &   (RvVisit.pk > max_rv_visit_pk)
        &   (RvVisit.mjd <= max_mjd)
        )
        .group_by(RvVisit.visit_pk)
        .order_by(RvVisit.visit_pk.desc())
    )
    sq = (
        RvVisit
        .select(
            RvVisit.pk,
            RvVisit.visit_pk,
            RvVisit.star_pk,
            RvVisit.bc,
            RvVisit.vrel,
            RvVisit.vrelerr,
            RvVisit.vrad,
            RvVisit.chisq,
            RvVisit.rv_teff,
            RvVisit.rv_tefferr,
            RvVisit.rv_logg,
            RvVisit.rv_loggerr,
            RvVisit.rv_feh,
            RvVisit.rv_feherr,
            RvVisit.xcorr_vrel,
            RvVisit.xcorr_vrelerr,
            RvVisit.xcorr_vrad,
            RvVisit.n_components,
            RvVisit.visitflag,
        )
        .join(
            ssq,
            on=(
                (RvVisit.visit_pk == ssq.c.visit_pk)
            &   (RvVisit.starver == ssq.c.max)
            )
        )
    )

    q_base = (
        Visit.select(
            Visit.apred,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.fiber,
            Visit.file,
            Visit.obj,
            Visit.pk.alias("visit_pk"),
            Visit.dateobs.alias("date_obs"),
            Visit.jd,
            Visit.exptime,
            Visit.nframes.alias("n_frames"),
            Visit.assigned,
            Visit.on_target,
            Visit.valid,
            Visit.snr,
            fn.COALESCE(sq.c.visitflag, Visit.visitflag).alias("visit_flags"),  # if no RV calculated, include flag from the visit
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),

            # Most recent radial velocity measurement.
            sq.c.bc,
            sq.c.vrel.alias("v_rel"),
            sq.c.vrelerr.alias("e_v_rel"),
            sq.c.vrad.alias("v_rad"),
            sq.c.chisq.alias("doppler_rchi2"),
            sq.c.rv_teff.alias("doppler_teff"),
            sq.c.rv_tefferr.alias("doppler_e_teff"),
            sq.c.rv_logg.alias("doppler_logg"),
            sq.c.rv_loggerr.alias("doppler_e_logg"),
            sq.c.rv_feh.alias("doppler_fe_h"),
            sq.c.rv_feherr.alias("doppler_e_fe_h"),
            sq.c.xcorr_vrel.alias("xcorr_v_rel"),
            sq.c.xcorr_vrelerr.alias("xcorr_e_v_rel"),
            sq.c.xcorr_vrad.alias("xcorr_v_rad"),
            sq.c.n_components,
            sq.c.pk.alias("rv_visit_pk"),
            sq.c.star_pk.alias("star_pk"),

            # Source information,
            Visit.catalogid,
            Visit.sdss_id,
            Visit.healpix,
            Visit.sdss5_target_catalogids,
            Visit.ra_sdss_id.alias("ra"),
            Visit.dec_sdss_id.alias("dec"),
            Visit.gaia_sourceid,
            Visit.gaia_release,
            SDSS_ID_Stacked.catalogid21,
            SDSS_ID_Stacked.catalogid25,
            SDSS_ID_Stacked.catalogid31,
            SDSS_ID_Flat.version_id,
            SDSS_ID_Flat.n_associated,

            Visit.jmag.alias("j_mag"),
            Visit.jerr.alias("e_j_mag"),
            Visit.hmag.alias("h_mag"),
            Visit.herr.alias("e_h_mag"),
            Visit.kmag.alias("k_mag"),
            Visit.kerr.alias("e_k_mag"),
        )
        .distinct(Visit.apred, Visit.mjd, Visit.plate, Visit.telescope, Visit.field, Visit.fiber)
        .join(sq, JOIN.LEFT_OUTER, on=(Visit.pk == sq.c.visit_pk))
        .switch(Visit)
        .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(Visit.sdss_id == SDSS_ID_Stacked.sdss_id))
        .join(SDSS_ID_Flat, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
        .where(
            (Visit.apred == apred)
        &   (Visit.pk > max_visit_pk)
        &   (Visit.mjd <= max_mjd)
        &   (SDSS_ID_Flat.rank == 1)
        &   (
                Tuple(Visit.obj, Visit.telescope).in_(restrict_to_stars)
                if restrict_to_stars is not None
                else SQL("TRUE")
            )
        )
        .limit(None if restrict_to_stars is not None else limit)
    )
    # Count before converting to dicts (peewee .count() doesn't work well with .dicts())
    q_total = q_base.count()
    q = q_base.dicts()

    # For each visit, pop out the source information and assign a source ID.
    source_keys = (
        "catalogid",
        "catalogid21",
        "catalogid25",
        "catalogid31",
        "version_id",
        "n_associated",
        "sdss_id",
        "gaia_sourceid",
        "gaia_release",
        "sdss5_target_catalogids",
        "ra",
        "dec",
        "healpix",
        "j_mag",
        "e_j_mag",
        "h_mag",
        "e_h_mag",
        "k_mag",
        "e_k_mag"
    )
    spectrum_data = parse_apogee_visit_spectrum_data(q, source_keys, queue, f"Parsing APOGEE {apred} visit spectra", total=q_total)

    preserve = list(
        set(ApogeeVisitSpectrum._meta.fields.values())
    -   {
            ApogeeVisitSpectrum.pk,
            ApogeeVisitSpectrum.created,
            ApogeeVisitSpectrum.spectrum_pk,
            ApogeeVisitSpectrum.source_pk,
        }
    )
    n_updated_visit_spectra = 0
    if spectrum_data:
        queue.put(dict(description=f"Upserting APOGEE {apred} visit spectra", total=len(spectrum_data), completed=0))
        with database.atomic():
            for chunk in chunked(spectrum_data, batch_size):
                _chunk = []
                for r in chunk:
                    r.pop("sdss_id")
                    _chunk.append(r)

                q = (
                    ApogeeVisitSpectrum
                    .insert_many(_chunk)
                    .returning(ApogeeVisitSpectrum.pk)
                    .on_conflict(
                        conflict_target=[
                            ApogeeVisitSpectrum.release,
                            ApogeeVisitSpectrum.apred,
                            ApogeeVisitSpectrum.mjd,
                            ApogeeVisitSpectrum.plate,
                            ApogeeVisitSpectrum.telescope,
                            ApogeeVisitSpectrum.field,
                            ApogeeVisitSpectrum.fiber,
                            ApogeeVisitSpectrum.prefix,
                            ApogeeVisitSpectrum.reduction
                        ],
                        preserve=preserve,
                        # These `where` conditions are the only scenarios where we would consider the spectrum as `modified`.
                        where=_where_modified
                    )
                    .tuples()
                    .execute()
                )
                for pk in q:
                    n_updated_visit_spectra += 1
                queue.put(dict(advance=min(batch_size, len(chunk))))

    n_new_visit_spectra = assign_spectrum_pks(ApogeeVisitSpectrum, batch_size, queue)

    return (n_new_visit_spectra, n_updated_visit_spectra)



def parse_apogee_visit_spectrum_data(q, source_keys, queue, description, k=1000, total=None):
    """
    Parse APOGEE visit spectrum data, keeping only spectrum-level fields.

    This removes source-only fields and prepares the data for upserting.
    """
    spectrum_data = []
    if total is None:
        total = q.count()
    if total > 0:
        queue.put(dict(description=description, total=total, completed=0))
        # Keys to remove (source-only, except catalogid which is kept for linking)
        keys_to_remove = set(source_keys) - {"catalogid", "sdss_id"}
        for i, r in enumerate(q.iterator()):
            # Add release and transform fields
            r.update(
                dict(
                    release="sdss5",
                    prefix=r.pop("file").lstrip()[:2],
                    plate=r["plate"].lstrip()
                )
            )
            # Remove source-only keys
            for key in keys_to_remove:
                r.pop(key, None)

            spectrum_data.append(r)
            if i > 0 and i % k == 0:
                queue.put(dict(advance=k))

    return spectrum_data


def parse_apogee_coadd_spectrum_data(q, source_keys, queue, description, k=1000, total=None):
    """
    Parse APOGEE coadd spectrum data, keeping only spectrum-level fields.

    This removes source-only fields and prepares the data for upserting.
    """
    spectrum_data = []
    if total is None:
        total = q.count()
    if total > 0:
        queue.put(dict(description=description, total=total, completed=0))
        # Keys to remove (source-only, except catalogid/healpix which are kept)
        keys_to_remove = set(source_keys) - {"catalogid", "healpix"}
        for i, r in enumerate(q.iterator()):
            # Remove source-only keys
            for key in keys_to_remove:
                r.pop(key, None)

            spectrum_data.append(r)
            if i > 0 and i % k == 0:
                queue.put(dict(advance=k))

    return spectrum_data


def assign_spectrum_pks(model, batch_size, queue):
    from astra.models.base import database

    q = (
        model
        .select(model.pk)
        .where(model.spectrum_pk.is_null())
    )
    n = 0
    if q:
        queue.put(dict(description="Assigning spectrum primary keys", total=q.count(), completed=0))
        with database.atomic():
            for batch in chunked(q.tuples(), batch_size):
                n += (
                    model
                    .update(
                        spectrum_pk=Case(None, [
                            (model.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(batch)
                        ])
                    )
                    .where(model.pk.in_(batch))
                    .execute()
                )
                queue.put(dict(advance=min(batch_size, len(batch))))
    return n


def migrate_sdss4_dr17_apogee_spectra_from_sdss5_catalogdb(batch_size: Optional[int] = 10_000, limit: Optional[int] = None, queue=None):
    """
    Migrate SDSS4 DR17 APOGEE spectrum-level information from the SDSS-V catalog database.

    This function only loads spectrum-level data. Source creation and spectrum-to-source
    linking should be handled separately (e.g., via create_sources_and_link_spectra).

    :param batch_size: [optional]
        The batch size to use when upserting data.

    :returns:
        A tuple of new spectrum identifiers (`astra.models.apogee.ApogeeVisitSpectrum.spectrum_id`)
        that were inserted.
    """
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.base import database
    from astra.models.spectrum import Spectrum

    if queue is None:
        queue = ProgressContext()

    from astra.migrations.sdss5db.catalogdb import (
        SDSS_DR17_APOGEE_Allvisits as Visit,
        AllStar_DR17_synspec_rev1 as Star
    )

    # Query visit spectra directly
    q = (
        Visit
        .select(
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.apogee_id.alias("obj"), # see notes in astra.models.apogee.ApogeeVisitSpectrum about this
            Visit.fiberid.alias("fiber"),
            Visit.jd,
            Visit.dateobs.alias("date_obs"),
            Visit.starflag.alias("visit_flags"),
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),
            Visit.snr,
            Visit.file,

            # Radial velocity information
            Visit.vrel.alias("v_rel"),
            Visit.vrelerr.alias("e_v_rel"),
            Visit.vhelio.alias("v_rad"),
            Visit.bc,
            Visit.rv_teff.alias("doppler_teff"),
            Visit.rv_logg.alias("doppler_logg"),
            Visit.rv_feh.alias("doppler_fe_h"),
            Visit.xcorr_vrel.alias("xcorr_v_rel"),
            Visit.xcorr_vrelerr.alias("xcorr_e_v_rel"),
            Visit.xcorr_vhelio.alias("xcorr_v_rad"),
            Visit.rv_chi2.alias("doppler_rchi2"),
            Visit.ccfwhm,
            Visit.autofwhm,
            Visit.n_components,
            Visit.rv_flag.alias("doppler_flags"),
        )
        .limit(limit)
        .dicts()
    )

    # Process in batches to avoid memory issues and provide progress
    apogee_visit_spectra = []
    row_count = 0
    with queue.subtask("Fetching APOGEE DR17 visit spectra", total=None) as fetch_step:
        for row in q.iterator():
            basename = row.pop("file")
            row["plate"] = row["plate"].lstrip()
            if row["telescope"] == "apo1m":
                row["reduction"] = row["obj"]

            apogee_visit_spectra.append({
                "release": "dr17",
                "apred": "dr17",
                "prefix": basename.lstrip()[:2],
                **row
            })
            row_count += 1

            # Update progress every 1000 rows
            if row_count % 1000 == 0:
                fetch_step.update(completed=row_count)

        fetch_step.update(completed=row_count, total=row_count)

    # Upsert the spectra
    pks = upsert_many(
        ApogeeVisitSpectrum,
        ApogeeVisitSpectrum.pk,
        apogee_visit_spectra,
        batch_size,
        queue,
        "Upserting APOGEE DR17 visit spectra"
    )

    # Assign spectrum_pk values using efficient bulk SQL
    table = f"{ApogeeVisitSpectrum._meta.schema}.{ApogeeVisitSpectrum._meta.table_name}"

    if pks:
        with queue.subtask("Assigning spectrum_pk to new spectra", total=len(pks)) as assign_step:
            for batch in chunked(pks, batch_size):
                with database.atomic():
                    # Generate new spectrum_pks in bulk
                    new_spectrum_pks = flatten(
                        Spectrum
                        .insert_many([{"spectrum_flags": 0}] * len(batch))
                        .returning(Spectrum.pk)
                        .tuples()
                        .execute()
                    )

                    # Build VALUES clause for bulk update
                    pk_pairs = list(zip(batch, new_spectrum_pks))
                    if pk_pairs:
                        values_sql = ", ".join(f"({pk}, {spk})" for pk, spk in pk_pairs)
                        sql = f"""
                            UPDATE {table}
                            SET spectrum_pk = v.spectrum_pk
                            FROM (VALUES {values_sql}) AS v(pk, spectrum_pk)
                            WHERE {table}.pk = v.pk
                        """
                        database.execute_sql(sql)

                assign_step.update(advance=len(batch))

    # Sanity check - assign spectrum_pk to any spectra missing it
    missing_pks = flatten(
        ApogeeVisitSpectrum
        .select(ApogeeVisitSpectrum.pk)
        .where(ApogeeVisitSpectrum.spectrum_pk.is_null())
        .tuples()
    )

    if missing_pks:
        with queue.subtask("Fixing missing spectrum_pk values", total=len(missing_pks)) as fix_step:
            for batch in chunked(missing_pks, batch_size):
                with database.atomic():
                    new_spectrum_pks = flatten(
                        Spectrum
                        .insert_many([{"spectrum_flags": 0}] * len(batch))
                        .returning(Spectrum.pk)
                        .tuples()
                        .execute()
                    )

                    pk_pairs = list(zip(batch, new_spectrum_pks))
                    if pk_pairs:
                        values_sql = ", ".join(f"({pk}, {spk})" for pk, spk in pk_pairs)
                        sql = f"""
                            UPDATE {table}
                            SET spectrum_pk = v.spectrum_pk
                            FROM (VALUES {values_sql}) AS v(pk, spectrum_pk)
                            WHERE {table}.pk = v.pk
                        """
                        database.execute_sql(sql)

                fix_step.update(advance=len(batch))

    assert not (
        ApogeeVisitSpectrum
        .select(ApogeeVisitSpectrum.pk)
        .where(ApogeeVisitSpectrum.spectrum_pk.is_null())
        .exists()
    )
    queue.put(Ellipsis)
    return None

    
def migrate_sdss4_dr17_apogee_coadded_spectra_from_visits(batch_size: Optional[int] = 10_000, queue=None):
    """
    Derive DR17 APOGEE coadded (apStar) spectra from already-ingested `ApogeeVisitSpectrum`
    rows and `Source.sdss4_apogee_id`.
 
    This MUST be run after both:
      - `migrate_sdss4_dr17_apogee_spectra_from_sdss5_catalogdb` (populates ApogeeVisitSpectrum)
      - `migrate_sdss4_apogee_id` (populates Source.sdss4_apogee_id)
 
    since coadds are matched to sources via `Source.sdss4_apogee_id`. Calling this before
    `Source.sdss4_apogee_id` is populated will silently produce zero coadded spectra --
    the lookup dict will simply be empty and every row will fail to match.
    """
    from astra.models.apogee import ApogeeVisitSpectrum, ApogeeCoaddedSpectrumInApStar
    from astra.models.base import database
    from astra.models.source import Source
    from astra.migrations.sdss5db.catalogdb import AllStar_DR17_synspec_rev1 as Star
 
    if queue is None:
        queue = ProgressContext()
 
    # Ingest ApogeeCoadded
    # Derive coadded spectra from already-ingested visit spectra (local DB) instead of
    # re-querying the remote catalogdb, which is much faster
    from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
    from astra.models.source import Source

    # Build lookup from apogee_id to source_pk
    with queue.subtask("Building APOGEE ID lookup", total=None) as lookup_step:
        lookup_source_pk_given_sdss4_apogee_id = {
            apogee_id: pk
            for pk, apogee_id in Source.select(Source.pk, Source.sdss4_apogee_id).where(Source.sdss4_apogee_id.is_null(False)).tuples()
        }
        lookup_step.update(total=len(lookup_source_pk_given_sdss4_apogee_id), completed=len(lookup_source_pk_given_sdss4_apogee_id))

    # Query our local ApogeeVisitSpectrum table for distinct (obj, field, telescope) combinations
    with queue.subtask("Querying APOGEE DR17 coadded spectra", total=None) as query_step:
        dr17_rows = list(
            ApogeeVisitSpectrum
            .select(
                ApogeeVisitSpectrum.obj,
                ApogeeVisitSpectrum.field,
                ApogeeVisitSpectrum.telescope,
            )
            .where(ApogeeVisitSpectrum.release == "dr17")
            .distinct()
            .tuples()
        )
        query_step.update(total=len(dr17_rows), completed=len(dr17_rows))

    # Lookup additional information
    q = (
        Star
        .select(
            Star.apogee_id.alias("obj"),
            Star.telescope,
            Star.field,
            Star.snr,
            Star.meanfib.alias("mean_fiber"),
            Star.sigfib.alias("std_fiber"),
            Star.vhelio_avg.alias("v_rad"),
            Star.verr.alias("e_v_rad"),
            Star.vscatter.alias("std_v_rad"),
            Star.rv_teff.alias("doppler_teff"),
            Star.rv_logg.alias("doppler_logg"),
            Star.rv_feh.alias("doppler_fe_h"),
            Star.rv_chi2.alias("doppler_rchi2"),
            Star.rv_flag.alias("doppler_flags"),
            Star.rv_ccfwhm.alias("ccfwhm"),
            Star.rv_autofwhm.alias("autofwhm"),
            Star.starflag.alias("star_flags"),
        )
        .dicts()
    )
    star_meta = {
        (s["obj"], s["field"], s["telescope"]): s for s in q
    }


    with queue.subtask("Processing APOGEE DR17 coadded spectra", total=len(dr17_rows)) as process_step:
        apogee_coadded_spectra = []
        for obj, field, telescope in dr17_rows:
            source_pk = lookup_source_pk_given_sdss4_apogee_id.get(obj)
            if source_pk is not None:

                key = (obj, field, telescope)
                s = dict(
                    source_pk=source_pk,
                    release="dr17",
                    filetype="apStar",
                    apred="dr17",
                    apstar="stars",
                    obj=obj,
                    telescope=telescope,
                    field=field,
                    prefix="ap" if telescope.startswith("apo") else "as",
                )
                s.update(star_meta[key])
                apogee_coadded_spectra.append(s)
        process_step.update(completed=len(dr17_rows))


    # Upsert the spectra
    pks = upsert_many(
        ApogeeCoaddedSpectrumInApStar,
        ApogeeCoaddedSpectrumInApStar.pk,
        apogee_coadded_spectra,
        batch_size,
        queue,
        "Upserting APOGEE DR17 coadded spectra"
    )

    # Update existing spectra with new information if it exists.
    for chunk in tqdm(chunked(apogee_coadded_spectra, batch_size), desc="Updating"):
        (
            ApogeeCoaddedSpectrumInApStar
            .insert_many(chunk)
            .on_conflict(
                conflict_target=[
                    ApogeeCoaddedSpectrumInApStar.release,
                    ApogeeCoaddedSpectrumInApStar.apred,
                    ApogeeCoaddedSpectrumInApStar.apstar,
                    ApogeeCoaddedSpectrumInApStar.obj,
                    ApogeeCoaddedSpectrumInApStar.telescope,
                    ApogeeCoaddedSpectrumInApStar.field,
                    ApogeeCoaddedSpectrumInApStar.prefix,
                ],
                preserve=(
                    ApogeeCoaddedSpectrumInApStar.snr,
                    ApogeeCoaddedSpectrumInApStar.mean_fiber,
                    ApogeeCoaddedSpectrumInApStar.std_fiber,
                    ApogeeCoaddedSpectrumInApStar.v_rad,
                    ApogeeCoaddedSpectrumInApStar.e_v_rad,
                    ApogeeCoaddedSpectrumInApStar.std_v_rad,
                    ApogeeCoaddedSpectrumInApStar.doppler_teff,
                    ApogeeCoaddedSpectrumInApStar.doppler_logg,
                    ApogeeCoaddedSpectrumInApStar.doppler_fe_h,
                    ApogeeCoaddedSpectrumInApStar.doppler_rchi2,
                    ApogeeCoaddedSpectrumInApStar.doppler_flags,
                    ApogeeCoaddedSpectrumInApStar.ccfwhm,
                    ApogeeCoaddedSpectrumInApStar.autofwhm,
                    ApogeeCoaddedSpectrumInApStar.star_flags,
                ),
                update={
                    ApogeeCoaddedSpectrumInApStar.modified: datetime.now()
                }
            )
            .execute()
        )

    # Assign spectrum_pk values to any spectra missing it.
    N = len(pks)
    if pks:
        with queue.subtask("Assigning primary keys to spectra", total=N) as assign_step:
            N_assigned = 0
            with database.atomic():
                for batch in chunked(pks, batch_size):
                    cases = []
                    for spectrum_pk, pk in enumerate_new_spectrum_pks(batch):
                        cases.append((ApogeeCoaddedSpectrumInApStar.pk == pk, spectrum_pk))

                    B = (
                        ApogeeCoaddedSpectrumInApStar
                        .update(spectrum_pk=Case(None, cases))
                        .where(ApogeeCoaddedSpectrumInApStar.pk.in_(batch))
                        .execute()
                    )
                    assign_step.update(advance=B)
                    N_assigned += B

    queue.put(Ellipsis)

    return None


def update_apogee_combined_spectra_from_coadds(batch_size=500, queue=None):
    """
    Update ApogeeCombinedSpectrum rows with field values from ApogeeCoaddedSpectrumInApStar
    where the coadd rows have since been populated (but the combined rows still have nulls).

    Joins on source_pk, apred, and telescope.
    """
    from astra.models.base import database
    from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
    from astra.models.mwm import ApogeeCombinedSpectrum

    queue = queue or ProgressContext()

    # These are the fields that exist on both models and could have been null
    # when ApogeeCombinedSpectrum was first created.
    shared_fields = [
        "min_mjd",
        "max_mjd",
        "n_entries",
        "n_visits",
        "n_good_visits",
        "n_good_rvs",
        "snr",
        "mean_fiber",
        "std_fiber",
        "star_flags",
        "v_rad",
        "e_v_rad",
        "std_v_rad",
        "median_e_v_rad",
        "doppler_teff",
        "doppler_e_teff",
        "doppler_logg",
        "doppler_e_logg",
        "doppler_fe_h",
        "doppler_e_fe_h",
        "doppler_rchi2",
        "doppler_flags",
        "xcorr_v_rad",
        "xcorr_v_rel",
        "xcorr_e_v_rel",
        "ccfwhm",
        "autofwhm",
        "n_components",
    ]

    Coadd = ApogeeCoaddedSpectrumInApStar
    Combined = ApogeeCombinedSpectrum

    # Find ApogeeCombinedSpectrum rows where at least one shared field is null,
    # but the corresponding ApogeeCoaddedSpectrumInApStar row has non-null values.
    # We use snr as a representative field: if it's null on Combined but not on Coadd,
    # the row likely needs updating.
    q = (
        Combined
        .select(Combined.pk, Combined.source_pk, Combined.apred, Combined.telescope)
        .join(
            Coadd,
            on=(
                (Combined.source_pk == Coadd.source_pk)
                & (Combined.apred == Coadd.apred)
                & (Combined.telescope == Coadd.telescope)
            ),
        )
        .where(
            Combined.snr.is_null()
            & Coadd.snr.is_null(False)
        )
        .tuples()
    )

    rows_to_update = list(q)
    n_total = len(rows_to_update)
    log.info(f"Found {n_total} ApogeeCombinedSpectrum rows to update from ApogeeCoaddedSpectrumInApStar")

    if n_total == 0:
        queue.put(Ellipsis)
        return 0

    n_updated = 0
    with queue.subtask(f"Updating {n_total} ApogeeCombinedSpectrum rows", total=n_total) as step:
        for batch in chunked(rows_to_update, batch_size):
            # Build a mapping of (source_pk, apred, telescope) -> combined_pk for this batch
            keys = [(source_pk, apred, telescope) for (_, source_pk, apred, telescope) in batch]
            combined_pks = [pk for (pk, _, _, _) in batch]

            # Fetch the corresponding coadd rows
            coadd_rows = (
                Coadd
                .select()
                .where(
                    fn.ROW(Coadd.source_pk, Coadd.apred, Coadd.telescope).in_(keys)
                )
                .dicts()
            )

            coadd_lookup = {}
            for row in coadd_rows:
                key = (row["source_pk"], row["apred"], row["telescope"])
                coadd_lookup[key] = row

            with database.atomic():
                for combined_pk, source_pk, apred, telescope in batch:
                    coadd_row = coadd_lookup.get((source_pk, apred, telescope))
                    if coadd_row is None:
                        continue

                    updates = {}
                    for field_name in shared_fields:
                        coadd_value = coadd_row.get(field_name)
                        if coadd_value is not None:
                            updates[field_name] = coadd_value

                    if updates:
                        updates["modified"] = datetime.now()
                        (
                            Combined
                            .update(**updates)
                            .where(Combined.pk == combined_pk)
                            .execute()
                        )
                        n_updated += 1

            step.update(advance=len(batch))

    log.info(f"Updated {n_updated} ApogeeCombinedSpectrum rows")
    queue.put(Ellipsis)
    return n_updated
