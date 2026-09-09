from typing import Optional
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import fitsio
import numpy as np
import subprocess
import concurrent.futures

from astra.utils import log, expand_path

from peewee import chunked, FloatField, IntegerField


def _parse_space_separated_floats_mean(arr):
    """Fast parsing of space-separated float strings to compute mean."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = arr[i]
        vals = s.split()
        nv = len(vals)
        if nv == 0:
            result[i] = np.nan
        elif nv == 1:
            result[i] = float(vals[0])
        else:
            total = 0.0
            for v in vals:
                total += float(v)
            result[i] = total / nv
    return result


def _parse_space_separated_floats_list(arr):
    """Fast parsing of space-separated float strings to list of floats."""
    n = len(arr)
    result = [None] * n  # Pre-allocate list
    for i in range(n):
        vals = arr[i].split()
        if vals:
            result[i] = [float(v) for v in vals]
        else:
            result[i] = []
    return result

def match_unlinked_boss_visit_spectra():
    from astra.models.base import database
    from astra.models.boss import BossVisitSpectrum
    from astra.models.source import Source

    catalogids = (
        Source.catalogid,
        Source.catalogid31,
        Source.catalogid25,
        Source.catalogid21
    )
    n = 0
    with database.atomic():
        for catalogid in catalogids:
            n += (
                BossVisitSpectrum
                .update(source_pk=Source.pk)
                .from_(Source)
                .where(
                    BossVisitSpectrum.source.is_null()
                    & (BossVisitSpectrum.catalogid == catalogid)
                )
                .execute()
            )
    return n


def migrate_from_spall_file(run2d, queue, max_mjd: Optional[int] = None, gzip=True, limit=None, batch_size=10_000, incremental=True):
    """
    Migrate all new BOSS visit spectrum-level information from the spAll file, which is generated
    by the SDSS-V BOSS data reduction pipeline.

    This function only loads spectrum-level data. Source creation and spectrum-to-source
    linking should be handled separately (e.g., via match_unlinked_boss_visit_spectra).
    """

    from astra.models.boss import BossVisitSpectrum
    from astra.models.base import database
    from astra.models.spectrum import Spectrum
    from astra.migrations.utils import ProgressContext
    from astra.utils import flatten

    if queue is None:
        queue = ProgressContext()

    # Handle frozen data path.
    if run2d == "v6_2_1":
        path = expand_path(f"$SAS_BASE_DIR/ipl-5/spectro/boss/redux/v6_2_1/summary/daily/spAll-v6_2_1.fits")
    else:
        path = expand_path(f"$BOSS_SPECTRO_REDUX/{run2d}/summary/daily/spAll-{run2d}.fits")
    if gzip:
        path += ".gz"

    translations = {
        "NEXP": "n_exp",
        "XCSAO_RV": "xcsao_v_rad",
        "XCSAO_ERV": "xcsao_e_v_rad",
        "XCSAO_RXC": "xcsao_rxc",
        "XCSAO_TEFF": "xcsao_teff",
        "XCSAO_ETEFF": "xcsao_e_teff",
        "XCSAO_LOGG": "xcsao_logg",
        "XCSAO_ELOGG": "xcsao_e_logg",
        "XCSAO_FEH": "xcsao_fe_h",
        "XCSAO_EFEH": "xcsao_e_fe_h",
        "ZWARNING": "zwarning_flags",
        "GRI_GAIA_TRANSFORM": "gri_gaia_transform_flags",
        "EXPTIME": "exptime",
        "AIRMASS": "airmass",
        "SEEING50": "seeing",
        "OBS": "telescope",
        "MOON_DIST": "moon_dist_mean",
        "MOON_PHASE": "moon_phase_mean",
        "FIELD": "fieldid",
        "MJD": "mjd",
        "CATALOGID": "catalogid",
        "HEALPIX": "healpix",
        "DELTA_RA_LIST": "delta_ra",
        "DELTA_DEC_LIST": "delta_dec",
        "SN_MEDIAN_ALL": "snr",
        "SPEC_FILE": "spec_file",
    }
    columns = list(translations.keys())

    with queue.subtask(f"Reading spAll file for {run2d}", total=None) as read_step:
        spAll = fitsio.read(path, ext=1, columns=columns)

        most_recent_mjd = 0
        if incremental:
            most_recent = (
                BossVisitSpectrum
                .select()
                .where(BossVisitSpectrum.run2d == run2d)
                .order_by(BossVisitSpectrum.mjd.desc())
                .first()
            )
            if most_recent is not None:
                most_recent_mjd = most_recent.mjd

        if max_mjd is not None:
            mask = (spAll["MJD"] >= most_recent_mjd) & (spAll["MJD"] <= max_mjd)
        else:
            mask = spAll["MJD"] >= most_recent_mjd

        if limit is not None:
            index = np.where(np.cumsum(mask) == limit)[0][0]
            mask[index:] = False

        total = int(np.sum(mask))
        read_step.update(total=total, completed=total)

    if total == 0:
        queue.put(Ellipsis)
        return None

    with queue.subtask(f"Processing BOSS {run2d} metadata", total=5) as meta_step:
        # Extract masked arrays for all columns at once
        masked_data = {to_key: spAll[from_key][mask] for from_key, to_key in translations.items()}
        meta_step.update(advance=1)

        # Vectorized telescope transformation: "APO" -> "apo25m", "LCO" -> "lco25m"
        telescope_raw = masked_data["telescope"]
        telescope_arr = np.char.add(np.char.lower(telescope_raw.astype(str)), "25m")
        meta_step.update(advance=1)

        # Vectorized moon_dist and moon_phase mean calculations
        moon_dist_arr = _parse_space_separated_floats_mean(masked_data["moon_dist_mean"].astype(str))
        moon_phase_arr = _parse_space_separated_floats_mean(masked_data["moon_phase_mean"].astype(str))
        meta_step.update(advance=1)

        # Parse delta_ra and delta_dec to lists of floats
        delta_ra_list = _parse_space_separated_floats_list(masked_data["delta_ra"].astype(str))
        delta_dec_list = _parse_space_separated_floats_list(masked_data["delta_dec"].astype(str))
        meta_step.update(advance=1)

        # Vectorized fiber_offset calculation
        fiber_offset_arr = np.array([
            any(abs(ra) > 0 or abs(dec) > 0 for ra, dec in zip(ra_list, dec_list)) if ra_list else False
            for ra_list, dec_list in zip(delta_ra_list, delta_dec_list)
        ], dtype=bool)

        # Handle catalogid masking - vectorized
        catalogid_raw = masked_data["catalogid"]
        if hasattr(catalogid_raw, 'mask'):
            catalogid_arr = np.where(catalogid_raw.mask, -1, catalogid_raw.data)
        else:
            catalogid_arr = np.asarray(catalogid_raw)
        meta_step.update(advance=1)

    # Pre-convert numpy arrays to Python types for faster iteration
    all_columns = (
        masked_data["n_exp"].tolist(),
        masked_data["xcsao_v_rad"].tolist(),
        masked_data["xcsao_e_v_rad"].tolist(),
        masked_data["xcsao_rxc"].tolist(),
        masked_data["xcsao_teff"].tolist(),
        masked_data["xcsao_e_teff"].tolist(),
        masked_data["xcsao_logg"].tolist(),
        masked_data["xcsao_e_logg"].tolist(),
        masked_data["xcsao_fe_h"].tolist(),
        masked_data["xcsao_e_fe_h"].tolist(),
        masked_data["zwarning_flags"].tolist(),
        masked_data["gri_gaia_transform_flags"].tolist(),
        masked_data["exptime"].tolist(),
        masked_data["airmass"].tolist(),
        masked_data["seeing"].tolist(),
        telescope_arr.tolist(),
        moon_dist_arr.tolist(),
        moon_phase_arr.tolist(),
        masked_data["fieldid"].tolist(),
        masked_data["mjd"].tolist(),
        catalogid_arr.tolist(),
        masked_data["healpix"].tolist(),
        delta_ra_list,
        delta_dec_list,
        masked_data["snr"].tolist(),
        masked_data["spec_file"].tolist(),
        fiber_offset_arr.tolist(),
    )

    # Build all rows at once using zip (much faster than index access)
    spectrum_data = [
        {
            "release": "sdss5",
            "run2d": run2d,
            "filetype": "specFull",
            "n_exp": row[0],
            "xcsao_v_rad": row[1],
            "xcsao_e_v_rad": row[2],
            "xcsao_rxc": row[3],
            "xcsao_teff": row[4],
            "xcsao_e_teff": row[5],
            "xcsao_logg": row[6],
            "xcsao_e_logg": row[7],
            "xcsao_fe_h": row[8],
            "xcsao_e_fe_h": row[9],
            "zwarning_flags": row[10],
            "gri_gaia_transform_flags": row[11],
            "exptime": row[12],
            "airmass": row[13],
            "seeing": row[14],
            "telescope": row[15],
            "moon_dist_mean": row[16],
            "moon_phase_mean": row[17],
            "fieldid": row[18],
            "mjd": row[19],
            "catalogid": row[20],
            "healpix": row[21],
            "delta_ra": row[22],
            "delta_dec": row[23],
            "snr": row[24],
            "spec_file": row[25],
            "fiber_offset": row[26],
        }
        for row in zip(*all_columns)
    ]

    # Upsert spectra in batches
    pks = []
    with queue.subtask(f"Upserting BOSS {run2d} spectra", total=total) as upsert_step:
        with database.atomic():
            for chunk in chunked(spectrum_data, batch_size):
                chunk_pks = flatten(
                    BossVisitSpectrum
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .returning(BossVisitSpectrum.pk)
                    .tuples()
                    .execute()
                )
                pks.extend(chunk_pks)
                upsert_step.update(advance=len(chunk))

    # Assign spectrum_pk values using bulk operations
    N = len(pks)
    if pks:
        with queue.subtask(f"Assigning spectrum_pk to {N} BOSS {run2d} spectra", total=N) as assign_step:
            with database.atomic():
                for batch in chunked(pks, batch_size):
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
                        # Use raw SQL for efficient bulk update with VALUES
                        values_sql = ", ".join(f"({pk}, {spk})" for pk, spk in pk_pairs)
                        table = f"{BossVisitSpectrum._meta.schema}.{BossVisitSpectrum._meta.table_name}"
                        sql = f"""
                            UPDATE {table}
                            SET spectrum_pk = v.spectrum_pk
                            FROM (VALUES {values_sql}) AS v(pk, spectrum_pk)
                            WHERE {table}.pk = v.pk
                        """
                        database.execute_sql(sql)

                    assign_step.update(advance=len(batch))

    queue.put(Ellipsis)
    return None


def _migrate_specfull_metadata(spectra, fields, raise_exceptions=True, full_output=False):

    K = len(fields)
    keys_str = "|".join([f"({k})" for k in fields.values()])

    # 80 chars per line, 150 lines -> 12000
    command_template = " | ".join([
        'hexdump -n 80000 -e \'80/1 "%_p" "\\n"\' {path}',
        f'egrep "{keys_str}"',
        f"head -n {K+5}"
    ])
    commands = ""
    for specFull in spectra:
        path = expand_path(specFull.path)
        commands += f"{command_template.format(path=path)}\n"

    outputs = subprocess.check_output(commands, shell=True, text=True)
    outputs = outputs.strip().split("\n")

    p, all_metadata = (-1, {})
    for line in outputs:
        try:
            key, *values = line.split("= ")
            key, value = (key.strip(), values[0].split()[0].strip(" '"))
        except (IndexError, ValueError): # binary data, probably
            continue

        for field, from_key in fields.items():
            if from_key == key:
                break
        else:
            continue

        name = field.name
        if line[8:10] != "= ": # not a key=value line
            #log.warning(f"Skipping line '{line}' because not a valid line")
            continue

        if name == "plateid":
            p += 1
        pk = spectra[p].pk
        all_metadata.setdefault(pk, {})
        if name in all_metadata[pk]:
            log.warning(f"Multiple key `{name}` found in {spectra[p]}: {expand_path(spectra[p].path)}")
            log.warning(f"\tKeeping existing (k, v) pair: {name}={all_metadata[pk][name]} and ignoring new value: {value}")
            continue

        if isinstance(field, IntegerField):
            try:
                value = int(float(value))
            except:
                value = -1
        elif isinstance(field, FloatField):
            try:
                value = float(value)
            except:
                value = np.nan

        all_metadata[pk][name] = value

    missing_key_counts, examples = ({}, {})
    for pk, meta in all_metadata.items():
        for field, from_key in fields.items():
            if field.name not in meta:
                missing_key_counts.setdefault(field.name, 0)
                missing_key_counts[field.name] += 1
                examples[field.name] = pk

    #if missing_key_counts:
    #    log.warning(f"There are missing keys in some spectra:")
    #    for key, count in missing_key_counts.items():
    #        log.warning(f"\t{key} is missing in {count} spectra in this batch. Example pk={examples[key]}")

    return (all_metadata, missing_key_counts, outputs) if full_output else (all_metadata, missing_key_counts)


def migrate_specfull_metadata_from_image_headers(
    max_workers: Optional[int] = 128,
    limit: Optional[int] = None,
    batch_size: Optional[int] = 100,
    queue = None
):
    from astra.models.boss import BossVisitSpectrum
    from astra.models.base import database
    from astra.migrations.utils import ProgressContext

    if queue is None:
        queue = ProgressContext()

    # Reconnect database in subprocess to avoid stale connections
    if not database.is_closed():
        database.close()
    database.connect()

    q = (
        BossVisitSpectrum
        .select()
        .where(BossVisitSpectrum.alt.is_null())# & (BossVisitSpectrum.catalogid > 0))
        .limit(limit)
    )
    if q.count() == 0:
        queue.put(Ellipsis)
        return None

    fields = {
        BossVisitSpectrum.plateid: "PLATEID",
        BossVisitSpectrum.cartid: "CARTID",
        BossVisitSpectrum.mapid: "MAPID",
        BossVisitSpectrum.slitid: "SLITID",
        BossVisitSpectrum.psfsky: "PSFSKY",
        BossVisitSpectrum.preject: "PREJECT",
        BossVisitSpectrum.n_std: "NSTD",
        BossVisitSpectrum.n_gal: "NGAL",
        BossVisitSpectrum.lowrej: "LOWREJ",
        BossVisitSpectrum.highrej: "HIGHREJ",
        BossVisitSpectrum.scatpoly: "SCATPOLY",
        BossVisitSpectrum.proftype: "PROFTYPE",
        BossVisitSpectrum.nfitpoly: "NFITPOLY",
        BossVisitSpectrum.alt: "ALT",
        BossVisitSpectrum.az: "AZ",
        BossVisitSpectrum.airmass: "AIRMASS",
        BossVisitSpectrum.airtemp: "AIRTEMP",
        BossVisitSpectrum.dewpoint: "DEWPOINT",
        BossVisitSpectrum.dust_a: "DUSTA",
        BossVisitSpectrum.dust_b: "DUSTB",
        BossVisitSpectrum.gust_speed: "GUSTS",
        BossVisitSpectrum.gust_direction: "GUSTD",
        BossVisitSpectrum.humidity: "HUMIDITY",
        BossVisitSpectrum.pressure: "PRESSURE",
        BossVisitSpectrum.wind_direction: "WINDD",
        BossVisitSpectrum.wind_speed: "WINDS",
        BossVisitSpectrum.tai_beg: "TAI-BEG",
        BossVisitSpectrum.tai_end: "TAI-END",
        BossVisitSpectrum.n_guide: "NGUIDE",
        BossVisitSpectrum.skychi2: "SKYCHI2",
        BossVisitSpectrum.schi2min: "SCHI2MIN",
        BossVisitSpectrum.schi2max: "SCHI2MAX",
    }

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    defaults = {
        "n_guide": -1,
        "airtemp": np.nan,
        "dewpoint": np.nan,
        "n_std": -1
    }

    specFulls, futures = ({}, [])
    all_missing_counts = {}
    try:
        with queue.subtask("Scraping specFull headers", total=None) as scrape_step:
            for chunk in chunked(q, batch_size):
                futures.append(executor.submit(_migrate_specfull_metadata, chunk, fields))
                for spec in chunk:
                    specFulls[spec.pk] = spec
                scrape_step.update(advance=len(chunk))
            scrape_step.update(total=len(specFulls), completed=len(specFulls))

        with queue.subtask("Parsing specFull metadata", total=len(futures)) as parse_step:
            for future in concurrent.futures.as_completed(futures):
                metadata, missing_counts = future.result()
                for name, missing_count in missing_counts.items():
                    all_missing_counts.setdefault(name, 0)
                    all_missing_counts[name] += missing_count

                for pk, meta in metadata.items():
                    for key, value in meta.items():
                        setattr(specFulls[pk], key, value)
                    for key, value in defaults.items():
                        if key not in meta:
                            setattr(specFulls[pk], key, value)

                parse_step.update(advance=1)
    finally:
        # Always tear down the worker pool; a leaked ProcessPoolExecutor can
        # deadlock interpreter shutdown (especially under the 'fork' start
        # method), which hangs the migration scheduler's process.join().
        executor.shutdown(wait=True)

    with queue.subtask("Ingesting specFull metadata", total=len(specFulls)) as ingest_step:
        for chunk in chunked(specFulls.values(), batch_size):
            (
                BossVisitSpectrum
                .bulk_update(
                    chunk,
                    fields=list(fields.keys())
                )
            )
            ingest_step.update(advance=len(chunk))

    queue.put(Ellipsis)

    return None
