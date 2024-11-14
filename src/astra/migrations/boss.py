from typing import Optional
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import numpy as np
import subprocess
import concurrent.futures

from astra.utils import log, expand_path, dict_to_iterable

from peewee import (
    chunked,
    JOIN, 
    Case,
    FloatField,
    IntegerField
)


def migrate_from_spall_file(run2d, queue, gzip=True, limit=None, batch_size=1000):
    """
    Migrate all new BOSS visit information (`specFull` files) stored in the spAll file, which is generated
    by the SDSS-V BOSS data reduction pipeline.
    """

    from astra.models.base import database
    from astra.models.boss import BossVisitSpectrum
    from astra.models.source import Source
    from astra.migrations.utils import enumerate_new_spectrum_pks, upsert_many, NoQueue

    path = expand_path(f"$BOSS_SPECTRO_REDUX/{run2d}/summary/daily/spAll-{run2d}.fits")
    if gzip:
        path += ".gz"
    
    from astra.migrations.sdss5db.catalogdb import (
        Catalog,
        CatalogToGaia_DR2,
        CatalogToGaia_DR3,
        CatalogdbModel
    )

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"
            
    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"


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

        # Some additional identifiers that we don't necessarily need, but will take for now
        "CATALOGID_V0": "catalogid_v0",
        "CATALOGID_V0P5": "catalogid_v0p5",
        "SDSS_ID": "sdss_id",
        "GAIA_ID": "gaia_dr2_source_id",
        "FIRSTCARTON": "carton_0"
    }
    source_keys_only = ("catalogid_v0", "catalogid_v0p5", "sdss_id", "gaia_dr2_source_id", "carton_0") 
    transformations = {
        "telescope": lambda x: f"{x.lower()}25m",
        "moon_dist_mean": lambda x: np.mean(tuple(map(float, x.split()))),
        "moon_phase_mean": lambda x: np.mean(tuple(map(float, x.split()))),
        "delta_ra": lambda x: list(map(float, x.split())),
        "delta_dec": lambda x: list(map(float, x.split()))
    }

    #with fits.open(path) as hdul:
    #queue.put(dict(de))
    hdul = fits.open(path)
    spAll = hdul[1].data
    if limit is not None:
        spAll = spAll[:limit]

    total = len(spAll)
    queue.put(dict(total=len(translations), description=f"Parsing BOSS {run2d} metadata"))

    spectrum_data_dicts = dict(
        release=["sdss5"] * total,
        run2d=[run2d] * total,
        filetype=["specFull"] * total,
    )
    for from_key, to_key in translations.items(): 
        queue.put(dict(advance=1, description=f"Parsing BOSS {run2d} {to_key}"))
        spectrum_data_dicts[to_key] = spAll[from_key]
    
    queue.put(dict(description=f"Transforming BOSS {run2d} metadata", total=len(transformations), completed=0))
    for key, fun in transformations.items():
        queue.put(dict(advance=1, description=f"Transforming BOSS {run2d} {key}")) 
        spectrum_data_dicts[key] = list(map(fun, spectrum_data_dicts[key]))
    
    spectrum_data_dicts["fiber_offset"] = [np.any((np.abs(ra) + np.abs(dec)) > 0) for ra, dec in zip(spectrum_data_dicts["delta_ra"], spectrum_data_dicts["delta_dec"])]

    queue.put(dict(description=f"Converting BOSS {run2d} data types", total=None, completed=0))

    spectrum_data = list(dict_to_iterable(spectrum_data_dicts))

    # We need to get sdss_id and catalog information for each source.
    source_data = {}
    queue.put(dict(description=f"Linking BOSS {run2d} spectra to catalog", total=total, completed=0))
    for chunk in chunked(spectrum_data, batch_size):
        chunk_catalogids = []
        gaia_dr2_source_id_given_catalogid = {}
        for row in chunk:
            for key in ("catalogid", "catalogid_v0", "catalogid_v0p5"):
                try:
                    if np.all(row[key].mask):
                        continue
                except:
                    chunk_catalogids.append(row[key])
                    gaia_dr2_source_id_given_catalogid[row[key]] = row["gaia_dr2_source_id"]

        q = (
            Catalog
            .select(
                Catalog.ra,
                Catalog.dec,
                Catalog.catalogid,
                Catalog.version_id.alias("version_id"),
                Catalog.lead,
                CatalogToGaia_DR3.target.alias("gaia_dr3_source_id"),
                SDSS_ID_Flat.sdss_id,
                SDSS_ID_Flat.n_associated,
                SDSS_ID_Stacked.catalogid21,
                SDSS_ID_Stacked.catalogid25,
                SDSS_ID_Stacked.catalogid31,
            )
            .join(SDSS_ID_Flat, JOIN.LEFT_OUTER, on=(Catalog.catalogid == SDSS_ID_Flat.catalogid))
            .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
            .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.catalogid31 == CatalogToGaia_DR3.catalog))
            .where(Catalog.catalogid.in_(chunk_catalogids))
            .dicts()
        )
                
        reference_key = "catalogid"
        for row in q:
            if row[reference_key] in source_data:
                for key, value in row.items():
                    if source_data[row[reference_key]][key] is None and value is not None:
                        if key == "sdss_id":
                            source_data[row[reference_key]][key] = min(source_data[row[reference_key]][key], value)
                        else:
                            source_data[row[reference_key]][key] = value
                continue

            source_data[row[reference_key]] = row
            gaia_dr2_source_id = gaia_dr2_source_id_given_catalogid[row[reference_key]]
            if gaia_dr2_source_id < 0:
                gaia_dr2_source_id = None
            source_data[row[reference_key]]["gaia_dr2_source_id"] = gaia_dr2_source_id
        
        queue.put({"advance": batch_size})

    # Upsert the sources
    with database.atomic():
        queue.put(dict(description=f"Upserting BOSS {run2d} sources", total=len(source_data), completed=0))
        for chunk in chunked(source_data.values(), batch_size):
            (
                Source
                .insert_many(chunk)
                .on_conflict_ignore()
                .execute()
            )
            queue.put(dict(advance=min(batch_size, len(chunk))))

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
    queue.put(dict(description=f"Querying source primary keys for BOSS {run2d} spectra", total=q.count(), completed=0))
    source_pk_by_catalogid = {}
    for pk, *catalogids in q.iterator():
        for catalogid in catalogids:
            source_pk_by_catalogid[catalogid] = pk
        queue.put(dict(advance=1))
    
    n_warnings = 0
    for each in spectrum_data:
        try:
            each["source_pk"] = source_pk_by_catalogid[each["catalogid"]]
        except:
            # log warning?
            n_warnings += 1
        finally:
            for source_key_only in source_keys_only:
                each.pop(source_key_only, None)

            try:
                # Missing catalogid!
                if np.all(each["catalogid"].mask):
                    each["catalogid"] = -1 # cannot be null
            except:
                None


    #if n_warnings > 0:
    #    log.warning(f"There were {n_warnings} spectra with no source_pk, probably because of missing or fake catalogids")
    
    pks = upsert_many(
        BossVisitSpectrum,
        BossVisitSpectrum.pk,
        spectrum_data,
        batch_size,
        queue,
        f"Upserting BOSS {run2d} spectra"
    )

    # Assign spectrum_pk values to any spectra missing it.
    N = len(pks)
    if pks:
        queue.put(dict(description=f"Assigning primary keys to BOSS {run2d} spectra", total=N, completed=0))
        N_assigned = 0
        for batch in chunked(pks, batch_size):
            B = (
                BossVisitSpectrum
                .update(
                    spectrum_pk=Case(None, (
                        (BossVisitSpectrum.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(batch)
                    ))
                )
                .where(BossVisitSpectrum.pk.in_(batch))
                .execute()
            )
            queue.put(dict(advance=B))
            N_assigned += B
        #log.info(f"There were {N} spectra inserted and we assigned {N_assigned} spectra with new spectrum_pk values")
    
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
            log.warning(f"Skipping line '{line}' because not a valid line")
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
    from astra.models.base import database
    from astra.models.boss import BossVisitSpectrum
    from astra.models.source import Source
    from astra.migrations.utils import enumerate_new_spectrum_pks, upsert_many, NoQueue

    if queue is None:
        queue = NoQueue()

    q = (
        BossVisitSpectrum
        .select()
        .where(BossVisitSpectrum.alt.is_null() & (BossVisitSpectrum.catalogid > 0))
        .limit(limit)
    )
    
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

    queue.put(dict(total=q.count()))
    specFulls, futures, total = ({}, [], 0)
    for chunk in chunked(q, batch_size):
        futures.append(executor.submit(_migrate_specfull_metadata, chunk, fields))
        #for total, spec in enumerate(chunk, start=1 + total):
        for spec in chunk:
            specFulls[spec.pk] = spec
    
    defaults = {
        "n_guide": -1,
        "airtemp": np.nan,
        "dewpoint": np.nan,
        "n_std": -1
    }

    all_missing_counts = {}
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
        
        queue.put(dict(advance=1))

    #if all_missing_counts:
    #    log.warning(f"There were missing keys:")
    #    for name, count in all_missing_counts.items():
    #        log.warning(f"\t{name}: {count} missing")

    queue.put(dict(total=len(specFulls), completed=0, description="Ingesting specFull metadata"))
    for chunk in chunked(specFulls.values(), batch_size):
        (
            BossVisitSpectrum  
            .bulk_update(
                chunk,
                fields=list(fields.keys())
            )
        )
        queue.put(dict(advance=batch_size))
    
    queue.put(Ellipsis)
    return None
