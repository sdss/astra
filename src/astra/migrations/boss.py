from typing import Optional
from astropy.table import Table
from astropy.time import Time
from tqdm import tqdm
import numpy as np
import subprocess
import concurrent.futures

from astra.utils import log, expand_path
from astra.models.base import database
from astra.models.boss import BossVisitSpectrum
from astra.models.source import Source
from astra.migrations.utils import enumerate_new_spectrum_pks, upsert_many

from peewee import (
    chunked,
    JOIN, 
    Case,
    FloatField,
    IntegerField
)

def migrate_spectra_from_spall_file(
    run2d: Optional[str] = "v6_1_1",
    gzip: Optional[bool] = True,        
    limit: Optional[int] = None,
    batch_size: Optional[int] = 1000
):
    """
    Migrate all new BOSS visit information (`specFull` files) stored in the spAll file, which is generated
    by the SDSS-V BOSS data reduction pipeline.
    """

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


    path = f"$BOSS_SPECTRO_REDUX/{run2d}/spAll-{run2d}.fits"
    if gzip:
        path += ".gz"

    spAll = Table.read(expand_path(path))
    spAll.sort(["CATALOGID"])

    if limit is not None:
        spAll = spAll[:limit]

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
        "ZWARNING": ("zwarning_flags", lambda x: x or 0),
        "EXPTIME": "exptime",

        # Not yet done: gri_gaia_transform, because it is accidentally missing from the IPL3 files
        "AIRMASS": "airmass",
        "SEEING50": "seeing",
    
        "OBS": ("telescope", lambda x: f"{x.lower()}25m"),
        "MOON_DIST": ("moon_dist_mean", lambda x: np.mean(tuple(map(float, x.split())))),
        "MOON_PHASE": ("moon_phase_mean", lambda x: np.mean(tuple(map(float, x.split())))),

        "FIELD": "fieldid",
        "MJD": "mjd",
        "CATALOGID": "catalogid",
        "HEALPIX": "healpix",
        "DELTA_RA_LIST": ("delta_ra", lambda x: np.array(x.split(), dtype=float)),
        "DELTA_DEC_LIST": ("delta_dec", lambda x: np.array(x.split(), dtype=float)),
        "SN_MEDIAN_ALL": "snr",

        # Some additional identifiers that we don't necessarily need, but will take for now

        "CATALOGID_V0": "catalogid_v0",
        "CATALOGID_V0P5": "catalogid_v0p5",
        "SDSS_ID": "sdss_id",
        "GAIA_ID": "gaia_dr2_source_id",
        "FIRSTCARTON": "carton_0"
    }
    source_keys_only = ("catalogid_v0", "catalogid_v0p5", "sdss_id", "gaia_dr2_source_id", "carton_0") 

    spectrum_data = []
    for i, row in enumerate(tqdm(spAll)):

        row_data = dict(zip(row.keys(), row.values()))
        
        sanitised_row_data = {
            "release": "sdss5",
            "run2d": run2d,
            "filetype": "specFull",
        }
        for from_key, to in translations.items():
            if isinstance(to, str):
                sanitised_row_data[to] = row_data[from_key]
            else:
                to_key, to_callable = to
                sanitised_row_data[to_key] = to_callable(row_data[from_key])
        
        offset = np.abs(sanitised_row_data["delta_ra"]) + np.abs(sanitised_row_data["delta_dec"])
        sanitised_row_data["fiber_offset"] = np.any(offset > 0)
        spectrum_data.append(sanitised_row_data)

    # We need to get sdss_id and catalog information for each source.
    source_data = {}
    with tqdm(total=len(spectrum_data), desc="Linking to Catalog") as pb:
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
            
            pb.update(batch_size)
    

    # Upsert the sources
    with database.atomic():
        with tqdm(desc="Upserting sources", total=len(source_data)) as pb:
            for chunk in chunked(source_data.values(), batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    log.info(f"Getting data for sources")

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

    source_pk_by_catalogid = {}
    for pk, *catalogids in q.iterator():
        for catalogid in catalogids:
            source_pk_by_catalogid[catalogid] = pk
    
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


    if n_warnings > 0:
        log.warning(f"There were {n_warnings} spectra with no source_pk, probably because of missing or fake catalogids")
    
    pks = upsert_many(
        BossVisitSpectrum,
        BossVisitSpectrum.pk,
        spectrum_data,
        batch_size,
        desc="Upserting spectra"
    )

    # Assign spectrum_pk values to any spectra missing it.
    N = len(pks)
    if pks:
        with tqdm(total=N, desc="Assigning primary keys to spectra") as pb:
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
                pb.update(B)
                N_assigned += B

        log.info(f"There were {N} spectra inserted and we assigned {N_assigned} spectra with new spectrum_pk values")
    else:
        log.info(f"No new spectra inserted")
    
    return N



def _migrate_specfull_metadata(spectra, fields, raise_exceptions=False, full_output=False):

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
    where=(BossVisitSpectrum.alt.is_null() & (BossVisitSpectrum.catalogid > 0)),
    max_workers: Optional[int] = 8,
    limit: Optional[int] = None,
    batch_size: Optional[int] = 100,
):

    q = (
        BossVisitSpectrum
        .select()
    )
    if where:
        q = q.where(where)
    
    q = (
        q
        .limit(limit)
        .iterator()
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

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)

    specFulls, futures, total = ({}, [], 0)
    with tqdm(total=limit or 0, desc="Submitting work", unit="spectra") as pb:
        for chunk in chunked(q, batch_size):
            futures.append(executor.submit(_migrate_specfull_metadata, chunk, fields))
            for total, spec in enumerate(chunk, start=1 + total):
                specFulls[spec.pk] = spec
                pb.update()

    defaults = {
        "n_guide": -1,
        "airtemp": np.nan,
        "dewpoint": np.nan,
        "n_std": -1
    }

    all_missing_counts = {}
    with tqdm(total=total, desc="Collecting headers", unit="spectra") as pb:
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
                pb.update()

    if all_missing_counts:
        log.warning(f"There were missing keys:")
        for name, count in all_missing_counts.items():
            log.warning(f"\t{name}: {count} missing")

    with tqdm(total=total, desc="Updating", unit="spectra") as pb:     
        for chunk in chunked(specFulls.values(), batch_size):
            pb.update(
                BossVisitSpectrum  
                .bulk_update(
                    chunk,
                    fields=list(fields.keys())
                )
            )

    return pb.n
