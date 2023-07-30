import concurrent
import subprocess
from peewee import chunked, fn, JOIN
from typing import Optional
from tqdm import tqdm
from astra.models.apogee import ApogeeVisitSpectrum, Spectrum, ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar
from astra.models.source import Source
from astra.models.base import database
from astra.utils import expand_path, flatten, log

def migrate_apogee_obj_from_source(batch_size: Optional[int] = 100, limit: Optional[int] = None):

    q = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum.spectrum_id,
            Source.sdss4_dr17_apogee_id
        )
        .join(Source, on=(ApogeeVisitSpectrum.source_id == Source.id))
        .where(
            ApogeeVisitSpectrum.obj.is_null()
        &   Source.sdss4_dr17_apogee_id.is_null(False)
        )
        .tuples()
    )

    total = limit or q.count()
    with tqdm(total=total, desc="Updating", unit="spectra") as pb:        
        for chunk in chunked(q.iterator(), batch_size):
            objs = { spectrum_id: obj for spectrum_id, obj in chunk }
            q = (
                ApogeeVisitSpectrum
                .select()
                .where(ApogeeVisitSpectrum.spectrum_id.in_(list(objs.keys())))
            )
            batch = list(q)
            for spectrum in batch:
                spectrum.obj = objs[spectrum.spectrum_id]
            
            pb.update(
                ApogeeVisitSpectrum
                .bulk_update(
                    batch,
                    fields=[ApogeeVisitSpectrum.obj]
                )
            )

    return pb.n
                  

def migrate_sdss4_dr17_apvisit_from_sdss5_catalogdb(batch_size: Optional[int] = 100, limit: Optional[int] = None):
    """
    Migrate all SDSS4 DR17 APOGEE visit information (`apVisit` files) stored in the SDSS-V database.
    
    :param batch_size: [optional]
        The batch size to use when upserting data.
    
    :returns:
        A tuple of new spectrum identifiers (`astra.models.apogee.ApogeeVisitSpectrum.spectrum_id`)
        that were inserted.
    """
    
    from astra.migrations.sdss5db.catalogdb import (
        Catalog,
        SDSS_DR17_APOGEE_Allvisits as Visit,
        SDSS_DR17_APOGEE_Allstarmerge as Star,
        CatalogToGaia_DR3,
        CatalogToGaia_DR2
    )

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
            Visit.starflag.alias("spectrum_flags"),
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),
            Visit.snr,

            Visit.file,

            # Source information
            Star.gaia_source_id.alias("gaia_dr3_source_id"),
            CatalogToGaia_DR2.target.alias("gaia_dr2_source_id"),
            CatalogToGaia_DR3.catalog.alias("sdss5_catalogid_v1"),
            Catalog.ra,
            Catalog.dec,
            Catalog.version_id.alias("version_id"),
            Catalog.lead,
            Star.apogee_id.alias("sdss4_dr17_apogee_id"),
            Visit.apogee_target1.alias("sdss4_apogee_target1_flags"),
            Visit.apogee_target2.alias("sdss4_apogee_target2_flags"),
            Visit.apogee2_target1.alias("sdss4_apogee2_target1_flags"),
            Visit.apogee2_target2.alias("sdss4_apogee2_target2_flags"),
            Visit.apogee2_target3.alias("sdss4_apogee2_target3_flags"),

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
            Visit.rv_chi2.alias("doppler_rchisq"),
            Visit.ccfwhm,
            Visit.autofwhm,
            Visit.n_components,
            Visit.rv_flag.alias("doppler_flags"),
        )
        .join(Star, JOIN.LEFT_OUTER, on=(Star.apogee_id == Visit.apogee_id))
        .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(Star.gaia_source_id == CatalogToGaia_DR3.target))
        .join(Catalog, JOIN.LEFT_OUTER, on=(Catalog.catalogid == CatalogToGaia_DR3.catalog))
        .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(Catalog.catalogid == CatalogToGaia_DR2.catalog))
        .limit(limit)
        .dicts()
    )

    q = list(q)
    
    N = limit or len(q)    

    spectrum_ids = []
    with database.atomic():
        # Need to chunk this to avoid SQLite limits.
        with tqdm(desc="Assigning spectrum identifiers", unit="spectra", total=N) as pb:
            for chunk in chunked([{"spectrum_type_flags": 0}] * N, batch_size):                
                spectrum_ids.extend(
                    flatten(
                        Spectrum
                        .insert_many(chunk)
                        .returning(Spectrum.spectrum_id)
                        .tuples()
                        .execute()
                    )
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    log.info(f"Spectrum IDs created. Preparing data for ingestion.")

    source_only_keys = (
        "sdss5_catalogid_v1",
        "gaia_dr3_source_id",
        "gaia_dr2_source_id",
        "sdss4_dr17_apogee_id",
        "ra",
        "dec",
        "version_id",
        "lead",
        "sdss4_apogee_target1_flags",
        "sdss4_apogee_target2_flags",
        "sdss4_apogee2_target1_flags",
        "sdss4_apogee2_target2_flags",
        "sdss4_apogee2_target3_flags",
    )
    targeting_keys = [k for k in source_only_keys if k.startswith("sdss4_apogee")]

    source_data, spectrum_data = ({}, [])
    for spectrum_id, row in zip(spectrum_ids, q):
        basename = row.pop("file")
        
        apogee_id = row["sdss4_dr17_apogee_id"]

        if row["telescope"] == "apo1m":
            row["reduction"] = row["sdss4_dr17_apogee_id"]
        
        # TODO: Should we just be using APOGEE_ID as a unique identifier?
        unique_identifier = "_".join([apogee_id])
        this_source_data = dict(zip(source_only_keys, [row.pop(k) for k in source_only_keys]))

        for key in targeting_keys:        
            this_source_data[key] = max(0, this_source_data[key])

        # TODO: merge targeting flags together when they are different
        if unique_identifier in source_data:
            for k in targeting_keys:
                # Bitwise OR on targeting flags
                source_data[unique_identifier][k] |= this_source_data[k]
        else:
            source_data[unique_identifier] = this_source_data
        
        row["plate"] = row["plate"].lstrip()
        
        spectrum_data.append({
            "apogee_id": apogee_id, # Will be removed later, just for matching source_id.
            "release": "dr17",
            "apred": "dr17",
            "spectrum_id": spectrum_id,
            "prefix": basename.lstrip()[:2],
            **row
        })

    # Upsert the sources
    with database.atomic():
        with tqdm(desc="Upserting source information", total=len(source_data)) as pb:
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
            Source.id,
            Source.sdss4_dr17_apogee_id
        )
        .tuples()
        .iterator()
    )
    source_ids = {}
    for source_id, sdss4_dr17_apogee_id in q:
        source_ids[sdss4_dr17_apogee_id] = source_id

    for each in spectrum_data:
        each["source_id"] = source_ids[each.pop("apogee_id")]
    
    # Upsert the spectra
    spectrum_ids = _upsert_many(
        ApogeeVisitSpectrum,
        ApogeeVisitSpectrum.spectrum_id,
        spectrum_data,
        batch_size,
        desc="Upserting spectra"
    )
    log.info(f"Ingested {len(spectrum_ids)} spectra")
    return len(spectrum_ids)
    


def migrate_apvisit_from_sdss5_apogee_drpdb(
    apred: Optional[str] = None,
    batch_size: Optional[int] = 100, 
    limit: Optional[int] = None,
    full_output=False
):
    """
    Migrate all new APOGEE visit information (`apVisit` files) stored in the SDSS-V database, which is reported
    by the SDSS-V APOGEE data reduction pipeline.

    :param apred: [optional]
        Limit the ingestion to spectra with a specified `apred` version.
                
    :param batch_size: [optional]
        The batch size to use when upserting data.
    
    :param limit: [optional]
        Limit the ingestion to `limit` spectra.

    :returns:
        A tuple of new spectrum identifiers (`astra.models.apogee.ApogeeVisitSpectrum.spectrum_id`)
        that were inserted.
    """

    from astra.migrations.sdss5db.apogee_drpdb import Visit, RvVisit
    from astra.migrations.sdss5db.catalogdb import (
        Catalog,
        CatalogToGaia_DR3,
        CatalogToGaia_DR2
    )

    '''
    min_rv_visit_pk = 0
    if restrict_to_new_visits:
        log.warning("`restrict_to_new_visits` is not fully tested yet; you could miss some spectra")
        try:
            min_rv_visit_pk = ApogeeVisitSpectrum.select(fn.MAX(ApogeeVisitSpectrum.rv_visit_pk)).scalar() or 0
        except:
            None

    sq = (
        RvVisit
        .select(
            RvVisit.visit_pk, 
            fn.MAX(RvVisit.starver).alias('max')
        )
        .where(RvVisit.pk > min_rv_visit_pk)
    )
    if apred is not None:
        sq = sq.where(RvVisit.apred_vers == apred)
    
    sq = (
        sq
        .group_by(RvVisit.visit_pk)
        .alias("rv_visit")
    )

    cte = (
        RvVisit
        .select(
            RvVisit.pk,
            RvVisit.visit_pk,
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
        )            
        .join(
            sq, 
            on=(
                (RvVisit.visit_pk == sq.c.visit_pk)
            &   (RvVisit.starver == sq.c.max)
            )
        )
        .alias("cte")
    )

    # Main query
    q = (
        Visit
        .select(
            Visit.apred,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.fiber,
            Visit.prefix,
            Visit.obj,
            Visit.pk.alias("visit_pk"),
            Visit.dateobs.alias("date_obs"),
            Visit.jd,
            Visit.exptime,
            Visit.nframes.alias("n_frames"),
            Visit.assigned,
            Visit.on_target,
            Visit.valid,
            Visit.starflag.alias("spectrum_flags"),
            Visit.catalogid.alias("input_catalogid"),
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),

            # Source information,
            Visit.gaiadr2_sourceid.alias("gaia_dr2_source_id"),
            CatalogToGaia_DR3.target_id.alias("gaia_dr3_source_id"),
            Catalog.catalogid.alias("sdss5_catalogid_v1"),
            Catalog.version_id.alias("version_id"),
            Catalog.lead,
            Catalog.ra,
            Catalog.dec,
            
            cte.c.bc,
            cte.c.vrel.alias("v_rel"),
            cte.c.vrelerr.alias("e_v_rel"),
            cte.c.vrad.alias("v_rad"),
            cte.c.chisq.alias("doppler_rchisq"),
            cte.c.rv_teff.alias("doppler_teff"),
            cte.c.rv_tefferr.alias("doppler_e_teff"),
            cte.c.rv_logg.alias("doppler_logg"),
            cte.c.rv_loggerr.alias("doppler_e_logg"),
            cte.c.rv_feh.alias("doppler_fe_h"),
            cte.c.rv_feherr.alias("doppler_e_fe_h"),
            cte.c.xcorr_vrel.alias("xcorr_v_rel"),
            cte.c.xcorr_vrelerr.alias("xcorr_e_v_rel"),
            cte.c.xcorr_vrad.alias("xcorr_v_rad"),
            cte.c.n_components,
            cte.c.pk.alias("rv_visit_pk")
        )
        .join(cte, JOIN.LEFT_OUTER, on=(cte.c.visit_pk == Visit.pk))
        .switch(Visit)
        .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(Visit.gaiadr2_sourceid == CatalogToGaia_DR2.target_id))
        .join(Catalog, JOIN.LEFT_OUTER, on=(Catalog.catalogid == CatalogToGaia_DR2.catalogid))
        .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(Catalog.catalogid == CatalogToGaia_DR3.catalogid))
        .switch(Visit)
        .where(cte.c.pk > min_rv_visit_pk)
    )

    if apred is not None:
        q = q.where(Visit.apred_vers == apred)        
    '''

    ssq = (
        RvVisit
        .select(
            RvVisit.visit_pk,
            fn.MAX(RvVisit.starver).alias("max")
        )
        .where(
            (RvVisit.apred_vers == apred)
        &   (RvVisit.catalogid > 0) # Some RM_COSMOS fields with catalogid=0 (e.g., apogee_drp.visit = 7494220)
        )  
        .group_by(RvVisit.visit_pk)
        .order_by(RvVisit.visit_pk.desc())
    )
    sq = (
        RvVisit
        .select(
            RvVisit.pk,
            RvVisit.visit_pk,
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
        )
        .join(
            ssq, 
            on=(
                (RvVisit.visit_pk == ssq.c.visit_pk)
            &   (RvVisit.starver == ssq.c.max)
            )
        )
    )

    q = (
        Visit.select(
            Visit.apred,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.fiber,
            Visit.prefix,
            Visit.obj,
            Visit.pk.alias("visit_pk"),
            Visit.dateobs.alias("date_obs"),
            Visit.jd,
            Visit.exptime,
            Visit.nframes.alias("n_frames"),
            Visit.assigned,
            Visit.on_target,
            Visit.valid,
            Visit.starflag.alias("spectrum_flags"),
            Visit.catalogid.alias("input_catalogid"),
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),

            # Source information,
            Visit.gaiadr2_sourceid.alias("gaia_dr2_source_id"),
            CatalogToGaia_DR3.target_id.alias("gaia_dr3_source_id"),
            Catalog.catalogid.alias("sdss5_catalogid_v1"),
            Catalog.version_id.alias("version_id"),
            Catalog.lead,
            Catalog.ra,
            Catalog.dec,
            Visit.jmag.alias("j_mag"),
            Visit.jerr.alias("e_j_mag"),            
            Visit.hmag.alias("h_mag"),
            Visit.herr.alias("e_h_mag"),
            Visit.kmag.alias("k_mag"),
            Visit.kerr.alias("e_k_mag"),
            
            sq.c.bc,
            sq.c.vrel.alias("v_rel"),
            sq.c.vrelerr.alias("e_v_rel"),
            sq.c.vrad.alias("v_rad"),
            sq.c.chisq.alias("doppler_rchisq"),
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
            sq.c.pk.alias("rv_visit_pk")            
        )
        .join(sq, on=(Visit.pk == sq.c.visit_pk))
        .switch(Visit)
        # Need to join by Catalog on the visit catalogid (not gaia DR2) because sometimes Gaia DR2 value is 0
        # Doing it like this means we might end up with some `sdss5_catalogid_v1` actually NOT being v1, but
        # we will have to fix that afterwards. It will be indicated by the `version_id`.
        .join(Catalog, JOIN.LEFT_OUTER, on=(Catalog.catalogid == Visit.catalogid))
        .switch(Visit)
        .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(Visit.gaiadr2_sourceid == CatalogToGaia_DR2.target_id))
        .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(CatalogToGaia_DR2.catalogid == CatalogToGaia_DR3.catalogid))
    )

    q = q.limit(limit).dicts()

    # TODO: this is bad because it basically means we have to do a huge expensive query twice
    #       we should consider writign a generator that creates Spectrum objects in bulk and yields from them
    #       so that we only create ~roughly the number of spectrum ids that we need
    N = limit or q.count()
    
    log.info(f"Bulk assigning {N} unique spectra")

    spectrum_ids = []
    with database.atomic():
        # Need to chunk this to avoid SQLite limits.
        with tqdm(desc="Assigning", unit="spectra", total=N) as pb:
            for chunk in chunked([{"spectrum_type_flags": 0}] * N, batch_size):                
                spectrum_ids.extend(
                    flatten(
                        Spectrum
                        .insert_many(chunk)
                        .returning(Spectrum.spectrum_id)
                        .tuples()
                        .execute()
                    )
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    log.info(f"Spectrum IDs created. Preparing data for ingestion.")
    
    source_only_keys = (
        "sdss5_catalogid_v1",
        "gaia_dr2_source_id",
        "gaia_dr3_source_id",
        "version_id",
        "lead",
        "ra",
        "dec",
        "j_mag",
        "e_j_mag",
        "h_mag",
        "e_h_mag",
        "k_mag",
        "e_k_mag",
    )
    source_data, spectrum_data, sdss5_catalogid_v1s = ({}, [], [])
    for spectrum_id, row in zip(spectrum_ids, tqdm(q.iterator(), total=N, desc="Extracting source data from spectra")):
        sdss5_catalogid_v1 = row["sdss5_catalogid_v1"]
        source_data.setdefault(sdss5_catalogid_v1, {})
        source_kwds = dict(zip(source_only_keys, [row.pop(k) for k in source_only_keys]))
        if source_kwds["gaia_dr2_source_id"] == 0:
            source_kwds["gaia_dr2_source_id"] = None

        for k, v in source_kwds.items():
            if v is not None:
                source_data[sdss5_catalogid_v1][k] = v
        
        spectrum_data.append({
            "spectrum_id": spectrum_id,
            "release": "sdss5",
            **row
        })
        sdss5_catalogid_v1s.append(sdss5_catalogid_v1)
    
    with database.atomic():
        with tqdm(desc="Upserting source information", total=len(source_data)) as pb:
            for chunk in chunked(source_data.values(), batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()    

    # Get corresponding source ids for each sdss5_catalogid_v1

    q = (
        Source
        .select(
            Source.id,
            Source.sdss5_catalogid_v1
        )
        .tuples()
        .iterator()
    )
    source_ids = {}
    for source_id, sdss5_catalogid_v1 in tqdm(q, total=len(source_data), desc="Linking source and catalog identifiers"):
        source_ids[sdss5_catalogid_v1] = source_id
    
    # Put the source identifier back in to the spectrum data row.
    for sdss5_catalogid_v1, sd in zip(sdss5_catalogid_v1s, spectrum_data):
        sd["source_id"] = source_ids[sdss5_catalogid_v1]

    spectrum_ids = _upsert_many(
        ApogeeVisitSpectrum,
        ApogeeVisitSpectrum.spectrum_id,
        spectrum_data,
        batch_size,
        desc="Upserting spectra"
    )
    if full_output:
        return (len(spectrum_ids), spectrum_ids)
    else:
        return len(spectrum_ids)
    

def _migrate_apvisit_metadata(apVisits, raise_exceptions=False):

    keys = ("NAXIS1", "SNR", "NCOMBINE")
    K = len(keys)
    keys_str = "|".join([f"({k})" for k in keys])

    # 80 chars per line, 150 lines -> 12000
    command_template = " | ".join([
        'hexdump -n 12000 -e \'80/1 "%_p" "\\n"\' {path}',
        f'egrep "{keys_str}"',
        f"head -n {K}"
    ])
    commands = ""
    for apVisit in apVisits:
        path = expand_path(apVisit.path)
        commands += f"{command_template.format(path=path)}\n"
    
    outputs = subprocess.check_output(commands, shell=True, text=True)
    outputs = outputs.strip().split("\n")

    if len(outputs) != (K * len(apVisits)):

        if raise_exceptions:
            raise OSError(f"Unexpected outputs from `hexdump` on {apVisits}")

        log.warning(f"Unexpected length of outputs from `hexdump`!")
        log.warning(f"Running this chunk one-by-one to be sure... this chunk goes from {apVisits[0]} to {apVisits[-1]}")
        
        # Do it manually
        all_metadata = {}
        for apVisit in apVisits:
            try:
                this_metadata = _migrate_apvisit_metadata([apVisit], raise_exceptions=True)
            except OSError:
                log.exception(f"Exception on {apVisit}:")
                # Failure mode values.
                all_metadata[apVisit.spectrum_id] = (False, -1, -1)
                continue
            else:
                all_metadata.update(this_metadata)    
        
        log.info(f"Finished chunk that goes from {apVisits[0]} to {apVisits[-1]} one-by-one")

    else:
        all_metadata = {}
        for apVisit, output in zip(apVisits, chunked(outputs, K)):
            metadata = {}
            for line in output:
                key, value = line.split("=")
                key, value = (key.strip(), value.split()[0].strip(" '"))
                if key in metadata:
                    log.warning(f"Multiple key `{key}` found in {apVisit}: {expand_path(apVisit.path)}")
                else:
                    metadata[key] = value

            # @Nidever: "if there’s 2048 then it hasn’t been dithered, if it’s 4096 then it’s dithered."
            dithered = int(metadata["NAXIS1"]) == 4096
            snr = float(metadata["SNR"])
            n_frames = int(metadata["NCOMBINE"])

            all_metadata[apVisit.spectrum_id] = (dithered, snr, n_frames)
    
    return all_metadata



def migrate_apvisit_metadata_from_image_headers(
    where=(ApogeeVisitSpectrum.dithered.is_null() | ApogeeVisitSpectrum.snr.is_null()), 
    max_workers: Optional[int] = 8, 
    batch_size: Optional[int] = 100, 
    limit: Optional[int] = None
):
    """
    Gather metadata information from the headers of apVisit files and put that information in to the database.
    
    The header keys it looks for include:
        - `SNR`: the estimated signal-to-noise ratio goes to the `ApogeeVisitSpectrum.snr` attribute
        - `NAXIS1`: for determining `ApogeeVisitSpectrum.dithered` status
        - `NCOMBINE`: for determining the number of frames combined (`ApogeeVisitSpectrum.n_frames`)
        
    :param where: [optional]
        A `where` clause for the `ApogeeVisitSpectrum.select()` statement.
    
    :param max_workers: [optional]
        Maximum number of parallel workers to use.
        
    :param batch_size: [optional]
        The batch size to use when updating `ApogeeVisitSpectrum` objects, and for chunking to workers.

    :param limit: [optional]
        Limit the number of apVisit files to query.
    """

    q = (
        ApogeeVisitSpectrum
        .select()
        .where(where)
        .limit(limit)
        .iterator()
    )

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)

    apVisits, futures, total = ({}, [], 0)
    with tqdm(total=limit or 0, desc="Retrieving metadata", unit="spectra") as pb:
        for chunk in chunked(q, batch_size):
            futures.append(executor.submit(_migrate_apvisit_metadata, chunk))
            for total, apVisit in enumerate(chunk, start=1 + total):
                apVisits[apVisit.spectrum_id] = apVisit
                pb.update()

    with tqdm(total=total, desc="Collecting results", unit="spectra") as pb:
        for future in concurrent.futures.as_completed(futures):
            for spectrum_id, (dithered, snr, n_frames) in future.result().items():
                apVisits[spectrum_id].dithered = dithered
                apVisits[spectrum_id].snr = snr
                apVisits[spectrum_id].n_frames = n_frames
                pb.update()

    with tqdm(total=total, desc="Updating", unit="spectra") as pb:     
        for chunk in chunked(apVisits.values(), batch_size):
            pb.update(
                ApogeeVisitSpectrum  
                .bulk_update(
                    chunk,
                    fields=[
                        ApogeeVisitSpectrum.dithered,
                        ApogeeVisitSpectrum.snr,
                        ApogeeVisitSpectrum.n_frames,
                    ]
                )
            )

    return pb.n


def migrate_coadd_in_apstar_from_existing_apvisits(limit=None, batch_size=100):

    q = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum.source_id,
            ApogeeVisitSpectrum.release,
            ApogeeVisitSpectrum.apred,
            ApogeeVisitSpectrum.apstar,
            ApogeeVisitSpectrum.obj,
            ApogeeVisitSpectrum.telescope,
            Source.healpix,
            ApogeeVisitSpectrum.field,
            ApogeeVisitSpectrum.prefix,
        )
        .distinct()
        .join(ApogeeCoaddedSpectrumInApStar, JOIN.LEFT_OUTER, on=(ApogeeVisitSpectrum.source_id == ApogeeCoaddedSpectrumInApStar.source_id))
        .switch(ApogeeVisitSpectrum)
        .join(Source)
        .where(
            (ApogeeCoaddedSpectrumInApStar.source_id.is_null())
        &   (Source.healpix.is_null(False))
        )
        .dicts()
        .limit(limit)
    )

    N = limit or q.count()
    
    log.info(f"Bulk assigning {N} unique spectra")

    spectrum_ids = []
    with database.atomic():
        # Need to chunk this to avoid SQLite limits.
        with tqdm(desc="Assigning", unit="spectra", total=N) as pb:
            for chunk in chunked([{"spectrum_type_flags": 0}] * N, batch_size):                
                spectrum_ids.extend(
                    flatten(
                        Spectrum
                        .insert_many(chunk)
                        .returning(Spectrum.spectrum_id)
                        .tuples()
                        .execute()
                    )
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()    

    data = []
    for spectrum_id, result in zip(spectrum_ids, q):
        result.update(spectrum_id=spectrum_id)
        data.append(result)
    
    return _upsert_many(
        ApogeeCoaddedSpectrumInApStar,
        ApogeeCoaddedSpectrumInApStar.spectrum_id,
        data,
        batch_size
    )



def migrate_apvisit_in_apstar_from_existing_apvisits(limit=None, batch_size=100):
    """
    Create `ApogeeVisitSpectrumInApStar` records for any `ApogeeVisitSpectrum` objects.

    :param limit:
        Limit the number of records.
    
    :param batch_size: [optional]
        The batch size to use when upserting data.
    """

    q = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum.source_id,
            ApogeeVisitSpectrum.spectrum_id.alias("drp_spectrum_id"),
            ApogeeVisitSpectrum.release,
            ApogeeVisitSpectrum.apred,
            ApogeeVisitSpectrum.obj,
            ApogeeVisitSpectrum.telescope,
            Source.healpix,
            Source.sdss4_dr17_apogee_id.alias("obj"),
            ApogeeVisitSpectrum.field,
            ApogeeVisitSpectrum.prefix,
            ApogeeVisitSpectrum.plate,
            ApogeeVisitSpectrum.mjd,
            ApogeeVisitSpectrum.fiber,
        )
        .join(ApogeeVisitSpectrumInApStar, JOIN.LEFT_OUTER, on=(ApogeeVisitSpectrum.spectrum_id == ApogeeVisitSpectrumInApStar.drp_spectrum_id))
        .switch(ApogeeVisitSpectrum)
        .join(Source)
        .where(
            ApogeeVisitSpectrumInApStar.drp_spectrum_id.is_null()
            &   
            (            
                (
                    # healpix is only needed for SDSS-V, not SDSS-4!
                    (ApogeeVisitSpectrum.release == "sdss5") 
                &   Source.healpix.is_null(False)
                )
            |   (ApogeeVisitSpectrum.release == "dr17")                
            )  
        )
        .dicts()
        .limit(limit)
    )

    N = limit or q.count()
    
    log.info(f"Bulk assigning {N} unique spectra")

    spectrum_ids = []
    with database.atomic():
        # Need to chunk this to avoid SQLite limits.
        with tqdm(desc="Assigning", unit="spectra", total=N) as pb:
            for chunk in chunked([{"spectrum_type_flags": 0}] * N, batch_size):                
                spectrum_ids.extend(
                    flatten(
                        Spectrum
                        .insert_many(chunk)
                        .returning(Spectrum.spectrum_id)
                        .tuples()
                        .execute()
                    )
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()    

    data = []
    for spectrum_id, result in zip(spectrum_ids, q):
        result.update(spectrum_id=spectrum_id)
        data.append(result)
    
    return _upsert_many(
        ApogeeVisitSpectrumInApStar,
        ApogeeVisitSpectrumInApStar.spectrum_id,
        data,
        batch_size
    )


def _upsert_many(model, returning, data, batch_size, desc="Upserting"):
    new_ids = []
    with database.atomic():
        with tqdm(desc=desc, total=len(data)) as pb:
            for chunk in chunked(data, batch_size):
                new_ids.extend(
                    flatten(
                        model
                        .insert_many(chunk)
                        .on_conflict_ignore()
                        .returning(returning)
                        .tuples()
                        .execute()
                    )
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    return tuple(new_ids)


"""
if __name__ == "__main__":
    from astra.models.source import Source
    from astra.models.spectrum import Spectrum
    from astra.models.apogee import ApogeeVisitSpectrum
    models = [Spectrum, ApogeeVisitSpectrum, Source]
    #database.drop_tables(models)
    if models[0].table_exists():
        database.drop_tables(models)
    database.create_tables(models)

    #from astra.migrations.apogee import migrate_apvisit_from_sdss5_apogee_drpdb, migrate_sdss4_dr17_apvisit_from_sdss5_catalogdb
    foo = migrate_sdss4_dr17_apvisit_from_sdss5_catalogdb()
"""