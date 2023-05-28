
from peewee import chunked
from typing import Optional
from astra.models.apogee import ApogeeVisitSpectrum, Spectrum
from astra.models.base import database
from astra.utils import flatten, log
from tqdm import tqdm


def migrate_apvisit_from_sdss5_apogee_drpdb(release: Optional[str] = "sdss5", batch_size: Optional[int] = 100):
    """
    Migrate all new APOGEE visit information (`apVisit` files) stored in the SDSS-V database, which is reported
    by the SDSS-V APOGEE data reduction pipeline.
    
    :param release: [optional]
        The `release` keyword to assign to the `astra.models.apogee.ApogeeVisitSpectrum` rows.
    
    :param batch_size: [optional]
        The batch size to use when upserting data.
    
    :returns:
        A tuple of new spectrum identifiers (`astra.models.apogee.ApogeeVisitSpectrum.spectrum_id`)
        that were inserted.
    """

    from astra.migrations.sdss5db.apogee_drpdb import Visit

apred_vers
catalogid,
gaiadr2_sourceid
starflag
dateobs
jd
created
v_apred
assigned
on_target
valid
cadence
program
category
exptime
nframes

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
            Visit.pk.alias("apvisit_pk"),
            Visit.apogee_id.alias("sdss4_dr17_apogee_id"),
            
        )
        .dicts()
    )

    N = q.count()
    
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
    
    data = [
        {
            "spectrum_id": spectrum_id,
            "release": release,
            **row
        }
        for spectrum_id, row in zip(spectrum_ids, q)
    ]

    log.info(f"Data prepared.")

    new_spectrum_ids = []
    with database.atomic():
        with tqdm(desc="Upserting", unit="spectra", total=N) as pb:
            for chunk in chunked(data, batch_size):
                new_spectrum_ids.extend(
                    flatten(
                        ApogeeVisitSpectrum
                        .insert_many(chunk)
                        .on_conflict_ignore()
                        .returning(ApogeeVisitSpectrum.spectrum_id)
                        .tuples()
                        .execute()
                    )
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    return tuple(new_spectrum_ids)