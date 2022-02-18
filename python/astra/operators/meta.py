
from astropy.io import fits
from sdss_access import SDSSPath
from sqlalchemy.sql import exists
from tqdm import tqdm

from astra.database import (astradb, session as astra_session)
from astra.database.sdssdb import (apogee_drpdb, catalogdb, session as sdssdb_session)
from astra.utils import (log, flatten)
from astra.database.utils import deserialize_pks

def add_meta_to_new_task_instances_without_meta():
    """
    Add meta to new task instances without meta.
    """
    pk, = astra_session.query(
            astradb.TaskInstance.pk
        ).filter(
            astradb.TaskInstanceMeta.ti_pk == astradb.TaskInstance.pk
        ).order_by(
            astradb.TaskInstance.pk.desc()
        ).first()
    
    stmt = exists().where(astradb.TaskInstance.pk == astradb.TaskInstanceMeta.ti_pk)
    q = astra_session.query(astradb.TaskInstance.pk).filter(astradb.TaskInstance.pk > pk).filter(~stmt)
    for pk in tqdm(q.yield_per(1), total=q.count(), desc="Adding metadata to task instances"):
        try:
            add_meta_to_task_instance(pk)
        except:
            log.exception(f"Unable to add meta to task instance with pk {pk}")
            continue
    return None



def add_meta_to_task_instances_without_meta():
    """
    Add meta to task instances without meta.
    """
    failed, total = (0, count_task_instances_without_meta())
    for pk in tqdm(yield_task_instance_pks_without_meta(), total=total, desc="Adding metadata to task instances"):
        try:
            add_meta_to_task_instance(pk)
        except:
            log.exception(f"Unable to add meta to task instance with pk {pk}")
            continue
    log.info(f"Added meta to {total - failed} task instances")
    return None


def _query_task_instances_without_meta(sdss5_only=False):
    stmt = exists().where(astradb.TaskInstance.pk == astradb.TaskInstanceMeta.ti_pk)
    q = astra_session.query(astradb.TaskInstance.pk).filter(~stmt)
    if sdss5_only:
        log.warning("Only doing SDSS5 task instances at the moment")
        # Only do sdss5 things so far.
        q = q.filter(astradb.TaskInstanceParameter.ti_pk == astradb.TaskInstance.pk)\
            .filter(astradb.TaskInstanceParameter.parameter_pk.in_((438829, 494337, 493889)))
    return q


def yield_task_instance_pks_without_meta():
    """ Yield primary keys of task instances that do not have metadata. """
    q = _query_task_instances_without_meta()
    yield from (pk for pk, in q.yield_per(1))


        

def count_task_instances_without_meta():
    """Count the number of task instances without meta""" 
    return _query_task_instances_without_meta().count()
    

def add_meta_to_task_instances(pks):
    pks = deserialize_pks(pks, flatten=True)
    for pk in pks:
        add_meta_to_task_instance(pk)
    return pks


def add_meta_to_task_instance(ti_pk):
    """
    Update the task instance meta table for a given task instance.
    """

    ti = astra_session.query(astradb.TaskInstance).filter_by(pk=ti_pk).first()
    parameters = ti.parameters

    try:
        release = parameters["release"].lower()
        filetype = parameters["filetype"]
    except KeyError:
        raise KeyError(f"Either missing `release` or `filetype` parameter for task instance {ti}")
    
    if release in ("sdss5", None, "null"):
        is_sdss5 = True
    elif release in ("dr17", "dr16"):
        is_sdss5 = False
    else:
        raise ValueError(f"Cannot figure out if {ti} is SDSS V or not based on release value '{release}'")
        
    # Is it a SDSS-V object? If so we can get information from apogee_drpdb / boss tables first.        
    is_apogee = filetype in ("apVisit", "apStar")
    is_boss = filetype in ("spec", )

    if not is_apogee and not is_boss:
        raise ValueError(f"Don't know what to do with filetype of '{filetype}'")
    
    tree = SDSSPath(release=release)

    meta = dict(ti_pk=ti_pk)
    if is_sdss5:
        if is_apogee:
            # Need the apogee_drp_star_pk and apogee_drp_visit_pks.
            star_pk, catalogid, gaia_dr2_source_id = sdssdb_session.query(
                apogee_drpdb.Star.pk,
                apogee_drpdb.Star.catalogid,
                apogee_drpdb.Star.gaiadr2_sourceid    
            ).filter_by(apogee_id=parameters["obj"]).first()

            if filetype == "apVisit":
                # Match on a single visit.
                visit_pks = sdssdb_session.query(apogee_drpdb.Visit.pk).filter_by(
                    apogee_id=parameters["obj"], # TODO: raise with Nidever
                    telescope=parameters["telescope"],
                    fiberid=parameters["fiber"], # TODO: raise with Nidever
                    plate=parameters["plate"],
                    field=parameters["field"],
                    mjd=parameters["mjd"],
                    apred_vers=parameters["apred"] # TODO: raise with Nidever
                ).one_or_none()

                visit_paths = [tree.full(**parameters)]
                
            elif filetype == "apStar":
                # Get all visits.
                visits = sdssdb_session.query(
                    apogee_drpdb.Visit.pk,
                    apogee_drpdb.Visit.mjd,
                    apogee_drpdb.Visit.field,
                    # We have apred.
                    apogee_drpdb.Visit.fiberid.label("fiber"),
                    # We have telescope
                    apogee_drpdb.Visit.plate
                ).filter_by(apogee_id=parameters["obj"]).all()

                # We will need the paths too.
                visit_pks = []
                visit_paths = []
                for visit_pk, mjd, field, fiber, plate in visits:
                    visit_pks.append(visit_pk)
                    visit_paths.append(
                        tree.full(
                            filetype="apVisit",
                            apred=parameters["apred"],
                            telescope=parameters["telescope"],
                            mjd=mjd,
                            field=field,
                            fiber=fiber,
                            plate=plate
                        )
                    )

            else:
                raise ValueError(f"Don't know what to do with SDSS-V APOGEE filetype of '{filetype}'")

            visit_pks = flatten(visit_pks)

            # For the visit files we have to open them to get the number of pixels.
            apogee_drp_visit_naxis1 = []
            for visit_path in visit_paths:
                try:
                    apogee_drp_visit_naxis1.append(fits.getval(visit_path, "NAXIS1", ext=1))
                except:
                    log.exception(f"Could not get NAXIS1 from path {visit_path}")
                    apogee_drp_visit_naxis1.append(-1)

            meta.update(
                catalogid=catalogid,
                gaia_dr2_source_id=gaia_dr2_source_id,
                apogee_drp_star_pk=star_pk,
                apogee_drp_visit_pks=visit_pks,
                apogee_drp_visit_naxis1=apogee_drp_visit_naxis1,
            )

        elif is_boss:
            # We need catalogid, gaia_dr2_source_id, and catalogdb_sdssv_boss_spall_pkey.
            catalogid = parameters["catalogid"]
            if catalogid > 0:
                gaia_dr2_source_id, = sdssdb_session.query(catalogdb.TICV8.gaia_int)\
                                                    .filter(catalogdb.TICV8.id == catalogdb.CatalogToTICV8.target_id)\
                                                    .filter(catalogdb.CatalogToTICV8.catalogid == catalogid).one_or_none()
            else:
                gaia_dr2_source_id = None

            
            pkey, = sdssdb_session.query(catalogdb.SDSSVBossSpall.pkey)\
                                  .filter(
                                      catalogdb.SDSSVBossSpall.catalogid == catalogid,
                                      catalogdb.SDSSVBossSpall.run2d == parameters["run2d"],
                                      catalogdb.SDSSVBossSpall.plate == parameters["plate"],
                                      catalogdb.SDSSVBossSpall.mjd == parameters["mjd"],
                                      catalogdb.SDSSVBossSpall.fiberid == parameters["fiberid"]
                                  ).one_or_none()
            meta.update(
                catalogid=catalogid,
                gaia_dr2_source_id=gaia_dr2_source_id,
                catalogdb_sdssv_boss_spall_pkey=pkey
            )
    
    else:
        # Need to get information from elsewhere.
        if is_apogee:
            if filetype == "apVisit":
                catalogid = sdssdb_session.query(catalogdb.CatalogToSDSSDR16ApogeeStar.catalogid).filter(
                    catalogdb.SDSSDR16ApogeeStar.target_id == catalogdb.CatalogToSDSSDR16ApogeeStar.target_id,
                    catalogdb.SDSSDR16ApogeeStar.apstar_id == catalogdb.SDSSDR16ApogeeStarVisit.apstar_id,
                    catalogdb.SDSSDR16ApogeeStarVisit.visit_id == catalogdb.SDSSDR16ApogeeVisit.visit_id,
                    catalogdb.SDSSDR16ApogeeVisit.plate == parameters["plate"],
                    catalogdb.SDSSDR16ApogeeVisit.mjd == parameters["mjd"],
                    catalogdb.SDSSDR16ApogeeVisit.fiberid == parameters["fiber"], # Raise with Nidever
                    catalogdb.SDSSDR16ApogeeVisit.location_id == parameters["field"],
                    catalogdb.SDSSDR16ApogeeVisit.apred_version == parameters["apred"] # Raise with Nidever
                ).first_or_none()
                        
            elif filetype == "apStar":
                catalogid = sdssdb_session.query(catalogdb.CatalogToSDSSDR16ApogeeStar.catalogid).filter(
                    catalogdb.SDSSDR16ApogeeStar.apogee_id.label("obj") == parameters["obj"],
                    catalogdb.SDSSDR16ApogeeStar.field == parameters["field"],
                    catalogdb.SDSSDR16ApogeeStar.telescope == parameters["telescope"],
                    catalogdb.SDSSDR16ApogeeStar.apstar_id == catalogdb.CatalogToSDSSDR16ApogeeStar.target_id
                ).one_or_none()
                if catalogid is not None:
                    catalogid, = catalogid

            else:
                raise ValueError(f"Don't know what to do with SDSS-IV APOGEE filetype of '{filetype}'")

            if catalogid > 0:
                gaia_dr2_source_id = sdssdb_session.query(catalogdb.TICV8.gaia_int)\
                                                   .filter(catalogdb.TICV8.id == catalogdb.CatalogToTICV8.target_id)\
                                                   .filter(catalogdb.CatalogToTICV8.catalogid == catalogid).one_or_none()
                if gaia_dr2_source_id is not None:
                    gaia_dr2_source_id, = gaia_dr2_source_id
            else:
                gaia_dr2_source_id = None

            meta.update(
                catalogid=catalogid,
                gaia_dr2_source_id=gaia_dr2_source_id
            )

        else:
            raise NotImplementedError(f"Can only retrieve metadata for APOGEE products in SDSS-IV.")


    # Add generic information from the catalog, if we have a valid catalogid.
    if meta["catalogid"] > 0:
        row = sdssdb_session.query(catalogdb.Catalog).filter(catalogdb.Catalog.catalogid==meta["catalogid"]).one_or_none()

        meta.update(
            iauname=row.iauname,
            ra=row.ra,
            dec=row.dec,
            pmra=row.pmra,
            pmdec=row.pmdec,
            parallax=row.parallax,
            catalog_lead=row.lead,
            catalog_version_id=row.version_id,
        )
    
    # Create or update an entry in the database.
    instance = astradb.TaskInstanceMeta(**meta)
    with astra_session.begin(subtransactions=True):
        astra_session.merge(instance)

    return instance